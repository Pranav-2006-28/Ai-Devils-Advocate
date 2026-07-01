import os
import glob
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────
# For a consumer GPU, we use a lighter Qwen model as a stand-in for the "8b" 
# parameter model, preventing Out-Of-Memory errors during QLoRA fine-tuning. 
# You can swap this to "microsoft/phi-4" if you have 24GB+ of VRAM.
MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct" 

DATA_DIR = os.path.join(os.path.dirname(__file__), "persona_data")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "adapters")

# ─────────────────────────────────────────────────────────────────────────────
# Main Training Loop
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print(f"Loading Base Model ({MODEL_ID}) in 4-bit...")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    # Find all JSONL files in the persona_data directory
    datasets = glob.glob(os.path.join(DATA_DIR, "*.jsonl"))
    
    if not datasets:
        print(f"Error: No JSONL files found in {DATA_DIR}")
        return

    # Train a separate LoRA adapter for each persona
    for data_file in datasets:
        persona_name = os.path.basename(data_file).replace(".jsonl", "")
        print(f"\n{'='*50}\nTraining LoRA Adapter for Persona: {persona_name.upper()}\n{'='*50}")
        
        # Load a fresh model for each adapter to avoid PEFT conflicts
        from transformers import BitsAndBytesConfig
        quantization_config = BitsAndBytesConfig(load_in_4bit=True)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            device_map="auto",
            torch_dtype=torch.float16,
            quantization_config=quantization_config,
        )
        model = prepare_model_for_kbit_training(model)
        
        # Load dataset
        dataset = load_dataset("json", data_files=data_file, split="train")
        
        # Updated formatting function for the new JSONL schema
        def format_prompt(example):
            system_prompt = example.get("system_prompt", "You are an AI assistant.")
            instruction = example["instruction"]
            output = example["output"]
            
            return f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n{output}<|im_end|>"
        
        # Configure LoRA
        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )
        
        from trl import SFTConfig
        
        # Training arguments
        training_args = SFTConfig(
            output_dir=os.path.join(OUTPUT_DIR, persona_name),
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
            optim="paged_adamw_32bit",
            save_steps=10,
            logging_steps=1,
            learning_rate=2e-4,
            max_steps=5,  # Keep it ultra short for the demo
            fp16=False,
            max_grad_norm=0.3,
            warmup_ratio=0.03,
            lr_scheduler_type="constant",
        )
        
        trainer = SFTTrainer(
            model=model,
            train_dataset=dataset,
            peft_config=lora_config,
            processing_class=tokenizer,
            args=training_args,
            formatting_func=format_prompt,
        )
        
        # Train
        trainer.train()
        
        # Save the adapter
        adapter_path = os.path.join(OUTPUT_DIR, persona_name, "final_adapter")
        trainer.model.save_pretrained(adapter_path)
        tokenizer.save_pretrained(adapter_path)
        print(f"Successfully saved {persona_name} adapter to {adapter_path}")
        
        # Clear VRAM for the next persona
        import gc
        del trainer
        del model
        torch.cuda.empty_cache()
        gc.collect()

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    main()
