import os
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
ADAPTERS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "..", "training", "adapters")

print("Loading base model in 4-bit...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    device_map="auto",
    torch_dtype=torch.float16,
    load_in_4bit=True,
)

# We load the base model as a PeftModel but without an active adapter yet.
# Actually, PeftModel.from_pretrained allows loading an adapter.
# To hot-swap, we can load multiple adapters into the same model!

def load_all_adapters():
    """Loads all trained adapters into memory so we can hot-swap them."""
    personas = ["lawyer", "optimist", "pessimist"]
    peft_model = None
    
    for persona in personas:
        adapter_path = os.path.join(ADAPTERS_DIR, persona, "final_adapter")
        if not os.path.exists(adapter_path):
            print(f"Warning: Adapter for {persona} not found at {adapter_path}")
            continue
            
        if peft_model is None:
            # First adapter initialization
            peft_model = PeftModel.from_pretrained(base_model, adapter_path, adapter_name=persona)
        else:
            # Load subsequent adapters into the same model
            peft_model.load_adapter(adapter_path, adapter_name=persona)
            
    return peft_model

def generate_with_adapter(model, tokenizer, prompt, persona_name):
    print(f"\n--- Generating with {persona_name.upper()} Persona ---")
    
    # 1. Hot-swap to the requested adapter!
    try:
        model.set_adapter(persona_name)
    except ValueError:
        print(f"Adapter '{persona_name}' not loaded. Falling back to base model.")
        model.disable_adapter()
        
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=150, 
            temperature=0.7, 
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
        
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return response

if __name__ == "__main__":
    model = load_all_adapters()
    
    if model is None:
        print("No adapters found. Did training complete?")
        exit(1)
        
    test_prompt = (
        "Clause to analyze:\n"
        "\"The Company may terminate this Agreement at any time for any reason upon 30 days written notice.\"\n\n"
        "Model Risk Assessment: HIGH\n\n"
        "Respond ONLY with valid JSON using the keys 'summary', 'key_points', and 'confidence'."
    )
    
    print("\n[INPUT PROMPT]")
    print(test_prompt)
    
    lawyer_resp = generate_with_adapter(model, tokenizer, test_prompt, "lawyer")
    print(lawyer_resp)
    
    optimist_resp = generate_with_adapter(model, tokenizer, test_prompt, "optimist")
    print(optimist_resp)
    
    pessimist_resp = generate_with_adapter(model, tokenizer, test_prompt, "pessimist")
    print(pessimist_resp)
