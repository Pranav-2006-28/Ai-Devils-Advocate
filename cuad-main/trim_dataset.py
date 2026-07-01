import json
import os

def trim_dataset():
    data_path = r"C:\Users\prana\Desktop\Projects\MYPROJECT\MYPROJECT ADVISE\ai-devils-advocate\cuad-main\data\train_separate_questions.json"
    
    print(f"Loading {data_path}...")
    with open(data_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
        
    original_len = len(dataset['data'])
    print(f"Original dataset has {original_len} contracts.")
    
    # Keep only the first 80 contracts to save RAM for the MVP
    dataset['data'] = dataset['data'][:80]
    
    print(f"Trimmed dataset down to {len(dataset['data'])} contracts to prevent MemoryError.")
    
    # Save back to the same file
    with open(data_path, 'w', encoding='utf-8') as f:
        json.dump(dataset, f)
        
    print("Successfully saved trimmed dataset! Ready to train.")

if __name__ == "__main__":
    trim_dataset()
