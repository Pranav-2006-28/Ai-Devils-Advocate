import json
import pandas as pd
import os
import re

def main():
    data_path = r"C:\Users\prana\Desktop\Projects\MYPROJECT\MYPROJECT ADVISE\cuad-main\data\train_separate_questions.json"
    print(f"Loading local CUAD dataset from {data_path}...")
    
    with open(data_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
        
    category_counts = {}
    print("Analyzing contracts and counting clauses...")
    
    # CUAD JSON is in SQuAD format
    for doc in dataset['data']:
        for paragraph in doc['paragraphs']:
            for qa in paragraph['qas']:
                question = qa['question']
                # The question format is usually: Highlight the parts (if any) of this contract related to "Document Name" that should be...
                match = re.search(r'related to "(.*?)"', question)
                if match:
                    category = match.group(1)
                else:
                    category = question
                
                answers = qa.get('answers', [])
                is_impossible = qa.get('is_impossible', False)
                
                # If it's not impossible and we have answers, count it
                if not is_impossible and len(answers) > 0:
                    category_counts[category] = category_counts.get(category, 0) + 1
            
    # Sort and display
    df = pd.DataFrame(list(category_counts.items()), columns=['Clause Category', 'Count'])
    df = df.sort_values(by='Count', ascending=False).reset_index(drop=True)
    
    print("\n--- All 41 Clause Category Counts ---")
    print(df.to_string())
    
    # High-risk recommendations from our playbook
    recommended = [
        "Indemnification", 
        "Limitation of Liability", 
        "Uncapped Liability",
        "Termination for Convenience",
        "Non-Compete",
        "Exclusivity",
        "Auto-Renewal",
        "Governing Law",
        "IP Ownership",
        "Most Favored Nation",
        "Change of Control",
        "Anti-Assignment",
        "Warranty Duration",
        "Insurance"
    ]
    
    print("\n--- Recommended Risk-Relevant Categories ---")
    rec_df = df[df['Clause Category'].isin(recommended)].copy()
    rec_df = rec_df.sort_values(by='Count', ascending=False).reset_index(drop=True)
    print(rec_df.to_string())

if __name__ == "__main__":
    main()
