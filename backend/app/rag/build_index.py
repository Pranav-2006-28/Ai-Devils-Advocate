import os
import chromadb
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
from tqdm import tqdm
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent
CHROMA_DB_PATH = BASE_DIR / "chroma_db"

def build_index():
    print(f"Setting up ChromaDB at {CHROMA_DB_PATH}...")
    client = chromadb.PersistentClient(path=str(CHROMA_DB_PATH))
    
    # We use cosine similarity for text embeddings
    collection = client.get_or_create_collection(
        name="cuad_clauses",
        metadata={"hnsw:space": "cosine"}
    )
    
    print("Loading embedding model (all-MiniLM-L6-v2)...")
    # A fast, lightweight embedding model suitable for semantic search
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    
    print("Loading CUAD dataset from HuggingFace...")
    # Load the CUAD QA dataset
    dataset = load_dataset("theatticusproject/cuad-qa", split="train", trust_remote_code=True)
    
    print(f"Loaded {len(dataset)} documents. Extracting clauses...")
    
    clauses = []
    metadata = []
    ids = []
    
    # Simple risk prior mapping (based on Phase 3 plan)
    # We map known CUAD questions/categories to risk tiers
    risk_mapping = {
        "Indemnification": "high",
        "Limitation of Liability": "high",
        "Uncapped Liability": "high",
        "Termination for Convenience": "medium",
        "Non-Compete": "high",
        "Exclusivity": "high",
        "Auto-Renewal": "medium",
        "Governing Law": "low",
        "IP Ownership": "medium",
        "Most Favored Nation": "high",
        "Change of Control": "medium",
        "Anti-Assignment": "low",
        "Warranty Duration": "low",
        "Insurance": "low"
    }
    
    clause_counter = 0
    
    for row in tqdm(dataset):
        # The CUAD dataset QA format has 'title' (doc name), 'context' (text), 
        # and 'answers' (which contains text spans for the categories).
        # We need to extract the actual clauses that were annotated.
        
        # The question usually contains the category name in CUAD, e.g. "Highlight the parts (if any) of this contract related to 'Limitation of Liability'..."
        question = row.get("question", "")
        
        # Determine category based on question text mapping
        category = "Other"
        risk_tier = "low"
        
        for cat, risk in risk_mapping.items():
            if cat.lower() in question.lower():
                category = cat
                risk_tier = risk
                break
                
        answers = row.get("answers", {})
        answer_texts = answers.get("text", [])
        
        for ans_text in answer_texts:
            # We only index clauses that are substantial
            if len(ans_text) > 20:
                clauses.append(ans_text)
                metadata.append({
                    "category": category,
                    "risk_tier": risk_tier,
                    "source_doc": row.get("title", "Unknown")
                })
                ids.append(f"clause_{clause_counter}")
                clause_counter += 1
                
    print(f"Extracted {len(clauses)} valid clauses. Generating embeddings and adding to ChromaDB...")
    
    # Process in batches to avoid OOM
    batch_size = 256
    for i in tqdm(range(0, len(clauses), batch_size)):
        batch_clauses = clauses[i:i + batch_size]
        batch_metadata = metadata[i:i + batch_size]
        batch_ids = ids[i:i + batch_size]
        
        # Generate embeddings
        embeddings = embedder.encode(batch_clauses).tolist()
        
        # Add to ChromaDB
        collection.add(
            embeddings=embeddings,
            documents=batch_clauses,
            metadatas=batch_metadata,
            ids=batch_ids
        )
        
    print(f"\n✅ Successfully built RAG index with {len(clauses)} clauses at {CHROMA_DB_PATH}")

if __name__ == "__main__":
    # Ensure directory exists
    CHROMA_DB_PATH.mkdir(parents=True, exist_ok=True)
    build_index()
