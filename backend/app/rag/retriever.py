import os
import chromadb
from sentence_transformers import SentenceTransformer
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
CHROMA_DB_PATH = BASE_DIR / "chroma_db"

class ClauseRetriever:
    def __init__(self):
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Connect to existing ChromaDB
        if not CHROMA_DB_PATH.exists():
            print(f"Warning: ChromaDB not found at {CHROMA_DB_PATH}. Run build_index.py first.")
            self.collection = None
        else:
            self.client = chromadb.PersistentClient(path=str(CHROMA_DB_PATH))
            self.collection = self.client.get_collection(name="cuad_clauses")
            
    def retrieve_similar(self, clause_text: str, k: int = 5) -> list:
        """
        Retrieve the top k most similar historical clauses from the database.
        Returns a list of dicts with keys: 'text', 'category', 'risk_tier', 'source_doc', 'similarity'
        """
        if not self.collection:
            return []
            
        # Generate embedding for the query
        query_embedding = self.embedder.encode([clause_text]).tolist()
        
        # Query ChromaDB
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=k
        )
        
        # Format results
        formatted_results = []
        if results and results.get("documents") and len(results["documents"]) > 0:
            docs = results["documents"][0]
            metadatas = results["metadatas"][0]
            distances = results["distances"][0]
            
            for i in range(len(docs)):
                formatted_results.append({
                    "text": docs[i],
                    "category": metadatas[i].get("category", "Unknown"),
                    "risk_tier": metadatas[i].get("risk_tier", "Unknown"),
                    "source_doc": metadatas[i].get("source_doc", "Unknown"),
                    "distance": distances[i] # Lower distance = higher similarity for cosine space in Chroma
                })
                
        return formatted_results

# Singleton instance for the FastAPI app to use
retriever = ClauseRetriever()

if __name__ == "__main__":
    # Test script
    print("Testing Retriever...")
    test_retriever = ClauseRetriever()
    
    test_query = "Either party may terminate this agreement for convenience upon 30 days written notice."
    print(f"\nQuery: '{test_query}'")
    
    results = test_retriever.retrieve_similar(test_query, k=3)
    
    for i, res in enumerate(results):
        print(f"\n--- Match {i+1} (Dist: {res['distance']:.4f}) ---")
        print(f"Category: {res['category']} | Risk: {res['risk_tier']}")
        print(f"Source: {res['source_doc']}")
        print(f"Text: {res['text'][:200]}...")
