from rag_pipeline import RAGPipeline
import os
import sys
from dotenv import load_dotenv

# Ensure we can find the modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

load_dotenv()
api_key = os.getenv("MISTRAL_API_KEY")

if not api_key:
    # mock key for testing if we just want to trigger the db lock
    # but retrieve uses MistralTextEmbedder which will fail without key.
    # We hope the user has one.
    print("Checking for API key...")

try:
    print("Initializing Pipeline...")
    pipeline = RAGPipeline(api_key=api_key or "mock_key")
    
    # We might need to ensure ./db exists or mock the store loading if we want to fail fast
    # But QdrantDocumentStore will try to lock on init.
    
    print("First retrieval attempt...")
    # This might fail if the DB path exists and is locked.
    # Note: retrieve() calls load_document_store() which creates QdrantDocumentStore
    
    # We'll pass a dummy query. The embedder might fail if key is invalid, 
    # but the lock happens inside QdrantDocumentStore init which happens BEFORE embedding?
    # Let's check: 
    # retrieve() -> load_document_store() -> QdrantDocumentStore() -> lock
    # THEN TextEmbedder.
    # So even with bad key, we should hit the lock if we do it twice?
    
    pipeline.retrieve("test 1", top_k=1)
    print("Success 1")
    
    print("Second retrieval attempt...")
    pipeline.retrieve("test 2", top_k=1)
    print("Success 2")
    
    with open("verification_result.txt", "w") as f:
        f.write("SUCCESS")

except Exception as e:
    print(f"Caught expected exception: {e}")
    with open("verification_result.txt", "w") as f:
        f.write(f"FAILED: {e}")
