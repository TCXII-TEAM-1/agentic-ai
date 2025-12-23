"""Script to run the RAG pipeline."""

import os
from dotenv import load_dotenv
from rag_pipeline import RAGPipeline

# Load environment variables from .env file
load_dotenv()


def main():
    # Get API key from environment
    api_key = os.getenv("MISTRAL_API_KEY")
    
    if not api_key:
        print("❌ Error: MISTRAL_API_KEY not found in environment variables.")
        print("Please set it in a .env file or as an environment variable.")
        return
    
    # Configure paths (you can modify these)
    docs_dir = "./docs"  # Directory containing your PDF files
    db_path = "./db"     # Directory where Qdrant database will be stored
    
    # Ensure docs directory exists
    if not os.path.exists(docs_dir):
        os.makedirs(docs_dir)
        print(f"📁 Created docs directory at: {docs_dir}")
        print("   Please add your PDF files there and run this script again.")
        return
    
    # Check if there are any PDFs in the docs directory
    pdf_files = [f for f in os.listdir(docs_dir) if f.lower().endswith(".pdf")]
    if not pdf_files:
        print(f"⚠️ No PDF files found in {docs_dir}")
        print("   Please add PDF files and run this script again.")
        return
    
    print(f"📚 Found {len(pdf_files)} PDF file(s) to process")
    
    # Initialize and run the pipeline
    pipeline = RAGPipeline(
        api_key=api_key,
        docs_dir=docs_dir,
        db_path=db_path
    )
    
    # Run the full pipeline
    document_store = pipeline.run_full_pipeline()
    
    if document_store:
        print(f"\n✅ RAG pipeline completed!")
        print(f"   Documents in store: {document_store.count_documents()}")
        print(f"   Database location: {db_path}")


if __name__ == "__main__":
    main()
