import base64
import os
from typing import List, Optional
from mistralai import Mistral
from langchain_text_splitters import MarkdownHeaderTextSplitter
from haystack import Document
from haystack.utils import Secret
from haystack_integrations.components.embedders.mistral.document_embedder import (
    MistralDocumentEmbedder,
)

from haystack_integrations.components.embedders.mistral.text_embedder import (
    MistralTextEmbedder,
)
from haystack_integrations.document_stores.qdrant import QdrantDocumentStore
from haystack_integrations.components.retrievers.qdrant import QdrantEmbeddingRetriever
from haystack.document_stores.types import DuplicatePolicy


class RAGPipeline:
    """RAG pipeline for document processing and embedding."""

    def __init__(self, api_key: str, docs_dir: str = "./docs", db_path: str = "./db"):
        self.api_key = api_key
        self.client = Mistral(api_key=api_key)
        self.docs_dir = docs_dir
        self.db_path = db_path

        os.makedirs(self.db_path, exist_ok=True)

        self.headers = [
            ("##", "Section"),
            ("###", "Subsection"),
        ]
        self.markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=self.headers, strip_headers=False
        )
        self._document_store = None

    def get_document_store(
        self, index_name: str = "doxa_docs", recreate_index: bool = False
    ) -> QdrantDocumentStore:
        """Get or create the document store."""
        if self._document_store is None or recreate_index:
            self._document_store = QdrantDocumentStore(
                path=self.db_path,
                index=index_name,
                embedding_dim=1024,
                recreate_index=recreate_index,
            )
        return self._document_store

    def encode_pdf(self, pdf_path: str) -> str:
        """Encode PDF to base64."""
        with open(pdf_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def process_pdf_to_documents(self, pdf_path: str, filename: str) -> List[Document]:
        """Process a single PDF directly to Haystack Documents."""
        print(f"Processing: {filename}")

        # OCR
        base64_pdf = self.encode_pdf(pdf_path)
        ocr_response = self.client.ocr.process(
            model="mistral-ocr-latest",
            document={
                "type": "document_url",
                "document_url": f"data:application/pdf;base64,{base64_pdf}",
            },
            table_format="html",
            include_image_base64=True,
        )

        # Extract markdown
        all_pages = ocr_response.pages if hasattr(ocr_response, "pages") else []
        markdown_content = "\n\n".join(
            page.markdown
            for page in all_pages
            if hasattr(page, "markdown") and page.markdown.strip()
        )

        if not markdown_content:
            print(f"⚠️ No markdown found in {filename}")
            return []

        # Chunk
        try:
            chunks = self.markdown_splitter.split_text(markdown_content)
            print(f"   ✓ Created {len(chunks)} chunks")
        except Exception as e:
            print(f"   ✗ Error chunking {filename}: {e}")
            return []

        # Create Haystack Documents
        documents = []
        for i, chunk in enumerate(chunks):
            doc = Document(
                content=chunk.page_content,
                meta={
                    "source_file": filename,
                    "chunk_id": i,
                    "total_chunks": len(chunks),
                    **chunk.metadata,
                },
            )
            documents.append(doc)

        return documents

    def process_all_pdfs(self) -> List[Document]:
        """Process all PDFs in docs directory."""
        print("\n🔄 Processing PDFs...")
        all_documents = []

        for filename in os.listdir(self.docs_dir):
            if not filename.lower().endswith(".pdf"):
                continue

            pdf_path = os.path.join(self.docs_dir, filename)
            documents = self.process_pdf_to_documents(pdf_path, filename)
            all_documents.extend(documents)

        print(f"✅ Processed {len(all_documents)} document chunks from PDFs")
        return all_documents

    def embed_documents(self, documents: List[Document]) -> List[Document]:
        """Embed documents using Mistral."""
        print("\n🔄 Embedding documents with Mistral...")

        embedder = MistralDocumentEmbedder(
            api_key=Secret.from_token(self.api_key), model="mistral-embed"
        )

        documents_with_embeddings = embedder.run(documents)["documents"]

        print(f"✅ Embedded {len(documents_with_embeddings)} documents")

        # Print statistics
        self._print_embedding_stats(documents_with_embeddings)

        return documents_with_embeddings

    def store_embeddings(
        self, documents: List[Document], index_name: str = "doxa_docs"
    ) -> QdrantDocumentStore:
        """Store embeddings in Qdrant."""
        print("\n🔄 Storing embeddings in Qdrant...")

        document_store = self.get_document_store(index_name=index_name, recreate_index=True)

        document_store.write_documents(documents, policy=DuplicatePolicy.OVERWRITE)

        print(f"✅ Stored {document_store.count_documents()} documents in Qdrant")
        print(f"📂 Database location: {self.db_path}")

        return document_store

    def _print_embedding_stats(self, documents: List[Document]) -> None:
        """Print embedding statistics."""
        print("\n" + "=" * 80)
        print("📊 EMBEDDING STATISTICS:")
        print("=" * 80)

        if documents:
            first_doc = documents[0]
            print(f"\n📄 First document sample:")
            print(f"Content preview: {first_doc.content[:200]}...")
            print(f"Metadata: {first_doc.meta}")
            print(f"Embedding dimension: {len(first_doc.embedding)}")
            print(f"First 10 values: {first_doc.embedding[:10]}")

        for i, doc in enumerate(documents[:3]):
            print(f"\nDocument {i+1}:")
            print(f"  Source: {doc.meta.get('source_file', 'Unknown')}")
            print(f"  Chunk ID: {doc.meta.get('chunk_id', 'N/A')}")
            print(f"  Content length: {len(doc.content)} chars")
            if doc.embedding:
                print(
                    f"  Embedding sample: [{doc.embedding[0]:.6f}, {doc.embedding[1]:.6f}, {doc.embedding[2]:.6f}, ...]"
                )

    def run_full_pipeline(self) -> QdrantDocumentStore:
        """Run the complete RAG pipeline: Process PDFs → Embed → Store in DB."""
        print("🚀 Starting full RAG pipeline...")

        # Step 1: Process all PDFs (OCR + Chunk in one step)
        documents = self.process_all_pdfs()

        if not documents:
            print("⚠️ No documents to process")
            return None

        # Step 2: Embed
        embedded_docs = self.embed_documents(documents)

        # Step 3: Store in DB
        document_store = self.store_embeddings(embedded_docs)

        print("\n🎉 Pipeline completed successfully!")
        return document_store
        
    
    def load_document_store(self, index_name: str = "doxa_docs") -> QdrantDocumentStore:
        """Load existing Qdrant document store."""
        return self.get_document_store(index_name=index_name, recreate_index=False)

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        index_name: str = "doxa_docs",
        document_store: Optional[QdrantDocumentStore] = None,
    ) -> List[Document]:
        """
        Retrieve relevant documents for a given query.
        
        Args:
            query: The search query string
            top_k: Number of documents to retrieve
            index_name: Name of the Qdrant index
            document_store: Optional existing document store (loads from disk if not provided)
            
        Returns:
            List of relevant documents sorted by similarity
        """
        print(f"\n🔍 Retrieving documents for: '{query}'")

        # Load document store if not provided
        if document_store is None:
            document_store = self.load_document_store(index_name)

        # Embed the query
        text_embedder = MistralTextEmbedder(
            api_key=Secret.from_token(self.api_key), model="mistral-embed"
        )
        query_result = text_embedder.run(query)
        query_embedding = query_result["embedding"]

        # Use QdrantEmbeddingRetriever instead of query_by_embedding
        retriever = QdrantEmbeddingRetriever(
            document_store=document_store,
            top_k=top_k,
        )
        
        results = retriever.run(query_embedding=query_embedding)
        documents = results["documents"]

        print(f"✅ Retrieved {len(documents)} documents")

        # Print results summary
        self._print_retrieval_results(documents)

        return documents

    def _print_retrieval_results(self, documents: List[Document]) -> None:
        """Print retrieval results summary."""
        print("\n" + "-" * 60)
        print("📋 RETRIEVAL RESULTS:")
        print("-" * 60)

        for i, doc in enumerate(documents):
            score = doc.score if hasattr(doc, "score") and doc.score else "N/A"
            source = doc.meta.get("source_file", "Unknown")
            chunk_id = doc.meta.get("chunk_id", "N/A")
            content_preview = doc.content[:150].replace("\n", " ")

            print(f"\n[{i+1}] Score: {score}")
            print(f"    Source: {source} (Chunk {chunk_id})")
            print(f"    Preview: {content_preview}...")