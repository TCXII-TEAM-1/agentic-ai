import os
from typing import List
from agno.agent import Agent
from agno.tools import tool
from agno.models.mistral import MistralChat
from models import AnalysisResult, RetrievalResult
from rag_pipeline import RAGPipeline
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

# Initialize RAG pipeline
_pipeline = None


def _get_pipeline() -> RAGPipeline:
    """Get or create the RAG pipeline instance."""
    global _pipeline
    if _pipeline is None:
        _pipeline = RAGPipeline(
            api_key=os.getenv("MISTRAL_API_KEY"),
            docs_dir="./docs",
            db_path="./db",
        )
    return _pipeline


@tool
def retrieve_from_knowledge_base(query: str, top_k: int = 5) -> str:
    """
    Retrieve relevant documents from the knowledge base.
    
    Args:
        query: The search query to find relevant documents
        top_k: Number of documents to retrieve (default: 5)
        
    Returns:
        A formatted string containing the retrieved documents
    """
    print(f"\n🔍 Tool called with query: '{query}'")
    
    pipeline = _get_pipeline()
    documents = pipeline.retrieve(query=query, top_k=top_k)
    
    if not documents:
        return "No relevant documents found."
    
    # Format results
    results = []
    for i, doc in enumerate(documents, 1):
        source = doc.meta.get("source_file", "Unknown")
        score = doc.score if hasattr(doc, "score") and doc.score else "N/A"
        results.append(
            f"[Document {i}]\n"
            f"Source: {source}\n"
            f"Score: {score}\n"
            f"Content:\n{doc.content}\n"
        )
    
    return "\n---\n".join(results)


# Initialize Mistral model
mistral = MistralChat(id="mistral-small-latest", temperature=0.2)

# Create agent with the retrieval tool
solution_finder = Agent(
    model=mistral,
    name="Solution Finder Agent",
    description="""
You are a Retrieval Agent.

Your job is to:
- Build a search query using the provided keywords
- If a summary is available, combine it with the keywords
- Use the retrieval tool to fetch relevant documents from the database

Rules:
- Use keywords as the main signal
- Use the summary only as additional context
- Do not generate answers or solutions
- Only retrieve information
""",
    tools=[retrieve_from_knowledge_base],
)

RAG_PROMPT = """
You are a Retrieval Agent.

INPUT:
Keywords:
{keywords}

Summary (may be null):
{summary}

TASK:
1. Build ONE short search query using the keywords.
2. If the summary is not null, append it to the query.
3. Call the retrieval tool with the final query.
4. Return only the retrieved documents.

RULES:
- Keywords are mandatory and come first.
- Summary is optional and only adds context.
- Do not generate explanations or solutions.
- Do not invent information.
"""


def find_solution(analysis: AnalysisResult, top_k: int = 5) -> RetrievalResult:
    """
    Use the agent to find relevant documents based on ticket analysis.
    
    Args:
        analysis: The analysis result from the query analyzer
        top_k: Number of documents to retrieve
        
    Returns:
        RetrievalResult with documents and sources
    """
    prompt = RAG_PROMPT.format(
        keywords=", ".join(analysis.keywords),
        summary=analysis.summary if analysis.summary else "null",
    )
    
    response = solution_finder.run(prompt)
    response_text = response.content if hasattr(response, "content") else str(response)
    
    # Build query for tracking
    query = " ".join(analysis.keywords)
    if analysis.summary:
        query += f" {analysis.summary}"
    
    return RetrievalResult(
        query=query,
        documents=[{"content": response_text, "meta": {}}],
        sources=["knowledge_base"],
    )