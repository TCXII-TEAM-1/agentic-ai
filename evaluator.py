import os
from typing import List
from agno.agent import Agent
from agno.models.mistral import MistralChat
from agno.tools import tool
from dotenv import load_dotenv, find_dotenv
from models import RetrievalResult, AnalysisResult, AgentResponse

# Load environment variables (e.g., Mistral API key)
load_dotenv(find_dotenv())

# Initialize Mistral model for the evaluator agent
mistral = MistralChat(id="mistral-small-latest", temperature=0.2)

# Prompt template for the evaluator agent
EVALUATOR_PROMPT = """
You are an evaluator agent. Your task is to assess the **quality and confidence** of the retrieved information in solving the customer's issue.

**Context:**
The system is a specific customer support AI for **"Doxa"** (a software product). 
- Valid topics: Account issues, Features, Billing, Technical Troubleshooting.
- **Invalid topics:** Cooking, Sports, Weather, General Knowledge (e.g. "Pizza recipe"), Competitors, etc.

Ticket Analysis:
- Category: {category}
- Keywords: {keywords}
- Summary: {summary}

Retrieved Documents:
{documents}

Retrieval Metrics:
- Average Cosine Similarity: {avg_similarity}

Task:
1. **First, check if the query is OFF-TOPIC.**
   - Is the user asking about something completely unrelated to Doxa or software support (e.g., "recipe", "weather")?
   - If YES -> Assign **Confidence 1.00** and set Reasoning to "Off-topic query. Refusal recommended." (STOP HERE).

2. If the query is RELEVANT to Doxa:
   - Read the retrieved documents.
   - Determine if they contain the solution.
   - Assign a **confidence score**:
     - 1.00 = Perfect match, full solution found.
     - 0.50 = Partial information.
     - 0.00 = Irrelevant documents (failed retrieval for a valid query).
   - Provide reasoning.

Return valid JSON:
{{
  "confidence_score": <float>,
  "reasoning": "<short explanation>"
}}
"""

# Create the evaluator agent
evaluator_agent = Agent(
    model=mistral,
    name="Evaluator Agent",
    description="Evaluates the relevance and confidence of retrieved documents for a given ticket.",
)

def evaluate_solution(analysis: AnalysisResult, retrieval: RetrievalResult) -> AgentResponse:
    """Run the evaluator agent and return an AgentResponse containing the evaluation.
    """
    # Build a simple string representation of the retrieved documents
    doc_strings = []
    total_score = 0.0
    valid_scores = 0
    
    for i, doc in enumerate(retrieval.documents, 1):
        content = doc.get("content", "")
        score = doc.get("score", 0.0)
        
        doc_strings.append(f"[Doc {i}] (Similarity: {score:.4f})\n{content}")
        
        # Accumulate score if valid
        if score > 0:
            total_score += score
            valid_scores += 1
            
    docs_text = "\n---\n".join(doc_strings)
    avg_similarity = total_score / valid_scores if valid_scores > 0 else 0.0

    prompt = EVALUATOR_PROMPT.format(
        category="Support", 
        keywords=", ".join(analysis.keywords),
        summary=analysis.summary or "",
        documents=docs_text,
        avg_similarity=f"{avg_similarity:.4f}"
    )
    response = evaluator_agent.run(prompt)
    # Extract the content (JSON string) from the response
    content = response.content if hasattr(response, "content") else str(response)
    return AgentResponse(
        ticket_id="",
        analysis=analysis,
        context=[content],
        response=content,
    )
