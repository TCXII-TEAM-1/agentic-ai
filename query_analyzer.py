import os
from typing import List, Dict, Optional
from dataclasses import dataclass
from mistralai import Mistral
from haystack import Pipeline
from haystack_integrations.components.retrievers.qdrant import QdrantEmbeddingRetriever
from haystack.utils import Secret
from agno.agent import Agent
from agno.models.mistral import MistralChat
from haystack_integrations.components.embedders.mistral.document_embedder import (
    MistralDocumentEmbedder,
)
import json
from models import Ticket, AnalysisResult
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())
mistral = MistralChat(id="mistral-small-latest", temperature=0.2)

query_analyzer = Agent(
    model=mistral,
    name="Query Analyzer Agent",
    description="You are an analysis agent that analyzes customer support tickets and provides sentiment, keywords, and a summary if the prompt is too long.",
)

ANALYSIS_PROMPT = """
You are analyzing a customer support ticket.

Rules:
- Always return sentiment and keywords
- ONLY generate a summary if the ticket description is long
- If the description is short, return "summary": null

Return ONLY valid JSON:
{{
  "sentiment": "positive | neutral | negative",
  "keywords": ["keyword1", "keyword2"],
  "language": "English" | "French" | "Arabic" | "etc",
  "summary": string | null
}}

Ticket subject:
{subject}

Ticket category:
{category}

Ticket description:
{description}
"""

def analyze_ticket(ticket: Ticket) -> AnalysisResult:
    prompt = ANALYSIS_PROMPT.format(
        subject=ticket.subject,
        category=ticket.category,
        description=ticket.description,
    )

    response = query_analyzer.run(prompt)
    
    # Extract the content from RunOutput
    response_text = response.content if hasattr(response, 'content') else str(response)
    
    # Clean up the response if it contains markdown code blocks
    if "```json" in response_text:
        response_text = response_text.split("```json")[1].split("```")[0].strip()
    elif "```" in response_text:
        response_text = response_text.split("```")[1].split("```")[0].strip()
    
    analysis_data = json.loads(response_text)

    return AnalysisResult(
        ticket_id=ticket.id,
        sentiment=analysis_data["sentiment"],
        keywords=analysis_data["keywords"],
        language=analysis_data.get("language", "English"),
        summary=analysis_data.get("summary"),
    )