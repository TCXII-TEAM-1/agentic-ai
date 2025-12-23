from typing import List, Optional
from pydantic import BaseModel, Field


class Ticket(BaseModel):
    id: str
    subject: str
    category: str
    description: str
    client_id: str
    timestamp: str


class AnalysisResult(BaseModel):
    ticket_id: str
    sentiment: str = Field(description="positive | neutral | negative")
    keywords: List[str]
    language: str = Field(description="ISO 639-1 code or full language name (e.g., 'en', 'User's Language')")
    summary: Optional[str] = None


class RetrievalResult(BaseModel):
    query: str
    documents: List[dict]
    sources: List[str]


class AgentResponse(BaseModel):
    ticket_id: str
    analysis: AnalysisResult
    context: List[str]
    response: str