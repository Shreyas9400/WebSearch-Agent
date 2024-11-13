# api/models.py
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from enum import Enum

class ChatMessage(BaseModel):
    role: str = Field(..., description="Role of the message sender (user/assistant)")
    content: str = Field(..., description="Content of the message")

class SearchMode(str, Enum):
    AUTO = "Auto (Knowledge Base + Web)"
    WEB_ONLY = "Web Search Only"

    def __str__(self):
        return self.value

class ChatRequest(BaseModel):
    message: str = Field(..., description="User's message")
    history: List[List[str]] = Field(default=[], description="Chat history")
    num_results: int = Field(10, ge=5, le=30, description="Number of search results to fetch")
    max_chars: int = Field(10000, ge=1000, le=50000, description="Maximum characters per article")
    score_threshold: float = Field(0.8, ge=0.0, le=1.0, description="Minimum relevance score")
    temperature: float = Field(0.1, ge=0.0, le=1.0, description="Temperature for text generation")
    scoring_method: str = Field("Combined", description="Scoring method (BM25, TF-IDF, Combined)")
    engines: List[str] = Field(default=["google", "bing", "duckduckgo"], description="List of search engines to use")
    safe_search: str = Field("Moderate (1)", description="Safe search level")
    language: str = Field("all - All Languages", description="Preferred language for results")
    search_mode: SearchMode = Field(SearchMode.AUTO, description="Search mode (Auto/Web Only)")

class ChatResponse(BaseModel):
    response: str = Field(..., description="Assistant's response")
    error: Optional[str] = Field(None, description="Error message if any")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata about the response")
