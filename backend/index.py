from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from enum import Enum
from datetime import datetime
from chatbot import ChatBot, ScoringMethod, SafeSearch, QueryType
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI()

# Configure CORS with environment-based origins
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:3000")

# Set up origins based on environment
origins = [
    "http://localhost:3000",  # Local development
    "http://127.0.0.1:3000",  # Alternative local development
]

# Add production frontend URL if it exists
if FRONTEND_URL and FRONTEND_URL not in origins:
    origins.append(FRONTEND_URL)

# If in development, allow all origins
if ENVIRONMENT == "development":
    origins = ["*"]

# Configure CORS with the determined origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize ChatBot
chatbot = ChatBot()

# Pydantic models
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

@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest):
    try:
        force_web_search = request.search_mode == SearchMode.WEB_ONLY
        
        # Get response from chatbot
        start_time = datetime.now()
        response = await chatbot.get_response(
            query=request.message,
            history=request.history,
            num_results=request.num_results,
            max_chars=request.max_chars,
            score_threshold=request.score_threshold,
            temperature=request.temperature,
            scoring_method=request.scoring_method,
            selected_engines=request.engines,
            safe_search=request.safe_search,
            language=request.language.split(" - ")[0],
            force_web_search=force_web_search
        )
        end_time = datetime.now()
        
        metadata = {
            "processing_time": (end_time - start_time).total_seconds(),
            "timestamp": datetime.now().isoformat(),
            "parameters": {
                "num_results": request.num_results,
                "max_chars": request.max_chars,
                "score_threshold": request.score_threshold,
                "temperature": request.temperature,
                "scoring_method": request.scoring_method,
                "engines": request.engines,
                "safe_search": request.safe_search,
                "language": request.language,
                "search_mode": request.search_mode
            }
        }
        
        return ChatResponse(
            response=response,
            metadata=metadata
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing chat request: {str(e)}"
        )

@app.get("/api/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0"
    }

@app.get("/api/engines")
async def get_available_engines():
    try:
        engines = chatbot.default_engines
        return {
            "engines": engines,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error fetching available engines: {str(e)}"
        )
