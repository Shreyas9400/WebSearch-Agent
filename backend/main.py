from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from enum import Enum
import uvicorn
import logging
import asyncio
from datetime import datetime

# Import your ChatBot class and other necessary components
from chatbot import ChatBot, ScoringMethod, SafeSearch, QueryType

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Sentinel AI ChatBot API",
    description="A FastAPI implementation of the Sentinel AI ChatBot with web search capabilities",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Modify this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize ChatBot
chatbot = ChatBot()

# Pydantic models for request/response validation
class ChatMessage(BaseModel):
    role: str = Field(..., description="Role of the message sender (user/assistant)")
    content: str = Field(..., description="Content of the message")

class ChatHistory(BaseModel):
    messages: List[List[str]] = Field(default=[], description="List of message pairs [user_message, assistant_message]")

class SearchMode(str, Enum):
    AUTO = "Auto (Knowledge Base + Web)"
    WEB_ONLY = "Web Search Only"

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

@app.post("/api/chat", response_model=ChatResponse, tags=["chat"])
async def chat_endpoint(request: ChatRequest):
    """
    Process a chat message and return a response.
    
    Args:
        request (ChatRequest): The chat request containing message and parameters
        
    Returns:
        ChatResponse: The assistant's response with metadata
    """
    try:
        logger.info(f"Received chat request: {request.message}")
        
        # Determine if we should force web search
        force_web_search = request.search_mode == SearchMode.WEB_ONLY
        
        # Get response from chatbot
        start_time = datetime.now()
        response = await asyncio.create_task(
            chatbot.get_response(
                query=request.message,
                history=request.history,
                num_results=request.num_results,
                max_chars=request.max_chars,
                score_threshold=request.score_threshold,
                temperature=request.temperature,
                scoring_method=request.scoring_method,
                selected_engines=request.engines,
                safe_search=request.safe_search,
                language=request.language.split(" - ")[0],  # Extract language code
                force_web_search=force_web_search
            )
        )
        end_time = datetime.now()
        
        # Prepare metadata
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
        logger.error(f"Error processing chat request: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing chat request: {str(e)}"
        )

@app.get("/api/health", tags=["health"])
async def health_check():
    """
    Check the health status of the API.
    """
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": app.version
    }

@app.get("/api/engines", tags=["configuration"])
async def get_available_engines():
    """
    Get list of available search engines.
    """
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

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        workers=4
    )
