# api/routes.py
from fastapi import APIRouter, HTTPException
from .models import ChatRequest, ChatResponse
from .chatbot import ChatBot
from datetime import datetime

chat_router = APIRouter()
chatbot = ChatBot()

@chat_router.post("/api/chat")
async def chat_endpoint(request: ChatRequest):
    try:
        force_web_search = request.search_mode == "Web Search Only"
        
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
            "parameters": request.dict()
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

@chat_router.get("/api/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0"
    }

@chat_router.get("/api/engines")
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
