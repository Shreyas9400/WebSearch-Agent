# api/index.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .routes import chat_router

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For production, specify your actual frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(chat_router)

# Required for Vercel
from mangum import Adapter
handler = Adapter(app)
