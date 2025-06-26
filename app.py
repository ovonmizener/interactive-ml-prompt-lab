"""
FastAPI entrypoint for Interactive ML & Prompting Playground
Serves data, handles LLM calls, and provides API endpoints for the frontend
"""

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
from typing import Dict, List, Any, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Interactive ML & Prompting Playground",
    description="A full-stack web app for experimenting with ML and prompt engineering",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501"],  # Streamlit frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    """Health check endpoint"""
    return {"message": "Interactive ML & Prompting Playground API", "status": "running"}

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "ml-prompt-playground"}

# TODO: Add data upload endpoints
@app.post("/api/upload-data")
async def upload_data(file: UploadFile = File(...)):
    """Upload CSV/TXT data for ML processing"""
    # TODO: Implement data upload and validation
    raise HTTPException(status_code=501, detail="Not implemented yet")

# TODO: Add model training endpoints
@app.post("/api/train-model")
async def train_model(model_config: Dict[str, Any]):
    """Train ML model with provided configuration"""
    # TODO: Implement model training pipeline
    raise HTTPException(status_code=501, detail="Not implemented yet")

# TODO: Add LLM prompt endpoints
@app.post("/api/generate-llm")
async def generate_llm_response(prompt: str, model: str = "gpt-3.5-turbo"):
    """Generate LLM response for given prompt"""
    # TODO: Implement LLM API calls
    raise HTTPException(status_code=501, detail="Not implemented yet")

# TODO: Add semantic search endpoints
@app.post("/api/semantic-search")
async def semantic_search(query: str, documents: List[str]):
    """Perform semantic search over documents"""
    # TODO: Implement semantic search with embeddings
    raise HTTPException(status_code=501, detail="Not implemented yet")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000) 