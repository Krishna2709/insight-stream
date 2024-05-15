import os 

from fastapi import FastAPI
from fastapi import HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from insight import VideoAnalysis
from insight import query_engine

port = int(os.getenv("PORT", 8080))
app = FastAPI()


# Allowing CORS for the frontend
origins = [
    "http://localhost",
    "http://localhost:8080"
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

analyzer = VideoAnalysis()

# Data models for the request and response payloads
class AnalyzeRequest(BaseModel):
    youtube_url: str

class QueryRequest(BaseModel):
    prompt: str

# Analyzer endpoint
@app.post("/analyzer")
def analyzer_endpoint(request: AnalyzeRequest):
    """
    Analyzes the youtube video transcript and returns a summary, technical questions, and relevant research papers.

    request: a ditcionary containing summay, questions, and papers
    """
    try:
        result = analyzer.analyze(request.youtube_url)
        summary = result['summary']
        questions = result['questions']
        papers = result['papers']

        return {
            'summary': summary, 
            'questions': questions,
            'papers': papers
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

# Query endpoint
@app.post("/query")
def query_endpoint(request: QueryRequest):
    """
    Queries the vector engine to get relevant research papers based on the user's prompt.

    request: a dictionary containing the response and papers.
    """
    if not (analyzer.vector_engine):
        raise HTTPException(status_code=400, detail="Analyzer not initialized. Engines are not ready.")
    try:
        response = query_engine(request.prompt, analyzer.vector_engine)

        response_text = response['response']
        papers = response['papers']

        return {
            'response': response_text,
            'papers': papers
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))