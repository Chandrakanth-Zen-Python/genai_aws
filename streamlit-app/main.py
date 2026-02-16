from fastapi import FastAPI, HTTPException # api handling
from pydantic import BaseModel # request and response schema
import boto3
from typing import List, Optional


app = FastAPI(
    title="GenAI RAG API",
    description="Production API for RAG-powered chatbot",
    version="1.0.0"
)


# Initialize Bedrock client
bedrock = boto3.client('bedrock-runtime', region_name='us-east-1')


# Request/Response models
class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: List[Message]
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 1000


class ChatResponse(BaseModel):
    response: str
    model: str
    tokens_used: int

@app.get("/")
async def root():
    """Health check endpoint"""
    return {"status": "healthy", "service": "GenAI RAG API"}


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Chat endpoint with conversation history
    
    - **messages**: List of conversation messages
    - **temperature**: Controls randomness (0.0-1.0)
    - **max_tokens**: Maximum response length
    """
    try:
        # Invoke Bedrock with Mistral model using Converse API
        response = bedrock.converse(
            modelId="mistral.mistral-large-2402-v1:0",
            messages=[
                {"role": msg.role, "content": [{"text": msg.content}]}
                for msg in request.messages
            ],
            inferenceConfig={
                "maxTokens": request.max_tokens,
                "temperature": request.temperature
            }
        )

        # Parse response
        assistant_message = response['output']['message']['content'][0]['text']

        return ChatResponse(
            response=assistant_message,
            model="mistral-large",
            tokens_used=response['usage']['outputTokens']
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))