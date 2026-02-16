# Lab 2: FastAPI RAG Backend

## Objective
Build a production-ready REST API for your RAG system.

## Steps

1. Create project: `mkdir lab2-fastapi && cd lab2-fastapi`

2. Install FastAPI, uvicorn, boto3, pydantic

3. Create `main.py` with:
   - FastAPI app initialization
   - Pydantic models for requests/responses
   - `/chat` endpoint
   - Error handling

4. Test at http://localhost:8000/docs