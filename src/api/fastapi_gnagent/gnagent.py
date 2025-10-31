"""
This is the FastAPI endpoint for the system
It is the one used by default for deployment
"""

import os
import sys

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from gnagent.agent import main


class Request(BaseModel):
    question: str


class Response(BaseModel):
    question: str
    answer: str


gnagent = FastAPI(
    title="FASTAPI endpoint for GNAgent",
    description="Abstract the process of solving a genomic problem with GNAgent",
    version="0.1.0",
)


@gnagent.post(
    "/ask",
    summary="Answer a question using GNAgent",
    response_model=Response,
)
def ask_question(request: Request):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="You have not asked a question!")
    try:
        answer = main(request.question)
        return Response(question=request.question, answer=answer)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Sorry, we were not able to generate an answer to your question: {str(e)}",
        )
