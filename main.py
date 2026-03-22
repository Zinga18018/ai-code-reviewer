"""
AI Code Review Agent
====================
TinyLlama-1.1B powered code review system.
Paste code, get line-by-line reviews covering bugs, security,
performance, and style -- served over FastAPI with streaming support.

Usage:
    python main.py
    # Then open http://localhost:8001/docs
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from core import CodeReviewer, ReviewConfig
from api import register_routes

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)

config = ReviewConfig()
reviewer = CodeReviewer(config)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    reviewer.load()
    yield


app = FastAPI(
    title="AI Code Review Agent",
    description="TinyLlama-powered code review with streaming and batch support",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

register_routes(app, reviewer)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=config.port, reload=True)
