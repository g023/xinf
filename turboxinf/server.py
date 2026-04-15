"""
g023's TurboXInf
Author: g023 https://github.com/g023
License: MIT

TurboXInf — FastAPI Inference Server

OpenAI-compatible API with streaming support.
"""

import asyncio
import json
import time
import uuid
from typing import AsyncGenerator, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from .config import TurboXInfConfig
from .engine import TurboXInfEngine


# ── Request/Response Models ──────────────────────────────────────────────────

class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str = "g023/Qwen3-1.77B-g023"
    messages: list[Message]
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    stream: bool = False
    repetition_penalty: Optional[float] = None


class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class Choice(BaseModel):
    index: int = 0
    message: Optional[Message] = None
    delta: Optional[dict] = None
    finish_reason: Optional[str] = None


class ChatResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[Choice]
    usage: Usage


# ── App ──────────────────────────────────────────────────────────────────────

app = FastAPI(title="TurboXInf", version="1.0.0")
engine: Optional[TurboXInfEngine] = None


def get_engine() -> TurboXInfEngine:
    if engine is None:
        raise HTTPException(status_code=503, detail="Engine not loaded")
    return engine


@app.on_event("startup")
async def startup():
    global engine
    config = TurboXInfConfig()
    engine = TurboXInfEngine(config)
    engine.load()
    engine.warmup(runs=10)


@app.get("/health")
async def health():
    e = get_engine()
    return {"status": "ok", "stats": e.get_stats()}


@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [{
            "id": "g023/Qwen3-1.77B-g023",
            "object": "model",
            "owned_by": "turboxinf",
        }],
    }


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatRequest):
    e = get_engine()
    messages = [{"role": m.role, "content": m.content} for m in request.messages]

    if request.stream:
        return StreamingResponse(
            _stream_response(e, messages, request),
            media_type="text/event-stream",
        )

    result = await asyncio.get_event_loop().run_in_executor(
        None,
        lambda: e.generate(
            messages=messages,
            max_new_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            repetition_penalty=request.repetition_penalty,
        ),
    )

    return ChatResponse(
        id=f"chatcmpl-{uuid.uuid4().hex[:8]}",
        created=int(time.time()),
        model=request.model,
        choices=[Choice(
            message=Message(role="assistant", content=result["content"]),
            finish_reason="stop",
        )],
        usage=Usage(
            prompt_tokens=result["usage"]["input_tokens"],
            completion_tokens=result["usage"]["output_tokens"],
            total_tokens=result["usage"]["total_tokens"],
        ),
    )


async def _stream_response(
    e: TurboXInfEngine,
    messages: list,
    request: ChatRequest,
) -> AsyncGenerator[str, None]:
    request_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"

    def _generate():
        return list(e.generate_stream(
            messages=messages,
            max_new_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            repetition_penalty=request.repetition_penalty,
        ))

    loop = asyncio.get_event_loop()
    chunks = await loop.run_in_executor(None, _generate)

    for chunk in chunks:
        if chunk["done"]:
            data = {
                "id": request_id,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": request.model,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            }
        else:
            data = {
                "id": request_id,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": request.model,
                "choices": [{"index": 0, "delta": {"content": chunk["token"]}, "finish_reason": None}],
            }
        yield f"data: {json.dumps(data)}\n\n"

    yield "data: [DONE]\n\n"


def run_server(host: str = "0.0.0.0", port: int = 8000):
    """Run the TurboXInf server."""
    import uvicorn
    uvicorn.run(app, host=host, port=port)
