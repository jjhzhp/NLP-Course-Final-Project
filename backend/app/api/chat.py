import json
from typing import AsyncIterator

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from app.core.agent_service import AgentService
from app.core.schemas import ChatRequest, ChatResponse
from app.db.database import get_db
from app.dependencies import get_agent_service


router = APIRouter(prefix="/api", tags=["chat"])


@router.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    db: Session = Depends(get_db),
    agent: AgentService = Depends(get_agent_service),
) -> ChatResponse:
    return await agent.handle(db, request)


@router.post("/chat/stream")
async def chat_stream(
    request: ChatRequest,
    db: Session = Depends(get_db),
    agent: AgentService = Depends(get_agent_service),
) -> StreamingResponse:
    async def event_source() -> AsyncIterator[bytes]:
        try:
            async for event in agent.handle_stream(db, request):
                payload = json.dumps(event, ensure_ascii=False)
                yield f"data: {payload}\n\n".encode("utf-8")
        except Exception as exc:  # noqa: BLE001
            err = json.dumps(
                {"type": "error", "message": str(exc)},
                ensure_ascii=False,
            )
            yield f"data: {err}\n\n".encode("utf-8")

    headers = {
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no",
        "Connection": "keep-alive",
    }
    return StreamingResponse(event_source(), media_type="text/event-stream", headers=headers)
