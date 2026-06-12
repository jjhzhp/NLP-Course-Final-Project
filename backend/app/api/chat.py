from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.core.schemas import ChatRequest, ChatResponse
from app.db.database import get_db
from app.dependencies import get_qa_tool, get_summary_tool
from app.tools.qa_tool import QATool
from app.tools.summary_tool import SummaryTool


router = APIRouter(prefix="/api", tags=["chat"])
SUMMARY_INTENT_WORDS = ("总结", "梳理", "归纳", "复习", "提纲")


@router.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    db: Session = Depends(get_db),
    qa_tool: QATool = Depends(get_qa_tool),
    summary_tool: SummaryTool = Depends(get_summary_tool),
) -> ChatResponse:
    task_type = _resolve_task_type(request.task_type, request.query)
    if task_type == "summary":
        return await summary_tool.run(
            db,
            query=request.query,
            use_pro_model=request.use_pro_model,
            top_k=request.top_k,
        )

    return await qa_tool.run(
        db,
        query=request.query,
        use_pro_model=request.use_pro_model,
        top_k=request.top_k,
    )


def _resolve_task_type(task_type: str | None, query: str) -> str:
    if task_type and task_type != "auto":
        return task_type
    if any(word in query for word in SUMMARY_INTENT_WORDS):
        return "summary"
    return "qa"
