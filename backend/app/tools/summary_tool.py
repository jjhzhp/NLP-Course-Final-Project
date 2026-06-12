from typing import Any, AsyncIterator

import httpx
from sqlalchemy.orm import Session

from app.config import Settings
from app.core.llm_client import DeepSeekClient
from app.core.prompts import SUMMARY_SYSTEM_PROMPT, build_summary_prompt
from app.core.schemas import ChatResponse
from app.retrieval.retriever import HybridRetriever, estimate_confidence


class SummaryTool:
    def __init__(self, retriever: HybridRetriever, llm: DeepSeekClient, settings: Settings):
        self.retriever = retriever
        self.llm = llm
        self.settings = settings

    async def run(
        self,
        db: Session,
        query: str,
        use_pro_model: bool = False,
        top_k: int | None = None,
        retrieval_query: str | None = None,
        retrieval_profile: str = "summary",
    ) -> ChatResponse:
        chunks = self.retriever.retrieve(
            db,
            retrieval_query or query,
            top_k or self.settings.summary_final_top_k,
            profile=retrieval_profile,
        )
        confidence = estimate_confidence(chunks)
        prompt = build_summary_prompt(query, chunks)
        model = self.settings.deepseek_pro_model if use_pro_model else self.settings.deepseek_model

        try:
            answer = await self.llm.chat(
                messages=[
                    {"role": "system", "content": SUMMARY_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                model=model,
                temperature=0.2,
                max_tokens=1600,
            )
        except (RuntimeError, httpx.HTTPError) as exc:
            return ChatResponse(
                task_type="summary",
                answer=f"调用 DeepSeek API 失败：{exc}",
                sources=chunks,
                confidence=confidence,
                message="llm_error",
            )

        message = "low_retrieval_confidence" if confidence == "low" else None
        return ChatResponse(
            task_type="summary",
            answer=answer,
            sources=chunks,
            confidence=confidence,
            message=message,
        )

    async def run_stream(
        self,
        db: Session,
        query: str,
        use_pro_model: bool = False,
        top_k: int | None = None,
        retrieval_query: str | None = None,
        retrieval_profile: str = "summary",
    ) -> AsyncIterator[dict[str, Any]]:
        chunks = self.retriever.retrieve(
            db,
            retrieval_query or query,
            top_k or self.settings.summary_final_top_k,
            profile=retrieval_profile,
        )
        confidence = estimate_confidence(chunks)
        message = "low_retrieval_confidence" if confidence == "low" else None
        yield {
            "type": "meta",
            "task_type": "summary",
            "confidence": confidence,
            "sources": [c.model_dump() for c in chunks],
            "message": message,
        }

        prompt = build_summary_prompt(query, chunks)
        model = self.settings.deepseek_pro_model if use_pro_model else self.settings.deepseek_model
        try:
            async for token in self.llm.chat_stream(
                messages=[
                    {"role": "system", "content": SUMMARY_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                model=model,
                temperature=0.2,
                max_tokens=1600,
            ):
                yield {"type": "delta", "text": token}
        except (RuntimeError, httpx.HTTPError) as exc:
            yield {"type": "error", "message": f"调用 DeepSeek API 失败：{exc}"}
            return

        yield {"type": "done"}
