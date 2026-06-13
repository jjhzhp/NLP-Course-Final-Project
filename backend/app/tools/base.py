from typing import Any, AsyncIterator, Callable

import httpx
from sqlalchemy.orm import Session

from app.config import Settings
from app.core.llm_client import DeepSeekClient
from app.core.schemas import AgentTrace, ChatResponse, RetrievedChunk
from app.retrieval.retriever import HybridRetriever, estimate_confidence


PromptBuilder = Callable[[str, list[RetrievedChunk]], str]


class AgentTool:
    name: str = ""
    description: str = ""
    default_profile: str = "qa"
    system_prompt: str = ""
    prompt_builder: PromptBuilder
    temperature: float = 0.2
    max_tokens: int = 1200
    force_pro_model: bool = False
    refuse_low_confidence: bool = False

    def __init__(self, retriever: HybridRetriever, llm: DeepSeekClient, settings: Settings):
        self.retriever = retriever
        self.llm = llm
        self.settings = settings

    def default_top_k(self) -> int:
        if self.default_profile == "summary":
            return self.settings.summary_final_top_k
        return self.settings.final_top_k

    async def generate_with_chunks(
        self,
        *,
        query: str,
        chunks: list[RetrievedChunk],
        confidence: str,
        use_pro_model: bool = False,
        agent_trace: AgentTrace | None = None,
    ) -> ChatResponse:
        low_response = self._low_confidence_response(chunks, confidence, agent_trace)
        if low_response is not None:
            return low_response

        model = self._select_model(use_pro_model)
        prompt = self.prompt_builder(query, chunks)
        try:
            answer = await self.llm.chat(
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt},
                ],
                model=model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                continue_on_length=True,
            )
        except (RuntimeError, httpx.HTTPError) as exc:
            return ChatResponse(
                task_type=self.name,
                answer=f"调用 DeepSeek API 失败：{exc}",
                sources=chunks,
                confidence=confidence,  # type: ignore[arg-type]
                message="llm_error",
                agent_trace=agent_trace,
            )

        return ChatResponse(
            task_type=self.name,
            answer=answer,
            sources=chunks,
            confidence=confidence,  # type: ignore[arg-type]
            message="low_retrieval_confidence" if confidence == "low" else None,
            agent_trace=agent_trace,
        )

    async def stream_with_chunks(
        self,
        *,
        query: str,
        chunks: list[RetrievedChunk],
        confidence: str,
        use_pro_model: bool = False,
        agent_trace: AgentTrace | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        message = "low_retrieval_confidence" if confidence == "low" else None
        meta: dict[str, Any] = {
            "type": "meta",
            "task_type": self.name,
            "confidence": confidence,
            "sources": [chunk.model_dump() for chunk in chunks],
            "message": message,
        }
        if agent_trace is not None:
            meta["agent_trace"] = agent_trace.model_dump()
        yield meta

        low_text = self._low_confidence_text(confidence)
        if low_text is not None:
            yield {"type": "delta", "text": low_text}
            yield {"type": "done"}
            return

        model = self._select_model(use_pro_model)
        prompt = self.prompt_builder(query, chunks)
        try:
            async for token in self.llm.chat_stream(
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt},
                ],
                model=model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                continue_on_length=True,
            ):
                yield {"type": "delta", "text": token}
        except (RuntimeError, httpx.HTTPError) as exc:
            yield {"type": "error", "message": f"调用 DeepSeek API 失败：{exc}"}
            return

        yield {"type": "done"}

    async def run(
        self,
        db: Session,
        query: str,
        use_pro_model: bool = False,
        top_k: int | None = None,
        retrieval_query: str | None = None,
        retrieval_profile: str | None = None,
    ) -> ChatResponse:
        profile = retrieval_profile or self.default_profile
        chunks = self.retriever.retrieve(
            db,
            retrieval_query or query,
            top_k or self.default_top_k(),
            profile=profile,
        )
        confidence = estimate_confidence(chunks)
        return await self.generate_with_chunks(
            query=query,
            chunks=chunks,
            confidence=confidence,
            use_pro_model=use_pro_model,
        )

    async def run_stream(
        self,
        db: Session,
        query: str,
        use_pro_model: bool = False,
        top_k: int | None = None,
        retrieval_query: str | None = None,
        retrieval_profile: str | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        profile = retrieval_profile or self.default_profile
        chunks = self.retriever.retrieve(
            db,
            retrieval_query or query,
            top_k or self.default_top_k(),
            profile=profile,
        )
        confidence = estimate_confidence(chunks)
        async for event in self.stream_with_chunks(
            query=query,
            chunks=chunks,
            confidence=confidence,
            use_pro_model=use_pro_model,
        ):
            yield event

    def _select_model(self, use_pro_model: bool) -> str:
        if self.force_pro_model or use_pro_model:
            return self.settings.deepseek_pro_model
        return self.settings.deepseek_model

    def _low_confidence_response(
        self,
        chunks: list[RetrievedChunk],
        confidence: str,
        agent_trace: AgentTrace | None,
    ) -> ChatResponse | None:
        text = self._low_confidence_text(confidence)
        if text is None:
            return None
        return ChatResponse(
            task_type=self.name,
            answer=text,
            sources=chunks,
            confidence="low",
            message="low_retrieval_confidence",
            agent_trace=agent_trace,
        )

    def _low_confidence_text(self, confidence: str) -> str | None:
        if self.refuse_low_confidence and confidence == "low":
            return "课程资料中未找到与该问题充分相关的内容。建议补充资料，或换一种更具体的问法。"
        return None
