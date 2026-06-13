from dataclasses import dataclass
from typing import Any, AsyncIterator

from sqlalchemy.orm import Session

from app.core.evidence import EvidenceJudge
from app.core.planner import AgentPlanner
from app.core.router_agent import RouterAgent
from app.core.schemas import (
    AgentState,
    AgentStep,
    AgentSubtask,
    AgentTrace,
    ChatRequest,
    ChatResponse,
    RetrievedChunk,
)
from app.retrieval.retriever import estimate_confidence
from app.tools.base import AgentTool
from app.tools.grading_tool import GradingTool
from app.tools.qa_tool import QATool
from app.tools.quiz_tool import QuizTool
from app.tools.summary_tool import SummaryTool


@dataclass
class PreparedSubtask:
    subtask: AgentSubtask
    tool: AgentTool
    chunks: list[RetrievedChunk]
    confidence: str
    use_pro_model: bool


class AgentService:
    def __init__(
        self,
        router: RouterAgent,
        qa_tool: QATool,
        summary_tool: SummaryTool,
        quiz_tool: QuizTool,
        grading_tool: GradingTool,
        planner: AgentPlanner | None = None,
        evidence_judge: EvidenceJudge | None = None,
    ):
        self.router = router
        self.tools: dict[str, AgentTool] = {
            qa_tool.name: qa_tool,
            summary_tool.name: summary_tool,
            quiz_tool.name: quiz_tool,
            grading_tool.name: grading_tool,
        }
        self.retriever = qa_tool.retriever
        self.planner = planner or AgentPlanner(router.llm)
        self.evidence_judge = evidence_judge or EvidenceJudge(router.llm)

    async def handle(self, db: Session, request: ChatRequest) -> ChatResponse:
        state, prepared = await self._prepare(db, request)
        debug_trace = _debug_trace_enabled(request)

        responses: list[ChatResponse] = []
        for item in prepared:
            response = await item.tool.generate_with_chunks(
                query=item.subtask.query,
                chunks=item.chunks,
                confidence=item.confidence,
                use_pro_model=item.use_pro_model,
                agent_trace=state.trace if debug_trace and len(prepared) == 1 else None,
            )
            responses.append(response)
            state.trace.steps.append(
                AgentStep(
                    step_type="tool_generate",
                    input={"tool": item.tool.name, "query": item.subtask.query},
                    output={"confidence": response.confidence, "message": response.message},
                )
            )

        if len(responses) == 1:
            response = responses[0]
            if debug_trace:
                response.agent_trace = state.trace
            return response

        return ChatResponse(
            task_type="multi",
            answer=_format_multi_answer(responses),
            sources=_merge_chunks([chunk for response in responses for chunk in response.sources]),
            confidence=_aggregate_confidence([response.confidence for response in responses]),
            message=_aggregate_message([response.message for response in responses]),
            agent_trace=state.trace if debug_trace else None,
        )

    async def handle_stream(
        self, db: Session, request: ChatRequest
    ) -> AsyncIterator[dict[str, Any]]:
        state, prepared = await self._prepare(db, request)
        debug_trace = _debug_trace_enabled(request)
        all_sources = _merge_chunks([chunk for item in prepared for chunk in item.chunks])
        confidence = _aggregate_confidence([item.confidence for item in prepared])
        message = "low_retrieval_confidence" if any(item.confidence == "low" for item in prepared) else None
        task_type = prepared[0].subtask.task_type if len(prepared) == 1 else "multi"

        meta: dict[str, Any] = {
            "type": "meta",
            "task_type": task_type,
            "confidence": confidence,
            "sources": [chunk.model_dump() for chunk in all_sources],
            "message": message,
        }
        if debug_trace:
            meta["agent_trace"] = state.trace.model_dump()
        yield meta

        for index, item in enumerate(prepared, start=1):
            if len(prepared) > 1:
                yield {"type": "delta", "text": f"\n\n### {_subtask_title(item.subtask, index)}\n\n"}

            async for event in item.tool.stream_with_chunks(
                query=item.subtask.query,
                chunks=item.chunks,
                confidence=item.confidence,
                use_pro_model=item.use_pro_model,
            ):
                if event.get("type") in {"meta", "done"}:
                    continue
                yield event

            state.trace.steps.append(
                AgentStep(
                    step_type="tool_stream",
                    input={"tool": item.tool.name, "query": item.subtask.query},
                    output={"confidence": item.confidence},
                )
            )

        yield {"type": "done"}

    async def _prepare(self, db: Session, request: ChatRequest) -> tuple[AgentState, list[PreparedSubtask]]:
        decision = await self.router.decide(request.query, request.task_type)
        base_use_pro_model = request.use_pro_model or decision.needs_pro_model
        subtasks = await self.planner.plan(
            query=request.query,
            decision=decision,
            user_task_type=request.task_type,
        )
        if not subtasks:
            subtasks = [
                AgentSubtask(
                    task_type=decision.task_type,
                    query=request.query,
                    retrieval_profile=decision.retrieval_profile,
                    rewritten_query=decision.rewritten_query or request.query,
                    reason=decision.reason,
                )
            ]

        state = AgentState(
            original_query=request.query,
            task_type=decision.task_type,
            retrieval_profile=decision.retrieval_profile,
            use_pro_model=base_use_pro_model,
            subtasks=subtasks,
        )
        state.trace.steps.append(
            AgentStep(
                step_type="route",
                input={"query": request.query, "user_task_type": request.task_type},
                output=decision.model_dump(),
            )
        )
        state.trace.steps.append(
            AgentStep(
                step_type="plan",
                input={"query": request.query},
                output={"subtasks": [subtask.model_dump() for subtask in subtasks]},
            )
        )

        prepared: list[PreparedSubtask] = []
        for subtask in subtasks:
            tool = self.tools[subtask.task_type]
            use_pro_model = base_use_pro_model or tool.force_pro_model
            chunks, confidence = await self._retrieve_with_evidence_loop(
                db=db,
                request=request,
                subtask=subtask,
                tool=tool,
                trace=state.trace,
            )
            prepared.append(
                PreparedSubtask(
                    subtask=subtask,
                    tool=tool,
                    chunks=chunks,
                    confidence=confidence,
                    use_pro_model=use_pro_model,
                )
            )

        state.final_sources = _merge_chunks([chunk for item in prepared for chunk in item.chunks])
        return state, prepared

    async def _retrieve_with_evidence_loop(
        self,
        *,
        db: Session,
        request: ChatRequest,
        subtask: AgentSubtask,
        tool: AgentTool,
        trace: AgentTrace,
    ) -> tuple[list[RetrievedChunk], str]:
        top_k = request.top_k or tool.default_top_k()
        profile = subtask.retrieval_profile or tool.default_profile
        first_query = subtask.rewritten_query or subtask.query

        chunks = self._retrieve_many(db, [first_query], top_k, profile)
        confidence = estimate_confidence(chunks)
        trace.steps.append(
            AgentStep(
                step_type="retrieve",
                input={"queries": [first_query], "profile": profile, "top_k": top_k},
                output={"chunk_count": len(chunks), "confidence": confidence},
            )
        )

        decision = await self.evidence_judge.judge(
            query=subtask.query,
            chunks=chunks,
            task_type=subtask.task_type,
            retrieval_profile=profile,
            confidence=confidence,
        )
        trace.steps.append(
            AgentStep(
                step_type="judge_evidence",
                input={"task_type": subtask.task_type, "profile": profile},
                output=decision.model_dump(),
            )
        )

        if decision.is_sufficient or decision.should_refuse:
            return chunks, "low" if decision.should_refuse else confidence

        second_queries = [query for query in decision.suggested_queries if query and query != first_query][:3]
        if not second_queries:
            return chunks, confidence

        second_profile = decision.suggested_profile or profile
        second_chunks = self._retrieve_many(db, second_queries, top_k, second_profile)
        chunks = _merge_chunks(chunks + second_chunks)[:top_k]
        confidence = estimate_confidence(chunks)
        trace.steps.append(
            AgentStep(
                step_type="retrieve_retry",
                input={"queries": second_queries, "profile": second_profile, "top_k": top_k},
                output={"chunk_count": len(chunks), "confidence": confidence},
            )
        )
        return chunks, confidence

    def _retrieve_many(
        self,
        db: Session,
        queries: list[str],
        top_k: int,
        profile: str,
    ) -> list[RetrievedChunk]:
        chunks: list[RetrievedChunk] = []
        for query in queries:
            chunks.extend(self.retriever.retrieve(db, query, top_k, profile=profile))
        return _merge_chunks(chunks)[:top_k]


def _debug_trace_enabled(request: ChatRequest) -> bool:
    return bool((request.extra_context or {}).get("debug_agent_trace"))


def _merge_chunks(chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
    merged: dict[str, RetrievedChunk] = {}
    for chunk in chunks:
        current = merged.get(chunk.chunk_id)
        if current is None or _chunk_rank_score(chunk) > _chunk_rank_score(current):
            merged[chunk.chunk_id] = chunk
    return sorted(merged.values(), key=_chunk_rank_score, reverse=True)


def _chunk_rank_score(chunk: RetrievedChunk) -> float:
    if chunk.rerank_score is not None:
        return chunk.rerank_score
    return chunk.final_score


def _aggregate_confidence(confidences: list[str]) -> str:
    if not confidences or "low" in confidences:
        return "low"
    if "medium" in confidences:
        return "medium"
    return "high"


def _aggregate_message(messages: list[str | None]) -> str | None:
    unique = [message for message in dict.fromkeys(messages) if message]
    return ";".join(unique) if unique else None


def _format_multi_answer(responses: list[ChatResponse]) -> str:
    parts: list[str] = []
    for index, response in enumerate(responses, start=1):
        parts.append(f"### {_task_label(response.task_type, index)}\n{response.answer}")
    return "\n\n".join(parts)


def _subtask_title(subtask: AgentSubtask, index: int) -> str:
    return _task_label(subtask.task_type, index)


def _task_label(task_type: str, index: int) -> str:
    labels = {
        "qa": "问答",
        "summary": "总结",
        "quiz": "练习题",
        "grade": "批改",
    }
    return f"{index}. {labels.get(task_type, task_type)}"
