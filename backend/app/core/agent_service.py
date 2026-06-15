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
from app.core.summary_coverage import SummaryCoveragePlanner, is_outline_like_chunk
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
    refusal_message: str | None = None


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
        summary_coverage_planner: SummaryCoveragePlanner | None = None,
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
        self.summary_coverage_planner = summary_coverage_planner or SummaryCoveragePlanner(router.llm)

    async def handle(self, db: Session, request: ChatRequest) -> ChatResponse:
        state, prepared = await self._prepare(db, request)
        debug_trace = _debug_trace_enabled(request)

        responses: list[ChatResponse] = []
        for item in prepared:
            if item.refusal_message:
                response = _refusal_response(item, state.trace if debug_trace and len(prepared) == 1 else None)
            else:
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
                    step_type="tool_refuse" if item.refusal_message else "tool_generate",
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
        if any(item.refusal_message for item in prepared):
            message = "evidence_refusal"
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

            if item.refusal_message:
                yield {"type": "delta", "text": item.refusal_message}
                state.trace.steps.append(
                    AgentStep(
                        step_type="tool_refuse",
                        input={"tool": item.tool.name, "query": item.subtask.query},
                        output={"confidence": item.confidence, "message": "evidence_refusal"},
                    )
                )
                continue

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
            if subtask.task_type == "summary":
                chunks, confidence, refusal_message = await self._retrieve_summary_coverage_loop(
                    db=db,
                    request=request,
                    subtask=subtask,
                    tool=tool,
                    trace=state.trace,
                )
            else:
                chunks, confidence, refusal_message = await self._retrieve_with_evidence_loop(
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
                    refusal_message=refusal_message,
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
    ) -> tuple[list[RetrievedChunk], str, str | None]:
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

        if decision.should_refuse:
            return chunks, "low", _build_refusal_message(decision.reason)
        if decision.is_sufficient:
            return chunks, confidence, None

        second_queries = [query for query in decision.suggested_queries if query and query != first_query][:3]
        if not second_queries:
            return chunks, confidence, None

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
        return chunks, confidence, None

    async def _retrieve_summary_coverage_loop(
        self,
        *,
        db: Session,
        request: ChatRequest,
        subtask: AgentSubtask,
        tool: AgentTool,
        trace: AgentTrace,
    ) -> tuple[list[RetrievedChunk], str, str | None]:
        target_top_k = _summary_target_top_k(request.top_k, tool.default_top_k())
        profile = subtask.retrieval_profile or tool.default_profile
        initial_queries = await self.summary_coverage_planner.plan_queries(
            subtask.query,
            subtask.rewritten_query,
        )
        round_queries = [subtask.query, subtask.rewritten_query or subtask.query]
        round_queries = _dedupe_strings([query for query in round_queries if query])
        all_queries: list[str] = []
        chunks: list[RetrievedChunk] = []
        previous_chunk_ids: set[str] = set()
        last_decision_reason = ""

        for round_index in range(1, 4):
            if round_index == 1:
                queries = round_queries
            elif round_index == 2:
                queries = [query for query in initial_queries if query not in all_queries][:6]
            else:
                queries = [query for query in round_queries if query not in all_queries][:4]

            if not queries:
                trace.steps.append(
                    AgentStep(
                        step_type=f"retrieve_round_{round_index}",
                        input={"queries": [], "profile": profile, "top_k": target_top_k},
                        output={
                            "chunk_count": len(chunks),
                            "confidence": estimate_confidence(chunks),
                            "stop_reason": "no_new_queries",
                        },
                    )
                )
                trace.steps.append(
                    AgentStep(
                        step_type="summary_coverage_stop",
                        input={"round": round_index},
                        output={"reason": "no_new_queries"},
                    )
                )
                break

            all_queries.extend(queries)
            round_chunks = self._retrieve_many(db, queries, target_top_k, profile)
            chunks = _select_summary_coverage_chunks(chunks + round_chunks, target_top_k)
            confidence = estimate_confidence(chunks)
            new_chunk_count = len({chunk.chunk_id for chunk in chunks} - previous_chunk_ids)
            previous_chunk_ids = {chunk.chunk_id for chunk in chunks}
            trace.steps.append(
                AgentStep(
                    step_type=f"retrieve_round_{round_index}",
                    input={"queries": queries, "profile": profile, "top_k": target_top_k},
                    output={
                        "chunk_count": len(chunks),
                        "new_chunk_count": new_chunk_count,
                        "confidence": confidence,
                    },
                )
            )

            decision = await self.summary_coverage_planner.judge(
                query=subtask.query,
                chunks=chunks,
                round_index=round_index,
                previous_queries=all_queries,
            )
            last_decision_reason = decision.reason
            trace.steps.append(
                AgentStep(
                    step_type=f"summary_coverage_judge_{round_index}",
                    input={"round": round_index},
                    output=decision.model_dump(),
                )
            )

            if decision.should_refuse:
                return chunks, "low", _build_refusal_message(decision.reason)
            if decision.is_sufficient:
                trace.steps.append(
                    AgentStep(
                        step_type="summary_coverage_stop",
                        input={"round": round_index},
                        output={"reason": "coverage_sufficient", "covered_topics": decision.covered_topics},
                    )
                )
                return chunks, confidence, None
            if round_index >= 3:
                trace.steps.append(
                    AgentStep(
                        step_type="summary_coverage_stop",
                        input={"round": round_index},
                        output={"reason": "max_rounds_reached", "missing_topics": decision.missing_topics},
                    )
                )
                return chunks, confidence, None
            if new_chunk_count == 0:
                trace.steps.append(
                    AgentStep(
                        step_type="summary_coverage_stop",
                        input={"round": round_index},
                        output={"reason": "no_new_chunks", "missing_topics": decision.missing_topics},
                    )
                )
                return chunks, confidence, None

            round_queries = decision.next_queries

        return chunks, estimate_confidence(chunks), None if chunks else _build_refusal_message(last_decision_reason)

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


def _summary_target_top_k(request_top_k: int | None, default_top_k: int) -> int:
    requested = request_top_k or default_top_k
    return min(max(requested, 12), 14)


def _select_summary_coverage_chunks(chunks: list[RetrievedChunk], top_k: int) -> list[RetrievedChunk]:
    ranked = sorted(chunks, key=_summary_chunk_score, reverse=True)
    selected: list[RetrievedChunk] = []
    selected_ids: set[str] = set()
    seen_groups: set[tuple[str, str]] = set()

    for chunk in ranked:
        if chunk.chunk_id in selected_ids:
            continue
        group = _summary_group(chunk)
        if group in seen_groups and len(selected) < max(top_k // 2, 1):
            continue
        selected.append(chunk)
        selected_ids.add(chunk.chunk_id)
        seen_groups.add(group)
        if len(selected) >= top_k:
            return selected

    for chunk in ranked:
        if chunk.chunk_id in selected_ids:
            continue
        selected.append(chunk)
        selected_ids.add(chunk.chunk_id)
        if len(selected) >= top_k:
            break
    return selected


def _summary_chunk_score(chunk: RetrievedChunk) -> float:
    score = _chunk_rank_score(chunk)
    text_length = len(chunk.text.strip())
    if is_outline_like_chunk(chunk):
        score -= 1.0
    if text_length >= 180:
        score += 0.2
    if any(word in chunk.text for word in ("示例", "例：", "算法", "模型", "概率", "优点", "缺点", "歧义", "未登录词")):
        score += 0.25
    return score


def _summary_group(chunk: RetrievedChunk) -> tuple[str, str]:
    if chunk.heading:
        return (chunk.document_id, f"heading:{chunk.heading}")
    if chunk.page is not None:
        return (chunk.document_id, f"page:{chunk.page}")
    return (chunk.document_id, chunk.chunk_id)


def _dedupe_strings(items: list[str]) -> list[str]:
    return [item for item in dict.fromkeys(text.strip() for text in items if text and text.strip())]


def _refusal_response(item: PreparedSubtask, agent_trace: AgentTrace | None) -> ChatResponse:
    return ChatResponse(
        task_type=item.subtask.task_type,
        answer=item.refusal_message or _build_refusal_message("evidence_refusal"),
        sources=item.chunks,
        confidence="low",
        message="evidence_refusal",
        agent_trace=agent_trace,
    )


def _build_refusal_message(reason: str) -> str:
    return (
        "课程资料中未找到足够可靠的依据来完成这个请求，因此我不能基于当前资料给出完整回答。"
        f"依据不足原因：{reason}"
    )


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
