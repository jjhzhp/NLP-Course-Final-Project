import json
import re

import httpx

from app.core.llm_client import DeepSeekClient
from app.core.schemas import EvidenceDecision, RetrievedChunk


EVIDENCE_SYSTEM_PROMPT = """
你是中文课程 RAG 系统的证据充分性判断器。
你只判断检索片段是否足以支持回答，不直接回答用户问题。
必须只输出 JSON 对象，不要输出 Markdown。

输出 JSON 字段：
{
  "is_sufficient": true,
  "reason": "简短原因",
  "suggested_queries": ["证据不足时给出的改写检索 query"],
  "suggested_profile": "qa|summary|null",
  "should_refuse": false
}

规则：
- 如果片段没有覆盖问题中的核心概念，应判为 is_sufficient=false。
- 如果任务是 summary/quiz/grade，片段应覆盖较多相关知识点。
- 如果课程资料明显不足以回答，应设置 should_refuse=true。
""".strip()


class EvidenceJudge:
    def __init__(self, llm: DeepSeekClient | None = None):
        self.llm = llm

    async def judge(
        self,
        *,
        query: str,
        chunks: list[RetrievedChunk],
        task_type: str,
        retrieval_profile: str,
        confidence: str,
    ) -> EvidenceDecision:
        rule_decision = self._judge_by_rules(query, chunks, task_type, retrieval_profile, confidence)
        if not chunks or rule_decision.is_sufficient:
            return rule_decision

        if self.llm is None:
            return rule_decision

        try:
            return await self._judge_with_llm(query, chunks, task_type, retrieval_profile)
        except (RuntimeError, httpx.HTTPError, ValueError, json.JSONDecodeError):
            return rule_decision

    def _judge_by_rules(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        task_type: str,
        retrieval_profile: str,
        confidence: str,
    ) -> EvidenceDecision:
        if not chunks:
            return EvidenceDecision(
                is_sufficient=False,
                reason="no_retrieved_chunks",
                suggested_queries=[query],
                suggested_profile=retrieval_profile if retrieval_profile in {"qa", "summary"} else "qa",
                should_refuse=task_type == "qa",
            )

        min_chunks = 2 if task_type == "qa" else 4
        if confidence == "low" or len(chunks) < min_chunks:
            return EvidenceDecision(
                is_sufficient=False,
                reason="low_confidence_or_too_few_chunks",
                suggested_queries=_suggest_queries(query, task_type),
                suggested_profile="summary" if task_type != "qa" else "qa",
                should_refuse=False,
            )

        terms = _extract_terms(query)
        if terms and not _chunks_contain_any(chunks[:3], terms):
            return EvidenceDecision(
                is_sufficient=False,
                reason="top_chunks_do_not_cover_query_terms",
                suggested_queries=_suggest_queries(query, task_type),
                suggested_profile="summary" if task_type != "qa" else "qa",
                should_refuse=False,
            )

        return EvidenceDecision(
            is_sufficient=True,
            reason="rule_sufficient",
            suggested_queries=[],
            suggested_profile=retrieval_profile if retrieval_profile in {"qa", "summary"} else None,
            should_refuse=False,
        )

    async def _judge_with_llm(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        task_type: str,
        retrieval_profile: str,
    ) -> EvidenceDecision:
        assert self.llm is not None
        context = "\n\n".join(
            f"[{index}] {chunk.source_file}: {chunk.text[:500]}"
            for index, chunk in enumerate(chunks[:8], start=1)
        )
        content = await self.llm.chat(
            messages=[
                {"role": "system", "content": EVIDENCE_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"任务类型：{task_type}\n"
                        f"检索策略：{retrieval_profile}\n"
                        f"用户问题：{query}\n\n"
                        f"检索片段：\n{context}"
                    ),
                },
            ],
            temperature=0.0,
            max_tokens=500,
        )
        payload = _parse_json_object(content)
        if payload.get("suggested_profile") not in {"qa", "summary", None}:
            payload["suggested_profile"] = None
        return EvidenceDecision.model_validate(payload)


def _extract_terms(query: str) -> list[str]:
    terms = re.findall(r"[A-Za-z][A-Za-z0-9_-]{1,}|[\u4e00-\u9fff]{2,}", query)
    stop_words = {"什么", "总结", "梳理", "归纳", "一下", "区别", "主要", "方法", "出题", "题目", "我的答案"}
    return [term for term in terms if term not in stop_words][:8]


def _chunks_contain_any(chunks: list[RetrievedChunk], terms: list[str]) -> bool:
    text = "\n".join(chunk.text for chunk in chunks).lower()
    return any(term.lower() in text for term in terms)


def _suggest_queries(query: str, task_type: str) -> list[str]:
    cleaned = query.strip()
    if task_type == "qa":
        return [cleaned, f"{cleaned} 定义 原理"]
    if task_type == "grade":
        return [cleaned, f"{cleaned} 标准答案 关键点"]
    if task_type == "quiz":
        return [cleaned, f"{cleaned} 知识点 考点"]
    return [cleaned, f"{cleaned} 定义 主要方法 优缺点"]


def _parse_json_object(content: str) -> dict:
    text = content.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("LLM output does not contain a JSON object")
    return json.loads(text[start : end + 1])
