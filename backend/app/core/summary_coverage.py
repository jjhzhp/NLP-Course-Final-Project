import json
import re

import httpx

from app.core.llm_client import DeepSeekClient
from app.core.schemas import RetrievedChunk, SummaryCoverageDecision


SUMMARY_COVERAGE_PLAN_PROMPT = """
你是中文课程 RAG 系统的复习总结检索规划器。
你只负责把模糊总结请求拆成覆盖主要知识结构的检索 query，不回答问题。
必须只输出 JSON 对象，不要输出 Markdown。

输出 JSON 字段：
{
  "queries": ["3 到 6 个检索 query"]
}

规则：
- query 应覆盖概念、方法分类、关键算法或例子、难点、对比关系。
- query 必须围绕用户主题，不要扩展到无关课程外知识。
- query 要具体，避免“主要知识”“总结”这类泛词。
""".strip()


SUMMARY_COVERAGE_JUDGE_PROMPT = """
你是中文课程 RAG 系统的复习总结覆盖度判断器。
你只判断当前检索片段是否足以支持一次较完整的复习总结，不回答问题。
必须只输出 JSON 对象，不要输出 Markdown。

输出 JSON 字段：
{
  "is_sufficient": true,
  "reason": "简短原因",
  "covered_topics": ["已覆盖主题"],
  "missing_topics": ["缺失主题"],
  "next_queries": ["证据不足时下一轮检索 query"],
  "should_refuse": false
}

规则：
- 如果片段主要是目录页、标题页、概览页，不能判为充分。
- 完整复习总结通常应覆盖：核心概念、方法分类、关键算法或机制、例子、优缺点或适用场景、易混点。
- 如果主题明显不在课程资料中，设置 should_refuse=true。
- 如果还缺少重要主题，给出 1 到 4 个具体 next_queries。
""".strip()


DEFAULT_SUMMARY_FACETS = ("概念 定义", "方法分类", "关键算法 原理", "典型例子", "优缺点 适用场景", "难点 易混点")


class SummaryCoveragePlanner:
    def __init__(self, llm: DeepSeekClient | None = None):
        self.llm = llm

    async def plan_queries(self, query: str, rewritten_query: str | None = None) -> list[str]:
        if self.llm is not None:
            try:
                planned = await self._plan_with_llm(query, rewritten_query)
                if planned:
                    return _dedupe_queries([query, rewritten_query or "", *planned])[:8]
            except (RuntimeError, httpx.HTTPError, ValueError, json.JSONDecodeError):
                pass
        return _dedupe_queries([query, rewritten_query or "", *_rule_summary_queries(query)])[:8]

    async def judge(
        self,
        *,
        query: str,
        chunks: list[RetrievedChunk],
        round_index: int,
        previous_queries: list[str],
    ) -> SummaryCoverageDecision:
        rule_decision = self._judge_by_rules(query, chunks, previous_queries)
        if self.llm is None or rule_decision.should_refuse:
            return rule_decision

        try:
            llm_decision = await self._judge_with_llm(query, chunks, round_index, previous_queries)
            if not llm_decision.next_queries and not llm_decision.is_sufficient:
                llm_decision.next_queries = rule_decision.next_queries
            return llm_decision
        except (RuntimeError, httpx.HTTPError, ValueError, json.JSONDecodeError):
            return rule_decision

    async def _plan_with_llm(self, query: str, rewritten_query: str | None) -> list[str]:
        assert self.llm is not None
        content = await self.llm.chat(
            messages=[
                {"role": "system", "content": SUMMARY_COVERAGE_PLAN_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"用户总结请求：{query}\n"
                        f"Router 改写 query：{rewritten_query or query}"
                    ),
                },
            ],
            temperature=0.0,
            max_tokens=500,
        )
        payload = _parse_json_object(content)
        queries = payload.get("queries") or []
        return [str(item).strip() for item in queries if str(item).strip()]

    async def _judge_with_llm(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        round_index: int,
        previous_queries: list[str],
    ) -> SummaryCoverageDecision:
        assert self.llm is not None
        context = "\n\n".join(
            f"[{index}] 来源：{chunk.source_file} 页码：{chunk.page or '-'}\n{chunk.text[:700]}"
            for index, chunk in enumerate(chunks[:12], start=1)
        )
        content = await self.llm.chat(
            messages=[
                {"role": "system", "content": SUMMARY_COVERAGE_JUDGE_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"用户总结请求：{query}\n"
                        f"当前轮次：{round_index}\n"
                        f"已尝试 query：{previous_queries}\n\n"
                        f"检索片段：\n{context}"
                    ),
                },
            ],
            temperature=0.0,
            max_tokens=700,
        )
        payload = _parse_json_object(content)
        return SummaryCoverageDecision.model_validate(payload)

    def _judge_by_rules(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        previous_queries: list[str],
    ) -> SummaryCoverageDecision:
        if not chunks:
            return SummaryCoverageDecision(
                is_sufficient=False,
                reason="no_retrieved_chunks",
                missing_topics=list(DEFAULT_SUMMARY_FACETS[:4]),
                next_queries=_rule_summary_queries(query),
                should_refuse=False,
            )

        non_outline = [chunk for chunk in chunks if not is_outline_like_chunk(chunk)]
        covered_topics = _covered_topics(chunks)
        missing_topics = [topic for topic in DEFAULT_SUMMARY_FACETS if topic not in covered_topics]
        outline_ratio = 1.0 - (len(non_outline) / len(chunks))
        diverse_groups = {_source_group(chunk) for chunk in non_outline}

        if outline_ratio >= 0.5:
            reason = "retrieved_chunks_are_mostly_outline_or_too_short"
            is_sufficient = False
        elif len(non_outline) < 6:
            reason = "too_few_detail_chunks_for_summary"
            is_sufficient = False
        elif len(diverse_groups) < 4:
            reason = "summary_sources_not_diverse_enough"
            is_sufficient = False
        elif len(missing_topics) >= 3:
            reason = "missing_major_summary_facets"
            is_sufficient = False
        else:
            reason = "summary_coverage_sufficient"
            is_sufficient = True

        next_queries = _queries_for_missing_topics(query, missing_topics)
        next_queries = [item for item in next_queries if item not in set(previous_queries)]
        return SummaryCoverageDecision(
            is_sufficient=is_sufficient,
            reason=reason,
            covered_topics=covered_topics,
            missing_topics=missing_topics,
            next_queries=next_queries[:4],
            should_refuse=False,
        )


def is_outline_like_chunk(chunk: RetrievedChunk) -> bool:
    text = re.sub(r"\s+", "", chunk.text)
    if len(text) < 90:
        return True
    outline_words = ("目录", "基本概念", "基本问题", "主要内容", "本章", "概述")
    outline_hits = sum(1 for word in outline_words if word in text)
    detail_words = ("算法", "公式", "示例", "例：", "计算", "参数", "概率", "模型", "训练", "优点", "缺点")
    detail_hits = sum(1 for word in detail_words if word in text)
    return outline_hits >= 3 and detail_hits == 0


def _rule_summary_queries(query: str) -> list[str]:
    topic = _clean_summary_topic(query)
    return [f"{topic} {facet}" for facet in DEFAULT_SUMMARY_FACETS]


def _clean_summary_topic(query: str) -> str:
    topic = query.strip()
    for word in ("总结", "梳理", "归纳", "复习", "提纲", "主要知识", "知识点"):
        topic = topic.replace(word, "")
    topic = re.sub(r"\s+", " ", topic).strip(" ，,。；;：:")
    topic = topic.rstrip("的之和与及")
    return topic or query.strip()


def _covered_topics(chunks: list[RetrievedChunk]) -> list[str]:
    text = "\n".join(chunk.text for chunk in chunks)
    covered: list[str] = []
    topic_patterns = {
        "概念 定义": ("定义", "概念", "是什么"),
        "方法分类": ("分类", "类别", "方法", "路线", "类型"),
        "关键算法 原理": ("算法", "原理", "机制", "模型", "公式", "训练"),
        "典型例子": ("例", "示例", "例如", "计算"),
        "优缺点 适用场景": ("优点", "缺点", "适用", "局限"),
        "难点 易混点": ("难点", "问题", "错误", "混淆", "歧义"),
    }
    for topic, patterns in topic_patterns.items():
        if any(pattern in text for pattern in patterns):
            covered.append(topic)
    return covered


def _queries_for_missing_topics(query: str, missing_topics: list[str]) -> list[str]:
    topic = _clean_summary_topic(query)
    return [f"{topic} {missing}" for missing in missing_topics]


def _source_group(chunk: RetrievedChunk) -> tuple[str, str]:
    if chunk.heading:
        return (chunk.document_id, f"heading:{chunk.heading}")
    if chunk.page is not None:
        return (chunk.document_id, f"page:{chunk.page}")
    return (chunk.document_id, chunk.chunk_id)


def _dedupe_queries(queries: list[str]) -> list[str]:
    return [query for query in dict.fromkeys(query.strip() for query in queries if query and query.strip())]


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
