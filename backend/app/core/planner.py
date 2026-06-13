import json

import httpx

from app.core.llm_client import DeepSeekClient
from app.core.schemas import AgentDecision, AgentSubtask


PLANNER_SYSTEM_PROMPT = """
你是中文课程学习助手的任务规划器。
你只负责判断用户请求是否需要拆成多个子任务，不回答问题。
必须只输出 JSON 对象，不要输出 Markdown。

输出 JSON 字段：
{
  "subtasks": [
    {
      "task_type": "qa|summary|quiz|grade",
      "query": "该子任务要处理的用户请求",
      "retrieval_profile": "qa|summary",
      "rewritten_query": "用于该子任务检索的 query",
      "reason": "简短原因"
    }
  ]
}

规则：
- 单一请求只输出一个子任务。
- 同时要求总结和出题时，拆成 summary 后 quiz。
- 同时要求批改和总结时，拆成 grade 后 summary。
- 最多输出 3 个子任务。
""".strip()


class AgentPlanner:
    def __init__(self, llm: DeepSeekClient | None = None):
        self.llm = llm

    async def plan(
        self,
        *,
        query: str,
        decision: AgentDecision,
        user_task_type: str | None,
    ) -> list[AgentSubtask]:
        if user_task_type and user_task_type != "auto":
            return [_subtask_from_decision(query, decision)]

        if self.llm is not None:
            try:
                planned = await self._plan_with_llm(query, decision)
                if planned:
                    return planned[:3]
            except (RuntimeError, httpx.HTTPError, ValueError, json.JSONDecodeError):
                pass

        return self._plan_by_rules(query, decision)

    async def _plan_with_llm(self, query: str, decision: AgentDecision) -> list[AgentSubtask]:
        assert self.llm is not None
        content = await self.llm.chat(
            messages=[
                {"role": "system", "content": PLANNER_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"Router 初判任务：{decision.task_type}\n"
                        f"Router 检索策略：{decision.retrieval_profile}\n"
                        f"Router 改写 query：{decision.rewritten_query or query}\n\n"
                        f"用户请求：{query}"
                    ),
                },
            ],
            temperature=0.0,
            max_tokens=700,
        )
        payload = _parse_json_object(content)
        subtasks = payload.get("subtasks") or []
        return [AgentSubtask.model_validate(item) for item in subtasks][:3]

    def _plan_by_rules(self, query: str, decision: AgentDecision) -> list[AgentSubtask]:
        text = query.strip()
        wants_summary = any(word in text for word in ("总结", "梳理", "归纳", "复习", "提纲"))
        wants_quiz = any(word in text for word in ("出题", "练习题", "道题", "考考我"))
        wants_grade = any(word in text for word in ("我的答案", "批改", "对不对", "评分", "评价一下"))

        if wants_summary and wants_quiz:
            return [
                AgentSubtask(
                    task_type="summary",
                    query=query,
                    retrieval_profile="summary",
                    rewritten_query=decision.rewritten_query or query,
                    reason="rule_summary_part",
                ),
                AgentSubtask(
                    task_type="quiz",
                    query=query,
                    retrieval_profile="summary",
                    rewritten_query=decision.rewritten_query or query,
                    reason="rule_quiz_part",
                ),
            ]
        if wants_grade and wants_summary:
            return [
                AgentSubtask(
                    task_type="grade",
                    query=query,
                    retrieval_profile="summary",
                    rewritten_query=decision.rewritten_query or query,
                    reason="rule_grade_part",
                ),
                AgentSubtask(
                    task_type="summary",
                    query=query,
                    retrieval_profile="summary",
                    rewritten_query=decision.rewritten_query or query,
                    reason="rule_summary_part",
                ),
            ]
        return [_subtask_from_decision(query, decision)]


def _subtask_from_decision(query: str, decision: AgentDecision) -> AgentSubtask:
    return AgentSubtask(
        task_type=decision.task_type,
        query=query,
        retrieval_profile=decision.retrieval_profile,
        rewritten_query=decision.rewritten_query or query,
        reason=decision.reason,
    )


def _parse_json_object(content: str) -> dict:
    text = content.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("Planner output does not contain a JSON object")
    return json.loads(text[start : end + 1])
