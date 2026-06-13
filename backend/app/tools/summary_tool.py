from app.core.prompts import SUMMARY_SYSTEM_PROMPT, build_summary_prompt
from app.tools.base import AgentTool


class SummaryTool(AgentTool):
    name = "summary"
    description = "根据课程资料总结和梳理知识点"
    default_profile = "summary"
    system_prompt = SUMMARY_SYSTEM_PROMPT
    prompt_builder = staticmethod(build_summary_prompt)
    temperature = 0.2
    max_tokens = 1600
