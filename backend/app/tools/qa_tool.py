from app.core.prompts import QA_SYSTEM_PROMPT, build_qa_prompt
from app.tools.base import AgentTool


class QATool(AgentTool):
    name = "qa"
    description = "回答课程资料中的具体问题"
    default_profile = "qa"
    system_prompt = QA_SYSTEM_PROMPT
    prompt_builder = staticmethod(build_qa_prompt)
    temperature = 0.2
    max_tokens = 1200
    refuse_low_confidence = True
