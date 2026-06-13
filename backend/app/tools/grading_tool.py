from app.core.prompts import GRADING_SYSTEM_PROMPT, build_grading_prompt
from app.tools.base import AgentTool


class GradingTool(AgentTool):
    name = "grade"
    description = "根据课程资料批改或评价学生答案"
    default_profile = "summary"
    system_prompt = GRADING_SYSTEM_PROMPT
    prompt_builder = staticmethod(build_grading_prompt)
    temperature = 0.2
    max_tokens = 1600
    force_pro_model = True
