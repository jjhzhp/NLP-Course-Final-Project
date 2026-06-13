from app.core.prompts import QUIZ_SYSTEM_PROMPT, build_quiz_prompt
from app.tools.base import AgentTool


class QuizTool(AgentTool):
    name = "quiz"
    description = "根据课程资料生成练习题和参考答案"
    default_profile = "summary"
    system_prompt = QUIZ_SYSTEM_PROMPT
    prompt_builder = staticmethod(build_quiz_prompt)
    temperature = 0.3
    max_tokens = 1600
