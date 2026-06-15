from app.core.schemas import RetrievedChunk


QA_SYSTEM_PROMPT = """
你是一个严谨的中文课程学习助手。
你必须优先根据给定的课程资料回答问题。
如果资料中没有充分依据，请明确说明“课程资料中未找到充分依据”，不要编造。
回答要清晰、简洁，适合学生复习使用。
""".strip()


SUMMARY_SYSTEM_PROMPT = """
你是一个严谨的中文课程学习助手。
你必须根据给定课程资料做知识点总结。
如果资料覆盖不足，请在总结开头明确说明“课程资料依据不足，以下仅基于已检索到的片段整理”。
不要编造课程资料中没有的信息。
""".strip()


QUIZ_SYSTEM_PROMPT = """
你是一个严谨的中文课程助教。
你必须根据给定课程资料生成练习题。
题目、答案和考察点都必须有资料依据，不要编造资料中没有的信息。
""".strip()


GRADING_SYSTEM_PROMPT = """
你是一个严谨的中文课程助教。
你必须根据给定课程资料批改学生答案并给出学习反馈。
如果资料依据不足，请明确说明依据不足，不要编造参考答案。
如果用户提交的是选择题，你的首要任务是判断学生选择的选项是否是题干给定选项中最符合课程资料的一项。
不要因为正确选项没有覆盖课程资料中的全部知识点，就把选择题答案判为“部分正确”。
""".strip()


def format_chunks(chunks: list[RetrievedChunk]) -> str:
    lines: list[str] = []
    for index, chunk in enumerate(chunks, start=1):
        location = chunk.source_file
        if chunk.page is not None:
            location += f"，第 {chunk.page} 页"
        if chunk.heading:
            location += f"，标题：{chunk.heading}"
        lines.append(f"[{index}] 来源：{location}\n{chunk.text}")
    return "\n\n".join(lines)


def build_qa_prompt(query: str, chunks: list[RetrievedChunk]) -> str:
    context = format_chunks(chunks)
    return f"""
【课程资料片段】
{context}

【用户问题】
{query}

【回答要求】
1. 先直接回答问题。
2. 再给出必要解释。
3. 如果资料依据不足，请说明依据不足。
4. 不要编造课程资料中没有的信息。
""".strip()


def build_summary_prompt(query: str, chunks: list[RetrievedChunk]) -> str:
    context = format_chunks(chunks)
    return f"""
【课程资料片段】
{context}

【用户要求】
{query}

【总结要求】
1. 只基于课程资料片段总结。
2. 如果资料依据不足，请先说明依据不足。
3. 如果用户问题比较宽泛，请先说明“按资料检索到的主题组织如下”，再展开复习提纲。
4. 输出要适合复习，覆盖概念、方法分类、关键机制或算法、典型例子、对比关系、易混点。
5. 不要编造课程资料中没有的信息。

【输出格式】
一、核心概念
二、方法分类与关键机制
三、典型例子或计算过程
四、方法之间的对比关系
五、易混淆点与常见难点
六、复习建议
""".strip()


def build_quiz_prompt(query: str, chunks: list[RetrievedChunk]) -> str:
    context = format_chunks(chunks)
    return f"""
【课程资料片段】
{context}

【出题要求】
{query}

【要求】
1. 题目必须基于课程资料片段。
2. 每道题给出参考答案。
3. 每道题标注考察知识点。
4. 严格遵守用户指定的题目数量；如果用户要求“一道”“1 道”“一个”题目，只能生成 1 道题。
5. 如果用户没有指定题目数量和题型，默认生成判断题、选择题、简答题各一道。
6. 如果用户只指定数量但没有指定题型，自行选择最适合该数量和要求的题型，不要额外增加题目。
7. 如果资料依据不足，请先说明依据不足。
""".strip()


def build_grading_prompt(query: str, chunks: list[RetrievedChunk]) -> str:
    context = format_chunks(chunks)
    return f"""
【课程资料片段】
{context}

【学生提交内容】
{query}

【批改要求】
1. 只依据课程资料片段评价答案。
2. 如果资料依据不足，请明确说明。
3. 不要编造课程资料中没有的信息。
4. 如果题目包含 A/B/C/D 等选择题选项：
   - 先判断“我的答案”所选选项是否是题干给定选项中的最佳答案。
   - 判分对象是学生在该题中的选择，而不是题目选项是否覆盖课程资料的全部知识点。
   - 不要因为最佳选项只列出了课程资料中的部分知识点，就判为“部分正确”。
   - 可以在“知识补充”中说明课程资料还有更完整表述，但该补充不能影响选择题判分。
   - 只有当所有选项都明显不符合课程资料，或题干与资料明显冲突时，才指出题目可能存在问题。

【输出格式】
总体评价：正确 / 错误 / 部分正确
判定依据：
选项分析：
需要补充或修改的点：
建议修改后的参考答案或正确选项：
知识补充：
相关资料依据：
""".strip()
