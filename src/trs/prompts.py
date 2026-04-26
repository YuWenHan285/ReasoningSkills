from __future__ import annotations

from textwrap import dedent

TRS_NORMAL = dedent(
    """\
    You are a helpful and harmless assistant.
    You may be given an optional Solving Hints
    section. Use it only if it is relevant to the
    problem; otherwise, ignore it completely.
    [Solving Hints] {solving_hints} [/Solving Hints]
    Problem: {problem}
    """
)

TRS_ONLY = dedent(
    """\
    You are a helpful and harmless assistant.
    You may be given an optional Solving Hints
    section. Use it only if it is relevant to the
    problem; otherwise, ignore it completely.
    [Solving Hints] {solving_hints} [/Solving Hints]
    Only try to reduce the number of tokens used
    if the solution hints are useful; otherwise,
    please think normally. Problem: {problem}
    """
)

TRS_TRY_TO = dedent(
    """\
    You are a helpful and harmless assistant.
    You may be given an optional Solving Hints
    section. Use it only if it is relevant to the
    problem; otherwise, ignore it completely.
    [Solving Hints] {solving_hints} [/Solving Hints]
    If you use the solving hints, please try to
    reduce the number of tokens used. Problem:
    {problem}
    """
)

TRS_SHORT = dedent(
    """\
    You are a helpful and harmless assistant.
    You may be given an optional Solving Hints
    section. Use it only if it is relevant to the
    problem; otherwise, ignore it completely.
    [Solving Hints] {solving_hints} [/Solving Hints]
    Let’s think step by step and use less than
    {budget} tokens: {problem}
    """
)

TRS_DRAFT = dedent(
    """\
    You are a helpful and harmless assistant.
    You may be given an optional Solving Hints
    section. Use it only if it is relevant to the
    problem; otherwise, ignore it completely.
    [Solving Hints] {solving_hints} [/Solving Hints]
    Think step by step, but only keep a minimum
    draft for each thinking step, with 5 words at
    most. Problem: {problem}
    """
)

COD_PROMPT = dedent(
    """\
    Question:
    {question}
    Think step by step, but only keep a minimum
    draft for each thinking step, with 5 words at
    most. Return the answer at the end of the
    response after a separator ####.
    """
)

NOWAIT_PROMPT = dedent(
    """\
    Question:
    {question}
    Think step by step. Do not use any of the
    following words in your thinking process:
    “wait”, “alternatively”, “hmm”, “but”,
    “however”, “alternative”, “another”, “check”,
    “double-check”, “oh”, “maybe”, “verify”,
    “other”, “again”, “now”, “ah”, “any”.
    """
)

TALE_SOLVE_PROMPT = dedent(
    """\
    Question:
    {question}
    Let’s think step by step and use less than
    {budget} tokens:
    ----
    Question:
    {question}
    """
)

TALE_BUDGET_ESTIMATE_PROMPT = dedent(
    """\
    Question:
    {question}
    Task: Analyze the given question and estimate
    the minimum number of tokens required to
    generate a complete and accurate response.
    Please give the response by strictly following
    this format: [[budget]], for example, Budget:
    [[12]].
    """
)

DIRECT_PROMPT = "You are a helpful and harmless assistant.\n Let's think step by step: {problem}"

PROMPT_STYLES = {
    "normal": TRS_NORMAL,
    "only": TRS_ONLY,
    "try_to": TRS_TRY_TO,
    "short": TRS_SHORT,
    "draft": TRS_DRAFT,
    "direct": DIRECT_PROMPT,
}


EXTRACTION_SYSTEM = dedent(
    """\
    You are a principal engineer and researcher specializing in distilling reusable reasoning skills from solved examples.

    Objective:
    I will provide a Problem, a reasoning trace, and the Correct Answer. Synthesize them into generalized advice for future retrieval and inference.

    Instructions:
    - If the trace is correct, extract the efficient logical path.
    - If the trace is incorrect, identify the failure mode and how to avoid it.
    - Keep the advice abstract and reusable; do not copy specific constants unless they are essential domain terms.
    - Produce retrieval anchors that a lexical/BM25 retriever can match against future questions.

    Section 1: Guidelines for <learned_heuristic>
    - Provide a numbered list with at most 5 items.
    - Use this shape: "When encountering [abstract pattern], adopt [specific strategy] because [reason]. Be cautious of [pitfall]."

    Section 2: Guidelines for <retrieval_keywords>
    - Provide 10 to 20 comma-separated keywords or short phrases.
    - Mix domain terms, constraint descriptors, and problem-style cues.

    Output Format:
    Output only this XML structure.

    <learned_heuristic>
    1. [Heuristic 1]
    2. [Heuristic 2]
    ...
    </learned_heuristic>

    <retrieval_keywords>
    [keyword 1], [keyword 2], [keyword 3], ...
    </retrieval_keywords>
    """
)

EXTRACTION_USER = dedent(
    """\
    Input Data:
    [Problem]:
    {question}

    [Reasoning Trace]:
    {trajectory}

    [Correct Answer]:
    {answer}
    """
)


def render_trs_prompt(
    *,
    problem: str,
    solving_hints: str,
    style: str,
    budget: int | None = None,
) -> str:
    template = PROMPT_STYLES[style]
    payload = {"problem": problem, "solving_hints": solving_hints, "budget": budget}
    return dedent(template.format(**payload)).strip()
