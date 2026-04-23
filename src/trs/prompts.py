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

DIRECT_PROMPT = "{problem}"

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
    You are a reasoning-skill distiller. Convert one problem-solving trajectory into a reusable,
    abstract skill card and 10-20 retrieval keywords. Do not copy instance-specific constants,
    literal answers, or full code. Produce compact, actionable procedural knowledge.
    Return strict JSON.
    """
)

# The paper explicitly states that the exact extraction prompts will be released later, so this is a
# schema-faithful reconstruction rather than an exact leaked prompt.
EXTRACTION_USER = dedent(
    """\
    Distill the following experience record into one reusable reasoning skill.

    Constraints:
    1. Abstraction: avoid copying instance-specific constants, final answers, or full code.
    2. Actionability: make the procedure executable as steps/checks.
    3. Compactness: each field should be brief bullet-style phrases.
    4. If correctness=1, summarize the essential solution pattern.
    5. If correctness=0, summarize an anti-pattern and a concrete reusable fix.
    6. Output 10-20 retrieval keywords.

    Required JSON schema:
    {{
      "keywords": ["..."],
      "card": {{
        "trigger": ["..."],
        "do": ["..."],
        "avoid": ["..."],
        "check": ["..."],
        "risk": ["..."]
      }}
    }}

    Question:
    {question}

    Reference answer:
    {answer}

    Model trajectory / final response:
    {trajectory}

    correctness={correctness}
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
