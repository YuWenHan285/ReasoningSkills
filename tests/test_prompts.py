from trs.prompts import render_trs_prompt


def test_render_prompt_normal():
    text = render_trs_prompt(problem="2+2?", solving_hints="Do arithmetic.", style="normal")
    assert "[Solving Hints]" in text
    assert "Problem: 2+2?" in text
