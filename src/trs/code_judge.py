from __future__ import annotations

import json
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path


@dataclass
class TestCase:
    stdin: str
    stdout: str


@dataclass
class JudgeResult:
    passed: bool
    passed_cases: int
    total_cases: int
    error: str | None = None


def _normalize_output(text: str) -> str:
    return "\n".join(line.rstrip() for line in text.strip().splitlines()).strip()


def run_python_solution(code: str, testcases: list[TestCase], timeout_sec: int = 2) -> JudgeResult:
    with tempfile.TemporaryDirectory() as tmpdir:
        script_path = Path(tmpdir) / "solution.py"
        script_path.write_text(code, encoding="utf-8")
        passed = 0
        for tc in testcases:
            try:
                proc = subprocess.run(
                    ["python", str(script_path)],
                    input=tc.stdin,
                    text=True,
                    capture_output=True,
                    timeout=timeout_sec,
                )
            except subprocess.TimeoutExpired:
                return JudgeResult(False, passed, len(testcases), error="timeout")
            if proc.returncode != 0:
                return JudgeResult(False, passed, len(testcases), error=proc.stderr[:500])
            if _normalize_output(proc.stdout) != _normalize_output(tc.stdout):
                return JudgeResult(False, passed, len(testcases), error="wrong_answer")
            passed += 1
        return JudgeResult(True, passed, len(testcases))


def parse_testcases(raw: object) -> list[TestCase]:
    if isinstance(raw, str):
        raw = json.loads(raw)
    out: list[TestCase] = []
    if not isinstance(raw, list):
        return out
    for item in raw:
        if not isinstance(item, dict):
            continue
        stdin = str(item.get("input", item.get("stdin", "")))
        stdout = str(item.get("output", item.get("stdout", "")))
        out.append(TestCase(stdin=stdin, stdout=stdout))
    return out
