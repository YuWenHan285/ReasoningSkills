from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Optional

import typer
from dotenv import load_dotenv

from trs.clients import LLMClient, env_or_value
from trs.datasets.deepmath import load_deepmath
from trs.datasets.generic import load_generic_local, load_hf_dataset
from trs.eval_math import exact_match
from trs.generation import generate_trajectories
from trs.indexing import load_skills
from trs.inference import SkillRetriever, run_inference
from trs.skill_extraction import extract_skills, load_generation_records
from trs.utils.io import append_jsonl, read_jsonl, write_jsonl

app = typer.Typer(add_completion=False, no_args_is_help=True)


def _load_examples(
    *,
    source: str,
    dataset_name: str | None,
    split: str,
    path: str | None,
    limit: int | None,
    start: int,
):
    if source == "deepmath":
        return load_deepmath(dataset_name=dataset_name or "zwhe99/DeepMath-103K", split=split, start=start, limit=limit)
    if source == "hf":
        if not dataset_name:
            raise typer.BadParameter("--dataset-name is required when source=hf")
        return load_hf_dataset(dataset_name, split, limit)
    if source == "local":
        if not path:
            raise typer.BadParameter("--path is required when source=local")
        return load_generic_local(path, limit)
    raise typer.BadParameter(f"Unsupported source: {source}")


def _make_client(model: str, api_base: str | None, api_key: str | None, temperature: float, max_tokens: int | None) -> LLMClient:
    return LLMClient(model=model, api_base=api_base, api_key=api_key, temperature=temperature, max_tokens=max_tokens)


@app.command()
def generate(
    source: str = typer.Option("deepmath", help="deepmath | hf | local"),
    dataset_name: Optional[str] = typer.Option(None),
    split: str = typer.Option("train"),
    path: Optional[str] = typer.Option(None, help="Local dataset path for source=local"),
    start: int = typer.Option(0),
    limit: Optional[int] = typer.Option(None),
    model: str = typer.Option(...),
    api_base: Optional[str] = typer.Option(None),
    api_key: Optional[str] = typer.Option(None),
    api_key_env: Optional[str] = typer.Option("OPENAI_API_KEY"),
    output_path: str = typer.Option(...),
    system_prompt: Optional[str] = typer.Option(None),
    temperature: float = typer.Option(0.0),
    max_tokens: Optional[int] = typer.Option(None),
    concurrency: int = typer.Option(8),
):
    """Stage 1: generate base trajectories using the base model."""
    load_dotenv()
    examples = _load_examples(source=source, dataset_name=dataset_name, split=split, path=path, limit=limit, start=start)
    client = _make_client(model, api_base, env_or_value(api_key, api_key_env), temperature, max_tokens)
    asyncio.run(
        generate_trajectories(
            examples=examples,
            llm=client,
            output_path=output_path,
            system_prompt=system_prompt,
            concurrency=concurrency,
        )
    )
    typer.echo(f"Wrote trajectories to {output_path}")


@app.command("score-math")
def score_math(
    generations_path: str = typer.Option(...),
    output_path: str = typer.Option(...),
):
    """Optional: add exact-match correctness for math generations."""
    rows = read_jsonl(generations_path)
    scored = []
    for row in rows:
        correctness = exact_match(row.get("raw_text", ""), row.get("answer"))
        metadata = row.get("metadata", {})
        metadata["correctness"] = None if correctness is None else int(bool(correctness))
        row["metadata"] = metadata
        scored.append(row)
    write_jsonl(scored, output_path)
    typer.echo(f"Wrote scored generations to {output_path}")


@app.command()
def extract(
    generations_path: str = typer.Option(...),
    model: str = typer.Option(...),
    api_base: Optional[str] = typer.Option(None),
    api_key: Optional[str] = typer.Option(None),
    api_key_env: Optional[str] = typer.Option("OPENAI_API_KEY"),
    output_path: str = typer.Option(...),
    temperature: float = typer.Option(0.0),
    max_tokens: Optional[int] = typer.Option(None),
    concurrency: int = typer.Option(8),
):
    """Stage 2: distill skill cards and keywords with a stronger model."""
    load_dotenv()
    records = load_generation_records(generations_path)
    client = _make_client(model, api_base, env_or_value(api_key, api_key_env), temperature, max_tokens)
    asyncio.run(extract_skills(records=records, llm=client, output_path=output_path, concurrency=concurrency))
    typer.echo(f"Wrote skills to {output_path}")


@app.command()
def infer(
    source: str = typer.Option("deepmath", help="deepmath | hf | local"),
    dataset_name: Optional[str] = typer.Option(None),
    split: str = typer.Option("train"),
    path: Optional[str] = typer.Option(None),
    start: int = typer.Option(0),
    limit: Optional[int] = typer.Option(None),
    skills_path: str = typer.Option(...),
    retrieval_backend: str = typer.Option("bm25", help="bm25 | dense | hybrid"),
    top_k: int = typer.Option(1),
    dense_model_name: str = typer.Option("BAAI/bge-m3"),
    prompt_style: str = typer.Option("normal", help="normal | only | try_to | short | draft"),
    budget: Optional[int] = typer.Option(None),
    model: str = typer.Option(...),
    api_base: Optional[str] = typer.Option(None),
    api_key: Optional[str] = typer.Option(None),
    api_key_env: Optional[str] = typer.Option("OPENAI_API_KEY"),
    output_path: str = typer.Option(...),
    temperature: float = typer.Option(0.0),
    max_tokens: Optional[int] = typer.Option(None),
    concurrency: int = typer.Option(8),
    bm25_weight: float = typer.Option(0.5),
    dense_weight: float = typer.Option(0.5),
):
    """Stage 3: retrieve skill cards and inject them into the prompt."""
    load_dotenv()
    examples = _load_examples(source=source, dataset_name=dataset_name, split=split, path=path, limit=limit, start=start)
    skills = load_skills(skills_path)
    retriever = SkillRetriever(skills, backend=retrieval_backend, dense_model_name=dense_model_name)
    client = _make_client(model, api_base, env_or_value(api_key, api_key_env), temperature, max_tokens)
    asyncio.run(
        run_inference(
            examples=examples,
            retriever=retriever,
            llm=client,
            output_path=output_path,
            prompt_style=prompt_style,
            budget=budget,
            concurrency=concurrency,
            top_k=top_k,
            bm25_weight=bm25_weight,
            dense_weight=dense_weight,
        )
    )
    typer.echo(f"Wrote inference outputs to {output_path}")


@app.command("end-to-end-math")
def end_to_end_math(
    dataset_name: str = typer.Option("zwhe99/DeepMath-103K"),
    train_split: str = typer.Option("train"),
    test_split: str = typer.Option("train"),
    train_start: int = typer.Option(0),
    train_limit: int = typer.Option(100),
    test_start: int = typer.Option(100),
    test_limit: int = typer.Option(20),
    base_model: str = typer.Option(...),
    summarizer_model: str = typer.Option(...),
    inference_model: str = typer.Option(...),
    api_base: Optional[str] = typer.Option(None),
    api_key: Optional[str] = typer.Option(None),
    api_key_env: Optional[str] = typer.Option("OPENAI_API_KEY"),
    workdir: str = typer.Option("runs/deepmath_demo"),
    prompt_style: str = typer.Option("normal"),
    budget: Optional[int] = typer.Option(None),
    temperature: float = typer.Option(0.0),
    max_tokens: Optional[int] = typer.Option(None),
    concurrency: int = typer.Option(8),
):
    """Convenience runner for the paper's math TRS pipeline."""
    load_dotenv()
    root = Path(workdir)
    root.mkdir(parents=True, exist_ok=True)
    resolved_api_key = env_or_value(api_key, api_key_env)

    train_examples = load_deepmath(dataset_name=dataset_name, split=train_split, start=train_start, limit=train_limit)
    test_examples = load_deepmath(dataset_name=dataset_name, split=test_split, start=test_start, limit=test_limit)

    base_client = _make_client(base_model, api_base, resolved_api_key, temperature, max_tokens)
    summarizer_client = _make_client(summarizer_model, api_base, resolved_api_key, 0.0, max_tokens)
    inference_client = _make_client(inference_model, api_base, resolved_api_key, temperature, max_tokens)

    raw_path = str(root / "01_generations.jsonl")
    scored_path = str(root / "02_generations_scored.jsonl")
    skills_path = str(root / "03_skills.jsonl")
    infer_path = str(root / "04_inference.jsonl")

    asyncio.run(generate_trajectories(examples=train_examples, llm=base_client, output_path=raw_path, concurrency=concurrency))

    rows = read_jsonl(raw_path)
    scored_rows = []
    for row in rows:
        metadata = row.get("metadata", {})
        correctness = exact_match(row.get("raw_text", ""), row.get("answer"))
        metadata["correctness"] = None if correctness is None else int(bool(correctness))
        row["metadata"] = metadata
        scored_rows.append(row)
    write_jsonl(scored_rows, scored_path)

    asyncio.run(
        extract_skills(
            records=load_generation_records(scored_path),
            llm=summarizer_client,
            output_path=skills_path,
            concurrency=concurrency,
        )
    )

    retriever = SkillRetriever(load_skills(skills_path), backend="bm25")
    asyncio.run(
        run_inference(
            examples=test_examples,
            retriever=retriever,
            llm=inference_client,
            output_path=infer_path,
            prompt_style=prompt_style,
            budget=budget,
            concurrency=concurrency,
            top_k=1,
        )
    )
    typer.echo(f"Finished. Outputs are under {root}")


if __name__ == "__main__":
    app()
