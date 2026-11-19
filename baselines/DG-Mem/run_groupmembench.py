#!/usr/bin/env python3
"""Run the final GroupMemRAG evaluation across GroupMemBench."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, Sequence

from tqdm.auto import tqdm

from src.config import ExperimentConfig
from src.dataset import format_message, load_conversation_messages, load_questions
from src.embeddings import build_embedder
from src.entities import build_entity_extractor
from src.evaluation import (
    evaluate_record,
    load_completed_ids,
    sort_and_deduplicate_results,
    summarize_judgments,
)
from src.llm import build_chat_model
from src.memory_graph import INDEX_FORMAT_VERSION, MemoryGraph
from src.rerankers import build_reranker
from src.user_context_graph import DualGraphRetriever, UserContextGraph


ROOT = Path(__file__).resolve().parent
DEFAULT_DOMAINS = ("Finance", "Technology", "Healthcare", "Manufacturing")
DEFAULT_QUESTION_TYPES = (
    "multi_hop",
    "knowledge_update",
    "temporal",
    "user_implicit",
    "term_ambiguity",
    "abstention",
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the final dual-graph GroupMemBench experiment over all four "
            "domains and six question types."
        )
    )
    parser.add_argument("--config", default="configs/groupmembench.toml")
    parser.add_argument("--experiment-name", default="groupmemrag-groupmembench-final")
    parser.add_argument("--llm-profile", default="groupmembench_final")
    parser.add_argument("--benchmark-root", default="../GroupMemBench")
    parser.add_argument("--output-root", default="outputs/groupmembench")
    parser.add_argument("--index-root", default="cache/groupmembench")
    parser.add_argument("--domains", nargs="+", default=list(DEFAULT_DOMAINS))
    parser.add_argument("--question-types", nargs="+", default=list(DEFAULT_QUESTION_TYPES))
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--qa-workers", type=int, default=4)
    parser.add_argument("--question-limit", type=int, default=0)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--force-rebuild", action="store_true")
    parser.add_argument("--skip-connectivity-check", action="store_true")
    parser.add_argument("--retrieval-only", action="store_true")
    parser.add_argument("--judge", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--agent-prompt", default=None)
    parser.add_argument("--judge-prompt", default=None)
    parser.add_argument("--embedding-backend", choices=["sentence_transformer", "hashing"], default=None)
    parser.add_argument("--embedding-model", default=None)
    parser.add_argument("--embedding-device", choices=["auto", "cpu", "cuda"], default=None)
    parser.add_argument("--embedding-batch-size", type=int, default=None)
    parser.add_argument("--entity-extractor", choices=["regex", "spacy"], default=None)
    parser.add_argument("--spacy-model", default=None)
    parser.add_argument("--spacy-device", choices=["cpu", "cuda", "auto"], default=None)
    parser.add_argument("--spacy-batch-size", type=int, default=None)
    parser.add_argument("--reranker-provider", default=None)
    parser.add_argument("--reranker-model", default=None)
    parser.add_argument("--reranker-device", choices=["auto", "cpu", "cuda"], default=None)
    parser.add_argument("--reranker-batch-size", type=int, default=None)
    parser.add_argument("--print-config", action="store_true")
    return parser.parse_args(argv)


def choose(command_line_value: Any, configured_value: Any) -> Any:
    return configured_value if command_line_value is None else command_line_value


def resolve_project_path(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def configured_path(config: ExperimentConfig, override: str | None, configured: str, name: str) -> Path:
    if override is not None:
        return resolve_project_path(override)
    if not configured:
        raise ValueError(f"Missing {name} in [run]")
    return config.resolve_path(configured).resolve()


def _split_values(values: Iterable[str]) -> list[str]:
    return [part.strip() for value in values for part in str(value).split(",") if part.strip()]


def normalize_selections(values: Iterable[str], allowed: Sequence[str], label: str) -> list[str]:
    canonical = {item.lower(): item for item in allowed}
    selected: list[str] = []
    for value in _split_values(values):
        match = canonical.get(value.lower())
        if match is None:
            raise ValueError(f"Unknown {label} {value!r}. Allowed: {', '.join(allowed)}")
        if match not in selected:
            selected.append(match)
    if not selected:
        raise ValueError(f"At least one {label} is required")
    return selected


def validate_args(args: argparse.Namespace) -> tuple[list[str], list[str]]:
    if args.top_k <= 0:
        raise ValueError("--top-k must be greater than zero")
    if args.qa_workers <= 0:
        raise ValueError("--qa-workers must be greater than zero")
    if args.question_limit < 0:
        raise ValueError("--question-limit cannot be negative")
    domains = normalize_selections(args.domains, DEFAULT_DOMAINS, "domain")
    question_types = normalize_selections(args.question_types, DEFAULT_QUESTION_TYPES, "question type")
    return domains, question_types


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def retrieval_record(
    question: dict[str, Any],
    question_index: int,
    retriever: Any,
    retriever_name: str,
    top_k: int,
    experiment: ExperimentConfig,
    fingerprint: str,
) -> dict[str, Any]:
    """Retrieve one question and serialize an auditable trace."""
    question_text = str(question.get("question", ""))
    asking_user_id = str(question.get("asking_user_id", ""))
    search_query = question_text
    if retriever_name in {"user_context_graph", "dual_graph"}:
        retrieved = retriever.retrieve(search_query, top_k=top_k, asking_user_id=asking_user_id)
    else:
        retrieved = retriever.retrieve(search_query, top_k=top_k)
    record = {
        "id": question.get("id", ""),
        "question_index": question_index,
        "question": question.get("question", ""),
        "answer": question.get("answer", ""),
        "asking_user_id": question.get("asking_user_id", ""),
        "retrieval_query": search_query,
        "retrieved_docs": [format_message(result.message) for result in retrieved],
        "retrieval_trace": [result.to_dict() for result in retrieved],
        "retriever": retriever_name,
        "experiment": {
            "name": experiment.name,
            "llm_profile": experiment.llm_profile_name,
            "agent_provider": experiment.llm_profile.agent.provider if experiment.llm_profile else "",
            "agent_model": experiment.llm_profile.agent.model if experiment.llm_profile else "",
            "judge_provider": experiment.llm_profile.judge.provider if experiment.llm_profile else "",
            "judge_model": experiment.llm_profile.judge.model if experiment.llm_profile else "",
            "reranker_provider": experiment.reranker.provider,
            "reranker_model": experiment.reranker.model,
        },
    }
    if retriever_name in {"memory_graph", "dual_graph"}:
        memory_graph = retriever.memory_graph if retriever_name == "dual_graph" else retriever
        record["memory_graph"] = {
            "messages": len(memory_graph.messages),
            "entities": len(memory_graph.entity_texts),
            "fingerprint": fingerprint,
            "index_format_version": INDEX_FORMAT_VERSION,
            "config": asdict(memory_graph.config),
        }
        if retriever_name == "dual_graph":
            user_graph = retriever.user_context_graph
            record["user_context_graph"] = user_graph_record(user_graph)
    elif retriever_name == "user_context_graph":
        record["user_context_graph"] = user_graph_record(retriever)
    return record


def user_graph_record(graph: UserContextGraph) -> dict[str, Any]:
    return {
        "users": len(graph.user_texts),
        "roles": len(graph.role_texts),
        "contexts": len(graph.context_keys),
        "nodes": len(graph.node_names),
        "fingerprint": graph.fingerprint(),
        "config": asdict(graph.config),
        "reranker": {
            "provider": str(getattr(graph.reranker, "provider", "none")),
            "model": str(getattr(graph.reranker, "model_name", "")),
        },
    }


def check_connectivity(experiment: ExperimentConfig, judge_enabled: bool) -> None:
    if experiment.llm_profile is None:
        raise ValueError("The selected run requires an LLM profile")
    roles = ("agent", "judge") if judge_enabled else ("agent",)
    for role in roles:
        settings = getattr(experiment.llm_profile, role)
        answer = build_chat_model(settings).complete([{"role": "user", "content": "Reply with exactly OK"}])
        if not str(answer).strip():
            raise RuntimeError(f"{role} connectivity check returned an empty response")
        print(f"  {role}: provider={settings.provider} model={settings.model} OK")


def build_or_load_retriever(
    *,
    messages: list[dict[str, Any]],
    index_path: Path,
    experiment: ExperimentConfig,
    embedder: Any,
    extractor: Any,
    reranker: Any,
    force_rebuild: bool,
) -> DualGraphRetriever:
    memory_path = index_path / "memory_graph"
    user_path = index_path / "user_context_graph"
    index_ready = (
        not force_rebuild
        and (memory_path / "metadata.json").is_file()
        and (user_path / "metadata.json").is_file()
    )
    if index_ready:
        print(f"  loading dual-graph index: {index_path}")
        memory_graph = MemoryGraph.load(memory_path, embedder, extractor, show_progress=True)
        if memory_graph.fingerprint() != MemoryGraph.corpus_fingerprint(messages):
            raise ValueError(
                f"Cached index {index_path} belongs to a different corpus. "
                "Use --force-rebuild to replace it."
            )
        user_graph = UserContextGraph.load(
            user_path,
            messages,
            experiment.user_context_graph,
            reranker=reranker,
        )
        return DualGraphRetriever(memory_graph, user_graph)

    print(f"  building dual-graph index: {index_path}")
    memory_graph = MemoryGraph(embedder, extractor, experiment.memory_graph).build(messages, show_progress=True)
    user_graph = UserContextGraph(experiment.user_context_graph, reranker=reranker).build(messages)
    index_path.mkdir(parents=True, exist_ok=True)
    memory_graph.save(memory_path)
    user_graph.save(user_path)
    return DualGraphRetriever(memory_graph, user_graph)


def write_retrieval_records(path: Path, records: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def evaluate_records(
    *,
    records: Sequence[dict[str, Any]],
    output_path: Path,
    agent: Any,
    judge: Any,
    agent_prompt: str,
    judge_prompt: str,
    qa_workers: int,
    resume: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    failed_path = output_path.with_name(f"{output_path.stem}.failed.jsonl")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not resume:
        output_path.write_text("", encoding="utf-8")
        failed_path.write_text("", encoding="utf-8")
    futures: dict[Any, dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=qa_workers) as executor:
        for record in records:
            future = executor.submit(
                evaluate_record,
                record,
                agent=agent,
                judge=judge,
                agent_system_prompt=agent_prompt,
                judge_system_prompt=judge_prompt,
            )
            futures[future] = record
        with tqdm(total=len(futures), desc="reader/judge", unit="question", dynamic_ncols=True) as progress:
            for future in as_completed(futures):
                source = futures[future]
                try:
                    append_jsonl(output_path, future.result())
                except Exception as error:
                    append_jsonl(
                        failed_path,
                        {
                            "id": source.get("id", ""),
                            "question_index": source.get("question_index", 0),
                            "question": source.get("question", ""),
                            "error": str(error),
                        },
                    )
                finally:
                    progress.update(1)
    final_records = sort_and_deduplicate_results(output_path)
    summary = summarize_judgments(final_records) if judge is not None else None
    if summary is not None:
        output_path.with_name(f"{output_path.stem}.summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    return final_records, summary


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return records


def build_domain_summary(
    domain: str,
    question_types: Sequence[str],
    expected: dict[str, int],
    output_dir: Path,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    total_correct = total_scored = total_results = 0
    complete = True
    for question_type in question_types:
        records = load_jsonl(output_dir / f"dual_graph__{question_type}.jsonl")
        summary = summarize_judgments(records)
        result_count = len(records)
        scored_count = summary["total"]
        expected_count = expected[question_type]
        rows.append({
            "question_type": question_type,
            "expected": expected_count,
            "results": result_count,
            "scored": scored_count,
            "correct": summary["correct"],
            "accuracy": summary["accuracy"],
        })
        total_correct += summary["correct"]
        total_scored += scored_count
        total_results += result_count
        complete = complete and result_count == expected_count and scored_count == expected_count
    report = {
        "domain": domain,
        "rows": rows,
        "correct": total_correct,
        "scored": total_scored,
        "results": total_results,
        "expected": sum(expected.values()),
        "accuracy": total_correct / total_scored if total_scored else 0.0,
        "complete": complete,
    }
    lines = [
        f"# {domain} GroupMemBench summary",
        "",
        "| question type | accuracy | completeness |",
        "|---|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['question_type']} | {row['accuracy'] * 100:.1f}% "
            f"({row['correct']}/{row['scored']}) | "
            f"{row['scored']}/{row['results']}/{row['expected']} |"
        )
    lines.extend([
        f"| **overall** | **{report['accuracy'] * 100:.1f}% "
        f"({total_correct}/{total_scored})** | "
        f"**{total_scored}/{total_results}/{report['expected']}** |",
        "",
        "Completeness cells are `scored / result records / expected questions`.",
        "",
        "All result files are complete." if complete else "Some result files are incomplete.",
        "",
    ])
    (output_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    (output_dir / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def build_overall_summary(reports: Sequence[dict[str, Any]], output_root: Path) -> None:
    correct = sum(report["correct"] for report in reports)
    scored = sum(report["scored"] for report in reports)
    results = sum(report["results"] for report in reports)
    expected = sum(report["expected"] for report in reports)
    accuracy = correct / scored if scored else 0.0
    lines = [
        "# GroupMemBench summary",
        "",
        "| domain | accuracy | completeness |",
        "|---|---:|---:|",
    ]
    for report in reports:
        lines.append(
            f"| {report['domain']} | {report['accuracy'] * 100:.1f}% "
            f"({report['correct']}/{report['scored']}) | "
            f"{report['scored']}/{report['results']}/{report['expected']} |"
        )
    lines.extend([
        f"| **overall** | **{accuracy * 100:.1f}% ({correct}/{scored})** | "
        f"**{scored}/{results}/{expected}** |",
        "",
        "Completeness cells are `scored / result records / expected questions`.",
        "",
    ])
    (output_root / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    domains, question_types = validate_args(args)
    config_path = resolve_project_path(args.config)
    experiment = ExperimentConfig.load(config_path, profile=args.llm_profile)
    experiment.name = args.experiment_name.strip() or experiment.name
    experiment.run.retriever = "dual_graph"
    if args.print_config:
        payload = experiment.safe_dict()
        payload["batch"] = {
            "benchmark_root": str(resolve_project_path(args.benchmark_root)),
            "output_root": str(resolve_project_path(args.output_root)),
            "index_root": str(resolve_project_path(args.index_root)),
            "domains": domains,
            "question_types": question_types,
            "top_k": args.top_k,
            "qa_workers": args.qa_workers,
            "question_limit": args.question_limit,
            "resume": args.resume,
            "force_rebuild": args.force_rebuild,
            "retrieval_only": args.retrieval_only,
            "judge": args.judge,
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    benchmark_root = resolve_project_path(args.benchmark_root)
    output_root = resolve_project_path(args.output_root)
    index_root = resolve_project_path(args.index_root)
    if not benchmark_root.is_dir():
        raise FileNotFoundError(f"GroupMemBench directory not found: {benchmark_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    index_root.mkdir(parents=True, exist_ok=True)

    evaluate = not args.retrieval_only
    judge_enabled = evaluate and args.judge
    if evaluate and experiment.llm_profile is None:
        raise ValueError("Evaluation requires an LLM profile")
    if evaluate and not args.skip_connectivity_check:
        print(f"[preflight] checking profile={experiment.llm_profile_name}")
        check_connectivity(experiment, judge_enabled)

    print("[models] preparing shared retrieval models")
    embedder = build_embedder(
        choose(args.embedding_backend, experiment.embedding.provider),
        choose(args.embedding_model, experiment.embedding.model),
        device=choose(args.embedding_device, experiment.embedding.device),
        batch_size=int(choose(args.embedding_batch_size, experiment.embedding.batch_size)),
    )
    extractor = build_entity_extractor(
        choose(args.entity_extractor, experiment.entity.provider),
        choose(args.spacy_model, experiment.entity.model),
        device=choose(args.spacy_device, experiment.entity.device),
        batch_size=int(choose(args.spacy_batch_size, experiment.entity.batch_size)),
    )
    reranker = build_reranker(
        choose(args.reranker_provider, experiment.reranker.provider),
        choose(args.reranker_model, experiment.reranker.model),
        device=choose(args.reranker_device, experiment.reranker.device),
        batch_size=int(choose(args.reranker_batch_size, experiment.reranker.batch_size)),
    ) if experiment.user_context_graph.candidate_refinement else None

    agent_prompt = judge_prompt = ""
    agent = judge = None
    if evaluate:
        agent_prompt_path = configured_path(experiment, args.agent_prompt, experiment.run.agent_prompt, "agent_prompt")
        agent_prompt = agent_prompt_path.read_text(encoding="utf-8").strip()
        agent = build_chat_model(experiment.llm_profile.agent)
        if judge_enabled:
            judge_prompt_path = configured_path(experiment, args.judge_prompt, experiment.run.judge_prompt, "judge_prompt")
            judge_prompt = judge_prompt_path.read_text(encoding="utf-8").strip()
            judge = build_chat_model(experiment.llm_profile.judge)

    run_number = 0
    run_total = len(domains) * len(question_types)
    domain_reports: list[dict[str, Any]] = []
    for domain in domains:
        print(f"\n=== Domain: {domain} ===")
        conversation_path = benchmark_root / "data" / "final" / domain / f"synthetic_domain_channels_rolevariants_{domain}.json"
        questions_dir = benchmark_root / "questions" / domain
        if not conversation_path.is_file():
            raise FileNotFoundError(f"Conversation file not found: {conversation_path}")
        if not questions_dir.is_dir():
            raise FileNotFoundError(f"Questions directory not found: {questions_dir}")
        messages = load_conversation_messages(conversation_path)
        print(f"  loaded {len(messages)} messages")
        retriever = build_or_load_retriever(
            messages=messages,
            index_path=index_root / domain.lower(),
            experiment=experiment,
            embedder=embedder,
            extractor=extractor,
            reranker=reranker,
            force_rebuild=args.force_rebuild,
        )
        fingerprint = MemoryGraph.corpus_fingerprint(messages)
        domain_output = output_root / domain
        domain_output.mkdir(parents=True, exist_ok=True)
        expected: dict[str, int] = {}

        for question_type in question_types:
            run_number += 1
            print(f"\n[{run_number}/{run_total}] {domain} / dual_graph / {question_type}")
            question_path = questions_dir / f"{question_type}.jsonl"
            if not question_path.is_file():
                raise FileNotFoundError(f"Question file not found: {question_path}")
            questions = load_questions(question_path)
            if args.question_limit > 0:
                questions = questions[: args.question_limit]
            expected[question_type] = len(questions)
            output_path = domain_output / f"dual_graph__{question_type}.jsonl"
            completed = load_completed_ids(output_path, require_judge=judge_enabled) if evaluate and args.resume else set()
            pending = [(i, q) for i, q in enumerate(questions) if str(q.get("id", "")) not in completed]
            print(f"  questions={len(questions)} completed={len(completed)} pending={len(pending)}")
            records = [
                retrieval_record(q, i, retriever, "dual_graph", args.top_k, experiment, fingerprint)
                for i, q in tqdm(pending, desc="dual_graph retrieval", unit="question", dynamic_ncols=True)
            ]
            if not evaluate:
                write_retrieval_records(output_path, records)
                print(f"  retrieval records: {output_path}")
                continue
            _, summary = evaluate_records(
                records=records,
                output_path=output_path,
                agent=agent,
                judge=judge,
                agent_prompt=agent_prompt,
                judge_prompt=judge_prompt,
                qa_workers=args.qa_workers,
                resume=args.resume,
            )
            if summary is not None:
                print(f"  accuracy={summary['accuracy']:.4f} ({summary['correct']}/{summary['total']})")
            print(f"  results: {output_path}")

        if judge_enabled:
            report = build_domain_summary(domain, question_types, expected, domain_output)
            domain_reports.append(report)
            print(f"  summary: {domain_output / 'summary.md'}")

    if judge_enabled:
        build_overall_summary(domain_reports, output_root)
        print(f"\nOverall summary: {output_root / 'summary.md'}")
    print(f"Finished {run_total} runs.")
    print(f"Results: {output_root}")
    print(f"Indexes: {index_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
