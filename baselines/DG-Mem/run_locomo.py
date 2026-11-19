#!/usr/bin/env python3
"""Run GroupMemRAG on LoCoMo and report F1/BLEU-1 by QA category."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from tqdm.auto import tqdm

from src.config import ExperimentConfig
from src.embeddings import build_embedder
from src.entities import build_entity_extractor
from src.evaluation import complete_with_metadata, extract_final
from src.llm import build_chat_model
from src.locomo import (
    CATEGORY_NAMES,
    build_reader_user_prompt,
    conversation_windows,
    load_locomo,
    prompt_path,
    score_answer,
    summarize_records,
    summary_markdown,
)
from src.memory_graph import MemoryGraph
from src.rerankers import build_reranker
from src.user_context_graph import DualGraphRetriever, UserContextGraph


ROOT = Path(__file__).resolve().parent
DEFAULT_PROMPT_DIR = ROOT / "prompts"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate GroupMemRAG on LoCoMo")
    parser.add_argument("--config", default="configs/locomo.toml")
    parser.add_argument("--llm-profile", default=None, help="siliconflow or relay2")
    parser.add_argument("--data-file", default=None)
    parser.add_argument("--output-jsonl", default=None)
    parser.add_argument(
        "--retrieval-input-jsonl",
        default=None,
        help="Use precomputed retrieval records and run only the reader/scorer",
    )
    parser.add_argument(
        "--retrieval-output-jsonl",
        default=None,
        help="Persist retrieval records before reader calls so they can be reused",
    )
    parser.add_argument("--index-dir", default=None)
    parser.add_argument("--prompt-dir", default=str(DEFAULT_PROMPT_DIR))
    parser.add_argument("--reuse-index", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--evaluate", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--qa-workers", type=int, default=None)
    parser.add_argument("--question-limit", type=int, default=None, help="Global limit after category filtering")
    parser.add_argument("--sample-limit", type=int, default=0)
    parser.add_argument("--categories", default="1,2,3,4")
    parser.add_argument("--embedding-backend", choices=["sentence_transformer", "hashing"], default=None)
    parser.add_argument("--embedding-model", default=None)
    parser.add_argument("--embedding-device", choices=["auto", "cpu", "cuda"], default=None)
    parser.add_argument("--entity-extractor", choices=["regex", "spacy"], default=None)
    parser.add_argument("--reranker-provider", default=None)
    parser.add_argument("--reranker-model", default=None)
    parser.add_argument("--reranker-device", choices=["auto", "cpu", "cuda"], default=None)
    parser.add_argument("--print-config", action="store_true")
    return parser.parse_args()


def choose(command_line_value: Any, configured_value: Any) -> Any:
    return configured_value if command_line_value is None else command_line_value


def configured_path(config: ExperimentConfig, override: str | None, configured: str, name: str) -> Path:
    if override is not None:
        path = Path(override)
    else:
        if not configured:
            raise ValueError(f"Missing {name} in [run]")
        path = config.resolve_path(configured)
    return path.resolve()


def parse_categories(value: str) -> set[int]:
    categories = {int(item.strip()) for item in str(value).split(",") if item.strip()}
    unsupported = categories - set(CATEGORY_NAMES)
    if unsupported:
        raise ValueError(f"Only LoCoMo categories 1-4 are scored; unsupported: {sorted(unsupported)}")
    if not categories:
        raise ValueError("At least one category is required")
    return categories


def load_records(path: Path) -> list[dict[str, Any]]:
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


def write_sorted_records(path: Path) -> list[dict[str, Any]]:
    by_id = {str(record.get("id", "")): record for record in load_records(path)}
    records = sorted(
        by_id.values(),
        key=lambda record: (int(record.get("sample_index", 0)), int(record.get("qa_index", 0))),
    )
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return records


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def prepare_retriever(
    messages: list[dict[str, Any]],
    sample_index_dir: Path,
    experiment: ExperimentConfig,
    embedder: Any,
    extractor: Any,
    reranker: Any,
    reuse_index: bool,
) -> DualGraphRetriever:
    memory_dir = sample_index_dir / "memory_graph"
    user_dir = sample_index_dir / "user_context_graph"
    if reuse_index:
        if not (memory_dir / "metadata.json").exists() or not (user_dir / "metadata.json").exists():
            raise FileNotFoundError(f"Missing cached dual-graph index under {sample_index_dir}")
        memory_graph = MemoryGraph.load(memory_dir, embedder, extractor, show_progress=False)
        if memory_graph.fingerprint() != MemoryGraph.corpus_fingerprint(messages):
            raise ValueError(f"Cached Memory Graph has a different LoCoMo corpus: {sample_index_dir}")
        user_graph = UserContextGraph.load(
            user_dir,
            messages,
            experiment.user_context_graph,
            reranker=reranker,
        )
    else:
        memory_graph = MemoryGraph(embedder, extractor, experiment.memory_graph).build(messages, show_progress=False)
        user_graph = UserContextGraph(experiment.user_context_graph, reranker=reranker).build(messages)
        memory_graph.save(memory_dir)
        user_graph.save(user_dir)
    return DualGraphRetriever(memory_graph, user_graph)


def retrieval_record(
    *,
    sample: dict[str, Any],
    sample_index: int,
    qa: dict[str, Any],
    qa_index: int,
    global_index: int,
    retriever: DualGraphRetriever,
    top_k: int,
    experiment: ExperimentConfig,
) -> dict[str, Any]:
    question = str(qa.get("question", ""))
    results = retriever.retrieve(question, top_k=top_k)
    retrieved_dialog_ids = {
        str(dialog_id)
        for result in results
        for dialog_id in result.message.get("dialog_ids", [])
    }
    evidence = [str(value).replace("(", "").replace(")", "") for value in qa.get("evidence", [])]
    evidence_hits = [value for value in evidence if value in retrieved_dialog_ids]
    sample_id = str(sample.get("sample_id", f"sample-{sample_index}"))
    return {
        "id": f"{sample_id}:{qa_index}",
        "sample_id": sample_id,
        "sample_index": sample_index,
        "qa_index": qa_index,
        "question_index": global_index,
        "category": int(qa.get("category", 0)),
        "category_name": CATEGORY_NAMES[int(qa.get("category", 0))],
        "question": question,
        "answer": qa.get("answer", ""),
        "evidence": evidence,
        "retrieved_docs": [str(result.message.get("content", "")) for result in results],
        "retrieval_trace": [result.to_dict() for result in results],
        "retrieval_recall": len(evidence_hits) / len(evidence) if evidence else None,
        "retrieved_evidence": evidence_hits,
        "retriever": "dual_graph",
        "status": "retrieved",
        "experiment": {
            "name": experiment.name,
            "llm_profile": experiment.llm_profile_name,
            "agent_provider": experiment.llm_profile.agent.provider if experiment.llm_profile else "",
            "agent_model": experiment.llm_profile.agent.model if experiment.llm_profile else "",
            "embedding_model": experiment.embedding.model,
            "reranker_model": experiment.reranker.model,
            "top_k": top_k,
            "window_size": 6,
            "window_stride": 3,
            "window_min_tail": 4,
        },
    }


def answer_record(
    record: dict[str, Any],
    model: Any,
    prompts: dict[int, str],
    experiment: ExperimentConfig,
) -> dict[str, Any]:
    category = int(record["category"])
    messages = [
        {"role": "system", "content": prompts[category]},
        {
            "role": "user",
            "content": build_reader_user_prompt(record["question"], record["retrieved_docs"], category),
        },
    ]
    raw_answer, response = complete_with_metadata(model, messages)
    answer = extract_final(raw_answer).strip()
    if not answer:
        raise ValueError(f"Reader returned an empty answer; metadata={response}")
    scores = score_answer(answer, record.get("answer", ""), category)
    output = dict(record)
    output["experiment"] = {
        **dict(record.get("experiment", {})),
        "name": experiment.name,
        "llm_profile": experiment.llm_profile_name,
        "agent_provider": experiment.llm_profile.agent.provider if experiment.llm_profile else "",
        "agent_model": experiment.llm_profile.agent.model if experiment.llm_profile else "",
    }
    output.update({
        "agent_answer_raw": raw_answer,
        "agent_answer": answer,
        "agent_response": response,
        "f1": scores["f1"],
        "bleu1": scores["bleu1"],
        "status": "ok",
    })
    return output


def write_summary(
    output_path: Path,
    records: list[dict[str, Any]],
    expected: dict[str, int],
    llm_profile: str,
) -> None:
    profile_records = [
        record for record in records
        if record.get("experiment", {}).get("llm_profile") == llm_profile
    ]
    summary = summarize_records(profile_records)
    summary["llm_profile"] = llm_profile
    summary["expected_counts"] = expected
    summary["failed_or_missing"] = max(0, sum(expected.values()) - summary["Average"]["count"])
    json_path = output_path.with_name(f"{output_path.stem}.summary.json")
    markdown_path = output_path.with_name(f"{output_path.stem}.summary.md")
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown_path.write_text(summary_markdown(summary), encoding="utf-8")
    print(summary_markdown(summary))
    print(f"JSON summary: {json_path}")
    print(f"Markdown summary: {markdown_path}")


def main() -> int:
    args = parse_args()
    experiment = ExperimentConfig.load(args.config, profile=args.llm_profile)
    if args.print_config:
        print(json.dumps(experiment.safe_dict(), ensure_ascii=False, indent=2))
        return 0

    data_path = configured_path(experiment, args.data_file, experiment.run.conversation_json, "conversation_json")
    output_path = configured_path(experiment, args.output_jsonl, experiment.run.output_jsonl, "output_jsonl")
    index_path = configured_path(experiment, args.index_dir, experiment.run.index_dir, "index_dir")
    prompt_dir = Path(args.prompt_dir).resolve()
    categories = parse_categories(args.categories)
    reuse_index = bool(choose(args.reuse_index, experiment.run.reuse_index))
    resume = bool(choose(args.resume, experiment.run.resume))
    evaluate = bool(choose(args.evaluate, experiment.run.evaluate))
    top_k = int(choose(args.top_k, experiment.run.top_k))
    qa_workers = max(1, int(choose(args.qa_workers, experiment.run.qa_workers)))
    question_limit = int(choose(args.question_limit, experiment.run.question_limit))

    samples = load_locomo(data_path)
    if args.sample_limit > 0:
        samples = samples[:args.sample_limit]
    prompts = {category: prompt_path(prompt_dir, category).read_text(encoding="utf-8").strip() for category in categories}

    selected: list[tuple[int, dict[str, Any], int, dict[str, Any]]] = []
    expected = {name: 0 for name in CATEGORY_NAMES.values()}
    for sample_index, sample in enumerate(samples):
        for qa_index, qa in enumerate(sample.get("qa", [])):
            category = int(qa.get("category", 0))
            if category not in categories:
                continue
            if question_limit > 0 and len(selected) >= question_limit:
                break
            selected.append((sample_index, sample, qa_index, qa))
            expected[CATEGORY_NAMES[category]] += 1
        if question_limit > 0 and len(selected) >= question_limit:
            break

    output_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.mkdir(parents=True, exist_ok=True)
    if not resume:
        output_path.write_text("", encoding="utf-8")
    existing = load_records(output_path)
    completed_ids = set()
    if resume:
        for record in existing:
            if evaluate:
                same_profile = record.get("experiment", {}).get("llm_profile") == experiment.llm_profile_name
                if same_profile and record.get("status") == "ok" and str(record.get("agent_answer", "")).strip():
                    completed_ids.add(str(record.get("id", "")))
            else:
                completed_ids.add(str(record.get("id", "")))
    pending = [item for item in selected if f"{item[1].get('sample_id', f'sample-{item[0]}')}:{item[2]}" not in completed_ids]

    print(
        f"LoCoMo data={data_path} | selected={len(selected)} | pending={len(pending)} | "
        f"top_k={top_k} | profile={experiment.llm_profile_name or '<none>'}"
    )
    if not pending:
        final_records = write_sorted_records(output_path)
        if evaluate:
            write_summary(output_path, final_records, expected, experiment.llm_profile_name)
        return 0

    retrieved_records: list[dict[str, Any]] = []
    if args.retrieval_input_jsonl:
        retrieval_input = Path(args.retrieval_input_jsonl).resolve()
        cached = {str(record.get("id", "")): record for record in load_records(retrieval_input)}
        pending_ids = [
            f"{sample.get('sample_id', f'sample-{sample_index}')}:{qa_index}"
            for sample_index, sample, qa_index, _ in pending
        ]
        missing = [record_id for record_id in pending_ids if record_id not in cached]
        if missing:
            raise ValueError(
                f"Retrieval input is missing {len(missing)} pending records; first missing id: {missing[0]}"
            )
        retrieved_records = [cached[record_id] for record_id in pending_ids]
        print(f"Loaded {len(retrieved_records)} precomputed retrieval records: {retrieval_input}")
    else:
        embedder = build_embedder(
            choose(args.embedding_backend, experiment.embedding.provider),
            choose(args.embedding_model, experiment.embedding.model),
            device=choose(args.embedding_device, experiment.embedding.device),
            batch_size=experiment.embedding.batch_size,
        )
        extractor = build_entity_extractor(
            choose(args.entity_extractor, experiment.entity.provider),
            experiment.entity.model,
            device=experiment.entity.device,
            batch_size=experiment.entity.batch_size,
        )
        reranker = build_reranker(
            choose(args.reranker_provider, experiment.reranker.provider),
            choose(args.reranker_model, experiment.reranker.model),
            device=choose(args.reranker_device, experiment.reranker.device),
            batch_size=experiment.reranker.batch_size,
        )

        by_sample: dict[int, list[tuple[int, dict[str, Any], int, dict[str, Any]]]] = {}
        for item in pending:
            by_sample.setdefault(item[0], []).append(item)
        global_positions = {
            f"{sample.get('sample_id', f'sample-{sample_index}')}:{qa_index}": index
            for index, (sample_index, sample, qa_index, _) in enumerate(selected)
        }
        for sample_index, items in tqdm(by_sample.items(), desc="LoCoMo conversations", unit="conversation"):
            sample = items[0][1]
            messages = conversation_windows(sample)
            sample_id = str(sample.get("sample_id", f"sample-{sample_index}"))
            retriever = prepare_retriever(
                messages,
                index_path / sample_id,
                experiment,
                embedder,
                extractor,
                reranker,
                reuse_index,
            )
            for _, _, qa_index, qa in tqdm(items, desc=sample_id, unit="question", leave=False):
                record_id = f"{sample_id}:{qa_index}"
                retrieved_records.append(retrieval_record(
                    sample=sample,
                    sample_index=sample_index,
                    qa=qa,
                    qa_index=qa_index,
                    global_index=global_positions[record_id],
                    retriever=retriever,
                    top_k=top_k,
                    experiment=experiment,
                ))

    if args.retrieval_output_jsonl:
        retrieval_output = Path(args.retrieval_output_jsonl).resolve()
        retrieval_output.parent.mkdir(parents=True, exist_ok=True)
        with retrieval_output.open("w", encoding="utf-8") as handle:
            for record in retrieved_records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"Reusable retrieval records: {retrieval_output}")

    if not evaluate:
        for record in retrieved_records:
            append_jsonl(output_path, record)
        write_sorted_records(output_path)
        print(f"Wrote {len(retrieved_records)} retrieval-only records: {output_path}")
        return 0
    if experiment.llm_profile is None:
        raise ValueError("LoCoMo evaluation requires --llm-profile or experiment.llm_profile")

    model = build_chat_model(experiment.llm_profile.agent)
    failed_path = output_path.with_name(f"{output_path.stem}.failed.jsonl")
    if not resume:
        failed_path.write_text("", encoding="utf-8")
    print(f"Running LoCoMo reader | questions={len(retrieved_records)} | workers={qa_workers}")
    with ThreadPoolExecutor(max_workers=qa_workers) as executor:
        futures = {
            executor.submit(answer_record, record, model, prompts, experiment): record
            for record in retrieved_records
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc="Reader", unit="question"):
            source = futures[future]
            try:
                append_jsonl(output_path, future.result())
            except Exception as error:
                append_jsonl(failed_path, {
                    "id": source["id"],
                    "sample_id": source["sample_id"],
                    "qa_index": source["qa_index"],
                    "question": source["question"],
                    "error": str(error),
                })

    final_records = write_sorted_records(output_path)
    write_summary(output_path, final_records, expected, experiment.llm_profile_name)
    print(f"Results: {output_path}")
    if failed_path.exists() and failed_path.stat().st_size:
        print(f"Retryable failures: {failed_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
