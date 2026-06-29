import argparse
import csv
import json
from pathlib import Path


DETAIL_FIELDS = [
    "model",
    "dataset",
    "attack",
    "query_id",
    "query",
    "response",
    "extracted_info",
    "extracted_count",
    "retrieved_count",
    "retrieved_indexes",
    "total_time",
    "rag_prompt_tokens",
    "rag_completion_tokens",
    "rag_total_tokens",
    "attack_prompt_tokens",
    "attack_completion_tokens",
    "attack_total_tokens",
    "source_log",
]


def read_json_objects(path):
    decoder = json.JSONDecoder()
    content = path.read_text(encoding="utf-8")
    position = 0
    while position < len(content):
        while position < len(content) and content[position].isspace():
            position += 1
        if position >= len(content):
            break
        record, position = decoder.raw_decode(content, position)
        yield record


def row_from_record(model, dataset, attack, record, source_log):
    rag_cost = record.get("rag_step_cost", {})
    attack_cost = record.get("attack_step_cost", {})
    extracted = record.get("extracted_info") or []
    retrieved = record.get("retrieved_docs") or []
    return {
        "model": model,
        "dataset": dataset,
        "attack": attack,
        "query_id": record.get("query_id"),
        "query": record.get("query", ""),
        "response": record.get("response", ""),
        "extracted_info": json.dumps(extracted, ensure_ascii=False),
        "extracted_count": len(extracted),
        "retrieved_count": len(retrieved),
        "retrieved_indexes": ",".join(str(doc.get("index", "")) for doc in retrieved),
        "total_time": record.get("total_time", 0),
        "rag_prompt_tokens": rag_cost.get("prompt_tokens", 0),
        "rag_completion_tokens": rag_cost.get("completion_tokens", 0),
        "rag_total_tokens": rag_cost.get("total_tokens", 0),
        "attack_prompt_tokens": attack_cost.get("prompt_tokens", 0),
        "attack_completion_tokens": attack_cost.get("completion_tokens", 0),
        "attack_total_tokens": attack_cost.get("total_tokens", 0),
        "source_log": str(source_log),
    }


def write_csv(path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs-root", type=Path, default=Path("logs/standard"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", action="append")
    parser.add_argument("--dataset", action="append")
    parser.add_argument("--attack", action="append")
    args = parser.parse_args()

    details = []
    for result_path in sorted(args.logs_root.glob("*/*/*/results.jsonl")):
        model = result_path.parts[-4]
        dataset = result_path.parts[-3]
        attack = result_path.parts[-2]
        if args.model and model not in args.model:
            continue
        if args.dataset and dataset not in args.dataset:
            continue
        if args.attack and attack not in args.attack:
            continue

        rows = [
            row_from_record(model, dataset, attack, record, result_path)
            for record in read_json_objects(result_path)
        ]
        details.extend(rows)
        write_csv(
            args.output_dir / model / dataset / f"{attack}.csv",
            DETAIL_FIELDS,
            rows,
        )

    write_csv(args.output_dir / "all_queries.csv", DETAIL_FIELDS, details)

    summary_fields = [
        "model",
        "dataset",
        "attack",
        "queries",
        "extracted_count",
        "retrieved_count",
        "total_time",
        "rag_total_tokens",
        "attack_total_tokens",
        "overall_tokens",
        "source_log",
    ]
    summary = []
    groups = {}
    for row in details:
        key = (row["model"], row["dataset"], row["attack"], row["source_log"])
        groups.setdefault(key, []).append(row)

    for (model, dataset, attack, source_log), rows in sorted(groups.items()):
        rag_tokens = sum(row["rag_total_tokens"] for row in rows)
        attack_tokens = sum(row["attack_total_tokens"] for row in rows)
        summary.append(
            {
                "model": model,
                "dataset": dataset,
                "attack": attack,
                "queries": len(rows),
                "extracted_count": sum(row["extracted_count"] for row in rows),
                "retrieved_count": sum(row["retrieved_count"] for row in rows),
                "total_time": sum(row["total_time"] for row in rows),
                "rag_total_tokens": rag_tokens,
                "attack_total_tokens": attack_tokens,
                "overall_tokens": rag_tokens + attack_tokens,
                "source_log": source_log,
            }
        )

    write_csv(args.output_dir / "summary.csv", summary_fields, summary)


if __name__ == "__main__":
    main()
