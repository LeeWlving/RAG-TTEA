import argparse
import csv
import json
from pathlib import Path


DETAIL_FIELDS = [
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


def detail_row(attack, record, source_log):
    rag_cost = record.get("rag_step_cost", {})
    attack_cost = record.get("attack_step_cost", {})
    extracted = record.get("extracted_info") or []
    retrieved = record.get("retrieved_docs") or []
    return {
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    details = []

    for attack, log_path in manifest.items():
        source_log = Path(log_path)
        rows = [
            detail_row(attack, record, source_log)
            for record in read_json_objects(source_log)
        ]
        details.extend(rows)
        with (args.output_dir / f"{attack}.csv").open(
            "w", newline="", encoding="utf-8-sig"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=DETAIL_FIELDS)
            writer.writeheader()
            writer.writerows(rows)

    with (args.output_dir / "all_queries.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=DETAIL_FIELDS)
        writer.writeheader()
        writer.writerows(details)

    summary_fields = [
        "attack",
        "queries",
        "extracted_count",
        "retrieved_count",
        "total_time",
        "rag_total_tokens",
        "attack_total_tokens",
        "overall_tokens",
    ]
    summary = []
    for attack in manifest:
        rows = [row for row in details if row["attack"] == attack]
        rag_tokens = sum(row["rag_total_tokens"] for row in rows)
        attack_tokens = sum(row["attack_total_tokens"] for row in rows)
        summary.append(
            {
                "attack": attack,
                "queries": len(rows),
                "extracted_count": sum(row["extracted_count"] for row in rows),
                "retrieved_count": sum(row["retrieved_count"] for row in rows),
                "total_time": sum(row["total_time"] for row in rows),
                "rag_total_tokens": rag_tokens,
                "attack_total_tokens": attack_tokens,
                "overall_tokens": rag_tokens + attack_tokens,
            }
        )

    with (args.output_dir / "summary.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerows(summary)


if __name__ == "__main__":
    main()
