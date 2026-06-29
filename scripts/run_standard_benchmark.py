import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


DATASETS = {
    "Enron": {
        "db_path": "enron_500k",
        "role": "email assistant",
        "topic": "enron emails",
    },
    "HarryPotter": {
        "db_path": "harrypotter_26k",
        "role": "book assistant",
        "topic": "harry potter",
    },
    "Pokemon": {
        "db_path": "pokemon_1k",
        "role": "pokemon assistant",
        "topic": "pokemon",
    },
}

ATTACKS = [
    "RandomToken",
    "RandomText",
    "RandomEmb",
    "DGEA",
    "CopyBreak",
    "IKEA",
]


def count_records(path):
    if not path.exists():
        return 0
    decoder = json.JSONDecoder()
    content = path.read_text(encoding="utf-8")
    position = 0
    count = 0
    while position < len(content):
        while position < len(content) and content[position].isspace():
            position += 1
        if position >= len(content):
            break
        _, position = decoder.raw_decode(content, position)
        count += 1
    return count


def attack_args(attack, model, topic):
    if attack == "RandomToken":
        return [
            "--ak_attack_template", "copybreak/attack_template.txt",
            "--ak_emb_model", "MiniLM",
            "--ak_pool_size", "25",
        ]
    if attack == "RandomText":
        return [
            "--ak_attack_template", "copybreak/attack_template.txt",
            "--ak_llm_model", model,
            "--ak_system_prompt", "random/gen_system.txt",
            "--ak_template", "random/gen_template.txt",
            "--ak_temperature", "0.9",
        ]
    if attack == "RandomEmb":
        return [
            "--ak_attack_template", "copybreak/attack_template.txt",
            "--ak_emb_model", "MiniLM",
            "--ak_iterations", "3",
            "--ak_info_prompt", "random/ak_suffix.txt",
            "--ak_pool_size", "512",
            "--ak_random_vec", "embedding_statistics.csv",
        ]
    if attack == "DGEA":
        return [
            "--ak_command_prompt", "copybreak/attack_template.txt",
            "--ak_emb_model", "MiniLM",
            "--ak_iterations", "3",
            "--ak_info_prompt", "dgea/ak_suffix.txt",
            "--ak_pool_size", "512",
            "--ak_random_vec", "embedding_statistics.csv",
        ]
    if attack == "CopyBreak":
        return [
            "--ak_llm_model", model,
            "--ak_emb_model", "MiniLM",
            "--ak_attack_template", "copybreak/attack_template.txt",
            "--ak_sim_thresh", "0.6",
            "--ak_iterations", "10",
            "--ak_explore_template", "copybreak/explore_template.txt",
            "--ak_exploit_template", "copybreak/exploit_template.txt",
            "--ak_exchange_rate", "5",
            "--ak_num_of_each_reason", "2",
            "--ak_explore_temperature", "0.7",
            "--ak_exploit_temperature", "0.3",
        ]
    if attack == "IKEA":
        return [
            "--ak_attack_llm", model,
            "--ak_attack_emb_model", "MiniLM",
            "--ak_device", "cuda:0",
            "--ak_topic_word", topic,
            "--ak_num_anchors", "50",
            "--ak_anchor_gen_template", "ikea/anchor_gen_template.txt",
            "--ak_query_gen_iterations", "5",
            "--ak_thresh_sim_topic", "0.3",
            "--ak_thresh_dissim_anchor", "0.5",
            "--ak_thresh_q_anchor", "0.7",
            "--ak_sample_temperature", "1.0",
            "--ak_anchor_query_gen_template", "ikea/anchor_query_gen_template.txt",
            "--ak_thresh_irrelevant", "0.7",
            "--ak_thresh_outlier", "0.7",
            "--ak_penalty_refusal", "10.0",
            "--ak_penalty_irrelevant", "7.0",
            "--ak_thresh_qy_sim", "0.5",
            "--ak_gamma", "0.5",
            "--ak_anchor_mutate_gen_template", "ikea/anchor_mutate.txt",
            "--ak_thresh_stop_q", "0.6",
            "--ak_thresh_stop_y", "0.6",
        ]
    raise ValueError(f"Unsupported attack: {attack}")


def run_one(root, python, model, dataset, attack, max_query):
    dataset_cfg = DATASETS[dataset]
    model_dir = model.replace("-", "_")
    save_dir = root / "logs" / "standard" / model_dir / dataset / attack
    save_dir.mkdir(parents=True, exist_ok=True)
    results_path = save_dir / "results.jsonl"
    completed = count_records(results_path)
    if completed >= max_query:
        print(f"SKIP {model}/{dataset}/{attack}: {completed}/{max_query}")
        return

    command = [
        str(python),
        "-s",
        "pipeline.py",
        "--continue_pipe",
        "--continue_dir", str(save_dir),
        "--des", f"Standard {attack} with {model}",
        "--dataset", dataset,
        "--rag", "TextRAG",
        "--attack", attack,
        "--defense", "None",
        "--seed", "42",
        "--gpu", "0",
        "--rg_db_path", dataset_cfg["db_path"],
        "--rg_retriever", "MiniLM",
        "--rg_generator", model,
        "--rg_device", "cuda:0",
        "--rg_retr_kwargs_topk", "3",
        "--rg_role", dataset_cfg["role"],
        "--rg_gen_kwargs_system_prompt", "textrag/system.txt",
        "--rg_gen_kwargs_template", "textrag/template.txt",
        "--rg_gen_kwargs_temperature", "0.1",
        "--ak_max_query", str(max_query),
        *attack_args(attack, model, dataset_cfg["topic"]),
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(root)
    env["LOG_DIR"] = str(root / "logs")
    env["DATA_PATH"] = str(root / "data")
    env["DB_PATH"] = str(root / "data" / "databases")
    env["KEYS_PATH"] = str(root / "keys.yaml")
    env["PROMPT_PATH"] = str(root / "prompts")
    env["EXTRA_PATH"] = str(root / "extra_data")
    env["REFUSAL_MODEL"] = model

    print(f"RUN {model}/{dataset}/{attack}: resume at {completed}/{max_query}")
    result = subprocess.run(command, cwd=root, env=env)
    if result.returncode != 0:
        raise SystemExit(result.returncode)
    completed = count_records(results_path)
    if completed < max_query:
        raise RuntimeError(
            f"{model}/{dataset}/{attack} stopped at {completed}/{max_query}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["deepseek-chat", "qwen-plus"],
        required=True,
    )
    parser.add_argument(
        "--datasets", nargs="+", choices=DATASETS, default=list(DATASETS)
    )
    parser.add_argument(
        "--attacks", nargs="+", choices=ATTACKS, default=ATTACKS
    )
    parser.add_argument("--max-query", type=int, default=200)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    python = Path(sys.executable)
    for model in args.models:
        required_key = (
            "DEEPSEEK_API_KEY" if model == "deepseek-chat" else "DASHSCOPE_API_KEY"
        )
        if not os.environ.get(required_key):
            raise RuntimeError(f"Missing environment variable: {required_key}")
        for dataset in args.datasets:
            for attack in args.attacks:
                run_one(
                    root,
                    python,
                    model,
                    dataset,
                    attack,
                    args.max_query,
                )


if __name__ == "__main__":
    main()
