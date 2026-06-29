import argparse
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download


DATASETS = {
    "HarryPotter": {
        "repo_id": "vapit/HarryPotterQA",
        "files": ["data/train-00000-of-00001.parquet"],
    },
    "Pokemon": {
        "repo_id": "tungdop2/pokemon",
        "files": ["data/train-00000-of-00001-5e017125a702cfbb.parquet"],
    },
    "Enron": {
        "repo_id": "MichaelR207/enron_qa_0922",
        "files": [
            "data/train-00000-of-00002.parquet",
            "data/train-00001-of-00002.parquet",
        ],
    },
}


def download_frames(repo_id, filenames):
    paths = [
        hf_hub_download(repo_id=repo_id, filename=name, repo_type="dataset")
        for name in filenames
    ]
    return [pd.read_parquet(path) for path in paths]


def prepare_harry_potter(root):
    spec = DATASETS["HarryPotter"]
    frame = download_frames(spec["repo_id"], spec["files"])[0]
    output = root / "HarryPotter" / "HarryPotterQA-26k.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    frame[["context", "question", "answer"]].fillna("").to_csv(
        output, index=False, encoding="utf-8"
    )
    return output, len(frame)


def prepare_pokemon(root):
    spec = DATASETS["Pokemon"]
    frame = download_frames(spec["repo_id"], spec["files"])[0]
    output = root / "Pokemon" / "Pokemon-1k.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    frame[["name", "type_1", "type_2", "caption"]].fillna("").to_csv(
        output, index=False, encoding="utf-8"
    )
    return output, len(frame)


def prepare_enron(root):
    spec = DATASETS["Enron"]
    frames = download_frames(spec["repo_id"], spec["files"])
    frame = pd.concat(frames, ignore_index=True)
    output = root / "Enron" / "emails.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.rename(columns={"email": "message"})[["message"]].fillna("").to_csv(
        output, index=False, encoding="utf-8"
    )
    return output, len(frame)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    args = parser.parse_args()

    for prepare in (prepare_harry_potter, prepare_pokemon, prepare_enron):
        output, rows = prepare(args.data_dir)
        print(f"{output}: {rows} rows")


if __name__ == "__main__":
    main()
