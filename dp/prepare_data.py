"""Download AI-MO/aimo-validation-aime and split by year (2022-23 dev, 2024 test)."""
import argparse
import re
from pathlib import Path
from .io import append_jsonl
from .answers import normalize_aime


def detect_year(url):
    match = re.search(r"\b(2022|2023|2024)(?=[_\W])", str(url))
    return int(match.group(1)) if match else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", default="data")
    args = parser.parse_args()
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    files = {"dev": out / "aime_dev.jsonl", "test": out / "aime_2024.jsonl"}
    for path in files.values():
        if path.exists():
            path.unlink()
    from datasets import load_dataset
    dataset = load_dataset("AI-MO/aimo-validation-aime", split="train")
    counts = {"dev": 0, "test": 0}
    for ix, row in enumerate(dataset):
        year = detect_year(row.get("url", ""))
        if year is None:
            raise ValueError(f"Unable to infer year for row {ix}: {row.get('url')}")
        answer = normalize_aime(row.get("answer"))
        if answer is None:
            raise ValueError(f"Unrecognized AIME answer on row {ix}: {row.get('answer')}")
        split = "test" if year == 2024 else "dev"
        append_jsonl(files[split], {
            "id": f"aime-{year}-{row['id']}",
            "year": year,
            "problem": row["problem"],
            "answer": answer,
            "url": row.get("url", "")
        })
        counts[split] += 1
    print(f"Prepared {counts}; files: {files}")
    if counts != {"dev": 60, "test": 30}:
        raise RuntimeError(f"Unexpected split sizes: {counts}, expected dev=60/test=30")


if __name__ == "__main__":
    main()
