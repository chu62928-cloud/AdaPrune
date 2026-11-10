"""Evaluate majority voting using prefixes of the same Best-of-N candidate pool."""
import argparse
import json
from pathlib import Path
from .answers import vote
from .io import read_jsonl


def evaluate(rows, ks):
    report = []
    for k in ks:
        n_ok, n_answerable, n_failed, total_generated = 0, 0, 0, 0
        for row in rows:
            selected = row["rollouts"][:k]
            if len(selected) < k:
                raise ValueError(f"Row {row['id']} has only {len(selected)} candidates; requested k={k}")
            failed = [r for r in selected if r["status"] != "ok"]
            if failed:
                n_failed += 1
            # No optimistic skipping: an API error is NOT silently ignored.
            pred, _ = vote([r.get("answer") for r in selected])
            n_answerable += (pred is not None)
            n_ok += (pred == row["gold"])
            total_generated += sum(int(r.get("completion_tokens") or 0) for r in selected)
        report.append({
            "k": k,
            "questions": len(rows),
            "correct": n_ok,
            "accuracy": round(n_ok / len(rows), 4) if rows else None,
            "answered": n_answerable,
            "questions_with_api_errors": n_failed,
            "generated_tokens": total_generated,
            "note": "Prefix-of-N retrospective comparison, not separately timed runs"
        })
    return report


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--ks", default="1,4,8")
    p.add_argument("--output", default="")
    args = p.parse_args()
    rows = list(read_jsonl(args.input))
    ks = [int(x) for x in args.ks.split(",")]
    result = evaluate(rows, ks)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
