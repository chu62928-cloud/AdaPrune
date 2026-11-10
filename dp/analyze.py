"""Retrospective entropy analysis only. It DOES NOT prune while generating.

Intended as diagnostic groundwork before implementing vLLM online abort.
"""
import argparse
import json
from pathlib import Path
from .answers import vote
from .entropy import (
    collect_token_entropies, entropy_windows, first_prune_window, average_valid
)
from .io import read_jsonl


def analyze_row(row, keep, window, delta, checkpoint_tokens):
    candidates = row["rollouts"]
    if keep > len(candidates):
        raise ValueError("keep cannot exceed candidate count")
    infos = []
    for r in candidates:
        token_ent = collect_token_entropies(r.get("logprobs"))
        prefix_ent = token_ent[:checkpoint_tokens]
        windows = entropy_windows(prefix_ent, window)
        trigger = first_prune_window(windows, delta)
        infos.append({
            "rollout_id": r["rollout_id"],
            "answer": r.get("answer"),
            "valid_entropy_tokens": sum(v is not None for v in prefix_ent),
            "mean_topk_cond_entropy_nats": average_valid(prefix_ent),
            "trigger_window": trigger,
            "trigger_after_tokens": min((trigger + 1) * window, len(prefix_ent)) if trigger is not None else None,
            "full_completion_tokens": r.get("completion_tokens"),
            "window_means": windows,
        })
    if any(i["mean_topk_cond_entropy_nats"] is None for i in infos):
        raise ValueError("Missing token logprobs. Generate baseline with --logprobs first.")
    # Prefer unflagged traces; tie-breaking by lower mean approximate entropy.
    ranked = sorted(infos, key=lambda i: (
        i["trigger_window"] is not None,
        i["mean_topk_cond_entropy_nats"],
        i["rollout_id"]
    ))
    selected_ids = {x["rollout_id"] for x in ranked[:keep]}
    selected = [r for r in candidates if r["rollout_id"] in selected_ids]
    prediction, _ = vote([r.get("answer") for r in selected])
    complete_correct = any(r.get("answer") == row["gold"] for r in candidates)
    retained_correct = any(r.get("answer") == row["gold"] for r in selected)
    return {
        "id": row["id"], "gold": row["gold"],
        "selected_ids": sorted(selected_ids),
        "prediction": prediction,
        "correct": prediction == row["gold"],
        "had_correct_original_candidate": complete_correct,
        "retained_any_correct_candidate": retained_correct,
        "candidates": infos,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", default="outputs/entropy_offline_analysis.json")
    p.add_argument("--keep", type=int, default=4)
    p.add_argument("--window", type=int, default=32)
    p.add_argument("--delta", type=float, default=0.08)
    p.add_argument("--checkpoint-tokens", type=int, default=256,
                   help="Only this generated prefix is permitted for selection (no look-ahead).")
    args = p.parse_args()
    rows = list(read_jsonl(args.input))
    if args.keep < 1 or args.window < 1 or args.checkpoint_tokens < args.window * 3:
        raise ValueError("keep/window positive, checkpoint-tokens >= 3*window")
    results = [analyze_row(r, args.keep, args.window, args.delta, args.checkpoint_tokens) for r in rows]
    n = len(results)
    meta = {
        "method": "offline_full_trajectory_diagnostic_NOT_online_pruning",
        "window_tokens": args.window, "delta_nats": args.delta,
        "checkpoint_tokens": args.checkpoint_tokens,
        "selection_uses_only_prefix_entropy": True,
        "kept_candidates": args.keep, "questions": n,
        "accuracy": sum(r["correct"] for r in results) / n if n else None,
        "oracle_retention_on_questions_with_a_correct_sample": (
            sum(r["retained_any_correct_candidate"] for r in results if r["had_correct_original_candidate"])
            / sum(r["had_correct_original_candidate"] for r in results)
            if any(r["had_correct_original_candidate"] for r in results) else None
        ),
        "warning": "All paths were already fully generated. Selection uses prefix-only entropy, but predictions use finished candidates. NO inference/latency/KV savings are demonstrated."
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"summary": meta, "rows": results}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
