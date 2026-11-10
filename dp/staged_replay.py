"""Two-stage, prefix-only *offline* counterfactual selection.

This is NOT online pruning, NOT sandbox verification, and saves zero GPU time.
All candidate answers come from full rollouts generated previously.
"""
import argparse
import json
import random
from pathlib import Path

from .answers import vote
from .entropy import average_valid, collect_token_entropies, entropy_windows, first_prune_window
from .io import read_jsonl


def stage_select(pool, keep, checkpoint, window, delta):
    if keep > len(pool):
        raise ValueError(f"Can only keep {len(pool)} candidates, got {keep}")
    if checkpoint < 3 * window:
        raise ValueError("checkpoint must contain >=3 entropy windows")
    stats = []
    for r in pool:
        if r.get("status") != "ok":
            raise ValueError(f"Inference failed for candidate {r.get('rollout_id')}; rerun this question")
        prefix = collect_token_entropies(r.get("logprobs"))[:checkpoint]
        mean = average_valid(prefix)
        if mean is None:
            raise ValueError("Missing logprobs: run baseline with --logprobs")
        fired = first_prune_window(entropy_windows(prefix, window), delta)
        stats.append({
            "rollout_id": r["rollout_id"], "prefix_len": len(prefix),
            "mean_topk_entropy_nats": mean, "trigger_window": fired,
        })
    # No correct answers or full-future entropies are used in ranking.
    stats.sort(key=lambda x: (
        x["trigger_window"] is not None,
        x["mean_topk_entropy_nats"], x["rollout_id"]
    ))
    selected_ids = {v["rollout_id"] for v in stats[:keep]}
    selected = [r for r in pool if r["rollout_id"] in selected_ids]
    return selected, stats


def replay_row(row, keep1, keep2, step1, step2, window, delta, seed):
    pool = row["rollouts"]
    s1, stats1 = stage_select(pool, keep1, step1, window, delta)
    s2, stats2 = stage_select(s1, keep2, step2, window, delta)
    vote_all, _ = vote([r.get("answer") for r in pool])
    vote_selected, _ = vote([r.get("answer") for r in s2])
    # Final full-length entropy may be used for final voting of survivors.
    means = [average_valid(collect_token_entropies(r.get("logprobs"))) for r in s2]
    weights = [1.0 / max(h, 0.05) if h is not None else 0.0 for h in means]
    weighted, _ = vote([r.get("answer") for r in s2], weights=weights)
    rand_pool = list(pool)
    random.Random(seed).shuffle(rand_pool)
    rand_choice, _ = vote([r.get("answer") for r in rand_pool[:keep2]])
    gold = row["gold"]
    return {
        "id": row["id"], "gold": gold,
        "full_pool_prediction": vote_all,
        "random_prediction": rand_choice,
        "staged_majority_prediction": vote_selected,
        "staged_weighted_prediction": weighted,
        "full_pool_correct": vote_all == gold,
        "random_correct": rand_choice == gold,
        "staged_majority_correct": vote_selected == gold,
        "staged_weighted_correct": weighted == gold,
        "selected_stage1_ids": [r["rollout_id"] for r in s1],
        "selected_stage2_ids": [r["rollout_id"] for r in s2],
        "original_contains_gold": any(r.get("answer") == gold for r in pool),
        "retained_contains_gold": any(r.get("answer") == gold for r in s2),
        "stage1_prefix_scores": stats1,
        "stage2_prefix_scores": stats2,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", default="outputs/staged_replay.json")
    p.add_argument("--stage1-keep", type=int, default=4)
    p.add_argument("--stage2-keep", type=int, default=2)
    p.add_argument("--stage1-tokens", type=int, default=128)
    p.add_argument("--stage2-tokens", type=int, default=256)
    p.add_argument("--window", type=int, default=32)
    p.add_argument("--delta", type=float, default=0.08)
    p.add_argument("--seed", type=int, default=2026)
    a = p.parse_args()
    if not (0 < a.stage2_keep <= a.stage1_keep) or a.stage2_tokens < a.stage1_tokens:
        raise ValueError("Require 0 < stage2-keep <= stage1-keep and step2 >= step1")
    rows = list(read_jsonl(a.input))
    results = [replay_row(r, a.stage1_keep, a.stage2_keep, a.stage1_tokens,
                          a.stage2_tokens, a.window, a.delta, a.seed + i)
               for i, r in enumerate(rows)]
    n = len(results)
    summary = {"questions": n, "stage1_keep": a.stage1_keep,
               "stage2_keep": a.stage2_keep,
               "stage1_checkpoint_tokens": a.stage1_tokens,
               "stage2_checkpoint_tokens": a.stage2_tokens,
               "window": a.window, "delta": a.delta,
               "baseline_full_accuracy": sum(x["full_pool_correct"] for x in results)/n if n else None,
               "random_keep_accuracy": sum(x["random_correct"] for x in results)/n if n else None,
               "staged_majority_accuracy": sum(x["staged_majority_correct"] for x in results)/n if n else None,
               "staged_weighted_accuracy": sum(x["staged_weighted_correct"] for x in results)/n if n else None,
               "warning": "Offline replay: all rollouts ALREADY fully generated; NO compute saved. "
                          "Entropy-only second stage, NOT the poster's Python sandbox."}
    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"summary": summary, "rows": results}, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
