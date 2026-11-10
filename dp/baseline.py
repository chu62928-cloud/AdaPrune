"""Best-of-N inference using a local vLLM OpenAI-compatible server.

All N candidates are fully generated. No online pruning occurs here.
"""
import argparse
import asyncio
import json
import time
from pathlib import Path

from .answers import extract_aime_answer, vote
from .io import append_jsonl, read_jsonl, write_json

SYSTEM_PROMPT = (
    "You are a careful mathematical problem solver. Show your reasoning, "
    "then put ONLY the final integer answer (from 0 through 999) in "
    "\\boxed{...}. Do not include units in the box."
)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/aime_dev.jsonl")
    parser.add_argument("--output", default="outputs/best_of_8.jsonl")
    parser.add_argument("--n", type=int, default=8)
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--model", default="qwen-math-7b")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--logprobs", action="store_true")
    parser.add_argument("--top-logprobs", type=int, default=5)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def compact_logprobs(response):
    content = getattr(getattr(response.choices[0], "logprobs", None), "content", None) or []
    result = []
    for p in content:
        result.append({"top_logprobs": [
            {"token": o.token, "logprob": o.logprob, "bytes": o.bytes}
            for o in (p.top_logprobs or [])
        ]})
    return result


async def generate_one(client, sem, q, rollout_id, seed, args):
    async with sem:
        started = time.perf_counter()
        try:
            params = dict(
                model=args.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": q["problem"]}
                ],
                temperature=args.temperature,
                top_p=args.top_p,
                max_tokens=args.max_tokens,
                seed=seed,
            )
            if args.logprobs:
                params.update(logprobs=True, top_logprobs=args.top_logprobs)
            response = await client.chat.completions.create(**params)
            text = response.choices[0].message.content or ""
            usage = response.usage
            return {
                "rollout_id": rollout_id,
                "seed": seed,
                "status": "ok",
                "text": text,
                "answer": extract_aime_answer(text),
                "finish_reason": response.choices[0].finish_reason,
                "prompt_tokens": getattr(usage, "prompt_tokens", None),
                "completion_tokens": getattr(usage, "completion_tokens", None),
                "elapsed_sec": round(time.perf_counter() - started, 3),
                "logprobs": compact_logprobs(response) if args.logprobs else None,
            }
        except Exception as exc:
            return {
                "rollout_id": rollout_id,
                "seed": seed,
                "status": "error",
                "error": repr(exc),
                "answer": None,
                "elapsed_sec": round(time.perf_counter() - started, 3)
            }


async def main():
    args = get_args()
    # Import after argparse so --help and CPU-only unit tests work before setup.
    from openai import AsyncOpenAI
    if args.n < 1 or args.concurrency < 1 or args.max_tokens < 1:
        raise ValueError("n, concurrency and max-tokens must be >= 1")
    if args.logprobs and not (1 <= args.top_logprobs <= 20):
        raise ValueError("top-logprobs must be in [1,20]")
    data = list(read_jsonl(args.input))
    if args.limit > 0:
        data = data[:args.limit]
    output = Path(args.output)
    metadata_path = output.with_suffix(output.suffix + ".meta.json")
    config = {k: v for k, v in vars(args).items() if k not in {"resume", "limit"}}
    if output.exists() and not args.resume:
        raise FileExistsError(f"{output} already exists; use --resume or choose a new --output")
    if output.exists() and args.resume:
        if not metadata_path.exists():
            raise FileNotFoundError("Resume requires matching .meta.json")
        old = json.loads(metadata_path.read_text(encoding="utf-8"))
        if old != config:
            raise ValueError("Resume config mismatch. Use the same configuration or a new output path.")
    else:
        write_json(metadata_path, config)
    completed = {row["id"] for row in read_jsonl(output)} if output.exists() else set()
    client = AsyncOpenAI(base_url=args.base_url, api_key="EMPTY", timeout=3600.0, max_retries=2)
    sem = asyncio.Semaphore(args.concurrency)
    attempted = 0
    try:
        models = await client.models.list()
        print("Server models:", [m.id for m in models.data])
        if args.model not in [m.id for m in models.data]:
            raise RuntimeError(f"Requested model '{args.model}' not advertised by server")
        for idx, q in enumerate(data):
            if q["id"] in completed:
                continue
            # Question-index-based seeds remain stable across resume runs.
            seeds = [args.seed + idx * 100003 + i for i in range(args.n)]
            started = time.perf_counter()
            results = await asyncio.gather(*[
                generate_one(client, sem, q, i, seeds[i], args)
                for i in range(args.n)
            ])
            results.sort(key=lambda r: r["rollout_id"])
            if all(r["status"] != "ok" for r in results):
                print("All rollouts failed. First error:", results[0].get("error"), flush=True)
                raise RuntimeError("All rollouts failed; check model server / logprobs support before continuing")
            answers = [r.get("answer") for r in results]
            pred, scores = vote(answers)
            record = {
                "id": q["id"], "year": q.get("year"),
                "gold": q["answer"], "prediction": pred,
                "correct": pred == q["answer"],
                "vote_scores": scores,
                "question_wall_sec": round(time.perf_counter() - started, 3),
                "rollouts": results,
            }
            append_jsonl(output, record)
            attempted += 1
            valid = sum(r["status"] == "ok" for r in results)
            print(f"{q['id']}: pred={pred}, gold={q['answer']}, valid={valid}/{args.n}", flush=True)
    finally:
        await client.close()
    print(f"Finished {attempted} new questions -> {output}")


if __name__ == "__main__":
    asyncio.run(main())
