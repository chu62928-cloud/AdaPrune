"""Top-k conditional entropy: an APPROXIMATE uncertainty proxy, not full vocab entropy."""
import math


def token_conditional_entropy(top_logprobs):
    """Return H(P|token is in returned top-k) in nats, or None.

    This metric discards unknown tail mass and is not equal to full softmax entropy.
    """
    if not top_logprobs:
        return None
    vals = []
    unique = set()
    for idx, entry in enumerate(top_logprobs):
        if entry is None:
            continue
        key = tuple(entry.get("bytes") or []) if entry.get("bytes") else entry.get("token", idx)
        if key in unique:
            continue
        unique.add(key)
        lp = entry.get("logprob")
        if isinstance(lp, (float, int)) and math.isfinite(lp):
            vals.append(float(lp))
    if not vals:
        return None
    shift = max(vals)
    probs = [math.exp(v - shift) for v in vals]
    z = sum(probs)
    probs = [p / z for p in probs]
    return -sum(p * math.log(p) for p in probs if p > 0)


def collect_token_entropies(logprob_content):
    results = []
    for entry in (logprob_content or []):
        h = token_conditional_entropy(entry.get("top_logprobs") or [])
        results.append(h)
    return results


def entropy_windows(values, window=32):
    if window <= 0:
        raise ValueError("window must be positive")
    # Keep original token offsets when measuring the first time a prune condition fires.
    out = []
    for start in range(0, len(values), window):
        part = [x for x in values[start:start + window] if x is not None]
        out.append(sum(part) / len(part) if part else None)
    return out


def first_prune_window(windows, delta=0.08):
    if delta < 0:
        raise ValueError("delta must be nonnegative")
    for i in range(2, len(windows)):
        a, b, c = windows[i - 2:i + 1]
        if a is not None and b is not None and c is not None:
            if b - a > delta and c - b > delta:
                return i  # zero based
    return None


def average_valid(values):
    valid = [x for x in values if x is not None]
    return sum(valid) / len(valid) if valid else None
