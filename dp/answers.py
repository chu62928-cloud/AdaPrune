"""Conservative AIME integer answer extraction and voting.

AIME answers are integers 0..999; do not treat arbitrary digits in reasoning as answers.
"""
import re
from collections import defaultdict


def _boxed_contents(text):
    """Parse simple and nested braces for each \\boxed{...}."""
    result = []
    for found in re.finditer(r"\\boxed\s*\{", text):
        start = found.end()
        depth = 1
        for idx in range(start, len(text)):
            if text[idx] == "{":
                depth += 1
            elif text[idx] == "}":
                depth -= 1
                if depth == 0:
                    result.append(text[start:idx])
                    break
    return result


def normalize_aime(value):
    if value is None:
        return None
    raw = str(value).strip().strip("$ ")
    # Permits \boxed{\text{042}} and \boxed{\mathbf{042}}.
    for _ in range(2):
        raw = re.sub(r"^\\(?:text|mathbf|mathrm)\s*\{([^{}]*)\}$", r"\1", raw).strip()
    if not re.fullmatch(r"\+?\d{1,3}", raw):
        return None
    n = int(raw)
    return str(n) if 0 <= n <= 999 else None


def extract_aime_answer(text):
    if not text:
        return None
    boxes = _boxed_contents(text)
    for b in reversed(boxes):
        answer = normalize_aime(b)
        if answer is not None:
            return answer
    # Only unambiguous 'final answer' phrases, not arbitrary last numbers.
    matches = list(re.finditer(
        r"(?:final\s+answer|answer\s+is|最终答案|答案(?:是|为)?)\s*[:：=]?\s*\$?\s*(\d{1,3})(?!\d)",
        text, flags=re.IGNORECASE
    ))
    return normalize_aime(matches[-1].group(1)) if matches else None


def vote(answers, weights=None):
    """Return (prediction, scores); deterministic ties prefer first seen answer."""
    if weights is None:
        weights = [1.0] * len(answers)
    if len(weights) != len(answers):
        raise ValueError("weights and answers lengths differ")
    scores = defaultdict(float)
    seen = {}
    for idx, (a, w) in enumerate(zip(answers, weights)):
        if a is not None and w > 0:
            scores[str(a)] += float(w)
            seen.setdefault(str(a), idx)
    if not scores:
        return None, {}
    result = min(scores, key=lambda a: (-scores[a], seen[a]))
    return result, dict(scores)
