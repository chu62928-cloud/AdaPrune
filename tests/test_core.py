import math
from dp.answers import extract_aime_answer, normalize_aime, vote
from dp.entropy import token_conditional_entropy, entropy_windows, first_prune_window
from dp.prepare_data import detect_year
from dp.evaluate import evaluate


def test_answer_extraction():
    assert extract_aime_answer("Working: 1,2,3. Hence \\boxed{042}.") == "42"
    assert extract_aime_answer("\\boxed{5} but revised \\boxed{009}") == "9"
    assert extract_aime_answer("Final answer: 123") == "123"
    assert extract_aime_answer("I considered 12 and 19") is None
    assert normalize_aime("1234") is None


def test_vote_tie_first_seen():
    assert vote(["9", "2", "2", "9"])[0] == "9"
    assert vote(["9", "2"], [0.1, 1.0])[0] == "2"
    assert vote([None, None])[0] is None


def test_entropy_math():
    hs = token_conditional_entropy([{"token": "a", "logprob": math.log(0.5)},
                                    {"token": "b", "logprob": math.log(0.5)}])
    assert abs(hs - math.log(2)) < 1e-8
    assert first_prune_window([0.2, 0.4, 0.6], delta=0.1) == 2
    assert first_prune_window([0.2, 0.4, 0.3], delta=0.1) is None
    assert entropy_windows([0.1, 0.3, None, 0.7], 2) == [0.2, 0.7]


def test_dataset_year():
    assert detect_year("https://artofproblemsolving.com/wiki/index.php/2024_AIME_II_Problems") == 2024
    assert detect_year("https://artofproblemsolving.com/wiki/index.php/2022_AIME_I_Problems") == 2022


def test_eval():
    rows = [{"id": "1", "gold": "42", "rollouts": [
        {"status": "ok", "answer": "42", "completion_tokens": 10},
        {"status": "ok", "answer": "9", "completion_tokens": 10},
        {"status": "ok", "answer": "42", "completion_tokens": 10}
    ]}]
    assert evaluate(rows, [1, 3])[1]["accuracy"] == 1.0
    assert evaluate(rows, [1, 3])[1]["generated_tokens"] == 30


def test_staged_replay_does_not_require_answers_for_selection():
    from dp.staged_replay import stage_select, replay_row
    # Six tokens per path, each with a valid two-token logprob distribution.
    def make(ident, ans, p):
        return {
            "rollout_id": ident, "status": "ok", "answer": ans,
            "logprobs": [{"top_logprobs": [
                {"token": "a", "logprob": math.log(p)},
                {"token": "b", "logprob": math.log(1 - p)},
            ]} for _ in range(8)]
        }
    pool = [make(0, "42", .95), make(1, "9", .8),
            make(2, "42", .6), make(3, "9", .55)]
    selected, _ = stage_select(pool, keep=2, checkpoint=6, window=2, delta=.1)
    # Low entropy must be preferred here, independent of candidate answer labels.
    assert [r["rollout_id"] for r in selected] == [0, 1]
    row = {"id": "test", "gold": "42", "rollouts": pool}
    result = replay_row(row, 3, 2, 6, 8, 2, .1, seed=4)
    assert len(result["selected_stage1_ids"]) == 3
    assert len(result["selected_stage2_ids"]) == 2
    assert result["original_contains_gold"]
