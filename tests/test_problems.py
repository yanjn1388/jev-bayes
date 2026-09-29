import math

from jev_bayes.problems import fmt_pct, posterior, sens_fpr_from_lr


def test_posterior_classic_example():
    # 有病率 1%・感度 90%・偽陽性率 9% → 約 9.2%
    p = posterior(0.01, 0.9, 0.09)
    assert math.isclose(p, 0.009 / (0.009 + 0.99 * 0.09))
    assert abs(p - 0.0917) < 1e-3


def test_posterior_uninformative_test_returns_prior():
    assert math.isclose(posterior(0.2, 0.5, 0.5), 0.2)


def test_sens_fpr_from_lr_above_one():
    sens, fpr = sens_fpr_from_lr(20)
    assert math.isclose(sens, 0.9)
    assert math.isclose(fpr, 0.045)
    assert math.isclose(sens / fpr, 20)


def test_sens_fpr_from_lr_below_one():
    sens, fpr = sens_fpr_from_lr(0.2)
    assert math.isclose(sens, 0.18)
    assert math.isclose(fpr, 0.9)
    assert math.isclose(sens / fpr, 0.2)


def test_sens_fpr_from_lr_rejects_nonpositive():
    import pytest

    with pytest.raises(ValueError):
        sens_fpr_from_lr(0)


def test_fmt_pct_strips_trailing_zeros():
    assert fmt_pct(0.045) == "4.5"
    assert fmt_pct(0.01) == "1"
    assert fmt_pct(0.9 * 0.2) == "18"
    assert fmt_pct(0.8) == "80"
from collections import Counter

from jev_bayes.problems import QID, generate_all, read_jsonl, write_jsonl


def _by_id(rows):
    return {r["id"]: r for r in rows}


def test_counts_by_stage():
    rows = generate_all()
    counts = Counter((r["stage"], r["lang"]) for r in rows)
    assert counts[(1, "en")] == 120
    assert counts[(2, "en")] == 200
    assert counts[(2, "ja")] == 25
    assert counts[(3, "en")] == 72


def test_ids_are_unique():
    ids = [r["id"] for r in generate_all()]
    assert len(ids) == len(set(ids))


def test_generation_is_deterministic():
    assert generate_all() == generate_all()


def test_stage2_probabilities_are_valid():
    for r in generate_all():
        if r["stage"] != 2:
            continue
        assert 0 < r["p_true"] < 1
        assert 0 < r["params"]["sens"] <= 1
        assert 0 < r["params"]["fpr"] <= 1


def test_stage2_medical_example_text_and_values():
    r = _by_id(generate_all())["s2-medical-pi0.01-lr20-noul-en"]
    state = r["request"]["state"]
    assert "1% of people" in state
    assert "90% of people who have" in state
    assert "4.5% of people who do not" in state
    assert abs(r["p_true"] - 0.009 / (0.009 + 0.99 * 0.045)) < 1e-12
    assert r["p_inv"] == 0.9
    assert r["request"]["questions"][QID]["type"] == "noul"
    assert r["target"] is None


def test_stage2_choice_has_yes_no_criteria_and_target():
    r = _by_id(generate_all())["s2-taxi-pi0.2-lr0.5-choice-en"]
    q = r["request"]["questions"][QID]
    assert q["type"] == "choice"
    assert set(q["criteria"]) == {"yes", "no"}
    assert r["target"] == "yes"


def test_stage1_hidden_outcomes():
    rows = generate_all()
    die_choice = [r for r in rows if r["id"].startswith("s1-die6-choice")]
    die_noul = [r for r in rows if r["id"].startswith("s1-die6-noul")]
    assert all(r["hidden_outcome"] in {"1", "2", "3", "4", "5", "6"} for r in die_choice)
    assert all(r["hidden_outcome"] in {0, 1} for r in die_noul)
    assert all(r["target"] is None for r in die_choice)
    assert all(abs(r["p_true"] - 1 / 6) < 1e-12 for r in die_choice)


def test_stage1_states_are_distinct_by_draw():
    rows = [r for r in generate_all() if r["id"].startswith("s1-die6-choice")]
    states = {r["request"]["state"] for r in rows}
    assert len(states) == len(rows)


def test_stage3_true_probability_is_stated_value():
    r = _by_id(generate_all())["s3-spam-x45-choice-en"]
    assert r["p_true"] == 0.45
    assert "45%" in r["request"]["state"]


def test_japanese_problems_are_medical_noul():
    ja = [r for r in generate_all() if r["lang"] == "ja"]
    assert {r["theme"] for r in ja} == {"medical"}
    assert {r["qtype"] for r in ja} == {"noul"}
    assert "病気 X" in ja[0]["request"]["state"]


def test_jsonl_round_trip(tmp_path):
    rows = generate_all()[:3]
    path = tmp_path / "p.jsonl"
    write_jsonl(path, rows)
    assert read_jsonl(path) == rows
    assert read_jsonl(tmp_path / "missing.jsonl") == []
