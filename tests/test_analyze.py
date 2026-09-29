import numpy as np
import pytest

from jev_bayes.analyze import (
    brier_parts,
    bootstrap,
    ci95,
    fit_bayes_regression,
    load_records,
    logit,
    n_clipped,
    step_fit,
    summarize,
)
from jev_bayes.problems import LRS, PRIORS, generate_all, posterior, sens_fpr_from_lr


def _sigmoid(z):
    return 1 / (1 + np.exp(-z))


def _grid():
    prior = np.array([p for p in PRIORS for _ in LRS])
    lr = np.array([lr for _ in PRIORS for lr in LRS])
    return prior, lr


def test_regression_recovers_bayes():
    prior, lr = _grid()
    q = np.array([posterior(p, *sens_fpr_from_lr(l)) for p, l in zip(prior, lr)])
    a, b1, b2 = fit_bayes_regression(prior, lr, q)
    assert a == pytest.approx(0, abs=1e-9)
    assert b1 == pytest.approx(1)
    assert b2 == pytest.approx(1)


def test_regression_recovers_base_rate_neglect():
    prior, lr = _grid()
    q = _sigmoid(0.5 * logit(prior) + np.log(lr))
    _, b1, b2 = fit_bayes_regression(prior, lr, q)
    assert b1 == pytest.approx(0.5)
    assert b2 == pytest.approx(1)


def test_clip_prevents_infinite():
    prior, lr = _grid()
    q = _sigmoid(logit(prior) + np.log(lr))
    q[0], q[-1] = 0.0, 1.0
    coef = fit_bayes_regression(prior, lr, q)
    assert np.all(np.isfinite(coef))
    assert n_clipped(q) == 2


def test_brier_identity():
    rng = np.random.default_rng(0)
    q, p = rng.uniform(size=50), rng.uniform(size=50)
    expected = np.mean(p * (1 - q) ** 2 + (1 - p) * q ** 2)
    parts = brier_parts(q, p)
    assert parts["model_error"] + parts["irreducible"] == pytest.approx(expected)


def test_step_fit_recovers_slope_and_center():
    x = np.array([0.05, 0.15, 0.3, 0.45, 0.5, 0.55, 0.7, 0.85, 0.95])
    k, c = step_fit(x, _sigmoid(8 * (x - 0.4)))
    assert k == pytest.approx(8)
    assert c == pytest.approx(0.4)


def test_bootstrap_ci_contains_truth():
    rng = np.random.default_rng(1)
    data = rng.normal(loc=2.0, scale=1.0, size=200)
    lo, hi = ci95(bootstrap(lambda idx: data[idx].mean(), len(data), n_boot=500, seed=2))
    assert lo < 2.0 < hi


def test_load_records_averages_attempts():
    problems = generate_all()[:2]
    pid = problems[0]["id"]
    responses = [
        {"problem_id": pid, "attempt": 0, "status": "ok", "q": 0.2, "chosen": "1"},
        {"problem_id": pid, "attempt": 1, "status": "ok", "q": 0.4, "chosen": "1"},
        {"problem_id": pid, "attempt": 2, "status": "invalid", "q": None, "chosen": None},
    ]
    (rec,) = load_records(problems, responses)
    assert rec["q"] == pytest.approx(0.3)
    assert rec["q_spread"] == pytest.approx(0.2)
    assert rec["n_ok"] == 2


def test_load_records_excludes_problems_without_ok():
    problems = generate_all()[:2]
    responses = [{"problem_id": problems[1]["id"], "attempt": 0, "status": "error", "q": None, "chosen": None}]
    assert load_records(problems, responses) == []


def _synthetic_records():
    """noul は基準率を無視（b1=0.5）、choice は 0/1 に倒す、③の choice は 50% で段差、という合成データ。"""
    records = []
    for p in generate_all():
        if p["stage"] == 1:
            q, chosen = (0.83, "1") if p["qtype"] == "choice" else (0.2, None)
        elif p["stage"] == 2:
            prior, lr = p["params"]["prior"], p["params"]["lr"]
            if p["qtype"] == "noul":
                q = float(_sigmoid(0.5 * np.log(prior / (1 - prior)) + np.log(lr)))
            else:
                q = float(_sigmoid(2 * (np.log(prior / (1 - prior)) + np.log(lr))))
            chosen = None
        else:
            x = p["p_true"]
            q = x if p["qtype"] == "noul" else (0.05 if x < 0.5 else 0.95)
            chosen = None
        records.append({**p, "q": q, "q_spread": 0.0, "n_ok": 1, "chosen": chosen})
    return records


def test_summarize_judges_hypotheses_on_synthetic_data():
    records = _synthetic_records()
    s = summarize(records, n_problems=len(records), n_boot=200)
    assert s["hypotheses"]["H1"]["holds"] is True
    assert s["hypotheses"]["H2"]["holds"] is True
    assert s["hypotheses"]["H3"]["holds"] is True
    assert s["stage2"]["noul-en"]["b1"] == pytest.approx(0.5)
    assert s["stage2"]["choice-en"]["b1"] > 1
    assert s["counts"]["records"] == len(records)
    assert s["counts"]["excluded"] == 0


def test_summarize_rejects_hypotheses_without_neglect():
    """② は事前確率をむしろ重く見る（b1=1.2）、③ は記載値どおり。H1 と H3 は成り立たない。"""
    records = []
    for p in generate_all():
        q = p["p_true"]
        if p["stage"] == 2:
            prior, lr = p["params"]["prior"], p["params"]["lr"]
            q = float(_sigmoid(1.2 * np.log(prior / (1 - prior)) + np.log(lr)))
        records.append({**p, "q": q, "q_spread": 0.0, "n_ok": 1, "chosen": p["hidden_outcome"]})
    s = summarize(records, n_problems=len(records), n_boot=200)
    assert s["hypotheses"]["H1"]["holds"] is False
    assert s["hypotheses"]["H3"]["holds"] is False
