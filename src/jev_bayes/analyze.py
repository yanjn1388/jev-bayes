"""応答を集計し、仮説 H1〜H3 を判定する（SPEC §4）。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable

import numpy as np

from jev_bayes.problems import SEED, THEMES, read_jsonl

EPS = 1e-3
N_BOOT = 2000
STAGE1_CONDITIONS = (("die6", "choice"), ("die6", "noul"), ("spin4", "noul"), ("spin20", "noul"))
STAGE2_CONDITIONS = (("noul", "en"), ("choice", "en"), ("noul", "ja"))


def clip(q) -> np.ndarray:
    return np.clip(np.asarray(q, dtype=float), EPS, 1 - EPS)


def logit(p) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    return np.log(p / (1 - p))


def n_clipped(q) -> int:
    q = np.asarray(q, dtype=float)
    return int(np.sum((q < EPS) | (q > 1 - EPS)))


def fit_bayes_regression(prior, lr, q) -> np.ndarray:
    """logit(q) = a + b1·logit(prior) + b2·log(LR) を最小二乗で当てはめ、[a, b1, b2] を返す。"""
    x = np.column_stack([np.ones(len(q)), logit(prior), np.log(np.asarray(lr, dtype=float))])
    coef, *_ = np.linalg.lstsq(x, logit(clip(q)), rcond=None)
    return coef


def bootstrap(stat: Callable[[np.ndarray], object], n: int, n_boot: int = N_BOOT, seed: int = SEED) -> np.ndarray:
    """添字を復元抽出して統計量を n_boot 回計算する。"""
    rng = np.random.default_rng(seed)
    return np.array([stat(rng.integers(0, n, size=n)) for _ in range(n_boot)])


def ci95(samples: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lo, hi = np.percentile(samples, [2.5, 97.5], axis=0)
    return lo, hi


def brier_parts(q, p) -> dict:
    """期待 Brier = (q-p)^2 + p(1-p) の2項の平均。"""
    q, p = np.asarray(q, dtype=float), np.asarray(p, dtype=float)
    return {"model_error": float(np.mean((q - p) ** 2)), "irreducible": float(np.mean(p * (1 - p)))}


def sampled_brier(q, y) -> float:
    q, y = np.asarray(q, dtype=float), np.asarray(y, dtype=float)
    return float(np.mean((q - y) ** 2))


def step_fit(x, q) -> tuple[float, float]:
    """logit(q) = k·(x - c) を直線で当てはめ、傾き k と中心 c を返す。"""
    k, intercept = np.polyfit(np.asarray(x, dtype=float), logit(clip(q)), 1)
    return float(k), float(-intercept / k)


def load_records(problems: list[dict], responses: list[dict]) -> list[dict]:
    """ok の応答がある問題だけを残し、試行の平均を q にする。"""
    ok: dict[str, list[dict]] = {}
    for r in responses:
        if r["status"] == "ok":
            ok.setdefault(r["problem_id"], []).append(r)
    records = []
    for p in problems:
        rows = ok.get(p["id"])
        if not rows:
            continue
        qs = [r["q"] for r in rows]
        records.append({
            **p,
            "q": float(np.mean(qs)),
            "q_spread": float(max(qs) - min(qs)),
            "n_ok": len(rows),
            "chosen": rows[0]["chosen"],
        })
    return records


def _select(records: list[dict], **conds) -> list[dict]:
    return [r for r in records if all(r.get(k) == v for k, v in conds.items())]


def _arr(records: list[dict], key: str) -> np.ndarray:
    return np.array([r[key] for r in records], dtype=float)


def summarize_stage1(records: list[dict]) -> dict:
    out = {}
    for theme, qtype in STAGE1_CONDITIONS:
        rs = _select(records, stage=1, theme=theme, qtype=qtype)
        if not rs:
            continue
        q, p = _arr(rs, "q"), _arr(rs, "p_true")
        if qtype == "choice":
            y = np.array([float(r["chosen"] == r["hidden_outcome"]) for r in rs])
        else:
            y = np.array([float(r["hidden_outcome"]) for r in rs])
        out[f"{theme}-{qtype}"] = {
            "n": len(rs),
            "mean_q": float(q.mean()),
            "p_true": float(p.mean()),
            "outcome_rate": float(y.mean()),
            "sampled_brier": sampled_brier(q, y),
            **brier_parts(q, p),
        }
    return out


def regression_summary(rs: list[dict], n_boot: int, seed: int) -> dict:
    prior = np.array([r["params"]["prior"] for r in rs])
    lr = np.array([r["params"]["lr"] for r in rs])
    q, p, p_inv = _arr(rs, "q"), _arr(rs, "p_true"), _arr(rs, "p_inv")
    a, b1, b2 = fit_bayes_regression(prior, lr, q)
    boots = bootstrap(lambda idx: fit_bayes_regression(prior[idx], lr[idx], q[idx]), len(rs), n_boot, seed)
    lo, hi = ci95(boots)
    return {
        "n": len(rs),
        "a": float(a),
        "b1": float(b1),
        "b2": float(b2),
        "ci_low": [float(v) for v in lo],
        "ci_high": [float(v) for v in hi],
        "n_clipped": n_clipped(q),
        "closer_to_inverse": int(np.sum(np.abs(q - p_inv) < np.abs(q - p))),
        **brier_parts(q, p),
    }


def summarize_stage2(records: list[dict], n_boot: int, seed: int) -> dict:
    out: dict = {"by_theme": {}}
    for qtype, lang in STAGE2_CONDITIONS:
        rs = _select(records, stage=2, qtype=qtype, lang=lang)
        if rs:
            out[f"{qtype}-{lang}"] = regression_summary(rs, n_boot, seed)
    for qtype in ("noul", "choice"):
        for theme in THEMES:
            rs = _select(records, stage=2, qtype=qtype, lang="en", theme=theme)
            if rs:
                out["by_theme"][f"{qtype}-en-{theme}"] = regression_summary(rs, n_boot, seed)
    return out


def summarize_stage3(records: list[dict]) -> dict:
    out = {}
    for qtype in ("noul", "choice"):
        rs = _select(records, stage=3, qtype=qtype)
        if not rs:
            continue
        x, q = _arr(rs, "p_true"), _arr(rs, "q")
        k, c = step_fit(x, q)
        jumps = []
        for theme in THEMES:
            by_x = {r["params"]["stated"]: r["q"] for r in rs if r["theme"] == theme}
            if 45 in by_x and 55 in by_x:
                jumps.append(by_x[55] - by_x[45])
        out[qtype] = {
            "n": len(rs),
            "mae": float(np.mean(np.abs(q - x))),
            "k": k,
            "c": c,
            "jump_45_55": float(np.mean(jumps)) if jumps else None,
            **brier_parts(q, x),
        }
    return out


def _pair_key(r: dict) -> tuple:
    return (r["theme"], r["params"]["prior"], r["params"]["lr"])


def _pairs(records: list[dict], a: tuple[str, str], b: tuple[str, str]) -> tuple[np.ndarray, np.ndarray]:
    left = {_pair_key(r): r["q"] for r in _select(records, stage=2, qtype=a[0], lang=a[1])}
    right = {_pair_key(r): r["q"] for r in _select(records, stage=2, qtype=b[0], lang=b[1])}
    keys = sorted(set(left) & set(right))
    return np.array([left[k] for k in keys]), np.array([right[k] for k in keys])


def summarize_pairs(records: list[dict], n_boot: int, seed: int) -> dict:
    out = {}
    q_choice, q_noul = _pairs(records, ("choice", "en"), ("noul", "en"))
    if len(q_choice):
        diff = np.abs(logit(clip(q_choice))) - np.abs(logit(clip(q_noul)))
        lo, hi = ci95(bootstrap(lambda idx: diff[idx].mean(), len(diff), n_boot, seed))
        out["extremity"] = {"n": len(diff), "mean": float(diff.mean()), "ci_low": float(lo), "ci_high": float(hi)}
    q_ja, q_en = _pairs(records, ("noul", "ja"), ("noul", "en"))
    if len(q_ja):
        out["japanese"] = {"n": len(q_ja), "mean_abs_diff": float(np.mean(np.abs(q_ja - q_en)))}
    return out


def judge(stage2: dict, stage3: dict, pairs: dict) -> dict:
    noul = stage2.get("noul-en")
    extremity = pairs.get("extremity")
    jump = stage3.get("choice", {}).get("jump_45_55")
    return {
        "H1": {
            "holds": bool(noul is not None and noul["ci_high"][1] < 1),
            "b1": None if noul is None else noul["b1"],
            "b1_ci": None if noul is None else [noul["ci_low"][1], noul["ci_high"][1]],
        },
        "H2": {
            "holds": bool(extremity is not None and extremity["ci_low"] > 0),
            "mean": None if extremity is None else extremity["mean"],
            "ci": None if extremity is None else [extremity["ci_low"], extremity["ci_high"]],
        },
        "H3": {"holds": bool(jump is not None and jump > 0.5), "jump_45_55": jump},
    }


def summarize(records: list[dict], n_problems: int, n_boot: int = N_BOOT, seed: int = SEED) -> dict:
    stage2 = summarize_stage2(records, n_boot, seed)
    stage3 = summarize_stage3(records)
    pairs = summarize_pairs(records, n_boot, seed)
    return {
        "counts": {
            "problems": n_problems,
            "records": len(records),
            "excluded": n_problems - len(records),
            "clipped": n_clipped([r["q"] for r in records]),
        },
        "stage1": summarize_stage1(records),
        "stage2": stage2,
        "stage3": stage3,
        "pairs": pairs,
        "hypotheses": judge(stage2, stage3, pairs),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="応答を集計して summary.json を書く")
    parser.add_argument("--problems", type=Path, default=Path("data/problems.jsonl"))
    parser.add_argument("--responses", type=Path, default=Path("data/responses.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("results/summary.json"))
    parser.add_argument("--n-boot", type=int, default=N_BOOT)
    args = parser.parse_args(argv)

    problems = read_jsonl(args.problems)
    records = load_records(problems, read_jsonl(args.responses))
    summary = summarize(records, n_problems=len(problems), n_boot=args.n_boot)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary["hypotheses"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
