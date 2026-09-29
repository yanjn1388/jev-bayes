"""Jev に渡す問題を生成する（純関数）。"""

from __future__ import annotations

import json
import random
from pathlib import Path

SEED = 20260929
# 尤度比から感度・偽陽性率を作るときの基準値（SPEC §3 ②）
BASE_RATE = 0.9


def posterior(prior: float, sens: float, fpr: float) -> float:
    """陽性だったときの事後確率（ベイズの定理）。"""
    numerator = prior * sens
    return numerator / (numerator + (1 - prior) * fpr)


def sens_fpr_from_lr(lr: float) -> tuple[float, float]:
    """尤度比から (感度, 偽陽性率) を作る。LR>=1 は感度 0.9 固定、LR<1 は偽陽性率 0.9 固定。"""
    if lr <= 0:
        raise ValueError(f"尤度比は正の数でなければならない: {lr}")
    if lr >= 1:
        return BASE_RATE, BASE_RATE / lr
    return BASE_RATE * lr, BASE_RATE


def fmt_pct(p: float) -> str:
    """確率をパーセント表記の文字列にする（末尾の 0 と小数点を落とす）。"""
    return f"{p * 100:.2f}".rstrip("0").rstrip(".")


QID = "q"
PRIORS = (0.01, 0.05, 0.2, 0.5, 0.8)
LRS = (0.2, 0.5, 2.0, 5.0, 20.0)
STATED = (5, 15, 30, 45, 50, 55, 70, 85, 95)
STAGE1_DRAWS = 30

UNCERTAINTY_EN = " Do not treat the most likely outcome as certain; keep the uncertainty in the probability."
UNCERTAINTY_JA = "最も起こりやすい結果を確実とみなさず、不確実さを確率に残すこと。"
REPORT_EN = " Report the probability that the answer is yes."
REPORT_JA = "答えが「はい」である確率を答えよ。"

# ② と ③ の題材。state は ② 用、stated は ③ 用のテンプレート
THEMES: dict[str, dict[str, str]] = {
    "medical": {
        "state": (
            "In a population, {prior}% of people have condition X. A screening test comes back "
            "positive for {sens}% of people who have the condition, and for {fpr}% of people who "
            "do not. A randomly selected person from this population tested positive."
        ),
        "stated": (
            "A forecasting report states that the probability that a randomly selected person "
            "has condition X is {x}%."
        ),
        "question": "Does this person have condition X?",
        "yes": "The person has condition X",
        "no": "The person does not have condition X",
    },
    "spam": {
        "state": (
            "Of all emails arriving at an inbox, {prior}% are spam. A filter flags {sens}% of spam "
            "emails as spam, and also flags {fpr}% of legitimate emails as spam. An email arriving "
            "at this inbox was flagged as spam."
        ),
        "stated": (
            "A forecasting report states that the probability that a newly arrived email is spam "
            "is {x}%."
        ),
        "question": "Is this email spam?",
        "yes": "The email is spam",
        "no": "The email is legitimate",
    },
    "factory": {
        "state": (
            "Of all parts produced by a factory, {prior}% are defective. An inspection machine "
            "flags {sens}% of defective parts, and also flags {fpr}% of good parts. A part from "
            "this factory was flagged by the machine."
        ),
        "stated": (
            "A forecasting report states that the probability that a part from this factory is "
            "defective is {x}%."
        ),
        "question": "Is this part defective?",
        "yes": "The part is defective",
        "no": "The part is good",
    },
    "taxi": {
        "state": (
            "In a city, {prior}% of taxis belong to the Blue company and the rest belong to the "
            "Green company. A taxi was involved in a hit-and-run at night, and a witness says the "
            "taxi was Blue. Under the same conditions, the witness says \"Blue\" for {sens}% of "
            "Blue taxis and for {fpr}% of Green taxis."
        ),
        "stated": (
            "A forecasting report states that the probability that the taxi in a hit-and-run was "
            "Blue is {x}%."
        ),
        "question": "Was the taxi Blue?",
        "yes": "The taxi was Blue",
        "no": "The taxi was Green",
    },
}

THEME_MEDICAL_JA = {
    "state": (
        "ある集団では、{prior}% の人が病気 X にかかっている。スクリーニング検査は、病気 X の人の "
        "{sens}% で陽性になり、病気 X でない人の {fpr}% でも陽性になる。"
        "この集団から無作為に選んだ1人が陽性だった。"
    ),
    "question": "この人は病気 X にかかっているか。",
}


def _problem(
    pid: str,
    stage: int,
    theme: str,
    qtype: str,
    lang: str,
    params: dict,
    p_true: float,
    state: str,
    question: dict,
    *,
    p_inv: float | None = None,
    hidden_outcome: str | int | None = None,
    target: str | None = None,
) -> dict:
    return {
        "id": pid,
        "stage": stage,
        "theme": theme,
        "qtype": qtype,
        "lang": lang,
        "params": params,
        "p_true": p_true,
        "p_inv": p_inv,
        "hidden_outcome": hidden_outcome,
        "target": target,
        "request": {"state": state, "questions": {QID: question}},
    }


def _noul(instructions: str) -> dict:
    return {"type": "noul", "instructions": instructions}


def _choice(instructions: str, criteria: dict[str, str]) -> dict:
    return {"type": "choice", "instructions": instructions, "criteria": criteria}


def _binary_question(theme: dict[str, str], qtype: str) -> dict:
    """② ③ 共通の Yes/No の問い。Noul は確率の報告を明示し、Choice は yes/no の2択にする。"""
    if qtype == "noul":
        return _noul(theme["question"] + REPORT_EN + UNCERTAINTY_EN)
    return _choice(theme["question"] + UNCERTAINTY_EN, {"yes": theme["yes"], "no": theme["no"]})


def stage1_problems(seed: int = SEED) -> list[dict]:
    """① 手がかりなし。試行番号で入力を区別し、隠れた結果をシード固定で生成する。"""
    rng = random.Random(seed)
    faces = [str(i) for i in range(1, 7)]
    out: list[dict] = []
    for n in range(1, STAGE1_DRAWS + 1):
        state = (
            f"Draw #{n}: A fair six-sided die was rolled once. Each face is equally likely. "
            "The result is hidden and has not been shown to anyone."
        )
        question = _choice(
            "Which face did the die show?" + UNCERTAINTY_EN,
            {f: f"The die shows {f}" for f in faces},
        )
        out.append(_problem(
            f"s1-die6-choice-d{n:02d}", 1, "die6", "choice", "en", {"draw": n}, 1 / 6,
            state, question, hidden_outcome=rng.choice(faces),
        ))
    for n in range(1, STAGE1_DRAWS + 1):
        state = (
            f"Draw #{n}: A fair six-sided die was rolled once. Each face is equally likely. "
            "The result is hidden and has not been shown to anyone."
        )
        question = _noul("Did the die show 1?" + REPORT_EN + UNCERTAINTY_EN)
        out.append(_problem(
            f"s1-die6-noul-d{n:02d}", 1, "die6", "noul", "en", {"draw": n}, 1 / 6,
            state, question, hidden_outcome=int(rng.randint(1, 6) == 1),
        ))
    for sections, theme in ((4, "spin4"), (20, "spin20")):
        for n in range(1, STAGE1_DRAWS + 1):
            state = (
                f"Draw #{n}: A fair spinner divided into {sections} equal sections, numbered 1 "
                f"to {sections}, was spun once. Each section is equally likely. The result is "
                "hidden and has not been shown to anyone."
            )
            question = _noul("Did the spinner stop on section 1?" + REPORT_EN + UNCERTAINTY_EN)
            out.append(_problem(
                f"s1-{theme}-noul-d{n:02d}", 1, theme, "noul", "en", {"draw": n}, 1 / sections,
                state, question, hidden_outcome=int(rng.randint(1, sections) == 1),
            ))
    return out


def stage2_problems() -> list[dict]:
    """② 計算すれば出る。4題材 × 事前確率 5 × 尤度比 5 × {noul, choice}。"""
    out: list[dict] = []
    for theme_id, theme in THEMES.items():
        for prior in PRIORS:
            for lr in LRS:
                sens, fpr = sens_fpr_from_lr(lr)
                state = theme["state"].format(prior=fmt_pct(prior), sens=fmt_pct(sens), fpr=fmt_pct(fpr))
                params = {"prior": prior, "lr": lr, "sens": sens, "fpr": fpr}
                p_true = posterior(prior, sens, fpr)
                for qtype in ("noul", "choice"):
                    out.append(_problem(
                        f"s2-{theme_id}-pi{prior:g}-lr{lr:g}-{qtype}-en", 2, theme_id, qtype, "en",
                        params, p_true, state, _binary_question(theme, qtype),
                        p_inv=sens, target="yes" if qtype == "choice" else None,
                    ))
    return out


def stage3_problems() -> list[dict]:
    """③ 確率が書いてある。4題材 × 記載値 9 点 × {noul, choice}。"""
    out: list[dict] = []
    for theme_id, theme in THEMES.items():
        for x in STATED:
            state = theme["stated"].format(x=x)
            for qtype in ("noul", "choice"):
                out.append(_problem(
                    f"s3-{theme_id}-x{x}-{qtype}-en", 3, theme_id, qtype, "en", {"stated": x},
                    x / 100, state, _binary_question(theme, qtype),
                    target="yes" if qtype == "choice" else None,
                ))
    return out


def japanese_problems() -> list[dict]:
    """日本語条件。② の medical を日本語にし、noul だけで問う。"""
    out: list[dict] = []
    for prior in PRIORS:
        for lr in LRS:
            sens, fpr = sens_fpr_from_lr(lr)
            state = THEME_MEDICAL_JA["state"].format(prior=fmt_pct(prior), sens=fmt_pct(sens), fpr=fmt_pct(fpr))
            params = {"prior": prior, "lr": lr, "sens": sens, "fpr": fpr}
            question = _noul(THEME_MEDICAL_JA["question"] + REPORT_JA + UNCERTAINTY_JA)
            out.append(_problem(
                f"s2-medical-pi{prior:g}-lr{lr:g}-noul-ja", 2, "medical", "noul", "ja",
                params, posterior(prior, sens, fpr), state, question, p_inv=sens,
            ))
    return out


def generate_all(seed: int = SEED) -> list[dict]:
    return stage1_problems(seed) + stage2_problems() + stage3_problems() + japanese_problems()


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    path.write_text(text, encoding="utf-8")


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def main() -> None:
    rows = generate_all()
    out = Path("data/problems.jsonl")
    write_jsonl(out, rows)
    print(f"{len(rows)} 問を {out} に書いた")


if __name__ == "__main__":
    main()
