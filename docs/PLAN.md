# jev-bayes 実装計画

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Jev（TypeSafe AI）に「手がかりなし／計算すれば出る／書いてある」の3段の確率問題を渡し、報告確率の較正と基準率の無視を測るための、問題生成・呼び出し・集計・作図の一式を作る。

**Architecture:** 問題の生成（`problems.py`）→ Jev の呼び出し（`client.py`）→ 追記専用の JSONL へ記録し途中から再開できる実行 CLI（`run.py`）→ 集計と仮説判定（`analyze.py`）→ 図（`figures.py`）。計算を担うモジュールは純関数にし、テストでは実際の API を呼ばない。

**Tech Stack:** Python 3.12・uv・httpx・numpy・matplotlib・pytest

**Spec:** `docs/SPEC.md`

## Global Constraints

- 依存は `httpx` / `numpy` / `matplotlib`（実行時）と `pytest`（開発時）だけ。scipy・pandas・python-dotenv は入れない
- エンドポイントは `https://openrouter.ai/api/v1/systemone`。モデルの既定値は `typesafe/jev-1.13` で、環境変数 `JEV_MODEL` で上書きできる
- 質問 ID は常に `"q"`（1リクエスト1問）
- 費用の上限は既定で 1 ドル（`usage.cost` の累計）
- ロジットの切り詰めは $[0.001,\,0.999]$。ブートストラップは 2,000 回・シード `20260929`
- 乱数のシードは `20260929` に固定する
- テストはネットワークに出ない（`httpx.MockTransport` と偽の client を使う）
- API キーは `.env` の `OPENROUTER_API_KEY`。値をログ・例外メッセージ・commit に出さない
- コード中のコメントと docstring は日本語。プロンプトの文面は英語（日本語条件のみ日本語）

## Review Focus

1. **Jev がちょうど 0 や 1 を返す** — ロジットが無限大になり、回帰が壊れてはいけない。切り詰めた上で件数を報告する（Task 6 `test_clip_prevents_infinite`）
2. **Choice の確率の対応表に、指定した選択肢名が無い**（大文字小文字の違いなど）— 黙って別のキーを読まず `invalid` にする（Task 3 `test_extract_rejects_missing_target_key`）
3. **中断後の再実行** — 同じ (問題, 試行番号) を二重に呼ばない。`error` だった行だけ呼び直す（Task 4 `test_run_resumes_skipping_ok` / `test_run_retries_error_rows_on_rerun`）
4. **応答に `usage.cost` が無い** — 0 として数え、実行を止めない（Task 4 `test_run_missing_cost_counts_zero`）
5. **ある問題の応答がすべて `error`** — 集計から外し、外した件数を数える（Task 6 `test_load_records_excludes_problems_without_ok`）

---

### Task 1: プロジェクトの骨組みとベイズの計算

**Files:**
- Create: `pyproject.toml`
- Create: `src/jev_bayes/__init__.py`
- Create: `src/jev_bayes/problems.py`
- Create: `tests/test_problems.py`
- Modify: `docs/SPEC.md`（§2 の Choice の形式を公式ドキュメントの内容に更新）

**Interfaces:**
- Produces: `posterior(prior: float, sens: float, fpr: float) -> float` / `sens_fpr_from_lr(lr: float) -> tuple[float, float]` / `fmt_pct(p: float) -> str` / 定数 `SEED = 20260929`

- [ ] **Step 1: `pyproject.toml` を作る**

```toml
[project]
name = "jev-bayes"
version = "0.1.0"
description = "Jev の報告確率の較正と基準率の無視を測る実験"
requires-python = ">=3.12"
dependencies = ["httpx>=0.27", "numpy>=2.0", "matplotlib>=3.9"]

[dependency-groups]
dev = ["pytest>=8"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/jev_bayes"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

`src/jev_bayes/__init__.py` は空ファイルで作る。

- [ ] **Step 2: 失敗するテストを書く**

`tests/test_problems.py`:

```python
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
```

- [ ] **Step 3: テストが失敗することを確かめる**

Run: `uv run pytest tests/test_problems.py -v`
Expected: FAIL（`ModuleNotFoundError` または `ImportError: cannot import name 'posterior'`）

- [ ] **Step 4: 最小の実装を書く**

`src/jev_bayes/problems.py`:

```python
"""Jev に渡す問題を生成する（純関数）。"""

from __future__ import annotations

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
```

- [ ] **Step 5: テストが通ることを確かめる**

Run: `uv run pytest tests/test_problems.py -v`
Expected: 6 passed

- [ ] **Step 6: SPEC §2 を公式ドキュメントの形式に更新する**

`docs/SPEC.md` §2 の「リクエスト例」から warning callout の終わりまでを、次の内容に置き換える。

````markdown
リクエストと応答の形（TypeSafe 公式ドキュメント `docs.typesafe.ai/primitives/choice`・`/noul` で確認）:

```json
{"model": "typesafe/jev-1.13",
 "state": "...",
 "questions": {
   "q": {"type": "choice", "instructions": "...",
         "criteria": {"yes": "説明", "no": "説明"}}}}
```

- Choice の応答: `answers.q.choice`（最も確率の高い選択肢名）・`answers.q.probabilities`（選択肢名 → 確率。合計 1）・`answers.q.confidence`
- Noul の応答: `answers.q.noul`（Yes の確率）。Noul に confidence は無い
- 質問 ID（`q`）はモデルに送られない。選択肢名と説明はモデルに送られる

> [!warning] 未確認
> OpenRouter 経由でのモデル名の書き方（`typesafe/jev-1.13` か `jev-1.13` か）は §8 の疎通確認で確定させる。
````

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml uv.lock src/jev_bayes/__init__.py src/jev_bayes/problems.py tests/test_problems.py docs/SPEC.md
git commit -m "feat: ベイズの計算と尤度比から感度・偽陽性率を作る関数を追加"
```

---

### Task 2: 問題の生成

**Files:**
- Modify: `src/jev_bayes/problems.py`
- Modify: `tests/test_problems.py`
- Create: `data/problems.jsonl`（生成物）

**Interfaces:**
- Consumes: Task 1 の `posterior` / `sens_fpr_from_lr` / `fmt_pct` / `SEED`
- Produces:
  - 定数 `QID = "q"` / `PRIORS` / `LRS` / `STATED` / `THEMES: dict[str, dict[str, str]]`
  - `generate_all(seed: int = SEED) -> list[dict]`
  - `write_jsonl(path: Path, rows: list[dict]) -> None` / `read_jsonl(path: Path) -> list[dict]`（ファイルが無ければ空リスト）
  - 問題の dict のキー: `id` `stage`(int) `theme` `qtype`(`"noul"`/`"choice"`) `lang`(`"en"`/`"ja"`) `params`(dict) `p_true`(float) `p_inv`(float|None) `hidden_outcome`(str|int|None) `target`(str|None) `request`(`{"state": str, "questions": {"q": {...}}}`)
  - `target` の意味: Choice で報告確率として読む選択肢名。`None` のときは Jev が選んだ選択肢の確率を読む（①の Choice）
  - `hidden_outcome` の意味: ①の Choice は隠れた出目（`"1"`〜`"6"`）、①の Noul は事象が起きたか（`0`/`1`）、それ以外は `None`

- [ ] **Step 1: 失敗するテストを追記する**

`tests/test_problems.py` の末尾に追加:

```python
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
```

- [ ] **Step 2: テストが失敗することを確かめる**

Run: `uv run pytest tests/test_problems.py -v`
Expected: FAIL（`ImportError: cannot import name 'QID'`）

- [ ] **Step 3: 実装を追記する**

`src/jev_bayes/problems.py` の先頭の import を次に置き換える:

```python
from __future__ import annotations

import json
import random
from pathlib import Path
```

ファイル末尾に追記する:

```python
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
```

- [ ] **Step 4: テストが通ることを確かめる**

Run: `uv run pytest tests/test_problems.py -v`
Expected: 17 passed

- [ ] **Step 5: 問題ファイルを生成して目で確かめる**

Run: `uv run python -m jev_bayes.problems && wc -l data/problems.jsonl && head -c 600 data/problems.jsonl`
Expected: `417 問を data/problems.jsonl に書いた`・行数 417・先頭行が `s1-die6-choice-d01` の JSON

- [ ] **Step 6: Commit**

```bash
git add src/jev_bayes/problems.py tests/test_problems.py data/problems.jsonl
git commit -m "feat: 3段のはしごと日本語条件の問題を生成する"
```

---

### Task 3: Jev クライアント

**Files:**
- Create: `src/jev_bayes/client.py`
- Create: `tests/test_client.py`

**Interfaces:**
- Consumes: Task 2 の `QID` と問題の dict（`qtype` `target` `request`）
- Produces:
  - `ENDPOINT: str` / `DEFAULT_MODEL: str`
  - `class InvalidResponse(Exception)` / `class FatalRequestError(Exception)`
  - `@dataclass(frozen=True) class CallResult: status: str; http_status: int | None; raw: dict | None; error: str | None`（`status` は `"ok"` / `"error"` / `"invalid"`）
  - `build_body(problem: dict, model: str) -> dict`
  - `extract_q(problem: dict, raw: dict) -> tuple[float, str | None]`（報告確率と、Choice のとき Jev が選んだ選択肢名）
  - `load_env(path: Path) -> dict[str, str]`
  - `class JevClient(api_key: str, model: str = DEFAULT_MODEL, http: httpx.Client | None = None, sleep = time.sleep, max_retries: int = 3, base_delay: float = 1.0)` とメソッド `call(problem: dict) -> CallResult`

- [ ] **Step 1: 失敗するテストを書く**

`tests/test_client.py`:

```python
import httpx
import pytest

from jev_bayes.client import (
    CallResult,
    FatalRequestError,
    InvalidResponse,
    JevClient,
    build_body,
    extract_q,
    load_env,
)

NOUL_P = {
    "id": "n1", "qtype": "noul", "target": None,
    "request": {"state": "s", "questions": {"q": {"type": "noul", "instructions": "i"}}},
}
CHOICE_P = {
    "id": "c1", "qtype": "choice", "target": "yes",
    "request": {"state": "s", "questions": {"q": {"type": "choice", "instructions": "i",
                                                  "criteria": {"yes": "y", "no": "n"}}}},
}
DIE_P = {
    "id": "d1", "qtype": "choice", "target": None,
    "request": {"state": "s", "questions": {"q": {"type": "choice", "instructions": "i",
                                                  "criteria": {str(i): str(i) for i in range(1, 7)}}}},
}
OK_BODY = {"model": "jev-1.13.0", "answers": {"q": {"type": "noul", "noul": 0.3}},
           "usage": {"input_tokens": 300, "cost": 0.00001}}


def _client(handler, **kwargs):
    http = httpx.Client(transport=httpx.MockTransport(handler))
    return JevClient("test-key", http=http, sleep=lambda s: None, **kwargs)


def test_build_body_adds_model_and_keeps_request():
    body = build_body(NOUL_P, "m")
    assert body["model"] == "m"
    assert body["state"] == "s"
    assert body["questions"] == NOUL_P["request"]["questions"]


def test_extract_noul():
    assert extract_q(NOUL_P, {"answers": {"q": {"type": "noul", "noul": 0.3}}}) == (0.3, None)


def test_extract_choice_reads_target():
    raw = {"answers": {"q": {"type": "choice", "choice": "no", "probabilities": {"yes": 0.2, "no": 0.8}}}}
    assert extract_q(CHOICE_P, raw) == (0.2, "no")


def test_extract_choice_reads_selected_when_no_target():
    probs = {str(i): 0.02 for i in range(2, 7)} | {"1": 0.9}
    raw = {"answers": {"q": {"type": "choice", "choice": "1", "probabilities": probs}}}
    q, chosen = extract_q(DIE_P, raw)
    assert chosen == "1"
    assert q == pytest.approx(0.9)


def test_extract_rejects_bad_sum():
    raw = {"answers": {"q": {"type": "choice", "choice": "no", "probabilities": {"yes": 0.5, "no": 0.6}}}}
    with pytest.raises(InvalidResponse):
        extract_q(CHOICE_P, raw)


def test_extract_rejects_out_of_range_noul():
    with pytest.raises(InvalidResponse):
        extract_q(NOUL_P, {"answers": {"q": {"type": "noul", "noul": 1.2}}})


def test_extract_rejects_missing_target_key():
    raw = {"answers": {"q": {"type": "choice", "choice": "No", "probabilities": {"Yes": 0.2, "No": 0.8}}}}
    with pytest.raises(InvalidResponse):
        extract_q(CHOICE_P, raw)


def test_extract_rejects_missing_answer():
    with pytest.raises(InvalidResponse):
        extract_q(NOUL_P, {"answers": {}})


def test_call_ok_sends_auth_and_body():
    seen = {}

    def handler(request):
        seen["auth"] = request.headers["Authorization"]
        seen["body"] = request.read()
        return httpx.Response(200, json=OK_BODY)

    result = _client(handler, model="m").call(NOUL_P)
    assert result == CallResult("ok", 200, OK_BODY, None)
    assert seen["auth"] == "Bearer test-key"
    assert b'"model":"m"' in seen["body"].replace(b" ", b"")


def test_call_retries_429_then_ok():
    calls = []

    def handler(request):
        calls.append(1)
        return httpx.Response(429) if len(calls) < 3 else httpx.Response(200, json=OK_BODY)

    result = _client(handler).call(NOUL_P)
    assert result.status == "ok"
    assert len(calls) == 3


def test_call_gives_up_after_retries():
    calls = []

    def handler(request):
        calls.append(1)
        return httpx.Response(503)

    result = _client(handler, max_retries=3).call(NOUL_P)
    assert result.status == "error"
    assert result.http_status == 503
    assert len(calls) == 4


def test_call_retries_transport_error():
    calls = []

    def handler(request):
        calls.append(1)
        if len(calls) == 1:
            raise httpx.ConnectError("boom", request=request)
        return httpx.Response(200, json=OK_BODY)

    assert _client(handler).call(NOUL_P).status == "ok"


def test_call_raises_on_client_error_without_leaking_key():
    def handler(request):
        return httpx.Response(400, json={"error": "bad model"})

    with pytest.raises(FatalRequestError) as exc:
        _client(handler).call(NOUL_P)
    assert "400" in str(exc.value)
    assert "test-key" not in str(exc.value)


def test_call_non_json_body_is_invalid():
    def handler(request):
        return httpx.Response(200, content=b"<html>")

    result = _client(handler).call(NOUL_P)
    assert result.status == "invalid"
    assert result.raw is None


def test_load_env(tmp_path):
    path = tmp_path / ".env"
    path.write_text("# comment\n\nOPENROUTER_API_KEY=abc123\nJEV_MODEL=m\n", encoding="utf-8")
    assert load_env(path) == {"OPENROUTER_API_KEY": "abc123", "JEV_MODEL": "m"}
```

- [ ] **Step 2: テストが失敗することを確かめる**

Run: `uv run pytest tests/test_client.py -v`
Expected: FAIL（`ModuleNotFoundError: No module named 'jev_bayes.client'`）

- [ ] **Step 3: 実装を書く**

`src/jev_bayes/client.py`:

```python
"""OpenRouter 経由で Jev を呼び、応答から報告確率を取り出す。"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import httpx

from jev_bayes.problems import QID

ENDPOINT = "https://openrouter.ai/api/v1/systemone"
DEFAULT_MODEL = "typesafe/jev-1.13"
PROB_SUM_TOLERANCE = 0.01
TIMEOUT_SECONDS = 30.0


class InvalidResponse(Exception):
    """応答の形が期待と違う（確率が範囲外・合計が 1 でない・キーが無い）。"""


class FatalRequestError(Exception):
    """再試行しても直らないリクエストの誤り（429 以外の 4xx）。"""


@dataclass(frozen=True)
class CallResult:
    status: str  # "ok" / "error" / "invalid"
    http_status: int | None
    raw: dict | None
    error: str | None


def build_body(problem: dict, model: str) -> dict:
    return {"model": model, **problem["request"]}


def _check_prob(value: object, label: str) -> float:
    if not isinstance(value, (int, float)) or not 0 <= value <= 1:
        raise InvalidResponse(f"{label} が [0, 1] の数値ではない: {value!r}")
    return float(value)


def extract_q(problem: dict, raw: dict) -> tuple[float, str | None]:
    """報告確率と、Choice のとき Jev が選んだ選択肢名を返す。"""
    try:
        answer = raw["answers"][QID]
    except (KeyError, TypeError) as e:
        raise InvalidResponse(f"answers.{QID} が無い") from e
    if problem["qtype"] == "noul":
        return _check_prob(answer.get("noul"), "noul"), None
    probs = answer.get("probabilities")
    chosen = answer.get("choice")
    if not isinstance(probs, dict) or not probs:
        raise InvalidResponse("probabilities が無い")
    for key, value in probs.items():
        _check_prob(value, f"probabilities[{key}]")
    total = sum(probs.values())
    if abs(total - 1) > PROB_SUM_TOLERANCE:
        raise InvalidResponse(f"probabilities の合計が 1 でない: {total}")
    key = problem["target"] or chosen
    if key not in probs:
        raise InvalidResponse(f"probabilities に {key!r} が無い: {sorted(probs)}")
    return float(probs[key]), chosen


def load_env(path: Path) -> dict[str, str]:
    env: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        env[key.strip()] = value.strip()
    return env


class JevClient:
    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_MODEL,
        http: httpx.Client | None = None,
        sleep: Callable[[float], None] = time.sleep,
        max_retries: int = 3,
        base_delay: float = 1.0,
    ) -> None:
        self._api_key = api_key
        self.model = model
        self._http = http or httpx.Client(timeout=TIMEOUT_SECONDS)
        self._sleep = sleep
        self._max_retries = max_retries
        self._base_delay = base_delay

    def call(self, problem: dict) -> CallResult:
        """1問を呼ぶ。429・5xx・通信エラーは倍々の待ちで再試行し、その他の 4xx は例外で止める。"""
        body = build_body(problem, self.model)
        headers = {"Authorization": f"Bearer {self._api_key}"}
        last_status: int | None = None
        last_error = ""
        for attempt in range(self._max_retries + 1):
            if attempt > 0:
                self._sleep(self._base_delay * 2 ** (attempt - 1))
            try:
                resp = self._http.post(ENDPOINT, json=body, headers=headers)
            except httpx.TransportError as e:
                last_status, last_error = None, f"通信エラー: {type(e).__name__}"
                continue
            last_status = resp.status_code
            if resp.status_code == 200:
                try:
                    return CallResult("ok", 200, resp.json(), None)
                except ValueError:
                    return CallResult("invalid", 200, None, "応答が JSON でない")
            if resp.status_code == 429 or resp.status_code >= 500:
                last_error = f"HTTP {resp.status_code}"
                continue
            raise FatalRequestError(f"HTTP {resp.status_code}: {resp.text[:500]}")
        return CallResult("error", last_status, None, f"再試行を使い切った: {last_error}")
```

- [ ] **Step 4: テストが通ることを確かめる**

Run: `uv run pytest tests/test_client.py -v`
Expected: 15 passed

- [ ] **Step 5: Commit**

```bash
git add src/jev_bayes/client.py tests/test_client.py
git commit -m "feat: Jev クライアントと応答の検査を追加"
```

---

### Task 4: 実行 CLI（再開・費用上限）

**Files:**
- Create: `src/jev_bayes/run.py`
- Create: `tests/test_run.py`

**Interfaces:**
- Consumes: Task 2 の `read_jsonl`、Task 3 の `CallResult` / `extract_q` / `InvalidResponse` / `FatalRequestError` / `JevClient` / `load_env` / `DEFAULT_MODEL`
- Produces:
  - `response_row(problem: dict, attempt: int, result: CallResult, called_at: str) -> dict`（SPEC §6 の responses.jsonl のキーに `chosen` を加えたもの）
  - `filter_problems(problems: list[dict], only: str | None, limit: int | None) -> list[dict]`
  - `run(problems: list[dict], client, out_path: Path, attempts: int = 1, budget: float = 1.0, now: Callable[[], str] = ...) -> dict`（戻り値のキー: `called` `skipped` `spent` `stopped`）
  - CLI: `uv run python -m jev_bayes.run [--problems P] [--out O] [--attempts N] [--budget USD] [--only PREFIX] [--limit N]`

- [ ] **Step 1: 失敗するテストを書く**

`tests/test_run.py`:

```python
import json

import pytest

from jev_bayes.client import CallResult, FatalRequestError
from jev_bayes.problems import read_jsonl
from jev_bayes.run import filter_problems, run

P1 = {"id": "a-1", "qtype": "noul", "target": None,
      "request": {"state": "s", "questions": {"q": {"type": "noul", "instructions": "i"}}}}
P2 = {**P1, "id": "a-2"}
P3 = {**P1, "id": "b-1"}


def _ok(noul=0.4, cost=0.001):
    raw = {"model": "jev-1.13.0", "answers": {"q": {"type": "noul", "noul": noul}}}
    if cost is not None:
        raw["usage"] = {"cost": cost}
    return CallResult("ok", 200, raw, None)


class FakeClient:
    def __init__(self, results):
        self.results = list(results)
        self.calls = []

    def call(self, problem):
        self.calls.append(problem["id"])
        return self.results.pop(0)


def _now():
    return "2026-09-30T00:00:00+09:00"


def test_run_writes_rows(tmp_path):
    out = tmp_path / "r.jsonl"
    summary = run([P1, P2], FakeClient([_ok(0.4), _ok(0.7)]), out, now=_now)
    rows = read_jsonl(out)
    assert summary["called"] == 2
    assert [r["problem_id"] for r in rows] == ["a-1", "a-2"]
    assert rows[0]["q"] == 0.4
    assert rows[0]["status"] == "ok"
    assert rows[0]["model"] == "jev-1.13.0"
    assert rows[0]["cost"] == 0.001
    assert rows[0]["called_at"] == _now()


def test_run_resumes_skipping_ok(tmp_path):
    out = tmp_path / "r.jsonl"
    run([P1, P2], FakeClient([_ok(), _ok()]), out, now=_now)
    client = FakeClient([])
    summary = run([P1, P2], client, out, now=_now)
    assert client.calls == []
    assert summary["skipped"] == 2
    assert len(read_jsonl(out)) == 2


def test_run_retries_error_rows_on_rerun(tmp_path):
    out = tmp_path / "r.jsonl"
    run([P1], FakeClient([CallResult("error", 503, None, "x")]), out, now=_now)
    client = FakeClient([_ok()])
    run([P1], client, out, now=_now)
    assert client.calls == ["a-1"]
    assert [r["status"] for r in read_jsonl(out)] == ["error", "ok"]


def test_run_attempts(tmp_path):
    out = tmp_path / "r.jsonl"
    run([P1], FakeClient([_ok(), _ok(), _ok()]), out, attempts=3, now=_now)
    assert [r["attempt"] for r in read_jsonl(out)] == [0, 1, 2]


def test_run_stops_at_budget(tmp_path):
    out = tmp_path / "r.jsonl"
    client = FakeClient([_ok(cost=0.6), _ok(cost=0.6), _ok(cost=0.6)])
    summary = run([P1, P2, P3], client, out, budget=1.0, now=_now)
    assert summary["called"] == 2
    assert summary["stopped"] == "budget"
    assert summary["spent"] == pytest.approx(1.2)


def test_run_budget_counts_previous_rows(tmp_path):
    out = tmp_path / "r.jsonl"
    run([P1], FakeClient([_ok(cost=1.5)]), out, now=_now)
    client = FakeClient([_ok()])
    summary = run([P1, P2], client, out, budget=1.0, now=_now)
    assert client.calls == []
    assert summary["stopped"] == "budget"


def test_run_marks_invalid(tmp_path):
    out = tmp_path / "r.jsonl"
    bad = CallResult("ok", 200, {"answers": {"q": {"type": "noul", "noul": 1.5}}}, None)
    run([P1], FakeClient([bad]), out, now=_now)
    row = read_jsonl(out)[0]
    assert row["status"] == "invalid"
    assert row["q"] is None
    assert "noul" in row["error"]


def test_run_missing_cost_counts_zero(tmp_path):
    out = tmp_path / "r.jsonl"
    summary = run([P1, P2], FakeClient([_ok(cost=None), _ok(cost=None)]), out, now=_now)
    assert summary["called"] == 2
    assert summary["spent"] == 0
    assert read_jsonl(out)[0]["cost"] is None


def test_fatal_error_propagates_and_keeps_previous_rows(tmp_path):
    out = tmp_path / "r.jsonl"

    class Boom(FakeClient):
        def call(self, problem):
            if problem["id"] == "a-2":
                raise FatalRequestError("HTTP 400")
            return super().call(problem)

    with pytest.raises(FatalRequestError):
        run([P1, P2, P3], Boom([_ok()]), out, now=_now)
    assert [r["problem_id"] for r in read_jsonl(out)] == ["a-1"]


def test_rows_are_valid_json_lines(tmp_path):
    out = tmp_path / "r.jsonl"
    run([P1], FakeClient([_ok()]), out, now=_now)
    for line in out.read_text(encoding="utf-8").splitlines():
        json.loads(line)


def test_filter_problems():
    assert [p["id"] for p in filter_problems([P1, P2, P3], "a-", None)] == ["a-1", "a-2"]
    assert [p["id"] for p in filter_problems([P1, P2, P3], None, 1)] == ["a-1"]
    assert filter_problems([P1, P2, P3], None, None) == [P1, P2, P3]
```

- [ ] **Step 2: テストが失敗することを確かめる**

Run: `uv run pytest tests/test_run.py -v`
Expected: FAIL（`ModuleNotFoundError: No module named 'jev_bayes.run'`）

- [ ] **Step 3: 実装を書く**

`src/jev_bayes/run.py`:

```python
"""問題を Jev に投げ、応答を追記専用の JSONL に記録する。中断しても続きから再開できる。"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Callable

from jev_bayes.client import DEFAULT_MODEL, CallResult, InvalidResponse, JevClient, extract_q, load_env
from jev_bayes.problems import read_jsonl


def _now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def response_row(problem: dict, attempt: int, result: CallResult, called_at: str) -> dict:
    status, error, q, chosen = result.status, result.error, None, None
    raw = result.raw
    if status == "ok":
        try:
            q, chosen = extract_q(problem, raw)
        except InvalidResponse as e:
            status, error = "invalid", str(e)
    usage = raw.get("usage", {}) if isinstance(raw, dict) else {}
    return {
        "problem_id": problem["id"],
        "attempt": attempt,
        "called_at": called_at,
        "status": status,
        "http_status": result.http_status,
        "raw": raw,
        "q": q,
        "chosen": chosen,
        "model": raw.get("model") if isinstance(raw, dict) else None,
        "cost": usage.get("cost") if isinstance(usage, dict) else None,
        "error": error,
    }


def filter_problems(problems: list[dict], only: str | None, limit: int | None) -> list[dict]:
    selected = [p for p in problems if only is None or p["id"].startswith(only)]
    return selected if limit is None else selected[:limit]


def _append(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def run(
    problems: list[dict],
    client,
    out_path: Path,
    attempts: int = 1,
    budget: float = 1.0,
    now: Callable[[], str] = _now_iso,
) -> dict:
    """`ok` 済みの (問題, 試行番号) を飛ばして呼ぶ。費用の累計が上限に達したら止める。"""
    existing = read_jsonl(out_path)
    done = {(r["problem_id"], r["attempt"]) for r in existing if r["status"] == "ok"}
    spent = sum(r.get("cost") or 0 for r in existing)
    called = skipped = 0
    for problem in problems:
        for attempt in range(attempts):
            if (problem["id"], attempt) in done:
                skipped += 1
                continue
            if spent >= budget:
                return {"called": called, "skipped": skipped, "spent": spent, "stopped": "budget"}
            row = response_row(problem, attempt, client.call(problem), now())
            _append(out_path, row)
            spent += row["cost"] or 0
            called += 1
    return {"called": called, "skipped": skipped, "spent": spent, "stopped": None}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Jev に問題を投げて応答を記録する")
    parser.add_argument("--problems", type=Path, default=Path("data/problems.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("data/responses.jsonl"))
    parser.add_argument("--attempts", type=int, default=1)
    parser.add_argument("--budget", type=float, default=1.0)
    parser.add_argument("--only", help="この接頭辞で始まる ID の問題だけを呼ぶ")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args(argv)

    env = {**load_env(Path(".env")), **os.environ} if Path(".env").exists() else dict(os.environ)
    api_key = env.get("OPENROUTER_API_KEY")
    if not api_key:
        print("OPENROUTER_API_KEY が .env にも環境変数にも無い", file=sys.stderr)
        return 2
    client = JevClient(api_key, model=env.get("JEV_MODEL", DEFAULT_MODEL))
    problems = filter_problems(read_jsonl(args.problems), args.only, args.limit)
    summary = run(problems, client, args.out, attempts=args.attempts, budget=args.budget)
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: テストが通ることを確かめる**

Run: `uv run pytest -v`
Expected: 43 passed（Task 1〜4 の合計）

- [ ] **Step 5: Commit**

```bash
git add src/jev_bayes/run.py tests/test_run.py
git commit -m "feat: 再開と費用上限つきの実行 CLI を追加"
```

---

### Task 5: 疎通確認（実際の API・手動）

実際の API を呼ぶ。費用は 0.001 ドル未満。結果は `data/smoke.jsonl` に残す（本番の `responses.jsonl` とは分ける）。

**Files:**
- Create: `data/smoke.jsonl`（生成物）
- Modify: `docs/SPEC.md`（§2 のモデル名の warning を解消）

- [ ] **Step 1: 1問だけ呼んで、リクエストの形とモデル名を確かめる**

Run: `uv run python -m jev_bayes.run --out data/smoke.jsonl --only s2-medical-pi0.01-lr20-noul-en --attempts 1`
Expected: `{"called": 1, ..., "stopped": null}`

`FatalRequestError: HTTP 400` などで止まったら、エラー本文を読み、モデル名が原因なら `JEV_MODEL=jev-1.13 uv run python -m jev_bayes.run ...` で再実行する。通ったほうのモデル名を `.env` に `JEV_MODEL=...` として書く（`DEFAULT_MODEL` で通った場合は書かない）。

- [ ] **Step 2: 同じ入力を5回呼び、答えが揺れるかを確かめる**

Run:
```bash
uv run python -m jev_bayes.run --out data/smoke.jsonl --only s2-medical-pi0.01-lr20- --attempts 5
uv run python -m jev_bayes.run --out data/smoke.jsonl --only s1-die6-choice-d01 --attempts 5
```
Expected: 1本目は noul と choice の2問 × 5回から Step 1 の済み分を除いた `called` 9。2本目は `called` 5

- [ ] **Step 3: 応答の中身を読む**

Run:
```bash
uv run python - <<'EOF'
import json
from collections import defaultdict
rows = [json.loads(l) for l in open("data/smoke.jsonl")]
by = defaultdict(list)
for r in rows:
    by[r["problem_id"]].append((r["status"], r["q"], r["chosen"]))
for pid, v in by.items():
    print(pid, v)
print("models:", {r["model"] for r in rows})
print("cost:", sum(r["cost"] or 0 for r in rows))
print(json.dumps(rows[-1]["raw"], indent=1)[:800])
EOF
```
Expected: 全行 `ok`。choice の `raw` に `choice` / `probabilities` / `confidence` がある

- [ ] **Step 4: 試行回数 $r$ を決める**

- 同じ問題の5回の `q` がすべて一致 → $r=1$
- 一致しない → 本番は約 420 問なので $r=5$（約 2,100 回・費用はなお 0.1 ドル未満の見込み）

決めた $r$ と、その根拠（5回の `q` の値）を Task 8 の事前登録に書く。

- [ ] **Step 5: SPEC §2 を確定させて commit**

`docs/SPEC.md` §2 の「未確認」の warning callout を、確定したモデル名・応答の `model` の値・$r$ の判断材料を書いた note callout に置き換える。

```bash
git add data/smoke.jsonl docs/SPEC.md
git commit -m "docs: 疎通確認の結果で API の形とモデル名を確定"
```

---

### Task 6: 集計と仮説の判定

**Files:**
- Create: `src/jev_bayes/analyze.py`
- Create: `tests/test_analyze.py`

**Interfaces:**
- Consumes: Task 2 の `SEED` / `THEMES` / `generate_all` / `posterior` / `read_jsonl`
- Produces:
  - 定数 `EPS = 1e-3` / `N_BOOT = 2000`
  - `clip(q) -> np.ndarray` / `logit(p) -> np.ndarray` / `n_clipped(q) -> int`
  - `fit_bayes_regression(prior, lr, q) -> np.ndarray`（`[a, b1, b2]`）
  - `bootstrap(stat: Callable[[np.ndarray], object], n: int, n_boot: int = N_BOOT, seed: int = SEED) -> np.ndarray`
  - `ci95(samples: np.ndarray) -> tuple[np.ndarray, np.ndarray]`
  - `brier_parts(q, p) -> dict`（`model_error` / `irreducible`）/ `sampled_brier(q, y) -> float`
  - `step_fit(x, q) -> tuple[float, float]`（傾き $k$ と中心 $c$。$x$ は 0〜1）
  - `load_records(problems: list[dict], responses: list[dict]) -> list[dict]`（問題の dict に `q`（ok の試行の平均）・`q_spread`・`n_ok`・`chosen` を加えたもの。ok が無い問題は含めない）
  - `summarize(records: list[dict], n_problems: int, n_boot: int = N_BOOT, seed: int = SEED) -> dict`（キー: `counts` `stage1` `stage2` `stage3` `pairs` `hypotheses`）
  - `summary["stage2"][key]` のキー: `n` `a` `b1` `b2` `ci_low`(list[3]) `ci_high`(list[3]) `n_clipped` `closer_to_inverse` `model_error` `irreducible`。`key` は `"noul-en"` / `"choice-en"` / `"noul-ja"`、題材別は `summary["stage2"]["by_theme"][f"{qtype}-en-{theme}"]`
  - CLI: `uv run python -m jev_bayes.analyze [--problems P] [--responses R] [--out results/summary.json] [--n-boot N]`

- [ ] **Step 1: 失敗するテストを書く**

`tests/test_analyze.py`:

```python
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
```

- [ ] **Step 2: テストが失敗することを確かめる**

Run: `uv run pytest tests/test_analyze.py -v`
Expected: FAIL（`ModuleNotFoundError: No module named 'jev_bayes.analyze'`）

- [ ] **Step 3: 実装を書く**

`src/jev_bayes/analyze.py`:

```python
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
```

- [ ] **Step 4: テストが通ることを確かめる**

Run: `uv run pytest tests/test_analyze.py -v`
Expected: 10 passed

- [ ] **Step 5: Commit**

```bash
git add src/jev_bayes/analyze.py tests/test_analyze.py
git commit -m "feat: 回帰・Brier 分解・段差の当てはめで仮説を判定する集計を追加"
```

---

### Task 7: 図（ブログ用3枚）

**Files:**
- Create: `src/jev_bayes/figures.py`
- Create: `tests/test_figures.py`

**Interfaces:**
- Consumes: Task 6 の `load_records` / `summarize` と `summary` の構造、Task 2 の `read_jsonl`
- Produces:
  - `write_figures(records: list[dict], summary: dict, out_dir: Path) -> list[Path]`（`ladder.png` / `coefficients.png` / `brier.png` の3つ）
  - CLI: `uv run python -m jev_bayes.figures [--problems P] [--responses R] [--summary S] [--out-dir results]`

- [ ] **Step 1: 失敗するテストを書く**

`tests/test_figures.py`:

```python
from jev_bayes.analyze import summarize
from jev_bayes.figures import write_figures
from jev_bayes.problems import generate_all


def test_write_figures_creates_three_pngs(tmp_path):
    records = [
        {**p, "q": p["p_true"], "q_spread": 0.0, "n_ok": 1, "chosen": p["hidden_outcome"]}
        for p in generate_all()
    ]
    summary = summarize(records, n_problems=len(records), n_boot=50)
    paths = write_figures(records, summary, tmp_path)
    assert sorted(p.name for p in paths) == ["brier.png", "coefficients.png", "ladder.png"]
    for p in paths:
        assert p.stat().st_size > 1000
```

- [ ] **Step 2: テストが失敗することを確かめる**

Run: `uv run pytest tests/test_figures.py -v`
Expected: FAIL（`ModuleNotFoundError: No module named 'jev_bayes.figures'`）

- [ ] **Step 3: 実装を書く**

`src/jev_bayes/figures.py`:

```python
"""記事用の図3枚を描く（SPEC §4.5）。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from jev_bayes.analyze import load_records  # noqa: E402
from jev_bayes.problems import read_jsonl  # noqa: E402

plt.rcParams["font.family"] = ["Hiragino Sans", "Noto Sans CJK JP", "DejaVu Sans"]

STAGE_COLORS = {1: "#7a7a7a", 2: "#1f6feb", 3: "#d97706"}
STAGE_LABELS = {1: "① 手がかりなし", 2: "② 計算すれば出る", 3: "③ 書いてある"}
CONDITION_COLORS = {"noul-en": "#1f6feb", "choice-en": "#d97706", "noul-ja": "#16a34a"}
DPI = 150


def fig_ladder(records: list[dict], path: Path) -> Path:
    """横軸 真の確率 p・縦軸 報告確率 q。Noul と Choice を左右に並べる。"""
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=True)
    for ax, qtype in zip(axes, ("noul", "choice")):
        ax.plot([0, 1], [0, 1], color="black", lw=1, ls="--", label="理想（報告 = 真の確率）")
        for stage in (1, 2, 3):
            rs = [r for r in records if r["stage"] == stage and r["qtype"] == qtype and r["lang"] == "en"]
            if rs:
                ax.scatter([r["p_true"] for r in rs], [r["q"] for r in rs], s=18, alpha=0.7,
                           color=STAGE_COLORS[stage], label=STAGE_LABELS[stage])
        ax.set_title(qtype.capitalize())
        ax.set_xlabel("真の確率 p")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    axes[0].set_ylabel("Jev の報告確率 q")
    axes[0].legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return path


def fig_coefficients(summary: dict, path: Path) -> Path:
    """(b1, b2) 平面にベイズの点 (1, 1) と推定値・95% 区間を置く。"""
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.scatter([1], [1], marker="*", s=220, color="black", label="ベイズ (1, 1)", zorder=3)
    for key, color in CONDITION_COLORS.items():
        s = summary["stage2"].get(key)
        if not s:
            continue
        b1, b2 = s["b1"], s["b2"]
        xerr = [[max(0.0, b1 - s["ci_low"][1])], [max(0.0, s["ci_high"][1] - b1)]]
        yerr = [[max(0.0, b2 - s["ci_low"][2])], [max(0.0, s["ci_high"][2] - b2)]]
        ax.errorbar(b1, b2, xerr=xerr, yerr=yerr, fmt="o", color=color, capsize=4, label=key)
    ax.axvline(1, color="gray", lw=0.8, ls=":")
    ax.axhline(1, color="gray", lw=0.8, ls=":")
    ax.set_xlabel("b1（事前確率の重み）")
    ax.set_ylabel("b2（尤度比の重み）")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return path


def _brier_rows(summary: dict) -> list[tuple[str, float, float]]:
    rows = [(f"① {k}", v["model_error"], v["irreducible"]) for k, v in summary["stage1"].items()]
    rows += [(f"② {k}", summary["stage2"][k]["model_error"], summary["stage2"][k]["irreducible"])
             for k in CONDITION_COLORS if k in summary["stage2"]]
    rows += [(f"③ {k}", v["model_error"], v["irreducible"]) for k, v in summary["stage3"].items()]
    return rows


def fig_brier(summary: dict, path: Path) -> Path:
    """条件ごとに「Jev の誤り」と「減らせない不確実性」を積み上げた横棒。"""
    rows = _brier_rows(summary)
    labels = [r[0] for r in rows]
    fig, ax = plt.subplots(figsize=(7, 0.45 * len(rows) + 1.5))
    ax.barh(labels, [r[2] for r in rows], color="#cbd5e1", label="減らせない不確実性 p(1-p)")
    ax.barh(labels, [r[1] for r in rows], left=[r[2] for r in rows], color="#dc2626",
            label="Jev の誤り (q-p)²")
    ax.invert_yaxis()
    ax.set_xlabel("期待 Brier スコア")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return path


def write_figures(records: list[dict], summary: dict, out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    return [
        fig_ladder(records, out_dir / "ladder.png"),
        fig_coefficients(summary, out_dir / "coefficients.png"),
        fig_brier(summary, out_dir / "brier.png"),
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="記事用の図3枚を描く")
    parser.add_argument("--problems", type=Path, default=Path("data/problems.jsonl"))
    parser.add_argument("--responses", type=Path, default=Path("data/responses.jsonl"))
    parser.add_argument("--summary", type=Path, default=Path("results/summary.json"))
    parser.add_argument("--out-dir", type=Path, default=Path("results"))
    args = parser.parse_args(argv)

    records = load_records(read_jsonl(args.problems), read_jsonl(args.responses))
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    for path in write_figures(records, summary, args.out_dir):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: テストが通ることを確かめる**

Run: `uv run pytest -v`
Expected: 54 passed（全テスト）

- [ ] **Step 5: 疎通確認のデータで図を描いて目で確かめる**

Run:
```bash
uv run python -m jev_bayes.analyze --responses data/smoke.jsonl --out /tmp/jev-smoke-summary.json --n-boot 50
uv run python -m jev_bayes.figures --responses data/smoke.jsonl --summary /tmp/jev-smoke-summary.json --out-dir /tmp/jev-smoke-figs
```
Expected: 3枚の png が出る。日本語のラベルが豆腐（□）になっていないことを Read ツールで画像を開いて確かめる。豆腐なら `plt.rcParams["font.family"]` の先頭を `fc-list :lang=ja family` で見つかるフォント名に替える

- [ ] **Step 6: Commit**

```bash
git add src/jev_bayes/figures.py tests/test_figures.py
git commit -m "feat: はしご・係数・Brier 内訳の図を追加"
```

---

### Task 8: 事前登録

本番を呼ぶ前に、仮説・判定基準・$r$・問題文と分析コードの版を固定する。

**Files:**
- Create: `docs/PREREGISTRATION.md`

- [ ] **Step 1: 固定する版を取る**

Run:
```bash
shasum -a 256 data/problems.jsonl
git rev-parse HEAD
uv run python -c "import json;print(json.loads(open('data/smoke.jsonl').readline())['model'])"
```

- [ ] **Step 2: `docs/PREREGISTRATION.md` を書く**

次の内容で書く。`<...>` は Step 1 と Task 5 の実際の値で埋める（未記入のまま commit しない）。

```markdown
# 事前登録（jev-bayes）

- 登録日: <YYYY-MM-DD>
- 本書は本番の呼び出しより前に commit する。以後、判定基準は変更しない

## 固定する版

| 対象 | 値 |
|---|---|
| 問題ファイル `data/problems.jsonl` の SHA-256 | <sha256> |
| 分析コード（本書を commit する直前の HEAD） | <commit hash> |
| 疎通確認で応答したモデル | <model> |
| 1問あたりの試行回数 r | <1 または 5>（根拠: 疎通確認で同じ入力を5回呼んだ q の値 <値>） |

## 仮説と判定基準

| ID | 仮説 | 判定基準 |
|---|---|---|
| H1 | ② の noul（英語）は基準率を無視する | 回帰係数 b1 の 95% ブートストラップ区間の上端が 1 未満 |
| H2 | ② の choice は noul より 0/1 に倒れる | 対にした \|logit q_choice\| − \|logit q_noul\| の平均の 95% 区間の下端が 0 より大きい |
| H3 | ③ の choice は 50% 付近で段差を作る | 題材平均で q(55) − q(45) > 0.5 |

- 判定は `uv run python -m jev_bayes.analyze` の `hypotheses` の出力をそのまま使う
- ロジットの切り詰めは [0.001, 0.999]、ブートストラップは 2,000 回・シード 20260929

## 探索的な分析（判定には使わない）

- 題材別の回帰係数、条件付き確率の逆転に近い件数、日本語と英語の差、① の抽選版 Brier
- 結果を見てから思いついた分析は、この節に追記せず、報告で「事後の探索」と明記する
```

- [ ] **Step 3: Commit**

```bash
git add docs/PREREGISTRATION.md
git commit -m "docs: 本番前に仮説・判定基準・版を事前登録"
```

---

### Task 9: 本番の実行と集計

**Files:**
- Create: `data/responses.jsonl`（生成物）
- Create: `results/summary.json` / `results/ladder.png` / `results/coefficients.png` / `results/brier.png`

- [ ] **Step 1: 本番を呼ぶ**

Run: `uv run python -m jev_bayes.run --attempts <Task 8 の r>`
Expected: `called` が 417 × r、`stopped` が `null`

途中で止まった場合は同じコマンドを再実行する（`ok` 済みは飛ばされる）。`FatalRequestError` で止まった場合は、エラー本文を読んで原因を直してから再実行する。

- [ ] **Step 2: 失敗と不正の件数を確かめる**

Run:
```bash
uv run python - <<'EOF'
import json
from collections import Counter
rows = [json.loads(l) for l in open("data/responses.jsonl")]
print(Counter(r["status"] for r in rows))
print("cost:", sum(r["cost"] or 0 for r in rows))
print("models:", Counter(r["model"] for r in rows))
EOF
```
Expected: `error` と `invalid` が 0。0 でなければ Step 1 を再実行し、それでも残る `invalid` は `error` 欄の理由を読んで記録する（分析からは自動で除外される）。`models` が1種類であること（途中で版が変わったら報告に書く）

- [ ] **Step 3: 集計と図**

Run:
```bash
uv run python -m jev_bayes.analyze
uv run python -m jev_bayes.figures
```
Expected: `hypotheses` の JSON が出力され、`results/` に `summary.json` と png 3枚ができる。png は Read ツールで開いて目で確かめる

- [ ] **Step 4: Commit**

```bash
git add data/responses.jsonl results/
git commit -m "feat: 本番の応答と集計結果を追加"
```

---

### Task 10: My-Knowladge への記録

このタスクは実験リポジトリの外（My-Knowladge）で行う。My-Knowladge の規約（`wiki-templates` スキル・`wiki-bookkeeping` スキル・`uv run scripts/wiki_lint_*.py`）に従う。

- [ ] **Step 1: 分析ページを作る**

`wiki/analyses/<YYYY-MM-DD>_Jevにベイズ問題を渡す_基準率無視と較正.md` を `wiki-templates` の analyses 形式で作る。冒頭の `> [!summary]` に H1〜H3 の判定結果を3行以内で書く。本文には `results/summary.json` の主要な値（段ごとの Brier の内訳、$b_1$・$b_2$ と区間、③ の段差）と、実験リポジトリのパス `~/projects/jev-bayes/` を書く

- [ ] **Step 2: 関連ページとつなぐ**

- `wiki/sources/2026-09-29_Jevはサイコロを振らない_較正された確率の落とし穴.md` の「メモ」に、追試と拡張の結果へのリンクを1行足す
- `wiki/concepts/数学/較正.md` と `wiki/entities/Jev.md` に、分析ページへのリンクを足す
- H1 が成り立った場合は、`wiki/concepts/数学/ベースレート.md` にも AI 側の実例として足し、`wiki-pattern-judgement` スキルで patterns/ への昇格を検討する

- [ ] **Step 3: bookkeeping と lint**

`wiki/index.md` と `wiki/log.md` を `wiki-bookkeeping` スキルに従って更新し、`uv run scripts/wiki_lint_links.py` / `wiki_lint_latex.py` / `wiki_lint_tags.py` / `wiki_lint_tables.py` がすべて errors 0 であることを確かめてから commit する
