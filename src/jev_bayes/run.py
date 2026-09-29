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
