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
