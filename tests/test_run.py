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
