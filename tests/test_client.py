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
