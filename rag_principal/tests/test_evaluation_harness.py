import json
import math
import os
import tempfile
import pandas as pd
import pytest
from unittest.mock import MagicMock, patch

from evaluate import (
    _derive_judge_provider,
    _sanitize_url,
    _sanitize_error_message,
    _sanitize_for_json,
    _atomic_write_json,
    _get_git_info,
    judge_records,
    run_judge_only,
)


def test_derive_judge_provider(monkeypatch):
    monkeypatch.delenv("RAGAS_JUDGE_PROVIDER", raising=False)
    assert _derive_judge_provider("https://api.deepseek.com") == "deepseek"
    assert _derive_judge_provider("https://chat.maritaca.ai/api") == "maritaca"
    assert _derive_judge_provider("https://openrouter.ai/api/v1") == "openrouter"
    assert _derive_judge_provider("https://api.openai.com/v1") == "openai"
    assert _derive_judge_provider("https://custom.llm.local/v1") == "custom"

    monkeypatch.setenv("RAGAS_JUDGE_PROVIDER", "meu_provider")
    assert _derive_judge_provider("https://chat.maritaca.ai/api") == "meu_provider"


def test_sanitize_url():
    assert _sanitize_url("https://chat.maritaca.ai/api?key=12345") == "https://chat.maritaca.ai/api"
    assert _sanitize_url("https://user:pass@host.com/v1/") == "https://host.com/v1"
    assert _sanitize_url("") == ""


def test_sanitize_error_message():
    raw_error = "HTTP 403: key=sk-maritaca-secret-key-1234567890 authorization failed"
    sanitized = _sanitize_error_message(raw_error)
    assert "sk-maritaca-secret-key-1234567890" not in sanitized
    assert "key=***REDACTED***" in sanitized


def test_sanitize_for_json():
    data = {
        "valid": 1.23,
        "nan_val": float("nan"),
        "inf_val": float("inf"),
        "nested": [1, float("nan"), {"inner": float("nan")}],
    }
    cleaned = _sanitize_for_json(data)
    assert cleaned["valid"] == 1.23
    assert cleaned["nan_val"] is None
    assert cleaned["inf_val"] is None
    assert cleaned["nested"][1] is None
    assert cleaned["nested"][2]["inner"] is None


def test_atomic_write_json():
    with tempfile.TemporaryDirectory() as tmpdir:
        target = os.path.join(tmpdir, "out.json")
        data = {"metric": float("nan"), "status": "ok"}
        _atomic_write_json(target, data)

        with open(target, "r", encoding="utf-8") as f:
            content = f.read()
            loaded = json.loads(content)

        assert "NaN" not in content
        assert loaded["metric"] is None
        assert loaded["status"] == "ok"


def test_get_git_info():
    info = _get_git_info()
    assert "git_commit" in info
    assert "git_dirty" in info
    assert isinstance(info["git_dirty"], bool)
    assert "git_diff_sha256" in info


def test_judge_records_detects_nan_and_marks_failed(monkeypatch):
    mock_df = pd.DataFrame([
        {
            "user_input": "q1",
            "faithfulness": float("nan"),
            "context_precision": 0.8,
            "context_recall": 0.9,
        }
    ])
    mock_scores = MagicMock()
    mock_scores.to_pandas.return_value = mock_df

    monkeypatch.setattr("evaluate.evaluate", lambda *args, **kwargs: mock_scores)

    records = [{"user_input": "q1", "response": "r1", "retrieved_contexts": ["c1"], "reference": "ref1"}]
    diagnostics = [{"id": "dev_01", "latency_ms": 100}]

    scores_dict, details, success, error_msg = judge_records(
        records, diagnostics, "mock-model", MagicMock()
    )

    assert success is False
    assert "NaN" in error_msg
    assert scores_dict["faithfulness"] is None
    assert details[0]["faithfulness"] is None


def test_judge_records_handles_exception_and_redacts(monkeypatch):
    def mock_eval(*args, **kwargs):
        raise RuntimeError("Falha 403: bearer token=secret_token_123456789")

    monkeypatch.setattr("evaluate.evaluate", mock_eval)

    records = [{"user_input": "q1", "response": "r1", "retrieved_contexts": ["c1"], "reference": "ref1"}]
    diagnostics = [{"id": "dev_01", "latency_ms": 100}]

    scores_dict, details, success, error_msg = judge_records(
        records, diagnostics, "mock-model", MagicMock()
    )

    assert success is False
    assert "secret_token_123456789" not in error_msg
    assert "token=***REDACTED***" in error_msg
    assert scores_dict["faithfulness"] is None


def test_run_judge_only_with_mocked_judge(monkeypatch):
    with tempfile.TemporaryDirectory() as tmpdir:
        artifact_path = os.path.join(tmpdir, "generation.json")
        gen_data = {
            "split": "dev",
            "summary": {"latency_p50_ms": 200, "numeric_precision": 0.95},
            "details": [
                {
                    "id": "dev_economy_01",
                    "user_input": "Qual foi o PIB?",
                    "response": "O PIB cresceu 1,8%",
                    "retrieved_contexts": ["Contexto 1"],
                    "reference": "1,8%",
                }
            ],
        }
        with open(artifact_path, "w", encoding="utf-8") as f:
            json.dump(gen_data, f)

        mock_df = pd.DataFrame([
            {
                "user_input": "Qual foi o PIB?",
                "faithfulness": 0.95,
                "context_precision": 0.88,
                "context_recall": 0.90,
            }
        ])
        mock_scores = MagicMock()
        mock_scores.to_pandas.return_value = mock_df

        monkeypatch.setattr("evaluate.evaluate", lambda *args, **kwargs: mock_scores)
        monkeypatch.setattr("evaluate._ragas_judge_llm", lambda: ("sabia-4", MagicMock()))

        output_path = os.path.join(tmpdir, "judged.json")
        scores, success = run_judge_only(artifact_path, output_path=output_path)

        assert success is True
        assert scores["faithfulness"] == 0.95
        assert scores["numeric_precision"] == 0.95
        assert scores["latency_p50_ms"] == 200

        with open(output_path, "r", encoding="utf-8") as f:
            result = json.load(f)

        assert result["status"] == "completed"
        assert result["summary"]["faithfulness"] == 0.95
        assert result["details"][0]["faithfulness"] == 0.95

