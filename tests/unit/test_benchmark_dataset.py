import json
from pathlib import Path

import pytest

import evaluate
from ragas.metrics.base import Metric


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
SPLIT_PATHS = {
    "dev": DATA_DIR / "golden_dataset_dev.json",
    "test": DATA_DIR / "golden_dataset_test.json",
    "adversarial": DATA_DIR / "golden_dataset_adversarial.json",
}
EXPECTED_DOMAINS = {
    "social_protection",
    "labor_market",
    "demography",
    "economic_conjuncture",
    "investment_trade",
    "sectoral_regional",
}


def _load(split: str) -> list[dict]:
    return json.loads(SPLIT_PATHS[split].read_text(encoding="utf-8"))


def test_benchmark_targets_rag_principal_with_sabia_4_judge(monkeypatch):
    monkeypatch.delenv("RAGAS_JUDGE_MODEL", raising=False)

    assert evaluate.RAG_NAME == "rag_principal"
    assert evaluate.RAGAS_JUDGE_MODEL == "sabia-4"
    assert issubclass(evaluate.Faithfulness, Metric)
    assert issubclass(evaluate.ContextPrecision, Metric)
    assert issubclass(evaluate.ContextRecall, Metric)
    assert evaluate._run_metadata(
        type("Args", (), {"seed": 42, "use_graph": False, "limit": None})()
    )["ragas_judge_model"] == "sabia-4"


def test_split_sizes_ids_and_domain_coverage():
    splits = {name: _load(name) for name in SPLIT_PATHS}

    assert len(splits["dev"]) == 15
    assert len(splits["test"]) == 15
    assert len(splits["adversarial"]) == 10

    ids = [item["id"] for items in splits.values() for item in items]
    assert len(ids) == len(set(ids))
    assert {item["domain"] for item in splits["dev"]} == EXPECTED_DOMAINS
    assert {item["domain"] for item in splits["test"]} == EXPECTED_DOMAINS


def test_supervised_items_have_ground_truth_and_existing_sources():
    for split in ("dev", "test"):
        for item in _load(split):
            assert item["question"].strip()
            assert item["ground_truth"].strip()
            assert item["source_files"]
            assert item["source_pages"]
            assert all(page >= 1 for page in item["source_pages"])
            for relative_path in item["source_files"]:
                path = Path(relative_path)
                assert not path.is_absolute(), relative_path
                assert ".." not in path.parts, relative_path
                assert path.suffix.lower() == ".pdf", relative_path


def test_supervised_source_files_exist_when_corpus_is_available():
    if not any(DATA_DIR.rglob("*.pdf")):
        pytest.skip("corpus PDF externo não está disponível")

    for split in ("dev", "test"):
        for item in _load(split):
            for relative_path in item["source_files"]:
                assert (DATA_DIR / relative_path).is_file(), relative_path


def test_dev_and_test_use_distinct_source_documents():
    dev_sources = {
        source for item in _load("dev") for source in item["source_files"]
    }
    test_sources = {
        source for item in _load("test") for source in item["source_files"]
    }

    assert dev_sources.isdisjoint(test_sources)


def test_adversarial_split_requires_refusal():
    for item in _load("adversarial"):
        assert item["type"] == "adversarial"
        assert item["expected_refusal"] is True
        assert item["source_files"] == []


def test_output_label_is_safe_for_model_ids():
    assert evaluate._safe_output_label("openai/gpt-oss-120b") == "_openai-gpt-oss-120b"
    assert (
        evaluate._safe_output_label("nvidia/nemotron-3-super-120b-a12b")
        == "_nvidia-nemotron-3-super-120b-a12b"
    )
