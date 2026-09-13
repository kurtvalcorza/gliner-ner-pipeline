"""Offline tests for the public validation and evaluation stage helpers (DAT24 / EVAL21)."""

from __future__ import annotations

import pytest

from gliner_ner_pipeline import (
    DEFAULT_THRESHOLD,
    ENCODER_MODEL_ID,
    ENCODER_REVISION,
    INPUT_SCHEMA,
    MAX_LABEL_CHARS,
    MAX_LABELS,
    MAX_TEXT_CHARS,
    MODEL_ID,
    MODEL_REVISION,
    evaluation_report,
    validate_inputs,
)

TEXT = "Marie Curie was born in Warsaw and later joined the University of Paris."
LABELS = ["person", "location", "organisation"]
GOLD = [
    {"text": "Marie Curie", "label": "person", "start": 0, "end": 11},
    {"text": "Warsaw", "label": "location", "start": 24, "end": 30},
    {"text": "University of Paris", "label": "organisation", "start": 51, "end": 70},
]


def _result(entities: list[dict], threshold: float = DEFAULT_THRESHOLD) -> dict:
    return {
        "entities": entities,
        "n_entities": len(entities),
        "labels": list(LABELS),
        "threshold": threshold,
    }


def test_validate_inputs_returns_manifest_with_schema_and_both_identities() -> None:
    manifest = validate_inputs(TEXT, LABELS, names=["curie_sentence"])
    assert manifest["verdict"] == "accepted"
    assert manifest["findings"] == []
    assert manifest["schema"] == INPUT_SCHEMA
    assert manifest["schema"]["text_chars"] == [1, MAX_TEXT_CHARS]
    assert manifest["schema"]["labels"] == [1, MAX_LABELS]
    assert manifest["schema"]["label_chars"] == [1, MAX_LABEL_CHARS]
    assert manifest["inputs"] == [{"id": "curie_sentence", "chars": len(TEXT), "words": len(TEXT.split())}]
    assert manifest["labels"] == LABELS
    assert manifest["threshold"] == DEFAULT_THRESHOLD
    assert (manifest["model_id"], manifest["model_revision"]) == (MODEL_ID, MODEL_REVISION)
    assert (manifest["encoder_model_id"], manifest["encoder_revision"]) == (
        ENCODER_MODEL_ID,
        ENCODER_REVISION,
    )


def test_validate_inputs_default_id_and_explicit_threshold() -> None:
    manifest = validate_inputs(TEXT, ["person"], 0.2)
    assert [entry["id"] for entry in manifest["inputs"]] == ["text-0"]
    assert manifest["threshold"] == 0.2


def test_validate_inputs_rejects_like_detect() -> None:
    with pytest.raises(TypeError, match="text must be str"):
        validate_inputs(123, LABELS)
    with pytest.raises(ValueError, match="text is empty"):
        validate_inputs("   ", LABELS)
    with pytest.raises(ValueError, match="ceiling is"):
        validate_inputs("x" * (MAX_TEXT_CHARS + 1), LABELS)
    with pytest.raises(TypeError, match="labels must be a list of str"):
        validate_inputs(TEXT, "person")
    with pytest.raises(ValueError, match=f"labels must hold 1..{MAX_LABELS}"):
        validate_inputs(TEXT, [f"label-{i}" for i in range(MAX_LABELS + 1)])
    with pytest.raises(ValueError, match="labels must be unique"):
        validate_inputs(TEXT, ["person", "person"])
    with pytest.raises(ValueError, match="ceiling is"):
        validate_inputs(TEXT, ["x" * (MAX_LABEL_CHARS + 1)])
    with pytest.raises(ValueError, match=r"threshold must be a number in \[0, 1\]"):
        validate_inputs(TEXT, LABELS, 1.5)
    with pytest.raises(ValueError, match="names must have exactly one entry"):
        validate_inputs(TEXT, LABELS, names=["a", "b"])


def test_evaluation_report_not_measurable_without_gold_spans() -> None:
    report = evaluation_report(_result(GOLD[:1]))
    assert report["verdict"] == "not-measurable"
    assert report["metrics"] == []
    assert report["n_entities"] == 1
    assert "entity_f1" in report["needs"]
    assert report["labels"] == LABELS
    assert (report["model_id"], report["model_revision"]) == (MODEL_ID, MODEL_REVISION)
    assert (report["encoder_model_id"], report["encoder_revision"]) == (ENCODER_MODEL_ID, ENCODER_REVISION)


def test_evaluation_report_sample_sanity_with_gold_spans() -> None:
    report = evaluation_report(_result(GOLD), GOLD, sample_kind="synthetic")
    assert report["verdict"] == "sample-sanity"
    metric = report["metrics"][0]
    assert metric["id"] == "entity_f1"
    assert metric["value"] == pytest.approx(1.0)
    assert (metric["precision"], metric["recall"], metric["hits"]) == (1.0, 1.0, 3)
    assert (metric["n_predicted"], metric["n_gold"]) == (3, 3)
    assert metric["matching"] == "exact (start, end, label) triple"
    assert metric["estimation"]


def test_evaluation_report_partial_match_scores_below_one() -> None:
    predicted = [GOLD[0], {"text": "Paris", "label": "location", "start": 65, "end": 70}]
    report = evaluation_report(_result(predicted), GOLD)
    metric = report["metrics"][0]
    assert metric["hits"] == 1
    assert metric["precision"] == pytest.approx(0.5)
    assert metric["recall"] == pytest.approx(1 / 3)
    assert 0.0 < metric["value"] < 1.0


def test_evaluation_report_empty_gold_list_is_still_measured_not_skipped() -> None:
    report = evaluation_report(_result(GOLD), [])
    assert report["verdict"] == "sample-sanity"
    metric = report["metrics"][0]
    assert (metric["hits"], metric["n_gold"]) == (0, 0)
    assert metric["value"] == 0.0
