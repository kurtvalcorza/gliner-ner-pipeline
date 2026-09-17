from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from gliner_ner_pipeline import (
    ADAPT_CLASSES,
    ARTIFACT_FORMAT,
    ARTIFACT_FORMAT_VERSION,
    DATASET_REPRESENTATION,
    DEFAULT_THRESHOLD,
    MODEL_ID,
    GLiNERPipeline,
    compute_span_f1,
    evaluate_ner_dataset,
    generate_synthetic_ner_dataset,
    load_byod_dataset,
    split_ner_dataset,
    validate_dataset,
)

# ---------------------------------------------------------------------------
# Dataset validation & splitting tests
# ---------------------------------------------------------------------------


def test_synthetic_dataset_conforms_to_schema():
    dataset = generate_synthetic_ner_dataset()
    assert len(dataset) == 24
    manifest = validate_dataset(dataset)
    assert manifest["verdict"] == "accepted"
    assert manifest["representation"] == DATASET_REPRESENTATION
    assert manifest["n_records"] == 24
    for cls_name in ADAPT_CLASSES:
        assert manifest["label_distribution"][cls_name] > 0


def test_validate_dataset_rejects_non_sequence():
    with pytest.raises(TypeError, match="records must be a sequence"):
        validate_dataset("not a list")  # type: ignore[arg-type]


def test_validate_dataset_rejects_too_few_records():
    dataset = generate_synthetic_ner_dataset()[:2]
    with pytest.raises(ValueError, match="dataset requires at least"):
        validate_dataset(dataset)


def test_validate_dataset_rejects_missing_keys():
    dataset = [
        {"id": "1", "tokenized_text": ["Aspirin"]},
        {"id": "2", "tokenized_text": ["Imatinib"], "ner": []},
        {"id": "3", "tokenized_text": ["Metformin"], "ner": []},
        {"id": "4", "tokenized_text": ["Penicillin"], "ner": []},
    ]
    with pytest.raises(KeyError, match="missing required key 'ner'"):
        validate_dataset(dataset)


def test_validate_dataset_rejects_duplicate_ids():
    dataset = generate_synthetic_ner_dataset()[:4]
    dataset[1]["id"] = dataset[0]["id"]
    with pytest.raises(ValueError, match="duplicates id"):
        validate_dataset(dataset)


def test_validate_dataset_rejects_invalid_tokens():
    dataset = generate_synthetic_ner_dataset()[:4]
    dataset[0]["tokenized_text"] = ["Aspirin", ""]
    with pytest.raises(TypeError, match="token\\[1\\] must be a non-empty string"):
        validate_dataset(dataset)


def test_validate_dataset_rejects_invalid_label():
    dataset = generate_synthetic_ner_dataset()[:4]
    dataset[0]["ner"] = [[0, 0, "alien_lifeform"]]
    with pytest.raises(ValueError, match="label 'alien_lifeform' not in"):
        validate_dataset(dataset)


def test_validate_dataset_rejects_out_of_bounds_span():
    dataset = generate_synthetic_ner_dataset()[:4]
    dataset[0]["ner"] = [[0, 999, "chemical_drug"]]
    with pytest.raises(ValueError, match="out of bounds"):
        validate_dataset(dataset)


def test_validate_dataset_rejects_overlapping_spans():
    dataset = generate_synthetic_ner_dataset()[:4]
    dataset[0]["ner"] = [[0, 2, "chemical_drug"], [1, 3, "disease"]]
    with pytest.raises(ValueError, match="overlaps with an existing entity span"):
        validate_dataset(dataset)


def test_split_ner_dataset_disjoint_and_deterministic():
    dataset = generate_synthetic_ner_dataset()
    train1, val1 = split_ner_dataset(dataset, val_fraction=0.25, seed=42)
    train2, val2 = split_ner_dataset(dataset, val_fraction=0.25, seed=42)

    assert len(train1) + len(val1) == len(dataset)
    assert len(train1) == 18
    assert len(val1) == 6

    # Determinism check
    assert [r["id"] for r in train1] == [r["id"] for r in train2]
    assert [r["id"] for r in val1] == [r["id"] for r in val2]

    # Disjointness check
    train_ids = {r["id"] for r in train1}
    val_ids = {r["id"] for r in val1}
    assert len(train_ids & val_ids) == 0


def test_split_ner_dataset_invalid_fraction():
    dataset = generate_synthetic_ner_dataset()
    with pytest.raises(ValueError, match="val_fraction must be in \\(0, 1\\)"):
        split_ner_dataset(dataset, val_fraction=1.5)


# ---------------------------------------------------------------------------
# BYOD Loader tests
# ---------------------------------------------------------------------------


def test_load_byod_json(tmp_path: Path):
    dataset = generate_synthetic_ner_dataset()[:6]
    json_path = tmp_path / "custom_ner.json"
    json_path.write_text(json.dumps(dataset), encoding="utf-8")

    loaded = load_byod_dataset(json_path)
    assert len(loaded) == 6
    assert loaded[0]["id"] == dataset[0]["id"]
    assert "spans" in loaded[0]


def test_load_byod_jsonl(tmp_path: Path):
    dataset = generate_synthetic_ner_dataset()[:6]
    jsonl_path = tmp_path / "custom_ner.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for item in dataset:
            f.write(json.dumps(item) + "\n")

    loaded = load_byod_dataset(jsonl_path)
    assert len(loaded) == 6
    assert loaded[0]["id"] == dataset[0]["id"]


def test_load_byod_missing_file():
    with pytest.raises(FileNotFoundError):
        load_byod_dataset("non_existent_file.json")


def test_load_byod_empty_file(tmp_path: Path):
    empty_file = tmp_path / "empty.json"
    empty_file.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="empty"):
        load_byod_dataset(empty_file)


# ---------------------------------------------------------------------------
# Metrics tests
# ---------------------------------------------------------------------------


def test_compute_span_f1():
    assert compute_span_f1(0, 0, 0) == {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    assert compute_span_f1(5, 5, 5) == {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    res = compute_span_f1(2, 4, 4)
    assert res == {"precision": 0.5, "recall": 0.5, "f1": 0.5}


def test_evaluate_ner_dataset():
    records = [
        {
            "id": "1",
            "text": "Aspirin reduces fever.",
            "spans": [
                {"start": 0, "end": 7, "label": "chemical_drug"},
                {"start": 16, "end": 21, "label": "disease"},
            ],
        },
        {
            "id": "2",
            "text": "Imatinib targets BCR-ABL1.",
            "spans": [
                {"start": 0, "end": 8, "label": "chemical_drug"},
                {"start": 17, "end": 25, "label": "gene_protein"},
            ],
        },
    ]

    # Perfect predictions
    preds_perfect = [
        [
            {"start": 0, "end": 7, "label": "chemical_drug"},
            {"start": 16, "end": 21, "label": "disease"},
        ],
        [
            {"start": 0, "end": 8, "label": "chemical_drug"},
            {"start": 17, "end": 25, "label": "gene_protein"},
        ],
    ]
    res_perfect = evaluate_ner_dataset(preds_perfect, records, ADAPT_CLASSES)
    assert res_perfect["micro_f1"] == 1.0
    assert res_perfect["macro_f1"] == 1.0
    assert res_perfect["hits"] == 4

    # Empty predictions
    preds_empty = [[], []]
    res_empty = evaluate_ner_dataset(preds_empty, records, ADAPT_CLASSES)
    assert res_empty["micro_f1"] == 0.0
    assert res_empty["hits"] == 0


# ---------------------------------------------------------------------------
# Pipeline evaluate & artifact save/load parity tests
# ---------------------------------------------------------------------------


class _MockModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 2)
        self.frozen_layer = nn.Linear(4, 4)
        for p in self.frozen_layer.parameters():
            p.requires_grad = False

    def predict_entities(self, text: str, labels: list[str], threshold: float = DEFAULT_THRESHOLD):
        # Deterministic mock prediction
        if "Aspirin" in text:
            return [{"start": 0, "end": 7, "label": "chemical_drug", "score": 0.95}]
        return []

    def _create_data_collator(self):
        def collate_fn(batch):
            return {"dummy": torch.zeros((len(batch), 4))}
        return collate_fn

    def forward(self, **kwargs):
        class Output:
            def __init__(self, loss):
                self.loss = loss
        return Output(loss=torch.tensor(0.5, requires_grad=True))


def test_pipeline_evaluate_with_mock():
    mock_model = _MockModel()
    pipe = GLiNERPipeline(
        _runner=lambda text, labels, threshold: [],
        device="cpu",
        model=mock_model,
    )

    records = [
        {
            "id": "1",
            "text": "Aspirin treats pain.",
            "tokenized_text": ["Aspirin", "treats", "pain", "."],
            "ner": [[0, 0, "chemical_drug"], [2, 2, "disease"]],
            "spans": [
                {"start": 0, "end": 7, "label": "chemical_drug"},
                {"start": 15, "end": 19, "label": "disease"},
            ],
        },
        {
            "id": "2",
            "text": "Imatinib targets BCR-ABL1.",
            "tokenized_text": ["Imatinib", "targets", "BCR-ABL1", "."],
            "ner": [[0, 0, "chemical_drug"], [2, 2, "gene_protein"]],
            "spans": [
                {"start": 0, "end": 8, "label": "chemical_drug"},
                {"start": 17, "end": 25, "label": "gene_protein"},
            ],
        },
        {
            "id": "3",
            "text": "Metformin treats diabetes.",
            "tokenized_text": ["Metformin", "treats", "diabetes", "."],
            "ner": [[0, 0, "chemical_drug"], [2, 2, "disease"]],
            "spans": [
                {"start": 0, "end": 9, "label": "chemical_drug"},
                {"start": 17, "end": 25, "label": "disease"},
            ],
        },
        {
            "id": "4",
            "text": "Penicillin treats pneumonia.",
            "tokenized_text": ["Penicillin", "treats", "pneumonia", "."],
            "ner": [[0, 0, "chemical_drug"], [2, 2, "disease"]],
            "spans": [
                {"start": 0, "end": 10, "label": "chemical_drug"},
                {"start": 18, "end": 27, "label": "disease"},
            ],
        },
    ]

    report = pipe.evaluate(records, labels=list(ADAPT_CLASSES))
    assert report["num_samples"] == 4
    assert report["micro"]["hits"] == 1
    assert "chemical_drug" in report["per_class"]


def test_pipeline_artifact_save_and_reload(tmp_path: Path):
    mock_model = _MockModel()
    pipe = GLiNERPipeline(
        _runner=lambda text, labels, threshold: [],
        device="cpu",
        model=mock_model,
    )

    artifact_file = tmp_path / "gliner_biomed_adapter.pt"
    pipe.save_artifact(
        artifact_file,
        metadata={"test_run": True, "epochs": 3},
    )
    assert artifact_file.is_file()

    # Verify contents of artifact dictionary
    payload = torch.load(artifact_file, map_location="cpu", weights_only=False)
    assert payload["format"] == ARTIFACT_FORMAT
    assert payload["format_version"] == ARTIFACT_FORMAT_VERSION
    assert payload["base_model"]["model_id"] == MODEL_ID
    assert "fc.weight" in payload["adapter_state_dict"]
    # Verify frozen layer was excluded from adapter weights
    assert "frozen_layer.weight" not in payload["adapter_state_dict"]

    # Reload into fresh pipeline instance
    fresh_mock = _MockModel()
    fresh_pipe = GLiNERPipeline(
        _runner=lambda text, labels, threshold: [],
        device="cpu",
        model=fresh_mock,
    )
    meta = fresh_pipe.load_artifact(artifact_file)
    assert meta.get("test_run") is True
    assert meta.get("epochs") == 3


def test_pipeline_load_artifact_validation_errors(tmp_path: Path):
    pipe = GLiNERPipeline(_runner=lambda text, labels, threshold: [], device="cpu", model=_MockModel())

    with pytest.raises(FileNotFoundError):
        pipe.load_artifact(tmp_path / "non_existent.pt")

    bad_format_file = tmp_path / "bad_format.pt"
    torch.save({"format": "wrong_format"}, bad_format_file)
    with pytest.raises(ValueError, match="Artifact format mismatch"):
        pipe.load_artifact(bad_format_file)

    bad_model_file = tmp_path / "bad_model.pt"
    torch.save({"format": ARTIFACT_FORMAT, "base_model": {"model_id": "wrong_model"}}, bad_model_file)
    with pytest.raises(ValueError, match="Artifact base model mismatch"):
        pipe.load_artifact(bad_model_file)


def test_pipeline_adapt_mock():
    mock_model = _MockModel()
    pipe = GLiNERPipeline(
        _runner=lambda text, labels, threshold: [],
        device="cpu",
        model=mock_model,
    )

    train = generate_synthetic_ner_dataset()[:6]
    val = generate_synthetic_ner_dataset()[6:10]

    result = pipe.adapt(
        train_records=train,
        val_records=val,
        epochs=2,
        batch_size=2,
        learning_rate=1e-4,
    )

    assert result["epochs"] == 2
    assert len(result["history"]) == 2
    assert "loss" in result["history"][0]
    assert "val_f1" in result["history"][0]
    assert result["final_eval"] is not None
