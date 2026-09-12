import hashlib
import json
import re
from pathlib import Path

import pytest

from gliner_ner_pipeline import (
    DEFAULT_THRESHOLD,
    DEFAULT_WEIGHTS_DIR,
    ENCODER_KEY,
    ENCODER_MODEL_ID,
    ENCODER_REVISION,
    ENCODER_WEIGHTS_DIR,
    MAX_LABELS,
    MAX_TEXT_CHARS,
    MODEL_ID,
    MODEL_KEY,
    MODEL_REVISION,
    GLiNERPipeline,
    entity_f1,
    stage_missing_encoder_files,
    stage_missing_files,
    verify_encoder_snapshot,
    verify_snapshot,
)

REPO = Path(__file__).resolve().parents[1]
MANIFEST = REPO / "weights" / MODEL_KEY / "dimer-base-manifest.json"
ENCODER_MANIFEST = REPO / "weights" / ENCODER_KEY / "dimer-base-manifest.json"
TEXT = "Marie Curie was born in Warsaw and worked in Paris."
GOLD = [
    {"start": 0, "end": 11, "label": "person"},
    {"start": 24, "end": 30, "label": "location"},
    {"start": 45, "end": 50, "label": "location"},
]


def _fake_pipeline(entities=None, calls=None):
    def runner(text, labels, threshold):
        if calls is not None:
            calls.append((text, list(labels), threshold))
        source = GOLD if entities is None else entities
        return [dict(e, text=TEXT[e["start"] : e["end"]], score=0.9) for e in source]

    return GLiNERPipeline(runner, "cpu")


def test_identity_constants_are_40_hex_and_match_manifest():
    assert re.fullmatch(r"[0-9a-f]{40}", MODEL_REVISION)
    assert DEFAULT_WEIGHTS_DIR == REPO / "weights" / MODEL_KEY
    if MANIFEST.is_file():
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        assert manifest["modelId"] == MODEL_ID
        assert manifest["revision"] == MODEL_REVISION
    cfg = REPO / "weights" / MODEL_KEY / "gliner_config.json"
    if cfg.is_file():
        config = json.loads(cfg.read_text(encoding="utf-8"))
        assert config["model_name"] == ENCODER_MODEL_ID
        assert config["max_types"] == MAX_LABELS


def test_encoder_identity_constants_are_40_hex_and_match_manifest():
    assert re.fullmatch(r"[0-9a-f]{40}", ENCODER_REVISION)
    assert ENCODER_WEIGHTS_DIR == REPO / "weights" / ENCODER_KEY
    if ENCODER_MANIFEST.is_file():
        manifest = json.loads(ENCODER_MANIFEST.read_text(encoding="utf-8"))
        assert manifest["modelId"] == ENCODER_MODEL_ID
        assert manifest["revision"] == ENCODER_REVISION
        assert {f["path"] for f in manifest["files"]} >= {"config.json", "tokenizer_config.json", "spm.model"}


def _write_snapshot(tmp_path: Path, content: bytes, sha256: str, revision: str = MODEL_REVISION) -> Path:
    (tmp_path / "gliner_config.json").write_bytes(content)
    manifest = {
        "modelId": MODEL_ID,
        "revision": revision,
        "files": [{"path": "gliner_config.json", "bytes": len(content), "sha256": sha256}],
    }
    (tmp_path / "dimer-base-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return tmp_path


def test_verify_snapshot_accepts_matching_digest(tmp_path):
    content = b'{"a": 1}'
    root = _write_snapshot(tmp_path, content, hashlib.sha256(content).hexdigest())
    assert verify_snapshot(root)["revision"] == MODEL_REVISION


def test_verify_snapshot_rejects_tampered_digest(tmp_path):
    content = b'{"a": 1}'
    good = hashlib.sha256(content).hexdigest()
    bad = ("0" if good[0] != "0" else "1") + good[1:]
    with pytest.raises(ValueError, match="sha256"):
        verify_snapshot(_write_snapshot(tmp_path, content, bad))


def test_verify_snapshot_rejects_wrong_revision_and_size(tmp_path):
    content = b"xyz"
    digest = hashlib.sha256(content).hexdigest()
    with pytest.raises(ValueError, match="revision"):
        verify_snapshot(_write_snapshot(tmp_path, content, digest, revision="f" * 40))
    root = _write_snapshot(tmp_path, content, digest)
    (tmp_path / "gliner_config.json").write_bytes(b"xyzw")
    with pytest.raises(ValueError, match="size"):
        verify_snapshot(root)


def test_stage_missing_files_fetches_only_absent_entries_then_verifies(tmp_path):
    """Fresh-clone shape: manifest committed, weight file absent. allow_download fetches exactly that file."""
    payload = b"weights-bytes"
    (tmp_path / "config.json").write_bytes(b"{}")
    manifest = {
        "modelId": MODEL_ID,
        "revision": MODEL_REVISION,
        "files": [
            {"path": "config.json", "bytes": 2, "sha256": hashlib.sha256(b"{}").hexdigest()},
            {"path": "model.bin", "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()},
        ],
    }
    (tmp_path / "dimer-base-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="allow_download=True"):
        stage_missing_files(tmp_path)
    fetched = []

    def fake_download(relative_path, root):
        fetched.append(relative_path)
        (root / relative_path).write_bytes(payload)

    assert stage_missing_files(tmp_path, allow_download=True, downloader=fake_download) == ["model.bin"]
    assert fetched == ["model.bin"]
    listed = verify_snapshot(tmp_path)["files"]
    assert (listed if isinstance(listed, int) else len(listed)) == 2
    assert stage_missing_files(tmp_path, allow_download=True, downloader=fake_download) == []


def test_stage_missing_files_refuses_foreign_manifest(tmp_path):
    manifest = {"modelId": "someone/else", "revision": MODEL_REVISION, "files": []}
    (tmp_path / "dimer-base-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="refusing to stage"):
        stage_missing_files(tmp_path, allow_download=True, downloader=lambda *_: None)


def test_from_pretrained_refuses_without_snapshot(tmp_path):
    with pytest.raises(FileNotFoundError):
        GLiNERPipeline.from_pretrained(weights_dir=tmp_path, allow_download=False)


def _write_encoder_snapshot(
    tmp_path: Path, content: bytes, sha256: str, model_id: str = ENCODER_MODEL_ID
) -> Path:
    (tmp_path / "tokenizer_config.json").write_bytes(content)
    manifest = {
        "modelId": model_id,
        "revision": ENCODER_REVISION,
        "files": [{"path": "tokenizer_config.json", "bytes": len(content), "sha256": sha256}],
    }
    (tmp_path / "dimer-base-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return tmp_path


def test_verify_encoder_snapshot_accepts_and_rejects_tampered_digest(tmp_path):
    content = b'{"do_lower_case": false}'
    good = hashlib.sha256(content).hexdigest()
    root = _write_encoder_snapshot(tmp_path, content, good)
    assert verify_encoder_snapshot(root)["revision"] == ENCODER_REVISION
    bad = ("0" if good[0] != "0" else "1") + good[1:]
    with pytest.raises(ValueError, match="sha256"):
        verify_encoder_snapshot(_write_encoder_snapshot(tmp_path, content, bad))


def test_encoder_snapshot_refuses_foreign_model_id(tmp_path):
    content = b"{}"
    root = _write_encoder_snapshot(tmp_path, content, hashlib.sha256(content).hexdigest(), model_id=MODEL_ID)
    with pytest.raises(ValueError, match="modelId"):
        verify_encoder_snapshot(root)
    with pytest.raises(ValueError, match="refusing to stage"):
        stage_missing_encoder_files(root, allow_download=True, downloader=lambda *_: None)


def test_stage_missing_encoder_files_fetches_absent_entry(tmp_path):
    content = b"{}"
    root = _write_encoder_snapshot(tmp_path, content, hashlib.sha256(content).hexdigest())
    (root / "tokenizer_config.json").unlink()
    with pytest.raises(FileNotFoundError, match="allow_download=True"):
        stage_missing_encoder_files(root)
    fetched = []

    def fake_download(relative_path, dst):
        fetched.append(relative_path)
        (dst / relative_path).write_bytes(content)

    staged = stage_missing_encoder_files(root, allow_download=True, downloader=fake_download)
    assert staged == ["tokenizer_config.json"]
    assert fetched == ["tokenizer_config.json"]
    assert verify_encoder_snapshot(root)["modelId"] == ENCODER_MODEL_ID


@pytest.mark.parametrize(
    ("text", "labels", "threshold", "exc"),
    [
        (42, ["person"], 0.5, TypeError),
        ("   ", ["person"], 0.5, ValueError),
        ("a" * (MAX_TEXT_CHARS + 1), ["person"], 0.5, ValueError),
        (TEXT, "person", 0.5, TypeError),
        (TEXT, [], 0.5, ValueError),
        (TEXT, [f"l{i}" for i in range(MAX_LABELS + 1)], 0.5, ValueError),
        (TEXT, ["person", 7], 0.5, TypeError),
        (TEXT, ["person", "person"], 0.5, ValueError),
        (TEXT, ["x" * 101], 0.5, ValueError),
        (TEXT, ["person"], 1.5, ValueError),
        (TEXT, ["person"], True, ValueError),
    ],
)
def test_detect_rejects_bad_input(text, labels, threshold, exc):
    calls = []
    pipe = _fake_pipeline(calls=calls)
    with pytest.raises(exc):
        pipe.detect(text, labels, threshold=threshold)
    assert calls == []  # the model never ran


def test_detect_output_fields_and_threshold_passthrough():
    calls = []
    pipe = _fake_pipeline(calls=calls)
    result = pipe.detect(TEXT, ["person", "location"])
    assert calls == [(TEXT, ["person", "location"], DEFAULT_THRESHOLD)]
    assert result["n_entities"] == 3
    first = {"text": "Marie Curie", "label": "person", "start": 0, "end": 11, "score": 0.9}
    assert result["entities"][0] == first
    assert [e["text"] for e in result["entities"]] == ["Marie Curie", "Warsaw", "Paris"]
    assert result["labels"] == ["person", "location"]
    assert result["threshold"] == DEFAULT_THRESHOLD
    assert result["model_id"] == MODEL_ID
    assert result["model_revision"] == MODEL_REVISION
    pipe.detect(TEXT, ["person", "location"], threshold=0.3)
    assert calls[-1][2] == 0.3


def test_detect_rejects_backend_entity_outside_contract():
    pipe = _fake_pipeline(entities=[{"start": 0, "end": 11, "label": "date"}])
    with pytest.raises(RuntimeError, match="invalid entity"):
        pipe.detect(TEXT, ["person"])


def test_entity_f1_exact_span_micro():
    # 2 exact hits, 1 span with the wrong label
    pred = [dict(g) for g in GOLD[:2]] + [{"start": 45, "end": 50, "label": "person"}]
    scores = entity_f1(pred, GOLD)
    assert scores["hits"] == 2
    assert abs(scores["precision"] - 2 / 3) < 1e-9
    assert abs(scores["recall"] - 2 / 3) < 1e-9
    assert abs(scores["f1"] - 2 / 3) < 1e-9
    assert entity_f1([], GOLD) == {"precision": 0.0, "recall": 0.0, "f1": 0.0, "hits": 0}
