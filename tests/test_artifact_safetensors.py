"""Adapter artifact format 2.0: safetensors only, plain-JSON metadata, reproducible bytes (ARTIFACT_SPEC 2.0
SER7, SER10, SER11). CI installs no torch, so these tests skip there and run in a torch environment."""

from __future__ import annotations

import json
import os
import pickle
import subprocess
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")

from gliner_ner_pipeline import (  # noqa: E402
    ARTIFACT_FORMAT,
    ARTIFACT_FORMAT_VERSION,
    ARTIFACT_METADATA_KEY,
    ARTIFACT_SUFFIX,
    MODEL_ID,
    GLiNERPipeline,
    read_artifact_header,
)

SRC = Path(__file__).resolve().parents[1] / "src"
METADATA = {
    "domain": "biomedical",
    "classes": ["disease", "gene_protein"],
    "epochs": 3,
    "metrics": {"f1": 0.5},
}


class _Adapter(torch.nn.Module):
    """A small model with trainable and frozen layers; its forward is a cheap CPU pass."""

    def __init__(self, seed: int):
        super().__init__()
        torch.manual_seed(seed)
        self.fc = torch.nn.Linear(4, 3)
        self.head = torch.nn.Linear(3, 2)
        self.frozen = torch.nn.Linear(4, 4)
        for p in self.frozen.parameters():
            p.requires_grad = False

    def forward(self, x):
        return self.head(torch.tanh(self.fc(self.frozen(x))))


def _pipe(seed: int) -> GLiNERPipeline:
    return GLiNERPipeline(_runner=lambda *a: [], device="cpu", model=_Adapter(seed))


def _header(path: Path) -> tuple[int, dict]:
    raw = path.read_bytes()
    n = int.from_bytes(raw[:8], "little")
    return n, json.loads(raw[8 : 8 + n])


def test_round_trip_into_a_fresh_model_gives_identical_tensors_and_outputs(tmp_path):
    trained = _pipe(seed=1)
    path = trained.save_artifact(tmp_path / f"adapter{ARTIFACT_SUFFIX}", metadata=METADATA)
    fresh = _pipe(seed=2)
    x = torch.linspace(-1, 1, 8).reshape(2, 4)
    assert not torch.equal(fresh.model(x), trained.model(x))
    # The frozen layer is base-model state, not adapter state: give the fresh model the same base.
    fresh.model.frozen.load_state_dict(trained.model.frozen.state_dict())

    assert fresh.load_artifact(path) == METADATA
    assert fresh.adapted is True
    for name, tensor in trained.model.state_dict().items():
        assert torch.equal(tensor, fresh.model.state_dict()[name]), name
    assert torch.equal(fresh.model(x), trained.model(x))


def test_file_holds_only_tensors_and_one_plain_json_metadata_key(tmp_path):
    path = _pipe(seed=1).save_artifact(tmp_path / f"a{ARTIFACT_SUFFIX}", metadata=METADATA)
    raw = path.read_bytes()
    n, header = _header(path)
    assert sorted(header["__metadata__"]) == [ARTIFACT_METADATA_KEY]
    tensors = {k: v for k, v in header.items() if k != "__metadata__"}
    assert sorted(tensors) == ["fc.bias", "fc.weight", "head.bias", "head.weight"]  # trainable only
    # The data section is exactly the tensor bytes: nothing else is stored in the file.
    assert len(raw) - 8 - n == max(v["data_offsets"][1] for v in tensors.values())
    assert not raw.startswith((b"PK", b"\x80"))  # neither a torch.save zip nor a raw pickle
    with pytest.raises(pickle.UnpicklingError):
        pickle.loads(raw)
    payload = json.loads(header["__metadata__"][ARTIFACT_METADATA_KEY])
    assert payload == read_artifact_header(path)
    assert payload["format"] == ARTIFACT_FORMAT
    assert payload["format_version"] == ARTIFACT_FORMAT_VERSION == "2.0"
    assert payload["base_model"]["model_id"] == MODEL_ID
    assert payload["metadata"] == METADATA
    assert sorted(payload["adapter_manifest"]) == sorted(tensors)


_SAVE_SNIPPET = (
    "import sys, torch\n"
    "sys.path.insert(0, sys.argv[1])\n"
    "from gliner_ner_pipeline import GLiNERPipeline\n"
    "torch.manual_seed(7)\n"
    "m = torch.nn.Sequential(torch.nn.Linear(4, 3), torch.nn.Linear(3, 2))\n"
    "p = GLiNERPipeline(_runner=lambda *a: [], device='cpu', model=m)\n"
    "p.save_artifact(sys.argv[2], metadata={'z': 1, 'a': [1, 2], 'm': {'y': 'x', 'b': None}})\n"
)


def test_saves_are_byte_identical_in_process_and_across_fresh_interpreters(tmp_path):
    torch.manual_seed(7)
    model = torch.nn.Sequential(torch.nn.Linear(4, 3), torch.nn.Linear(3, 2))
    pipe = GLiNERPipeline(_runner=lambda *a: [], device="cpu", model=model)
    meta = {"z": 1, "a": [1, 2], "m": {"y": "x", "b": None}}
    first = pipe.save_artifact(tmp_path / "one.safetensors", metadata=meta).read_bytes()
    reordered = dict(reversed(meta.items()))
    second = pipe.save_artifact(tmp_path / "two.safetensors", metadata=reordered).read_bytes()
    assert first == second

    outputs = []
    for i, hash_seed in enumerate(("0", "12345")):
        out = tmp_path / f"fresh{i}.safetensors"
        env = {**os.environ, "PYTHONHASHSEED": hash_seed, "CUDA_VISIBLE_DEVICES": "-1"}
        subprocess.run([sys.executable, "-c", _SAVE_SNIPPET, str(SRC), str(out)], check=True, env=env)
        outputs.append(out.read_bytes())
    assert outputs[0] == outputs[1] == first


def test_wrong_format_truncated_and_tampered_files_are_refused(tmp_path):
    pipe = _pipe(seed=1)
    good = pipe.save_artifact(tmp_path / "good.safetensors", metadata=METADATA)
    raw = good.read_bytes()
    n = int.from_bytes(raw[:8], "little")

    legacy = tmp_path / "legacy.pt"  # a format-1.x pickle artifact is no longer read
    torch.save({"format": ARTIFACT_FORMAT, "format_version": "1.1"}, legacy)
    truncated = tmp_path / "truncated.safetensors"
    truncated.write_bytes(raw[:-5])
    cut_header = tmp_path / "cut_header.safetensors"
    cut_header.write_bytes(raw[: 8 + n // 2])
    flipped = tmp_path / "flipped.safetensors"  # same length, one data byte changed
    flipped.write_bytes(raw[:-1] + bytes([raw[-1] ^ 0xFF]))
    for bad, message in [
        (legacy, "not a readable safetensors file"),
        (truncated, "not a readable safetensors file|could not be read"),
        (cut_header, "not a readable safetensors file"),
        (flipped, "differ from manifest"),
    ]:
        fresh = _pipe(seed=2)
        with pytest.raises(ValueError, match=message):
            fresh.load_artifact(bad)
        assert fresh.adapted is False, bad.name

    from safetensors.torch import save_file

    no_meta = tmp_path / "no_meta.safetensors"
    save_file({"fc.weight": torch.zeros(3, 4)}, str(no_meta))
    not_json = tmp_path / "not_json.safetensors"
    save_file({"fc.weight": torch.zeros(3, 4)}, str(not_json), metadata={ARTIFACT_METADATA_KEY: "{nope"})
    for bad, message in [(no_meta, "no 'dimer_artifact' metadata"), (not_json, "not valid JSON")]:
        with pytest.raises(ValueError, match=message):
            read_artifact_header(bad)


def test_metadata_that_is_not_plain_json_is_refused_before_writing(tmp_path):
    pipe = _pipe(seed=1)
    for bad in ({"path": Path("x")}, {"score": float("nan")}, {"obj": object()}):
        target = tmp_path / "bad.safetensors"
        with pytest.raises(ValueError, match="plain JSON data"):
            pipe.save_artifact(target, metadata=bad)
        assert not target.exists()
