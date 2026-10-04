"""Regression tests for the Notebook Review Framework v1 findings on `tutorials/gliner_ner_colab.ipynb`
(review PR #10: GL-M1..M4, GL-m1..m5).

The notebook's own cells are executed from the committed JSON in a namespace of the package's modules and inert
stand-ins (a fake `google.colab`, a fake pipeline). Nothing here loads the pinned checkpoint; CI installs neither
torch nor numpy, so the two artifact tests that need torch skip there and run in a torch environment.
"""
# ruff: noqa: E501  -- assertion messages and cell sources are kept on one line

from __future__ import annotations

import ast
import contextlib
import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest

# Windows conda trap (fleet note, bioclip2 row 6): import torch before anything else touches NumPy in this process.
with contextlib.suppress(ImportError):
    import torch  # noqa: F401

from gliner_ner_pipeline import (  # noqa: E402
    ADAPT_CLASSES,
    GLiNERPipeline,
    generate_synthetic_ner_dataset,
    load_byod_dataset,
    minimum_records,
    split_label_coverage,
    split_ner_dataset,
    validate_dataset,
)
from gliner_ner_pipeline import pipeline as pipeline_module  # noqa: E402
from gliner_ner_pipeline import samples as samples_module  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "tutorials" / "gliner_ner_colab.ipynb"

PEOPLE = [
    (["Ada", "Lovelace", "worked", "with", "Charles", "Babbage", "in", "London", "."], [[0, 1, "person"], [4, 5, "person"], [7, 7, "location"]]),
    (["Grace", "Hopper", "joined", "the", "United", "States", "Navy", "."], [[0, 1, "person"], [4, 6, "organization"]]),
    (["Marie", "Curie", "taught", "at", "the", "University", "of", "Paris", "."], [[0, 1, "person"], [5, 7, "organization"]]),
    (["Alan", "Turing", "worked", "at", "Bletchley", "Park", "in", "England", "."], [[0, 1, "person"], [4, 5, "location"], [7, 7, "location"]]),
    (["Rosalind", "Franklin", "studied", "at", "King's", "College", "London", "."], [[0, 1, "person"], [4, 6, "organization"]]),
    (["Nikola", "Tesla", "moved", "from", "Paris", "to", "New", "York", "."], [[0, 1, "person"], [4, 4, "location"], [6, 7, "location"]]),
    (["Katherine", "Johnson", "worked", "for", "NASA", "in", "Virginia", "."], [[0, 1, "person"], [4, 4, "organization"], [6, 6, "location"]]),
    (["Tim", "Berners-Lee", "worked", "at", "CERN", "near", "Geneva", "."], [[0, 1, "person"], [4, 4, "organization"], [6, 6, "location"]]),
    (["Jose", "Rizal", "studied", "medicine", "in", "Madrid", "."], [[0, 1, "person"], [5, 5, "location"]]),
    (["Linus", "Torvalds", "studied", "at", "the", "University", "of", "Helsinki", "."], [[0, 1, "person"], [5, 7, "organization"]]),
    (["Hedy", "Lamarr", "lived", "in", "Vienna", "before", "Hollywood", "."], [[0, 1, "person"], [4, 4, "location"], [6, 6, "location"]]),
    (["Wangari", "Maathai", "founded", "the", "Green", "Belt", "Movement", "in", "Kenya", "."], [[0, 1, "person"], [4, 6, "organization"], [8, 8, "location"]]),
    (["Fe", "del", "Mundo", "worked", "in", "Quezon", "City", "."], [[0, 2, "person"], [5, 6, "location"]]),
    (["Srinivasa", "Ramanujan", "travelled", "to", "Cambridge", "from", "Madras", "."], [[0, 1, "person"], [4, 4, "location"], [6, 6, "location"]]),
    (["Chien-Shiung", "Wu", "joined", "Columbia", "University", "in", "New", "York", "."], [[0, 1, "person"], [3, 4, "organization"], [6, 7, "location"]]),
    (["Jagadish", "Chandra", "Bose", "taught", "at", "Presidency", "College", "."], [[0, 2, "person"], [5, 6, "organization"]]),
]


def _jsonl(tmp_path: Path, name: str, rows, *, ner_as_objects: bool = False) -> Path:
    path = tmp_path / name
    lines = []
    for i, (tokens, ner) in enumerate(rows):
        spans = [{"start": s, "end": e, "label": lab} for s, e, lab in ner] if ner_as_objects else ner
        lines.append(json.dumps({"id": f"p-{i:02d}", "tokens": tokens, "ner": spans}))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _cells():
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"]


def _source(cell) -> str:
    return "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]


def _code_after(heading: str) -> str:
    cells = _cells()
    for i, cell in enumerate(cells):
        if cell["cell_type"] == "markdown" and heading in _source(cell):
            for nxt in cells[i + 1 :]:
                if nxt["cell_type"] == "code":
                    return _source(nxt)
    raise AssertionError(f"no code cell after {heading!r}")


def _markdown() -> str:
    return "\n".join(_source(c) for c in _cells() if c["cell_type"] == "markdown")


def _namespace() -> dict:
    """The kernel globals the carried module cells define (their public and private names)."""
    ns: dict = {}
    for module in (samples_module, pipeline_module):
        ns.update({k: v for k, v in vars(module).items() if not k.startswith("__")})
    return ns


def _fake_colab(monkeypatch, uploads: list[dict]):
    queue = list(uploads)
    files = types.ModuleType("google.colab.files")
    files.upload = lambda: queue.pop(0)
    colab = types.ModuleType("google.colab")
    colab.files = files
    google = types.ModuleType("google")
    google.colab = colab
    monkeypatch.setitem(sys.modules, "google", google)
    monkeypatch.setitem(sys.modules, "google.colab", colab)
    monkeypatch.setitem(sys.modules, "google.colab.files", files)


def _section4(monkeypatch, tmp_path, **fields) -> dict:
    """Execute Section 4 verbatim with USE_BYOD = True and the given form literals substituted."""
    monkeypatch.chdir(tmp_path)
    source = _code_after("## 4. Build, validate and split the sentences")
    source = source.replace("USE_BYOD = False", "USE_BYOD = True", 1)
    for key, value in fields.items():
        old = f"{key} = ''"
        assert old in source, key
        source = source.replace(old, f"{key} = {value!r}", 1)
    ns = _namespace()
    exec(compile(source, "<section 4>", "exec"), ns)
    return ns


# --- GL-M3: any label set through Section 4; schema and limits stated ----------------------------------------------


def test_sixteen_record_person_organization_location_jsonl_passes_section4_by_path(monkeypatch, tmp_path, capsys):
    path = _jsonl(tmp_path, "people.jsonl", PEOPLE)
    ns = _section4(monkeypatch, tmp_path, BYOD_PATH=str(path), BYOD_INFERENCE_TEXT="Hidilyn Diaz trained in Kuala Lumpur.")
    assert ns["LABELS"] == ["location", "organization", "person"]
    assert (len(ns["train_records"]), len(ns["val_records"])) == (12, 4)
    assert ns["data_source"] == "BYOD (people.jsonl)"
    assert ns["coverage"]["labels_missing_from_validation"] == []
    out = capsys.readouterr().out
    assert "'records_min': 14" in out and "record_schema" in out
    assert out.index("record_schema") < out.index("data_source")  # schema and limits printed before the data is read


def test_declared_labels_are_enforced_and_byod_labels_restrict_the_set(monkeypatch, tmp_path):
    path = _jsonl(tmp_path, "people.jsonl", PEOPLE)
    with pytest.raises(ValueError, match=r"label 'organization' not in \['location', 'person'\]"):
        _section4(monkeypatch, tmp_path, BYOD_PATH=str(path), BYOD_LABELS="person, location")


def test_sample_path_keeps_the_three_biomedical_labels():
    records = generate_synthetic_ner_dataset()
    train, val = split_ner_dataset(records, val_fraction=0.25, seed=42, labels=ADAPT_CLASSES)
    assert (len(train), len(val)) == (18, 6)
    assert split_label_coverage(train, val, ADAPT_CLASSES)["validation_spans"] == {"disease": 6, "chemical_drug": 6, "gene_protein": 6}


def test_evaluation_sets_need_not_contain_every_label():
    records = generate_synthetic_ner_dataset()[:4]
    for r in records:
        r["ner"] = [s for s in r["ner"] if s[2] != "gene_protein"]
    with pytest.raises(ValueError, match="missing examples for required labels"):
        validate_dataset(records, ADAPT_CLASSES)
    assert validate_dataset(records, ADAPT_CLASSES, require_all_labels=False)["label_distribution"]["gene_protein"] == 0


# --- GL-m5: failure messages name the rule and the fix, before Section 5 -------------------------------------------


def test_minimum_is_fourteen_records_at_the_default_fraction():
    assert minimum_records(0.25) == 14
    rows = [(t, n) for t, n in PEOPLE[:13]]
    split_ner_dataset([{"id": str(i), "ner": n} for i, (_t, n) in enumerate(PEOPLE[:14])], seed=42, labels=[])
    with pytest.raises(ValueError, match=r"the validation split would hold 3 records .* Supply at least 14 records"):
        split_ner_dataset([{"id": str(i), "ner": n} for i, (_t, n) in enumerate(rows)], seed=42, labels=[])


def test_ten_records_stop_in_section4_naming_the_split(monkeypatch, tmp_path):
    path = _jsonl(tmp_path, "small.jsonl", PEOPLE[:10])
    with pytest.raises(ValueError, match="the validation split would hold 2 records"):
        _section4(monkeypatch, tmp_path, BYOD_PATH=str(path))


def test_label_only_in_validation_is_named():
    records = [{"id": str(i), "ner": [[0, 0, "person"]]} for i in range(15)] + [{"id": "x", "ner": [[0, 0, "rare"]]}]
    seed = next(s for s in range(200) if any(r["id"] == "x" for r in split_ner_dataset(records, seed=s, labels=[])[1]))
    with pytest.raises(ValueError, match=r"label\(s\) \['rare'\] occur in no training record"):
        split_ner_dataset(records, seed=seed)


def test_csv_and_object_spans_and_bad_json_get_schema_messages(tmp_path):
    csv_file = tmp_path / "people.csv"
    csv_file.write_text("id,tokens\n1,a b\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"expects a \.json file .* or a \.jsonl file .*Convert the file"):
        load_byod_dataset(csv_file, allowed_labels=None)
    objects = _jsonl(tmp_path, "objects.jsonl", PEOPLE, ner_as_objects=True)
    with pytest.raises(ValueError, match=r"must be a list \[start_token, end_token, label\] \(inclusive end"):
        load_byod_dataset(objects, allowed_labels=None)
    broken = tmp_path / "broken.json"
    broken.write_text("[{", encoding="utf-8")
    with pytest.raises(ValueError, match="is not valid JSON .*one array of records"):
        load_byod_dataset(broken, allowed_labels=None)


def test_cancelled_upload_missing_path_and_no_colab_are_actionable(monkeypatch, tmp_path):
    _fake_colab(monkeypatch, [{}])
    with pytest.raises(ValueError, match=r"Upload exactly one \.json or \.jsonl file \(received 0\)"):
        _section4(monkeypatch, tmp_path)
    with pytest.raises(FileNotFoundError, match="is not a file in this runtime"):
        _section4(monkeypatch, tmp_path, BYOD_PATH=str(tmp_path / "nope.jsonl"))
    monkeypatch.setitem(sys.modules, "google.colab", None)
    with pytest.raises(RuntimeError, match="needs Google Colab. Elsewhere, .* set BYOD_PATH"):
        _section4(monkeypatch, tmp_path)


def test_uploaded_file_goes_through_section4(monkeypatch, tmp_path):
    payload = _jsonl(tmp_path, "src.jsonl", PEOPLE).read_bytes()
    _fake_colab(monkeypatch, [{"people.jsonl": payload}])
    ns = _section4(monkeypatch, tmp_path)
    assert ns["byod_path"] == Path("work") / "people.jsonl" and len(ns["raw_dataset"]) == 16


# --- GL-M2: Sections 5 and 6 start from the pretrained model ------------------------------------------------------


def test_adapt_refuses_an_already_adapted_pipeline():
    pipe = GLiNERPipeline(_runner=lambda *a: [], device="cpu", model=object(), adapted=True)
    with pytest.raises(RuntimeError, match="already adapted"):
        pipe.adapt(generate_synthetic_ner_dataset()[:6])


def test_sections_5_and_6_call_reset_first():
    assert _code_after("## 5. Measure the zero-shot baseline").startswith("reset_to_pretrained()\nbaseline_eval")
    section6 = _code_after("## 6. Adapt the layers above the frozen encoder")
    assert "reset_to_pretrained()\nstarted = time.perf_counter()\nadapt_result = pipe.adapt(" in section6


def test_reset_to_pretrained_reloads_only_an_adapted_pipeline(capsys):
    source = _code_after("### What the loader reported, and a clean starting point")
    loads = []

    class FakePipeline:
        @staticmethod
        def from_pretrained(**kwargs):
            loads.append(kwargs)
            return types.SimpleNamespace(adapted=False, load_warnings=[])

    fake_torch = types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: False))
    ns = {"pipe": types.SimpleNamespace(adapted=False, load_warnings=["w"]), "torch": fake_torch, "GLiNERPipeline": FakePipeline, "WEIGHTS_DIR": "w", "ENCODER_WEIGHTS_DIR": "e"}
    exec(compile(source, "<reset>", "exec"), ns)
    assert "'load_warnings': ['w']" in capsys.readouterr().out
    ns["reset_to_pretrained"]()
    assert loads == []
    ns["pipe"] = types.SimpleNamespace(adapted=True)
    ns["reset_to_pretrained"]()
    assert loads == [{"weights_dir": "w", "encoder_dir": "e"}] and ns["pipe"].adapted is False


def test_activity_names_the_cell_and_the_run_after_scope():
    md = _markdown()
    assert "## 11. Your turn — change one thing: the number of epochs" in md
    assert "In Section 6 set `EPOCHS = 1`" in md and "Select the Section 6 cell and choose **Runtime → Run after**" in md
    assert "re-run from that cell" not in md


# --- GL-m2: adapter manifest and refusals (torch) ------------------------------------------------------------------


def _mock_pipe():
    torch = pytest.importorskip("torch")
    nn = torch.nn

    class Mock(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(4, 2)
            self.frozen = nn.Linear(4, 4)
            for p in self.frozen.parameters():
                p.requires_grad = False

    return torch, GLiNERPipeline(_runner=lambda *a: [], device="cpu", model=Mock())


def test_tampered_adapters_are_refused_with_named_errors(tmp_path):
    torch, pipe = _mock_pipe()
    path = pipe.save_artifact(tmp_path / "a.pt")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    assert sorted(payload["adapter_manifest"]) == ["fc.bias", "fc.weight"] and payload["format_version"] == "1.1"
    cases = {
        "empty or missing": {**payload, "adapter_state_dict": {}, "adapter_manifest": {}},
        "does not have": {**payload, "adapter_state_dict": {"x." + k: v for k, v in payload["adapter_state_dict"].items()}},
        "model_revision mismatch": {**payload, "base_model": {**payload["base_model"], "model_revision": "0" * 40}},
        "encoder_revision mismatch": {**payload, "base_model": {**payload["base_model"], "encoder_revision": "0" * 40}},
        "differ from manifest": {**payload, "adapter_state_dict": {k: v + 1 for k, v in payload["adapter_state_dict"].items()}},
    }
    for message, tampered in cases.items():
        bad = tmp_path / "bad.pt"
        torch.save(tampered, bad)
        fresh = _mock_pipe()[1]
        with pytest.raises(ValueError, match=message):
            fresh.load_artifact(bad)
        assert fresh.adapted is False


def test_reload_marks_adapted_and_tensor_parity_is_exact(tmp_path):
    _torch, pipe = _mock_pipe()
    path = pipe.save_artifact(tmp_path / "a.pt")
    fresh = _mock_pipe()[1]
    assert fresh.adapter_parity(pipe, ["fc.weight", "fc.bias"])["tensors_identical"] == 0
    fresh.load_artifact(path)
    assert fresh.adapted is True
    assert fresh.adapter_parity(pipe, ["fc.weight", "fc.bias"]) == {"tensors_compared": 2, "tensors_identical": 2, "differing": []}


# --- GL-M1 / GL-M4 / GL-m1: isolated runtime, guided layer, records ------------------------------------------------


def test_exactly_two_kernel_cells_and_no_restart_text():
    kernel = [c for c in _cells() if c["cell_type"] == "code" and "# dimer: kernel cell" in _source(c)]
    assert len(kernel) == 2
    install = _source(kernel[0])
    assert "--require-hashes" in install and "--managed-python" in install and "LOCK_SHA256" in install
    md = _markdown()
    assert "Restart the runtime" not in md and "its restart" not in md and "installs the pinned dependencies" not in md
    tutorials = (ROOT / "tutorials" / "README.md").read_text(encoding="utf-8")
    assert "verified — clean-runtime" not in tutorials and "needed a manual restart" in tutorials


@pytest.mark.parametrize("real_google", [False, True])
def test_worker_colab_stubs_have_specs(monkeypatch, real_google):
    """Colab only: accelerate calls importlib.util.find_spec("google.colab"), which raises on a spec-less stub."""
    router = [_source(c) for c in _cells() if c["cell_type"] == "code"][1]
    worker = next(
        node.value.value
        for node in ast.parse(router).body
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") == "_WORKER_SOURCE"
    )
    start = worker.index('if os.environ.get("DIMER_KERNEL_IS_COLAB") == "1":')
    shim = worker[start : worker.index('_main = types.ModuleType("__main__")', start)]
    fake_google = types.ModuleType("google")
    fake_google.__path__ = []
    monkeypatch.setitem(sys.modules, "google", fake_google if real_google else None)
    monkeypatch.delitem(sys.modules, "google.colab", raising=False)
    monkeypatch.delitem(sys.modules, "google.colab.files", raising=False)
    monkeypatch.setenv("DIMER_KERNEL_IS_COLAB", "1")
    try:
        exec(compile(shim, "worker-colab-shim", "exec"), {"os": __import__("os"), "sys": sys, "types": types, "_send": None, "_recv": None})
        for name in ("google.colab", "google.colab.files"):
            spec = importlib.util.find_spec(name)
            assert spec is not None and spec.name == name
        assert sys.modules["google.colab"].__path__ == [] and callable(sys.modules["google.colab.files"].upload)
        if not real_google:
            assert importlib.util.find_spec("google") is not None
    finally:
        for name in ("google", "google.colab", "google.colab.files"):
            sys.modules.pop(name, None)  # monkeypatch then restores whatever was there before


def test_guided_layer_and_infrastructure_labels():
    md = _markdown()
    for marker, least in (
        ("**Who this is for.**", 1),
        ("**Input → Model → Output.**", 1),
        ("**How to use this notebook.**", 1),
        ("**Roadmap:**", 1),
        ("**Predict before running:**", 6),
        ("**What to notice:**", 7),
        ("<summary>Check your reasoning</summary>", 7),
        ("## Troubleshooting", 1),
        ("## Glossary", 1),
        ("## Conclusion (your notes)", 1),
    ):
        assert md.count(marker) >= least, marker
    code = [c for c in _cells() if c["cell_type"] == "code"]
    setup = code[:7]  # install, router, runtime record, 3 carried modules, model
    assert all(c["metadata"].get("cellView") == "form" and _source(c).startswith("# @title Infrastructure: ") for c in setup)
    assert "@@" not in md


def test_claims_match_behaviour_split_trainable_warnings_and_evidence():
    md = _markdown()
    assert "class coverage preserved" not in md and "class balance preserved" not in md
    assert "only the span representation and prompt projection layers are updated" not in md
    assert "`token_rep_layer` projection), the BiLSTM (`rnn`)" in md
    assert "is captured in `load_warnings`" not in md and "parameter match" not in md
    assert "**tutorial evidence, not a benchmark**" in md and "6 validation sentences" in md
    section7 = _code_after("## 7. Compare before and after")
    assert "print('Reading: ' + evidence_note + '.')" in section7


def test_no_learner_cell_uses_a_bare_assert():
    learner = [_source(c) for c in _cells() if c["cell_type"] == "code" and "embedded_module" not in c["metadata"].get("dimer", {}) and "# dimer: kernel cell" not in _source(c)]
    assert not any(line.lstrip().startswith("assert ") for src in learner for line in src.splitlines())


def test_release_documents_describe_this_notebook():
    verification = (ROOT / "docs" / "release-verification.md").read_text(encoding="utf-8")
    for stale in ("TASK-INFERENCE", "not-measurable", "GOLD_JSON", "THRESHOLD = 0.5", "input_manifest.json"):
        assert stale not in verification, stale
    assert "only after a manual restart" in verification and "Specification 2.2" in verification
    for rel in ("README.md", "STATUS.md"):
        text = (ROOT / rel).read_text(encoding="utf-8")
        assert "manual restart" in text and "never been executed" not in text, rel
