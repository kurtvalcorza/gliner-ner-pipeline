"""DAT19/DAT20: a refused BYOD input shows one `Refused: <file>: <reason>` line and no traceback.

Two halves, both read from the generated notebook:
- the router (Section 1, template `dat20-router-refusal/1` from ml-worker
  `integrations/dimer/fleet-specs/notebook-router/`) relays an exception carrying `dimer_refusal` as one
  line, while every other exception keeps its traceback;
- the BYOD cells turn every user-input problem into `BYODRefusal` and remove the earlier run's results and
  `outputs/` files before they stop.
"""

# ruff: noqa: E501  -- refusal messages and parametrized cases read best on one line
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import types
from multiprocessing.connection import Connection
from pathlib import Path

import pytest

from gliner_ner_pipeline import pipeline as pipeline_module
from gliner_ner_pipeline import samples as samples_module

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "tutorials/gliner_ner_colab.ipynb"
BYOD_CELL = "gliner_ner-19"
CARRIED = (
    "class BYODRefusal(ValueError):\n"
    '    """A refused BYOD input: ``BYODRefusal(file_name, reason_and_fix)``."""\n\n'
    "    dimer_refusal = True\n\n"
    "    def __init__(self, file_name, reason):\n"
    '        super().__init__(f"{file_name}: {reason}")\n'
    "        self.file_name = file_name\n"
    "        self.reason = reason\n\n"
    "    def _render_traceback_(self):\n"
    '        return [f"Refused: {self}"]\n'
)


def _code_cells(path: Path = NOTEBOOK) -> list[dict]:
    return [c for c in json.loads(path.read_text(encoding="utf-8"))["cells"] if c["cell_type"] == "code"]


def _source(cell: dict) -> str:
    return "".join(cell["source"])


def _cell(cell_id: str, path: Path = NOTEBOOK) -> str:
    return next(_source(c) for c in _code_cells(path) if c.get("id") == cell_id)


# --- router, kernel side (portable) -------------------------------------------------------------------------------


def _router_namespace(start: str) -> dict:
    source = next(_source(c) for c in _code_cells() if '_WORKER_SOURCE = r"""' in _source(c))
    chunk = source[source.index(start) : source.index("def _is_user_cell():")]
    namespace = {"os": os, "sys": sys, "signal": signal, "subprocess": subprocess, "Connection": Connection, "__name__": "router"}
    exec(compile(chunk, "<router>", "exec"), namespace)
    return namespace


class _Pipe:
    def __init__(self, messages=()):
        self.messages, self.sent = list(messages), []

    def recv(self):
        return self.messages.pop(0)

    def send(self, message):
        self.sent.append(message)


def _kernel_runtime(messages):
    namespace = _router_namespace("class IsolatedCellError(RuntimeError):")
    runtime = object.__new__(namespace["IsolatedRuntime"])
    runtime._send, runtime._recv, runtime._display = _Pipe(), _Pipe(messages), lambda *a, **k: None
    return namespace, runtime


def test_router_relays_a_refusal_as_one_line_without_a_traceback(capsys):
    """Acceptance: fails on main, whose router has no refusal channel."""
    namespace, runtime = _kernel_runtime([("refusal", "hostname: not readable")])
    with pytest.raises(namespace["IsolatedCellRefusal"]) as caught:
        runtime.run("x = 1\n")
    assert caught.value._render_traceback_() == ["Refused: hostname: not readable"]
    assert caught.value.__cause__ is None and caught.value.__suppress_context__
    assert capsys.readouterr().err == ""
    assert runtime._send.sent == [("run", "x = 1\n")]


def test_router_keeps_the_traceback_of_a_genuine_error(capsys):
    namespace, runtime = _kernel_runtime([("error", "Traceback (most recent call last):\nRuntimeError: bug\n", "RuntimeError: bug")])
    with pytest.raises(namespace["IsolatedCellError"], match="RuntimeError: bug"):
        runtime.run("x = 1\n")
    assert "Traceback (most recent call last)" in capsys.readouterr().err


def test_router_normal_cell_output_is_unchanged(capsys):
    _, runtime = _kernel_runtime([("stream", "stdout", "hello\n"), ("done",)])
    assert runtime.run("print('hello')\n") is None
    assert capsys.readouterr() == ("hello\n", "")


def test_router_template_is_applied():
    source = next(_source(c) for c in _code_cells() if '_WORKER_SOURCE = r"""' in _source(c))
    assert source.index('elif kind == "refusal":') < source.index('elif kind == "error":')
    assert 'if getattr(exc, "dimer_refusal", False):' in source
    header = "Its router refusal channel (DAT20) is applied from ml-worker integrations/dimer/fleet-specs/notebook-router/."
    assert header in (ROOT / "tools" / "build_notebook.py").read_text(encoding="utf-8")


@pytest.mark.skipif(os.name == "nt", reason="the isolated-cell router is POSIX-only (pass_fds, start_new_session)")
def test_router_worker_end_to_end(capsys):
    namespace = _router_namespace('_WORKER_SOURCE = r"""')
    runtime = namespace["IsolatedRuntime"](sys.executable, display=lambda *a, **k: None)
    try:
        runtime.run("class R(ValueError):\n    dimer_refusal = True\nkept = 41\n")
        with pytest.raises(namespace["IsolatedCellRefusal"]) as caught:
            runtime.run("raise R('hostname: not readable')\n")
        assert caught.value._render_traceback_() == ["Refused: hostname: not readable"]
        assert "Traceback" not in capsys.readouterr().err
        runtime.run("print(kept + 1)\n")  # the worker namespace survives a refusal
        assert capsys.readouterr().out == "42\n"
        with pytest.raises(namespace["IsolatedCellError"], match="RuntimeError: bug"):
            runtime.run("raise RuntimeError('bug')\n")
        assert "Traceback (most recent call last)" in capsys.readouterr().err
    finally:
        runtime.close()


def test_byod_refusal_is_carried_verbatim():
    assert CARRIED in _cell(BYOD_CELL)


def _refusal_line(exc: BaseException) -> str:
    assert getattr(exc, "dimer_refusal", False), f"not a refusal: {exc!r}"
    (line,) = exc._render_traceback_()
    assert "\n" not in line and "Traceback" not in line
    return line


package = samples_module

STEM = "gliner_ner"
EARLIER = {"raw_dataset": "old", "train_records": "old", "val_records": "old", "inference_result": "old", "adapt_result": "old", "result_payload": "old", "LABELS": "old"}


def _namespace() -> dict:
    """The kernel globals the carried module cells define (their public and private names)."""
    ns: dict = {}
    for module in (samples_module, pipeline_module):
        ns.update({k: v for k, v in vars(module).items() if not k.startswith("__")})
    ns.update({"pipe": "kept", "__name__": "__main__"})
    return ns


def _seed_outputs(tmp_path: Path) -> None:
    out = tmp_path / "outputs"
    out.mkdir()
    for name in ("evaluation_report.json", "adapter.safetensors", "entities.csv", "result.json"):
        (out / f"{STEM}_{name}").write_text("x", encoding="utf-8")


def _section4(monkeypatch, tmp_path, path: str = "", *, use_byod: bool = True, source: str | None = None, extra: dict | None = None, text: str = "") -> dict:
    monkeypatch.chdir(tmp_path)
    _seed_outputs(tmp_path)
    source = source or _cell(BYOD_CELL)
    if use_byod:
        source = source.replace("USE_BYOD = False", "USE_BYOD = True", 1)
    source = source.replace("BYOD_PATH = ''", f"BYOD_PATH = {path!r}", 1).replace("BYOD_INFERENCE_TEXT = ''", f"BYOD_INFERENCE_TEXT = {text!r}", 1)
    ns = _namespace()
    ns.update(EARLIER, **(extra or {}))
    exec(compile(source, "<section 4>", "exec"), ns)
    return ns


def _good_records() -> list:
    return [{"id": r["id"], "tokenized_text": r["tokenized_text"], "ner": r["ner"]} for r in package.generate_synthetic_ner_dataset()]


def _cases(tmp_path: Path) -> dict:
    (tmp_path / "hostname").write_text("runtime\n", encoding="utf-8")
    (tmp_path / "bad.json").write_text("[{", encoding="utf-8")
    (tmp_path / "empty.jsonl").write_text("", encoding="utf-8")
    (tmp_path / "folder").mkdir()
    few = _good_records()[:3]
    (tmp_path / "few.json").write_text(json.dumps(few), encoding="utf-8")
    objects = [dict(r, ner=[{"start": s, "end": e, "label": lab} for s, e, lab in r["ner"]]) for r in _good_records()]
    (tmp_path / "objects.json").write_text(json.dumps(objects), encoding="utf-8")
    return {
        "not a dataset suffix (the scripted /etc/hostname run)": ("hostname", "BYOD expects a .json file (one JSON array of records) or a .jsonl file"),
        "invalid JSON": ("bad.json", "is not valid JSON"),
        "empty file": ("empty.jsonl", "is empty"),
        "a folder": ("folder", "is not a file in this runtime"),
        "missing": ("nope.jsonl", "is not a file in this runtime"),
        "too few records": ("few.json", "Error"),
        "spans written as objects": ("objects.json", "must be a list [start_token, end_token, label]"),
    }


CASES = ["not a dataset suffix (the scripted /etc/hostname run)", "invalid JSON", "empty file", "a folder", "missing", "too few records", "spans written as objects"]


@pytest.mark.parametrize("case", CASES)
def test_byod_problems_are_one_refused_line_and_remove_earlier_results(monkeypatch, tmp_path, case):
    monkeypatch.setitem(sys.modules, "google.colab", None)
    file_name, rule = _cases(tmp_path)[case]
    with pytest.raises(ValueError) as caught:
        _section4(monkeypatch, tmp_path, str(tmp_path / file_name))
    line = _refusal_line(caught.value)
    assert line.startswith(f"Refused: {file_name}: ") and rule in line, line
    assert "again" in line and "earlier results of this session and their outputs/ files were removed" in line
    assert sorted(p.name for p in (tmp_path / "outputs").iterdir()) == []


def test_refusal_clears_the_session(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "google.colab", None)
    monkeypatch.chdir(tmp_path)
    source = _cell(BYOD_CELL).replace("USE_BYOD = False", "USE_BYOD = True", 1).replace("BYOD_PATH = ''", "BYOD_PATH = 'nope.jsonl'", 1)
    ns = _namespace()
    ns.update(EARLIER)
    with pytest.raises(ValueError, match="^nope.jsonl: BYOD_PATH = 'nope.jsonl' is not a file"):
        exec(compile(source, "<section 4>", "exec"), ns)
    assert not set(EARLIER) & set(ns), "an earlier run's results survived the refusal"
    assert ns["pipe"] == "kept"


def test_too_long_inference_text_is_one_refused_line(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "google.colab", None)
    (tmp_path / "good.json").write_text(json.dumps(_good_records()), encoding="utf-8")
    with pytest.raises(ValueError) as caught:
        _section4(monkeypatch, tmp_path, str(tmp_path / "good.json"), text="x" * (pipeline_module.MAX_TEXT_CHARS + 1))
    assert _refusal_line(caught.value).startswith("Refused: BYOD_INFERENCE_TEXT: the text has")


def test_upload_outside_colab_is_one_refused_line(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "google.colab", None)
    with pytest.raises(ValueError) as caught:
        _section4(monkeypatch, tmp_path)
    assert _refusal_line(caught.value).startswith("Refused: (no file): the upload dialog needs Google Colab")


@pytest.mark.parametrize(("uploaded", "file_name", "rule"), [({}, "(no file)", "received 0"), ({"a.json": b"1", "b.json": b"2"}, "a.json, b.json", "received 2")])
def test_cancelled_or_multiple_upload_is_one_refused_line(monkeypatch, tmp_path, uploaded, file_name, rule):
    colab = types.ModuleType("google.colab")
    colab.files = types.SimpleNamespace(upload=lambda: uploaded)
    monkeypatch.setitem(sys.modules, "google.colab", colab)
    with pytest.raises(ValueError) as caught:
        _section4(monkeypatch, tmp_path)
    line = _refusal_line(caught.value)
    assert line.startswith(f"Refused: {file_name}: upload exactly one") and rule in line


def test_valid_byod_file_still_splits(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "google.colab", None)
    (tmp_path / "good.json").write_text(json.dumps(_good_records()), encoding="utf-8")
    ns = _section4(monkeypatch, tmp_path, str(tmp_path / "good.json"))
    assert ns["data_source"] == "BYOD (good.json)" and ns["train_records"] and ns["val_records"]


def test_genuine_internal_error_keeps_its_traceback(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "google.colab", None)
    (tmp_path / "good.json").write_text(json.dumps(_good_records()), encoding="utf-8")

    def broken(*args, **kwargs):
        raise RuntimeError("internal bug")

    with pytest.raises(RuntimeError, match="internal bug") as caught:
        _section4(monkeypatch, tmp_path, str(tmp_path / "good.json"), extra={"load_byod_dataset": broken})
    assert not getattr(caught.value, "dimer_refusal", False)
    assert (tmp_path / "outputs" / f"{STEM}_result.json").exists()  # nothing is removed for an internal error


def test_sample_path_errors_are_not_refusals(monkeypatch, tmp_path):
    def broken(*args, **kwargs):
        raise ValueError("sample generator bug")

    with pytest.raises(ValueError, match="sample generator bug") as caught:
        _section4(monkeypatch, tmp_path, use_byod=False, extra={"generate_synthetic_ner_dataset": broken})
    assert not getattr(caught.value, "dimer_refusal", False)


def test_default_path_is_unchanged(monkeypatch, tmp_path):
    ns = _section4(monkeypatch, tmp_path, use_byod=False)
    assert ns["data_source"].startswith("Synthetic") and ns["LABELS"] == list(package.ADAPT_CLASSES)


@pytest.mark.skipif(not Path("/etc/hostname").is_file(), reason="needs a POSIX /etc/hostname, as in a fresh Colab runtime")
def test_scripted_refusal_input_is_the_real_etc_hostname(monkeypatch, tmp_path):
    """The hosted refusal run sets USE_BYOD=True;BYOD_PATH='/etc/hostname': a file present in every fresh runtime."""
    monkeypatch.setitem(sys.modules, "google.colab", None)
    with pytest.raises(ValueError) as caught:
        _section4(monkeypatch, tmp_path, "/etc/hostname")
    assert _refusal_line(caught.value).startswith("Refused: hostname: ValueError: BYOD expects a .json file")
