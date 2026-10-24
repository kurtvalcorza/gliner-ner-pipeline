"""Static guard (no torch needed, so it runs in CI): the package and the tutorial notebooks never write
a pickle artifact or read one without the restricted loader (ARTIFACT_SPEC 2.0 SER7, SER12).

The torch-backed round-trip tests in test_artifact_safetensors.py are skipped in CI, which installs no
torch; this test is what catches a regression of save_artifact to ``torch.save`` there.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN = {
    "torch.save": re.compile(r"\btorch\.save\s*\("),
    "pickle dump/load": re.compile(r"\bpickle\.(?:dumps?|loads?)\s*\("),
    "unrestricted torch.load": re.compile(r"\btorch\.load\s*\((?![^)]*weights_only\s*=\s*True)"),
}


def _sources():
    for path in sorted((ROOT / "src").rglob("*.py")):
        yield path, path.read_text(encoding="utf-8")
    for path in sorted((ROOT / "tutorials").glob("*.ipynb")):
        nb = json.loads(path.read_text(encoding="utf-8"))
        code = "\n".join(
            "".join(cell.get("source", [])) for cell in nb.get("cells", []) if cell.get("cell_type") == "code"
        )
        yield path, code


def test_no_pickle_serialization_in_package_or_notebooks():
    hits = [
        f"{path.relative_to(ROOT).as_posix()}: {label}"
        for path, text in _sources()
        for label, pattern in FORBIDDEN.items()
        if pattern.search(text)
    ]
    assert not hits, hits


def test_the_guard_detects_a_torch_save_regression():
    assert FORBIDDEN["torch.save"].search("torch.save(payload, path)")
    assert FORBIDDEN["unrestricted torch.load"].search("torch.load(path, map_location='cpu')")
    assert not FORBIDDEN["unrestricted torch.load"].search("torch.load(path, weights_only=True)")
