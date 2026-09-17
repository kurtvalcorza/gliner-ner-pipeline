from __future__ import annotations

import hashlib
import json
import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

MODEL_ID = "urchade/gliner_multi-v2.1"
MODEL_REVISION = "443d26d654e0324125a96bebd8e796c14ff2efe6"
MODEL_LICENSE = "apache-2.0"
MODEL_KEY = "gliner-multi-v2.1"
ARTIFACT_FORMAT = "org.valcorza.gliner-ner.adapter.v1"
ARTIFACT_FORMAT_VERSION = "1.0"
_WEIGHTS_ROOT = Path(__file__).resolve().parents[2] / "weights"
DEFAULT_WEIGHTS_DIR = _WEIGHTS_ROOT / MODEL_KEY
MANIFEST_NAME = "dimer-base-manifest.json"

# The GLiNER snapshot holds only gliner_config.json + model.safetensors; its `model_name` names the
# encoder whose tokenizer and config the gliner library resolves at load time. Those files are pinned
# here as a SECOND snapshot with its own manifest (no encoder weights: model.safetensors carries them),
# and the loader points the library at that verified directory instead of the Hub or an HF cache.
ENCODER_MODEL_ID = "microsoft/mdeberta-v3-base"
ENCODER_REVISION = "a0484667b22365f84929a935b5e50a51f71f159d"
ENCODER_LICENSE = "mit"
ENCODER_KEY = "mdeberta-v3-base-tokenizer"
ENCODER_WEIGHTS_DIR = _WEIGHTS_ROOT / ENCODER_KEY
DEFAULT_THRESHOLD = 0.5  # upstream predict_entities default; the caller owns tuning it
MAX_LABELS = 25  # gliner_config.json max_types: the most entity types seen per example in training
MAX_TEXT_CHARS = 5_000  # per call; gliner_config.json max_len is 384 words, longer text is cut by the library
MAX_LABEL_CHARS = 100


def _verify_manifest(root: Path, model_id: str, revision: str) -> dict[str, Any]:
    manifest_path = root / MANIFEST_NAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"snapshot manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("modelId") != model_id:
        raise ValueError(f"manifest modelId {manifest.get('modelId')!r} != {model_id!r}")
    if manifest.get("revision") != revision:
        raise ValueError(f"manifest revision {manifest.get('revision')!r} != {revision!r}")
    for entry in manifest["files"]:
        file_path = root / entry["path"]
        if not file_path.is_file():
            raise FileNotFoundError(f"snapshot file missing: {file_path}")
        size = file_path.stat().st_size
        if size != entry["bytes"]:
            raise ValueError(f"{entry['path']}: size {size} != manifest {entry['bytes']}")
        digest = hashlib.sha256()
        with open(file_path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                digest.update(chunk)
        if digest.hexdigest() != entry["sha256"]:
            raise ValueError(f"{entry['path']}: sha256 {digest.hexdigest()} != manifest {entry['sha256']}")
    return manifest


def verify_snapshot(path: str | Path | None = None) -> dict[str, Any]:
    """Check the GLiNER snapshot against its DIMER manifest; raise naming the first mismatch."""
    root = Path(path) if path is not None else DEFAULT_WEIGHTS_DIR
    return _verify_manifest(root, MODEL_ID, MODEL_REVISION)


def verify_encoder_snapshot(path: str | Path | None = None) -> dict[str, Any]:
    """Check the mDeBERTa tokenizer/config snapshot against its own manifest (second supply-chain check)."""
    root = Path(path) if path is not None else ENCODER_WEIGHTS_DIR
    return _verify_manifest(root, ENCODER_MODEL_ID, ENCODER_REVISION)


def _hub_download(
    relative_path: str, root: Path, model_id: str = MODEL_ID, revision: str = MODEL_REVISION
) -> None:
    """Fetch one manifest-listed file at the pinned revision straight into the snapshot directory."""
    from huggingface_hub import hf_hub_download

    hf_hub_download(model_id, relative_path, revision=revision, local_dir=str(root))


def _stage_missing(
    root: Path,
    model_id: str,
    revision: str,
    allow_download: bool,
    downloader: Callable[[str, Path], None] | None,
) -> list[str]:
    manifest_path = root / MANIFEST_NAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"manifest not found: {manifest_path}")
    with open(manifest_path, encoding="utf-8") as fh:
        manifest = json.load(fh)
    if manifest.get("modelId") != model_id or manifest.get("revision") != revision:
        raise ValueError(
            f"manifest names {manifest.get('modelId')}@{manifest.get('revision')}, "
            f"package pins {model_id}@{revision}; refusing to stage"
        )
    missing = [entry["path"] for entry in manifest["files"] if not (root / entry["path"]).is_file()]
    if not missing:
        return []
    if not allow_download:
        raise FileNotFoundError(
            f"snapshot at {root} is missing {missing}; pass allow_download=True to fetch them at {revision}"
        )
    fetch = downloader or (lambda rel, dst: _hub_download(rel, dst, model_id, revision))
    for relative_path in missing:
        fetch(relative_path, root)
    return missing


def stage_missing_files(
    path: str | Path | None = None,
    *,
    allow_download: bool = False,
    downloader: Callable[[str, Path], None] | None = None,
) -> list[str]:
    """Fetch GLiNER manifest entries that are absent locally (a fresh clone commits the manifest but
    git-ignores the weights). Returns the relative paths fetched; `verify_snapshot` still runs after."""
    root = Path(path) if path is not None else DEFAULT_WEIGHTS_DIR
    return _stage_missing(root, MODEL_ID, MODEL_REVISION, allow_download, downloader)


def stage_missing_encoder_files(
    path: str | Path | None = None,
    *,
    allow_download: bool = False,
    downloader: Callable[[str, Path], None] | None = None,
) -> list[str]:
    """Same as `stage_missing_files` for the mDeBERTa tokenizer/config snapshot at ENCODER_REVISION."""
    root = Path(path) if path is not None else ENCODER_WEIGHTS_DIR
    return _stage_missing(root, ENCODER_MODEL_ID, ENCODER_REVISION, allow_download, downloader)


def entity_f1(predicted: Sequence[dict[str, Any]], gold: Sequence[dict[str, Any]]) -> dict[str, float]:
    """Exact-span micro precision/recall/F1: a hit is an identical (start, end, label) triple."""
    pred_set = {(int(e["start"]), int(e["end"]), str(e["label"])) for e in predicted}
    gold_set = {(int(e["start"]), int(e["end"]), str(e["label"])) for e in gold}
    hits = len(pred_set & gold_set)
    precision = hits / len(pred_set) if pred_set else 0.0
    recall = hits / len(gold_set) if gold_set else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "hits": hits}


INPUT_SCHEMA: dict[str, Any] = {
    "input": "one text string plus 1..MAX_LABELS unique, non-empty, caller-supplied entity-type labels",
    "text_chars": [1, MAX_TEXT_CHARS],
    "labels": [1, MAX_LABELS],
    "label_chars": [1, MAX_LABEL_CHARS],
    "threshold": [0.0, 1.0],
    "preprocessing": (
        "the gliner library tokenizes with the pinned mDeBERTa-v3 tokenizer and truncates at "
        "gliner_config.json max_len = 384 words, so text beyond that is silently cut; returned spans are "
        "character offsets into the exact string you passed"
    ),
}


def _check_inputs(text: Any, labels: Any, threshold: Any) -> tuple[str, list[str], float]:
    """Raise TypeError/ValueError naming the first violated ceiling; return the checked request.

    ``GLiNERPipeline.detect`` and ``validate_inputs`` both route through this function so their
    acceptance criteria cannot diverge.
    """
    if not isinstance(text, str):
        raise TypeError(f"text must be str, got {type(text).__name__}")
    if not text.strip():
        raise ValueError("text is empty")
    if len(text) > MAX_TEXT_CHARS:
        raise ValueError(f"text has {len(text)} chars; ceiling is {MAX_TEXT_CHARS} (chunk it first)")
    if isinstance(labels, str | bytes) or not isinstance(labels, Sequence):
        raise TypeError("labels must be a list of str")
    if not 1 <= len(labels) <= MAX_LABELS:
        raise ValueError(f"labels must hold 1..{MAX_LABELS} items, got {len(labels)}")
    for i, label in enumerate(labels):
        if not isinstance(label, str) or not label.strip():
            raise TypeError(f"labels[{i}] must be a non-empty str")
        if len(label) > MAX_LABEL_CHARS:
            raise ValueError(f"labels[{i}] has {len(label)} chars; ceiling is {MAX_LABEL_CHARS}")
    if len(set(labels)) != len(labels):
        raise ValueError("labels must be unique")
    bad_type = isinstance(threshold, bool) or not isinstance(threshold, int | float)
    if bad_type or not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be a number in [0, 1]")
    return text, list(labels), float(threshold)


def validate_inputs(
    text: str,
    labels: Sequence[str],
    threshold: float = DEFAULT_THRESHOLD,
    *,
    names: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Validation stage: return the input manifest (schema, observations, request, verdict).

    Rejection is reported by raising exactly as ``detect`` would; a caller that wants the finding
    recorded catches the exception and stores ``str(exc)`` under ``findings``. Both pinned snapshot
    identities are recorded, because this pipeline verifies two (GLiNER and its mDeBERTa encoder).
    """
    checked_text, checked_labels, checked_threshold = _check_inputs(text, labels, threshold)
    if names is not None and len(names) != 1:
        raise ValueError("names must have exactly one entry (detect takes one text)")
    return {
        "schema": dict(INPUT_SCHEMA),
        "inputs": [
            {
                "id": names[0] if names else "text-0",
                "chars": len(checked_text),
                "words": len(checked_text.split()),
            }
        ],
        "labels": checked_labels,
        "threshold": checked_threshold,
        "verdict": "accepted",
        "findings": [],
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "encoder_model_id": ENCODER_MODEL_ID,
        "encoder_revision": ENCODER_REVISION,
    }


def evaluation_report(
    result: Mapping[str, Any],
    gold: Sequence[Mapping[str, Any]] | None = None,
    *,
    sample_kind: str = "synthetic",
) -> dict[str, Any]:
    """Evaluation stage: a machine-readable report even when nothing is measurable.

    With ``gold`` spans — dicts carrying ``start``, ``end`` and ``label`` — the report carries the
    repository's ``entity_f1`` (exact-span micro precision/recall/F1) as sample-sanity evidence;
    without them the verdict is ``not-measurable`` and the report says what labelled data would make
    the task measurable.
    """
    entities = list(result["entities"])
    base = {
        "task": "zero-shot named-entity recognition with a caller-supplied label set",
        "decision_rule": (
            f"a span is kept when its score reaches the caller's threshold "
            f"(default DEFAULT_THRESHOLD={DEFAULT_THRESHOLD}); the pipeline ships no tuned operating point"
        ),
        "score_semantics": (
            "each entity score is the model's own uncalibrated span score, not a probability that the "
            "span is correct"
        ),
        "threshold": result.get("threshold", DEFAULT_THRESHOLD),
        "labels": list(result.get("labels", [])),
        "sample_kind": sample_kind,
        "n_entities": len(entities),
        "baselines": [],
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "encoder_model_id": ENCODER_MODEL_ID,
        "encoder_revision": ENCODER_REVISION,
    }
    if gold is None:
        return {
            **base,
            "metrics": [],
            "verdict": "not-measurable",
            "reason": "no gold spans were supplied for the evaluated text",
            "needs": (
                "gold (start, end, label) spans over the same label set on text from your domain, scored "
                "with entity_f1 across enough documents to state a dispersion; exact-span matching also "
                "requires the annotation guideline to agree with the model's span boundaries"
            ),
        }
    gold_spans = list(gold)
    scores = entity_f1(entities, gold_spans)
    return {
        **base,
        "metrics": [
            {
                "id": "entity_f1",
                "value": scores["f1"],
                "precision": scores["precision"],
                "recall": scores["recall"],
                "hits": scores["hits"],
                "n_predicted": len(entities),
                "n_gold": len(gold_spans),
                "matching": "exact (start, end, label) triple",
                "estimation": "one text, no dispersion estimate",
            }
        ],
        "verdict": "sample-sanity",
        "reason": (
            f"{len(gold_spans)} gold span(s) on one tutorial text; exact-span sanity evidence, not an "
            "NER benchmark"
        ),
        "needs": (
            "a labelled span set from the deployment domain, with the same label vocabulary and the same "
            "boundary convention, for any generalisable precision/recall/F1 claim"
        ),
    }


def _local_encoder_class(root: Path, encoder_dir: Path) -> type:
    """The concrete GLiNER class for this config, with `model_name` redirected to the verified encoder
    directory so the library reads the tokenizer and AutoConfig from disk (local_files_only, no cache)."""
    from gliner import GLiNER

    config_dict = json.loads((root / "gliner_config.json").read_text(encoding="utf-8"))
    base = GLiNER._get_gliner_class(GLiNER._config_from_dict(config_dict))

    class LocalEncoderGLiNER(base):
        @classmethod
        def _load_config(cls, config_file: Path, **overrides: Any) -> Any:
            config = super()._load_config(config_file, **overrides)
            if config.model_name != ENCODER_MODEL_ID:
                raise ValueError(f"gliner_config.json names encoder {config.model_name!r}, not the pinned id")
            config.model_name = str(encoder_dir)
            return config

    return LocalEncoderGLiNER


@dataclass
class GLiNERPipeline:
    """Zero-shot and adapted NER pipeline. `_runner(text, labels, threshold)` returns gliner entity dicts."""

    _runner: Callable[[str, list[str], float], list[dict[str, Any]]]
    device: str
    load_warnings: list[str] = field(default_factory=list)
    model: Any = None

    @classmethod
    def from_pretrained(
        cls,
        device: str | None = None,
        weights_dir: str | Path | None = None,
        allow_download: bool = False,
        encoder_dir: str | Path | None = None,
    ) -> GLiNERPipeline:
        root = Path(weights_dir) if weights_dir is not None else DEFAULT_WEIGHTS_DIR
        enc = Path(encoder_dir) if encoder_dir is not None else ENCODER_WEIGHTS_DIR
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            if (root / MANIFEST_NAME).is_file():
                stage_missing_files(root, allow_download=allow_download)
                verify_snapshot(root)
                stage_missing_encoder_files(enc, allow_download=allow_download)
                verify_encoder_snapshot(enc)
                loader = _local_encoder_class(root, enc)
                model = loader.from_pretrained(
                    str(root), model_dir=str(root), local_files_only=True, map_location="cpu"
                )
            elif allow_download:
                from gliner import GLiNER  # Hub path: the library fetches the encoder assets unpinned

                model = GLiNER.from_pretrained(MODEL_ID, revision=MODEL_REVISION, map_location="cpu")
            else:
                raise FileNotFoundError(f"no verified snapshot at {root} and allow_download=False")
        # Validate and verify the snapshots before importing model libraries.
        import torch
        resolved_device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        model = model.to(resolved_device).eval()
        messages = [f"{w.category.__name__}: {w.message}" for w in caught]

        def runner(text: str, labels: list[str], threshold: float) -> list[dict[str, Any]]:
            with torch.inference_mode():
                return model.predict_entities(text, labels, threshold=threshold)

        return cls(runner, resolved_device, messages, model=model)

    def detect(
        self,
        text: str,
        labels: Sequence[str],
        threshold: float = DEFAULT_THRESHOLD,
    ) -> dict[str, Any]:
        """Extract spans for the caller-supplied `labels`; `threshold` is the upstream score cutoff."""
        text, labels, threshold = _check_inputs(text, labels, threshold)
        if self.model is not None:
            import torch

            with torch.inference_mode():
                raw = self.model.predict_entities(text, list(labels), threshold=threshold)
        else:
            raw = self._runner(text, list(labels), threshold)

        entities = []
        for e in raw:
            start, end = int(e["start"]), int(e["end"])
            if not 0 <= start < end <= len(text) or e["label"] not in labels:
                raise RuntimeError(f"backend returned an invalid entity: {e}")
            entities.append(
                {
                    "text": text[start:end],
                    "label": str(e["label"]),
                    "start": start,
                    "end": end,
                    "score": float(e["score"]),
                }
            )
        return {
            "entities": entities,
            "n_entities": len(entities),
            "labels": list(labels),
            "threshold": threshold,
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "encoder_model_id": ENCODER_MODEL_ID,
            "encoder_revision": ENCODER_REVISION,
        }

    def evaluate(
        self,
        records: Sequence[dict[str, Any]],
        labels: Sequence[str] | None = None,
        threshold: float = DEFAULT_THRESHOLD,
    ) -> dict[str, Any]:
        """Evaluate exact-span metrics over a dataset of NER records."""
        from .metrics import evaluate_ner_dataset
        from .samples import validate_dataset

        validate_dataset(records)
        if labels is None:
            extracted_labels = sorted({
                s["label"]
                for r in records
                for s in (r.get("spans") or r.get("entities", []))
            })
            if not extracted_labels:
                raise ValueError(
                    "Cannot infer labels: records contain no entity spans and labels was not provided"
                )
            eval_labels = extracted_labels
        else:
            eval_labels = list(labels)

        predictions: list[list[dict[str, Any]]] = []
        for r in records:
            res = self.detect(r["text"], eval_labels, threshold=threshold)
            predictions.append(res["entities"])

        metrics = evaluate_ner_dataset(predictions, records, eval_labels)
        return {
            "micro": {
                "precision": metrics["micro_precision"],
                "recall": metrics["micro_recall"],
                "f1": metrics["micro_f1"],
                "hits": metrics["hits"],
            },
            "macro": {
                "f1": metrics["macro_f1"],
            },
            "per_class": metrics["per_class"],
            "num_samples": len(records),
            "labels": eval_labels,
            "threshold": threshold,
        }

    def adapt(
        self,
        train_records: Sequence[dict[str, Any]],
        val_records: Sequence[dict[str, Any]] | None = None,
        *,
        epochs: int = 3,
        learning_rate: float = 5e-5,
        batch_size: int = 4,
        freeze_text_encoder: bool = True,
        weight_decay: float = 0.01,
        threshold: float = DEFAULT_THRESHOLD,
        seed: int = 42,
    ) -> dict[str, Any]:
        """Run bounded fine-tuning loop over domain records without external dependencies."""
        if self.model is None:
            raise RuntimeError("Cannot adapt: pipeline has no underlying PyTorch model loaded.")
        from .samples import validate_dataset

        validate_dataset(train_records)
        if not train_records:
            raise ValueError("train_records cannot be empty")
        if val_records is not None:
            validate_dataset(val_records)

        import random

        import torch

        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        if freeze_text_encoder:
            if hasattr(self.model, "freeze_component"):
                self.model.freeze_component("text_encoder")
            elif hasattr(self.model, "model") and hasattr(self.model.model, "token_rep_layer"):
                for p in self.model.model.token_rep_layer.parameters():
                    p.requires_grad = False

        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        if not trainable_params:
            raise RuntimeError("No trainable parameters found for adaptation.")

        optimizer = torch.optim.AdamW(trainable_params, lr=learning_rate, weight_decay=weight_decay)
        collator = self.model._create_data_collator()

        all_labels = sorted({
            span[2]
            for r in (list(train_records) + (list(val_records) if val_records else []))
            for span in r.get("ner", [])
        })

        history: list[dict[str, Any]] = []
        n_train = len(train_records)
        train_examples = [
            {"tokenized_text": r["tokenized_text"], "ner": r["ner"]}
            for r in train_records
        ]

        self.model.train()
        for epoch in range(1, epochs + 1):
            epoch_loss = 0.0
            num_batches = 0

            indices = list(range(n_train))
            random.shuffle(indices)

            for i in range(0, n_train, batch_size):
                batch_indices = indices[i : i + batch_size]
                batch_data = [train_examples[idx] for idx in batch_indices]
                batch = collator(batch_data)
                batch = {
                    k: v.to(self.device) if hasattr(v, "to") else v
                    for k, v in batch.items()
                }

                optimizer.zero_grad()
                output = self.model(**batch)
                loss = output.loss
                loss.backward()
                optimizer.step()

                epoch_loss += float(loss.item())
                num_batches += 1

            avg_loss = epoch_loss / max(1, num_batches)
            epoch_summary: dict[str, Any] = {
                "epoch": epoch,
                "loss": round(avg_loss, 6),
                "num_batches": num_batches,
            }

            if val_records:
                self.model.eval()
                val_res = self.evaluate(val_records, labels=all_labels, threshold=threshold)
                epoch_summary["val_f1"] = val_res["micro"]["f1"]
                epoch_summary["val_precision"] = val_res["micro"]["precision"]
                epoch_summary["val_recall"] = val_res["micro"]["recall"]
                self.model.train()

            history.append(epoch_summary)

        self.model.eval()

        final_eval: dict[str, Any] | None = None
        if val_records:
            final_eval = self.evaluate(val_records, labels=all_labels, threshold=threshold)

        return {
            "epochs": epochs,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "freeze_text_encoder": freeze_text_encoder,
            "train_samples": n_train,
            "val_samples": len(val_records) if val_records else 0,
            "history": history,
            "final_eval": final_eval,
        }

    def save_artifact(
        self,
        output_path: str | Path,
        metadata: dict[str, Any] | None = None,
    ) -> Path:
        """Export adapter state dict and lineage metadata to a .pt artifact."""
        if self.model is None:
            raise RuntimeError("Cannot save artifact: pipeline has no model loaded.")
        import torch

        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        trainable_names = {name for name, p in self.model.named_parameters() if p.requires_grad}
        if trainable_names:
            adapter_weights = {
                k: v.detach().cpu().clone()
                for k, v in self.model.state_dict().items()
                if k in trainable_names
            }
        else:
            adapter_weights = {
                k: v.detach().cpu().clone()
                for k, v in self.model.state_dict().items()
            }

        payload: dict[str, Any] = {
            "format": ARTIFACT_FORMAT,
            "format_version": ARTIFACT_FORMAT_VERSION,
            "base_model": {
                "model_id": MODEL_ID,
                "model_revision": MODEL_REVISION,
                "encoder_model_id": ENCODER_MODEL_ID,
                "encoder_revision": ENCODER_REVISION,
            },
            "adapter_state_dict": adapter_weights,
            "metadata": metadata or {},
        }
        torch.save(payload, path)
        return path

    def load_artifact(self, artifact_path: str | Path) -> dict[str, Any]:
        """Load adapter state dict into the current pipeline's model."""
        if self.model is None:
            raise RuntimeError("Cannot load artifact: pipeline has no model loaded.")
        import torch

        path = Path(artifact_path)
        if not path.is_file():
            raise FileNotFoundError(f"Artifact file not found: {path}")

        payload = torch.load(path, map_location=self.device, weights_only=True)

        if not isinstance(payload, dict):
            raise ValueError(f"Invalid artifact format: expected dict, got {type(payload).__name__}")
        if payload.get("format") != ARTIFACT_FORMAT:
            raise ValueError(
                f"Artifact format mismatch: {payload.get('format')!r} != {ARTIFACT_FORMAT!r}"
            )
        base = payload.get("base_model", {})
        if base.get("model_id") != MODEL_ID:
            raise ValueError(
                f"Artifact base model mismatch: {base.get('model_id')!r} != {MODEL_ID!r}"
            )

        weights = payload["adapter_state_dict"]
        weights = {
            k: v.to(self.device) if hasattr(v, "to") else v
            for k, v in weights.items()
        }
        self.model.load_state_dict(weights, strict=False)
        self.model.eval()
        return payload.get("metadata", {})

    @classmethod
    def from_artifact(
        cls,
        artifact_path: str | Path,
        device: str | None = None,
        weights_dir: str | Path | None = None,
        allow_download: bool = False,
        encoder_dir: str | Path | None = None,
    ) -> GLiNERPipeline:
        """Construct GLiNERPipeline from base weights and overlay adapter artifact weights."""
        pipeline = cls.from_pretrained(
            device=device,
            weights_dir=weights_dir,
            allow_download=allow_download,
            encoder_dir=encoder_dir,
        )
        pipeline.load_artifact(artifact_path)
        return pipeline
