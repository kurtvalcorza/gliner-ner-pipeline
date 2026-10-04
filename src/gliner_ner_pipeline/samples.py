"""Deterministic in-code sample data and contracts for GLiNER multi-v2.1 domain adaptation.

Provides deterministic synthetic domain NER datasets (biomedical/clinical entities: disease,
chemical_drug, gene_protein) and BYOD validation routines conforming to DIMER NOTEBOOK_SPEC 2.2.
The sample uses the three biomedical labels; a BYOD dataset may use any label set (`allowed_labels=None`
infers it from the records), and every declared label must occur in the training split.
"""
# ruff: noqa: E501

from __future__ import annotations

import json
import random
from collections.abc import Sequence
from pathlib import Path
from typing import Any

# Canonical biomedical / clinical NER adaptation vocabulary
ADAPT_CLASSES: tuple[str, ...] = ("disease", "chemical_drug", "gene_protein")
DATASET_REPRESENTATION = "io.github.kurtvalcorza.dataset.nlp.ner-spans.v1"

# Pinned ceilings from pipeline.py / gliner_config.json
MIN_DATASET_EXAMPLES = 4  # per split: the training and the validation split each need at least this many records
MAX_DATASET_EXAMPLES = 1_000
MAX_TOKENS_PER_EXAMPLE = 384
MAX_TOKEN_CHARS = 100
BYOD_SUFFIXES = (".json", ".jsonl")
RECORD_SCHEMA_HINT = (
    "each record is {id: str, tokenized_text (or tokens): [str, ...], ner: [[start_token, end_token, label], ...]} "
    "with token indices counted from 0 and an inclusive end"
)

TUTORIAL_TEXT = "Marie Curie conducted pioneering research on radioactivity in Paris and Warsaw."
TUTORIAL_LABELS = ["person", "location", "scientific_field"]
TUTORIAL_SPANS = [
    {"start": 0, "end": 11, "label": "person", "text": "Marie Curie"},
    {"start": 62, "end": 67, "label": "location", "text": "Paris"},
    {"start": 72, "end": 78, "label": "location", "text": "Warsaw"},
]

# 24 deterministic domain sentences covering disease, chemical_drug, gene_protein
_RAW_DOMAIN_EXAMPLES: list[dict[str, Any]] = [
    {
        "id": "bio-000",
        "tokens": ["Aspirin", "inhibits", "PTGS2", "to", "reduce", "inflammation", "and", "pain", "."],
        "ner": [
            [0, 0, "chemical_drug"],
            [2, 2, "gene_protein"],
            [5, 5, "disease"],
            [7, 7, "disease"],
        ],
    },
    {
        "id": "bio-001",
        "tokens": ["Imatinib", "targets", "BCR-ABL1", "fusion", "in", "chronic", "myeloid", "leukemia", "."],
        "ner": [
            [0, 0, "chemical_drug"],
            [2, 2, "gene_protein"],
            [5, 7, "disease"],
        ],
    },
    {
        "id": "bio-002",
        "tokens": ["Trastuzumab", "binds", "ERBB2", "receptors", "overexpressed", "in", "breast", "cancer", "."],
        "ner": [
            [0, 0, "chemical_drug"],
            [2, 2, "gene_protein"],
            [6, 7, "disease"],
        ],
    },
    {
        "id": "bio-003",
        "tokens": ["Metformin", "activates", "PRKAA1", "for", "glycemic", "control", "in", "type", "2", "diabetes", "."],
        "ner": [
            [0, 0, "chemical_drug"],
            [2, 2, "gene_protein"],
            [7, 9, "disease"],
        ],
    },
    {
        "id": "bio-004",
        "tokens": ["Erlotinib", "inhibits", "mutant", "EGFR", "in", "non-small", "cell", "lung", "carcinoma", "."],
        "ner": [
            [0, 0, "chemical_drug"],
            [3, 3, "gene_protein"],
            [5, 8, "disease"],
        ],
    },
    {
        "id": "bio-005",
        "tokens": ["Pembrolizumab", "blocks", "PDCD1", "pathway", "activation", "in", "metastatic", "melanoma", "."],
        "ner": [
            [0, 0, "chemical_drug"],
            [2, 2, "gene_protein"],
            [6, 7, "disease"],
        ],
    },
    {
        "id": "bio-006",
        "tokens": ["Rituximab", "targets", "MS4A1", "surface", "antigen", "on", "cells", "in", "lymphoma", "."],
        "ner": [
            [0, 0, "chemical_drug"],
            [2, 2, "gene_protein"],
            [8, 8, "disease"],
        ],
    },
    {
        "id": "bio-007",
        "tokens": ["Olaparib", "inhibits", "PARP1", "in", "tumors", "harboring", "mutant", "BRCA1", "ovarian", "cancer", "."],
        "ner": [
            [0, 0, "chemical_drug"],
            [2, 2, "gene_protein"],
            [7, 7, "gene_protein"],
            [8, 9, "disease"],
        ],
    },
    {
        "id": "bio-008",
        "tokens": ["Vemurafenib", "selectively", "inhibits", "mutated", "BRAF", "kinase", "in", "cutaneous", "melanoma", "."],
        "ner": [
            [0, 0, "chemical_drug"],
            [4, 4, "gene_protein"],
            [7, 8, "disease"],
        ],
    },
    {
        "id": "bio-009",
        "tokens": ["Dabrafenib", "combined", "with", "Trametinib", "delays", "resistance", "in", "thyroid", "carcinoma", "."],
        "ner": [
            [0, 0, "chemical_drug"],
            [3, 3, "chemical_drug"],
            [7, 8, "disease"],
        ],
    },
    {
        "id": "bio-010",
        "tokens": ["Bevacizumab", "binds", "circulating", "VEGFA", "to", "suppress", "angiogenesis", "in", "glioblastoma", "."],
        "ner": [
            [0, 0, "chemical_drug"],
            [3, 3, "gene_protein"],
            [8, 8, "disease"],
        ],
    },
    {
        "id": "bio-011",
        "tokens": ["Sunitinib", "inhibits", "KDR", "and", "PDGFRB", "in", "renal", "cell", "carcinoma", "."],
        "ner": [
            [0, 0, "chemical_drug"],
            [2, 2, "gene_protein"],
            [4, 4, "gene_protein"],
            [6, 8, "disease"],
        ],
    },
    {
        "id": "bio-012",
        "tokens": ["Paclitaxel", "stabilizes", "TUBB", "microtubules", "to", "induce", "apoptosis", "in", "pancreatic", "cancer", "."],
        "ner": [
            [0, 0, "chemical_drug"],
            [2, 2, "gene_protein"],
            [8, 9, "disease"],
        ],
    },
    {
        "id": "bio-013",
        "tokens": ["Cisplatin", "forms", "DNA", "crosslinks", "inducing", "cell", "death", "in", "testicular", "cancer", "."],
        "ner": [
            [0, 0, "chemical_drug"],
            [8, 9, "disease"],
        ],
    },
    {
        "id": "bio-014",
        "tokens": ["Doxorubicin", "intercalates", "DNA", "and", "inhibits", "TOP2A", "in", "sarcoma", "."],
        "ner": [
            [0, 0, "chemical_drug"],
            [5, 5, "gene_protein"],
            [7, 7, "disease"],
        ],
    },
    {
        "id": "bio-015",
        "tokens": ["Tamoxifen", "antagonizes", "ESR1", "signaling", "in", "estrogen-receptor-positive", "breast", "cancer", "."],
        "ner": [
            [0, 0, "chemical_drug"],
            [2, 2, "gene_protein"],
            [6, 7, "disease"],
        ],
    },
    {
        "id": "bio-016",
        "tokens": ["Anastrozole", "suppresses", "CYP19A1", "reducing", "estrogen", "synthesis", "in", "postmenopausal", "breast", "cancer", "."],
        "ner": [
            [0, 0, "chemical_drug"],
            [2, 2, "gene_protein"],
            [8, 9, "disease"],
        ],
    },
    {
        "id": "bio-017",
        "tokens": ["Bortezomib", "disrupts", "PSMB5", "proteasome", "activity", "in", "multiple", "myeloma", "."],
        "ner": [
            [0, 0, "chemical_drug"],
            [2, 2, "gene_protein"],
            [6, 7, "disease"],
        ],
    },
    {
        "id": "bio-018",
        "tokens": ["Lenalidomide", "modulates", "CRBN", "ubiquitin", "ligase", "complex", "in", "myelodysplastic", "syndrome", "."],
        "ner": [
            [0, 0, "chemical_drug"],
            [2, 2, "gene_protein"],
            [7, 8, "disease"],
        ],
    },
    {
        "id": "bio-019",
        "tokens": ["Ibrutinib", "irreversibly", "binds", "BTK", "blocking", "B-cell", "activation", "in", "mantle", "cell", "lymphoma", "."],
        "ner": [
            [0, 0, "chemical_drug"],
            [3, 3, "gene_protein"],
            [8, 10, "disease"],
        ],
    },
    {
        "id": "bio-020",
        "tokens": ["Ruxolitinib", "inhibits", "JAK1", "and", "JAK2", "signaling", "in", "primary", "myelofibrosis", "."],
        "ner": [
            [0, 0, "chemical_drug"],
            [2, 2, "gene_protein"],
            [4, 4, "gene_protein"],
            [7, 8, "disease"],
        ],
    },
    {
        "id": "bio-021",
        "tokens": ["Gefitinib", "blocks", "tyrosine", "kinase", "phosphorylation", "of", "EGFR", "in", "pulmonary", "adenocarcinoma", "."],
        "ner": [
            [0, 0, "chemical_drug"],
            [6, 6, "gene_protein"],
            [8, 9, "disease"],
        ],
    },
    {
        "id": "bio-022",
        "tokens": ["Lapatinib", "co-inhibits", "EGFR", "and", "ERBB2", "pathways", "in", "metastatic", "breast", "cancer", "."],
        "ner": [
            [0, 0, "chemical_drug"],
            [2, 2, "gene_protein"],
            [4, 4, "gene_protein"],
            [7, 9, "disease"],
        ],
    },
    {
        "id": "bio-023",
        "tokens": ["Crizotinib", "inhibits", "ALK", "and", "ROS1", "rearrangements", "in", "anaplastic", "large", "cell", "lymphoma", "."],
        "ner": [
            [0, 0, "chemical_drug"],
            [2, 2, "gene_protein"],
            [4, 4, "gene_protein"],
            [7, 10, "disease"],
        ],
    },
]


def _build_char_spans(tokens: list[str], ner_triples: list[list[Any]]) -> tuple[str, list[dict[str, Any]]]:
    """Reconstruct text and calculate exact character offsets from token spans."""
    token_offsets: list[tuple[int, int]] = []
    text_parts: list[str] = []
    current_char = 0
    for idx, token in enumerate(tokens):
        if idx > 0 and token not in {".", ",", ";", ":", ")", "]"}:
            # Add space before token unless it's closing punctuation
            text_parts.append(" ")
            current_char += 1
        start_char = current_char
        text_parts.append(token)
        current_char += len(token)
        token_offsets.append((start_char, current_char))

    full_text = "".join(text_parts)
    spans: list[dict[str, Any]] = []
    for start_t, end_t, label in ner_triples:
        char_start = token_offsets[start_t][0]
        char_end = token_offsets[end_t][1]
        spans.append(
            {
                "start": char_start,
                "end": char_end,
                "label": str(label),
                "text": full_text[char_start:char_end],
            }
        )
    return full_text, spans


def generate_synthetic_ner_dataset() -> list[dict[str, Any]]:
    """Return the deterministic 24-sentence biomedical/clinical NER dataset."""
    records: list[dict[str, Any]] = []
    for item in _RAW_DOMAIN_EXAMPLES:
        full_text, spans = _build_char_spans(item["tokens"], item["ner"])
        records.append(
            {
                "id": item["id"],
                "tokenized_text": list(item["tokens"]),
                "ner": [list(span) for span in item["ner"]],
                "text": full_text,
                "spans": spans,
            }
        )
    return records


def split_ner_dataset(
    records: Sequence[dict[str, Any]],
    val_fraction: float = 0.25,
    seed: int = 42,
    labels: Sequence[str] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split dataset into disjoint train and validation sets with a seeded shuffle (no stratification).

    Each split must hold at least MIN_DATASET_EXAMPLES records, and every label in `labels` (default: every
    label found in the records) must occur in the training split; otherwise a ValueError names the split or
    the label and what to change. A label may still be absent from the validation split: see
    `split_label_coverage`.
    """
    if not 0.0 < val_fraction < 1.0:
        raise ValueError(f"val_fraction must be in (0, 1), got {val_fraction}")
    n_val = max(1, round(len(records) * val_fraction))
    n_train = len(records) - n_val
    if n_val < MIN_DATASET_EXAMPLES or n_train < MIN_DATASET_EXAMPLES:
        small = "validation" if n_val < MIN_DATASET_EXAMPLES else "training"
        raise ValueError(
            f"the {small} split would hold {min(n_val, n_train)} records (train {n_train}, validation {n_val} of "
            f"{len(records)} at val_fraction={val_fraction}); each split needs at least {MIN_DATASET_EXAMPLES}. "
            f"Supply at least {minimum_records(val_fraction)} records at this val_fraction, or change VAL_FRACTION."
        )

    record_list = [dict(r) for r in records]
    rng = random.Random(seed)
    rng.shuffle(record_list)

    val_records = record_list[:n_val]
    train_records = record_list[n_val:]

    wanted = _labels_in(records) if labels is None else list(labels)
    in_train = _labels_in(train_records)
    absent = [lbl for lbl in wanted if lbl not in in_train]
    if absent:
        raise ValueError(
            f"label(s) {absent} occur in no training record with seed={seed}; the model cannot learn them. "
            "Add training examples of these labels, or change SEED."
        )
    return train_records, val_records


def minimum_records(val_fraction: float = 0.25) -> int:
    """Smallest dataset size whose seeded split leaves MIN_DATASET_EXAMPLES records in each split."""
    if not 0.0 < val_fraction < 1.0:
        raise ValueError(f"val_fraction must be in (0, 1), got {val_fraction}")
    n = 2 * MIN_DATASET_EXAMPLES
    while True:
        n_val = max(1, round(n * val_fraction))
        if n_val >= MIN_DATASET_EXAMPLES and n - n_val >= MIN_DATASET_EXAMPLES:
            return n
        n += 1


def _labels_in(records: Sequence[dict[str, Any]]) -> list[str]:
    """Sorted labels of the well-formed `[start, end, label]` spans in `records`."""
    found = {
        span[2]
        for r in records
        if isinstance(r, dict)
        for span in (r.get("ner") or [])
        if isinstance(span, list | tuple) and len(span) == 3 and isinstance(span[2], str) and span[2].strip()
    }
    return sorted(found)


def split_label_coverage(
    train_records: Sequence[dict[str, Any]],
    val_records: Sequence[dict[str, Any]],
    labels: Sequence[str],
) -> dict[str, Any]:
    """Span counts per label in each split, and the labels the validation split cannot measure."""

    def counts(records: Sequence[dict[str, Any]]) -> dict[str, int]:
        out = dict.fromkeys(labels, 0)
        for r in records:
            for span in r.get("ner", []):
                if span[2] in out:
                    out[span[2]] += 1
        return out

    train, val = counts(train_records), counts(val_records)
    return {
        "train_spans": train,
        "validation_spans": val,
        "labels_missing_from_validation": [lbl for lbl in labels if val[lbl] == 0],
    }


def validate_dataset(
    records: Sequence[dict[str, Any]],
    allowed_labels: Sequence[str] = ADAPT_CLASSES,
    *,
    require_all_labels: bool = True,
    min_records: int = MIN_DATASET_EXAMPLES,
) -> dict[str, Any]:
    """Validate that NER records conform strictly to tokenized span schema and ceilings.

    Every span label must be one of `allowed_labels`. With `require_all_labels` (the default) every allowed label
    must also occur at least once; evaluation sets are checked with `require_all_labels=False`.
    """
    if not isinstance(records, Sequence) or isinstance(records, str | bytes):
        raise TypeError(f"records must be a sequence of dicts, got {type(records).__name__}")
    if len(records) < min_records:
        raise ValueError(f"dataset requires at least {min_records} records, got {len(records)}")
    if len(records) > MAX_DATASET_EXAMPLES:
        raise ValueError(f"dataset exceeds ceiling of {MAX_DATASET_EXAMPLES} records, got {len(records)}")

    label_set = set(allowed_labels)
    seen_ids: set[str] = set()
    label_counts: dict[str, int] = {lbl: 0 for lbl in label_set}
    total_spans = 0

    for idx, r in enumerate(records):
        if not isinstance(r, dict):
            raise TypeError(f"record[{idx}] must be a dict, got {type(r).__name__}; {RECORD_SCHEMA_HINT}")
        if "id" not in r:
            raise KeyError(f"record[{idx}] missing required key 'id'; {RECORD_SCHEMA_HINT}")
        if "tokenized_text" not in r:
            raise KeyError(f"record[{idx}] missing required key 'tokenized_text'; {RECORD_SCHEMA_HINT}")
        if "ner" not in r:
            raise KeyError(f"record[{idx}] missing required key 'ner'; {RECORD_SCHEMA_HINT}")

        doc_id = str(r["id"]).strip()
        if not doc_id:
            raise ValueError(f"record[{idx}] id cannot be empty")
        if doc_id in seen_ids:
            raise ValueError(f"record[{idx}] duplicates id {doc_id!r}")
        seen_ids.add(doc_id)

        tokens = r["tokenized_text"]
        if not isinstance(tokens, list) or not tokens:
            raise TypeError(f"record[{idx}] tokenized_text must be a non-empty list of tokens")
        if len(tokens) > MAX_TOKENS_PER_EXAMPLE:
            raise ValueError(f"record[{idx}] has {len(tokens)} tokens; ceiling is {MAX_TOKENS_PER_EXAMPLE}")
        for t_idx, token in enumerate(tokens):
            if not isinstance(token, str) or not token.strip():
                raise TypeError(f"record[{idx}] token[{t_idx}] must be a non-empty string")
            if len(token) > MAX_TOKEN_CHARS:
                raise ValueError(f"record[{idx}] token[{t_idx}] exceeds {MAX_TOKEN_CHARS} chars")

        ner_spans = r["ner"]
        if not isinstance(ner_spans, list):
            raise TypeError(f"record[{idx}] ner must be a list of spans")

        occupied_tokens: set[int] = set()
        for span_idx, span in enumerate(ner_spans):
            if not isinstance(span, list | tuple) or len(span) != 3:
                raise ValueError(
                    f"record[{idx}] span[{span_idx}] must be a list [start_token, end_token, label] (inclusive end, "
                    f'e.g. [0, 1, "person"]), got {type(span).__name__} {str(span)[:80]}'
                )
            start, end, label = span
            if isinstance(start, bool) or not isinstance(start, int) or start < 0:
                raise TypeError(f"record[{idx}] span[{span_idx}] start must be non-negative int")
            if isinstance(end, bool) or not isinstance(end, int) or end < start:
                raise ValueError(f"record[{idx}] span[{span_idx}] end must be >= start")
            if end >= len(tokens):
                raise ValueError(
                    f"record[{idx}] span[{span_idx}] end index {end} out of bounds for {len(tokens)} tokens"
                )
            if not isinstance(label, str) or not label.strip():
                raise TypeError(f"record[{idx}] span[{span_idx}] label must be non-empty string")
            if label not in label_set:
                raise ValueError(f"record[{idx}] span[{span_idx}] label {label!r} not in {sorted(label_set)}")

            # Check for overlapping spans
            span_range = set(range(start, end + 1))
            if span_range & occupied_tokens:
                raise ValueError(f"record[{idx}] span[{span_idx}] overlaps with an existing entity span")
            occupied_tokens.update(span_range)

            label_counts[label] += 1
            total_spans += 1

    missing_labels = sorted(lbl for lbl, count in label_counts.items() if count == 0)
    if missing_labels and require_all_labels:
        raise ValueError(
            f"dataset missing examples for required labels: {missing_labels}; remove them from the declared "
            "label set or add examples"
        )

    return {
        "verdict": "accepted",
        "representation": DATASET_REPRESENTATION,
        "n_records": len(records),
        "n_spans": total_spans,
        "label_distribution": label_counts,
        "labels": list(allowed_labels),
    }


def load_byod_dataset(
    source: str | Path,
    allowed_labels: Sequence[str] | None = ADAPT_CLASSES,
) -> list[dict[str, Any]]:
    """Load and validate a user-supplied JSON or JSONL NER dataset.

    `allowed_labels=None` takes the label set from the records themselves (every label that occurs); otherwise
    every span label must be one of `allowed_labels` and each of them must occur.
    """
    path = Path(source)
    if not path.is_file():
        raise FileNotFoundError(f"BYOD dataset file not found: {path}")
    if path.suffix.lower() not in BYOD_SUFFIXES:
        raise ValueError(
            f"BYOD expects a .json file (one JSON array of records) or a .jsonl file (one record per line), got "
            f"{path.name!r}; {RECORD_SCHEMA_HINT}. Convert the file (for example from CSV) and supply it again."
        )

    raw_text = path.read_text(encoding="utf-8").strip()
    if not raw_text:
        raise ValueError(f"BYOD dataset file is empty: {path}")

    records: list[dict[str, Any]] = []
    if path.suffix.lower() == ".jsonl":
        for line_no, line in enumerate(raw_text.splitlines(), start=1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{path.name} line {line_no} is not valid JSON ({exc}); a .jsonl file holds one JSON object per line"
                ) from exc
            records.append(item)
    else:
        try:
            data = json.loads(raw_text)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"{path.name} is not valid JSON ({exc}); a .json file holds one array of records: {RECORD_SCHEMA_HINT}"
            ) from exc
        if not isinstance(data, list):
            raise TypeError(f"JSON dataset must contain a top-level array of objects, got {type(data).__name__}")
        records = data

    for r in records:
        if isinstance(r, dict) and "tokens" in r and "tokenized_text" not in r:
            r["tokenized_text"] = r.pop("tokens")
    labels = _labels_in(records) if allowed_labels is None else list(allowed_labels)
    if not labels:
        # No well-formed span anywhere: report the first malformed record or span, if there is one.
        validate_dataset(records, [], require_all_labels=False, min_records=0)
        raise ValueError(f"no entity labels found in {path.name}; {RECORD_SCHEMA_HINT}")
    # Validate before deriving text and character spans, so a malformed span gets the schema message.
    validate_dataset(records, labels)
    for r in records:
        if "text" not in r:
            full_text, spans = _build_char_spans(r["tokenized_text"], r["ner"])
            r["text"] = full_text
            r["spans"] = spans
    return records


# Pre-built immutable instance of the synthetic biomedical dataset
SAMPLE_BIOMEDICAL_DATASET: list[dict[str, Any]] = generate_synthetic_ner_dataset()
