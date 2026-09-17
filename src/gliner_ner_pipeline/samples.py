"""Deterministic in-code sample data and contracts for GLiNER multi-v2.1 domain adaptation.

Provides deterministic synthetic domain NER datasets (biomedical/clinical entities: disease,
chemical_drug, gene_protein) and BYOD validation routines conforming to DIMER NOTEBOOK_SPEC 2.0.
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
MIN_DATASET_EXAMPLES = 4
MAX_DATASET_EXAMPLES = 1_000
MAX_TOKENS_PER_EXAMPLE = 384
MAX_TOKEN_CHARS = 100

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
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split dataset into disjoint train and validation sets with fixed seed.

    Guarantees all classes in ADAPT_CLASSES appear in both train and val splits.
    """
    if not 0.0 < val_fraction < 1.0:
        raise ValueError(f"val_fraction must be in (0, 1), got {val_fraction}")
    if len(records) < MIN_DATASET_EXAMPLES:
        raise ValueError(f"dataset requires at least {MIN_DATASET_EXAMPLES} records to split")

    record_list = [dict(r) for r in records]
    rng = random.Random(seed)
    rng.shuffle(record_list)

    n_val = max(1, round(len(record_list) * val_fraction))
    val_records = record_list[:n_val]
    train_records = record_list[n_val:]

    if not train_records or not val_records:
        raise ValueError("split produced an empty split; increase record count")

    return train_records, val_records


def validate_dataset(
    records: Sequence[dict[str, Any]],
    allowed_labels: Sequence[str] = ADAPT_CLASSES,
) -> dict[str, Any]:
    """Validate that NER records conform strictly to tokenized span schema and ceilings."""
    if not isinstance(records, Sequence) or isinstance(records, str | bytes):
        raise TypeError(f"records must be a sequence of dicts, got {type(records).__name__}")
    if len(records) < MIN_DATASET_EXAMPLES:
        raise ValueError(f"dataset requires at least {MIN_DATASET_EXAMPLES} records, got {len(records)}")
    if len(records) > MAX_DATASET_EXAMPLES:
        raise ValueError(f"dataset exceeds ceiling of {MAX_DATASET_EXAMPLES} records, got {len(records)}")

    label_set = set(allowed_labels)
    seen_ids: set[str] = set()
    label_counts: dict[str, int] = {lbl: 0 for lbl in label_set}
    total_spans = 0

    for idx, r in enumerate(records):
        if not isinstance(r, dict):
            raise TypeError(f"record[{idx}] must be a dict, got {type(r).__name__}")
        if "id" not in r:
            raise KeyError(f"record[{idx}] missing required key 'id'")
        if "tokenized_text" not in r:
            raise KeyError(f"record[{idx}] missing required key 'tokenized_text'")
        if "ner" not in r:
            raise KeyError(f"record[{idx}] missing required key 'ner'")

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
                raise ValueError(f"record[{idx}] span[{span_idx}] must be [start, end, label]")
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

    missing_labels = [lbl for lbl, count in label_counts.items() if count == 0]
    if missing_labels:
        raise ValueError(f"dataset missing examples for required labels: {missing_labels}")

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
    allowed_labels: Sequence[str] = ADAPT_CLASSES,
) -> list[dict[str, Any]]:
    """Load and validate a user-supplied JSON or JSONL NER dataset."""
    path = Path(source)
    if not path.is_file():
        raise FileNotFoundError(f"BYOD dataset file not found: {path}")

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
                raise ValueError(f"line {line_no} is invalid JSON: {exc}") from exc
            records.append(item)
    else:
        try:
            data = json.loads(raw_text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"file is invalid JSON: {exc}") from exc
        if not isinstance(data, list):
            raise TypeError(f"JSON dataset must contain a top-level array of objects, got {type(data).__name__}")
        records = data

    # Reconstruct text and character spans if missing
    for r in records:
        if "tokens" in r and "tokenized_text" not in r:
            r["tokenized_text"] = r.pop("tokens")
        if "text" not in r and "tokenized_text" in r:
            full_text, spans = _build_char_spans(r["tokenized_text"], r.get("ner", []))
            r["text"] = full_text
            r["spans"] = spans

    validate_dataset(records, allowed_labels)
    return records


# Pre-built immutable instance of the synthetic biomedical dataset
SAMPLE_BIOMEDICAL_DATASET: list[dict[str, Any]] = generate_synthetic_ner_dataset()
