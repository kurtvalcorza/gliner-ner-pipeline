"""Evaluation metrics and reporting for GLiNER multi-v2.1 named-entity recognition.

Computes exact-span micro and macro Precision, Recall, and F1 across dataset records,
per-class performance breakdowns, and baseline delta comparisons.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


def compute_span_f1(
    hits: int,
    n_predicted: int,
    n_gold: int,
) -> dict[str, float]:
    """Calculate Precision, Recall, and F1 from counts."""
    precision = hits / n_predicted if n_predicted > 0 else 0.0
    recall = hits / n_gold if n_gold > 0 else 0.0
    f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0.0 else 0.0
    return {
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1": round(float(f1), 4),
    }


def evaluate_ner_dataset(
    predictions_per_record: Sequence[Sequence[Mapping[str, Any]]],
    gold_records: Sequence[Mapping[str, Any]],
    labels: Sequence[str],
) -> dict[str, Any]:
    """Compute micro/macro exact-span NER evaluation metrics across records.

    Each prediction is expected to have 'start', 'end', and 'label'.
    Gold records must have 'spans' with 'start', 'end', and 'label'.
    """
    if len(predictions_per_record) != len(gold_records):
        raise ValueError(
            f"number of prediction sets ({len(predictions_per_record)}) != gold records ({len(gold_records)})"
        )

    label_set = list(labels)
    per_class_counts: dict[str, dict[str, int]] = {
        lbl: {"hits": 0, "predicted": 0, "gold": 0} for lbl in label_set
    }

    total_hits = 0
    total_predicted = 0
    total_gold = 0

    for preds, gold_rec in zip(predictions_per_record, gold_records, strict=True):
        gold_spans = gold_rec.get("spans") or gold_rec.get("entities", [])
        gold_tuples = {(int(s["start"]), int(s["end"]), str(s["label"])) for s in gold_spans}
        pred_tuples = {(int(p["start"]), int(p["end"]), str(p["label"])) for p in preds}

        total_predicted += len(pred_tuples)
        total_gold += len(gold_tuples)

        for p in pred_tuples:
            lbl = p[2]
            if lbl in per_class_counts:
                per_class_counts[lbl]["predicted"] += 1

        for g in gold_tuples:
            lbl = g[2]
            if lbl in per_class_counts:
                per_class_counts[lbl]["gold"] += 1

        hits = pred_tuples & gold_tuples
        total_hits += len(hits)

        for h in hits:
            lbl = h[2]
            if lbl in per_class_counts:
                per_class_counts[lbl]["hits"] += 1

    micro = compute_span_f1(total_hits, total_predicted, total_gold)

    class_metrics: dict[str, dict[str, Any]] = {}
    f1_values: list[float] = []

    for lbl in label_set:
        c = per_class_counts[lbl]
        res = compute_span_f1(c["hits"], c["predicted"], c["gold"])
        class_metrics[lbl] = {
            **res,
            "hits": c["hits"],
            "predicted": c["predicted"],
            "gold": c["gold"],
        }
        if c["gold"] > 0:
            f1_values.append(res["f1"])

    macro_f1 = round(sum(f1_values) / len(f1_values), 4) if f1_values else 0.0

    return {
        "micro_precision": micro["precision"],
        "micro_recall": micro["recall"],
        "micro_f1": micro["f1"],
        "macro_f1": macro_f1,
        "hits": total_hits,
        "n_predicted": total_predicted,
        "n_gold": total_gold,
        "per_class": class_metrics,
        "labels": list(labels),
    }
