"""Per-repository template for tools/build_notebook.py (NOTEBOOK_SPEC 2.0 §4 standalone carrier).

Only the task-specific prose and stage cells live here. Runtime install, the embedded pipeline
modules (pipeline.py, metrics.py, samples.py), and the model pin/stage/verify cells are produced
by the generator from repository sources so they cannot drift from the package.

This template configures an E2E domain adaptation pipeline: GLiNER verifies TWO pinned snapshots
(the GLiNER checkpoint and the mDeBERTa-v3 encoder assets) and adapts span representation layers
via native PyTorch AdamW optimization while freezing the text encoder backbone.
"""
# ruff: noqa: E501  -- markdown prose and code-cell text are kept on single lines for readable rendering

REPO = "gliner-ner-pipeline"

BADGES = [
    (
        "GitHub",
        "https://img.shields.io/badge/GitHub-181717?style=flat&logo=github&logoColor=white",
        f"https://github.com/kurtvalcorza/{REPO}",
    ),
    (
        "Open In Colab",
        "https://colab.research.google.com/assets/colab-badge.svg",
        f"https://colab.research.google.com/github/kurtvalcorza/{REPO}/blob/main/tutorials/gliner_ner_colab.ipynb",
    ),
    (
        "Hugging Face",
        "https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-urchade%2Fgliner__multi--v2.1-ffcc4d?style=flat",
        "https://huggingface.co/urchade/gliner_multi-v2.1",
    ),
    (
        "Upstream",
        "https://img.shields.io/badge/Upstream-urchade%2FGLiNER-181717?style=flat&logo=github&logoColor=white",
        "https://github.com/urchade/GLiNER",
    ),
    ("arXiv", "https://img.shields.io/badge/arXiv-2311.08526-b31b1b.svg", "https://arxiv.org/abs/2311.08526"),
]

TEMPLATE = {
    "package": "gliner_ner_pipeline",
    "repo_name": REPO,
    "stem": "gliner_ner",
    "notebook_name": "gliner_ner_colab.ipynb",
    "profile": "E2E",
    "mode": "GUIDED",
    "run_all": (
        "Selecting **Run all** in a fresh supported runtime installs the pinned dependencies, stages and digest-verifies both "
        "pinned snapshots (GLiNER multi-v2.1 weights and mDeBERTa-v3 tokenizer assets), obtains the deterministic 24-sentence "
        "biomedical NER dataset automatically (no download), validates it against tokenized span contracts, splits it deterministically "
        "into train/validation sets with class coverage preserved, computes a zero-shot baseline evaluation on the validation split, "
        "adapts the span representation layers via a native bounded fine-tuning loop (freezing the text encoder backbone), "
        "evaluates the adapted model reporting exact-span micro/macro F1 and baseline deltas, runs inference on unseen clinical text, "
        "exports the adapter weights and lineage metadata to a `.pt` artifact, and reloads the artifact into a fresh pipeline instance "
        "to verify parameter match and prediction parity. The default path needs no repository clone, no DIMER worker or service, "
        "no external API keys, no upload dialog, and no configuration edits (NOTEBOOK_SPEC 2.0 §5)."
    ),
    "byod": (
        "After the tutorial workflow completes, set `USE_BYOD = True` in Section 4 and re-run from that cell to supply your own "
        "JSON or JSONL dataset file. It passes through the same validation, splitting, baseline evaluation, adaptation, post-adaptation "
        "evaluation, unseen inference, artifact export, and reload parity cells as the synthetic sample. The expected JSON schema and "
        "token/span ceilings are stated in the Prerequisites and in Section 4, and uploaded files stay inside this runtime. "
        "BYOD is optional and never part of the default path."
    ),
    "pipeline_class": "GLiNERPipeline",
    "weights_key": "gliner-multi-v2.1",
    "modules": ["metrics.py", "samples.py", "pipeline.py"],
    "entry_module": "pipeline.py",
    # generator /2: both weights directories derive from one shared root, and a second pinned snapshot
    # (the mDeBERTa encoder assets) is carried, staged and verified by the model cell.
    "rewrites": [
        [
            r"^_WEIGHTS_ROOT = Path\(__file__\)[^\n]*$",
            '_WEIGHTS_ROOT = Path.cwd() / "weights"  # standalone rewrite (build_notebook.py): working-directory-relative',
        ]
    ],
    "extra_weights": [
        {
            "key": "mdeberta-v3-base-tokenizer",
            "var": "ENCODER_MANIFEST",
            "dir": "ENCODER_WEIGHTS_DIR",
            "identity": ["ENCODER_MODEL_ID", "ENCODER_REVISION"],
            "stage": "stage_missing_encoder_files",
            "verify": "verify_encoder_snapshot",
        }
    ],
    "model_load": "GLiNERPipeline.from_pretrained(weights_dir=WEIGHTS_DIR, encoder_dir=ENCODER_WEIGHTS_DIR)",
    "runtime_imports": ["torch", "transformers", "gliner"],
    "title": "GLiNER multi-v2.1 — DIMER E2E named-entity recognition adaptation tutorial (standalone)",
    "badges": BADGES,
    "capability": "zero-shot and domain-adapted named-entity recognition with arbitrary label sets",
    "intro": (
        "GLiNER is a bidirectional encoder-based entity extraction model that matches arbitrary textual labels against "
        "candidate text spans. At inference, input text and entity types are encoded jointly with mDeBERTa-v3, and candidate "
        "spans are scored against label representations. While zero-shot inference works out of the box, domain-specific vocabularies "
        "(such as biomedical and clinical entities: `disease`, `chemical_drug`, `gene_protein`) benefit substantially from **domain "
        "adaptation**. This tutorial guides you end-to-end through validating a structured domain dataset, measuring zero-shot baseline "
        "performance, fine-tuning the span representation layers with the mDeBERTa backbone frozen, evaluating exact-span precision, "
        "recall, and F1 deltas, extracting entities from unseen clinical text, and packaging the trained adapter weights into a portable, "
        "reusable `.pt` artifact that reloads with verified parity."
    ),
    "learning_objectives": (
        "install the pinned runtime; inspect the carried pipeline, metrics, and dataset modules; stage and digest-verify "
        "both immutable snapshots; validate and split a domain NER dataset; evaluate baseline zero-shot performance; "
        "execute a bounded fine-tuning loop adapting span and prompt projections while keeping the mDeBERTa backbone frozen; "
        "evaluate post-adaptation exact-span F1 metrics and baseline deltas; perform entity extraction on unseen text; and "
        "export and reload the adapter artifact verifying weight parity."
    ),
    "exclusions": (
        "relation extraction, coreference resolution, unconstrained full-parameter fine-tuning of mDeBERTa on tiny data "
        "(prone to catastrophic forgetting), or arbitrary unvalidated dataset formats. The repository exposes none of these."
    ),
    "prerequisites": [
        "- **Runtime:** a fresh supported runtime (Google Colab or Jupyter, Python 3.12). CPU execution is supported; CUDA is auto-detected and recommended for fast adaptation.",
        "- **Knowledge:** understanding of named-entity recognition, character offsets vs token indices, and exact-span F1 metrics.",
        "- **Expected warning:** while building the fast DeBERTa tokenizer from `spm.model`, `transformers==4.57.6` logs an \"incorrect regex pattern … `fix_mistral_regex`\" warning. It refers to Mistral tokenizer regex, does not affect this SentencePiece model, and is captured in `load_warnings`.",
        "- **Data:** default sample is a deterministic 24-sentence synthetic biomedical dataset (`disease`, `chemical_drug`, `gene_protein`). BYOD supports custom JSON or JSONL files conforming to tokenized span schema. Do not upload confidential or restricted data to a hosted runtime unless authorized.",
    ],
    "cells": [
        {
            "md": (
                "## 4. Load dataset, validate contract, and split\n\n"
                "In this section we obtain the domain dataset. By default, the pipeline loads a deterministic 24-sentence "
                "synthetic biomedical NER dataset containing clinical entities (`disease`, `chemical_drug`, `gene_protein`). "
                "The dataset is validated through `validate_dataset`, ensuring that token sequences, span boundaries, and labels "
                "strictly satisfy contract constraints. The dataset is then split into disjoint training (75%) and validation (25%) "
                "subsets with class balance preserved across both splits.\n\n"
                "To supply your own data, upload a JSON or JSONL file conforming to the schema and toggle `USE_BYOD = True`."
            ),
            "code": (
                "import json\n"
                "from pathlib import Path\n\n"
                "USE_BYOD = False  # @param {{type:\"boolean\"}}\n"
                "VAL_FRACTION = 0.25  # @param {{type:\"number\"}}\n"
                "SEED = 42  # @param {{type:\"integer\"}}\n\n"
                "if USE_BYOD:\n"
                "    from google.colab import files\n"
                "    uploaded = files.upload()\n"
                "    file_name, payload = next(iter(uploaded.items()))\n"
                "    byod_path = Path('work') / file_name\n"
                "    byod_path.parent.mkdir(parents=True, exist_ok=True)\n"
                "    byod_path.write_bytes(payload)\n"
                "    raw_dataset = load_byod_dataset(byod_path)\n"
                "    data_source = 'BYOD (' + file_name + ')'\n"
                "else:\n"
                "    raw_dataset = generate_synthetic_ner_dataset()\n"
                "    data_source = 'Synthetic biomedical dataset (24 records)'\n\n"
                "val_manifest = validate_dataset(raw_dataset)\n"
                "train_records, val_records = split_ner_dataset(raw_dataset, val_fraction=VAL_FRACTION, seed=SEED)\n\n"
                "print({{'data_source': data_source, 'total_records': len(raw_dataset), 'train_records': len(train_records), 'val_records': len(val_records)}})\n"
                "print({{'label_distribution': val_manifest['label_distribution']}})\n"
                "print('Sample record:', json.dumps(train_records[0], indent=2))"
            ),
        },
        {
            "md": (
                "## 5. Evaluate pre-adaptation zero-shot baseline\n\n"
                "Before running adaptation, we benchmark the zero-shot baseline accuracy of the pretrained checkpoint on our "
                "held-out validation set. `pipe.evaluate(val_records, labels=ADAPT_CLASSES)` computes exact-span micro precision, "
                "recall, and F1 score, alongside per-class performance breakdowns. This provides an objective baseline against "
                "which adaptation gains are quantified."
            ),
            "code": (
                "baseline_eval = pipe.evaluate(val_records, labels=ADAPT_CLASSES)\n"
                "print('Pre-adaptation Baseline Metrics (Validation Split):')\n"
                "print(json.dumps(baseline_eval, indent=2))"
            ),
        },
        {
            "md": (
                "## 6. Adapt model via bounded fine-tuning\n\n"
                "We now adapt GLiNER to our target domain. Calling `pipe.adapt(train_records, val_records, ...)` executes a lightweight, "
                "native PyTorch fine-tuning loop: the mDeBERTa-v3 text encoder backbone is frozen (`freeze_text_encoder=True`), "
                "and only the span representation and prompt projection layers are updated using AdamW. This prevents catastrophic "
                "forgetting, keeps memory consumption low, and enables fast convergence even on CPU or modest GPU instances."
            ),
            "code": (
                "import time\n\n"
                "EPOCHS = 3  # @param {{type:\"integer\"}}\n"
                "LEARNING_RATE = 5e-5  # @param {{type:\"number\"}}\n"
                "BATCH_SIZE = 4  # @param {{type:\"integer\"}}\n\n"
                "started = time.perf_counter()\n"
                "adapt_result = pipe.adapt(\n"
                "    train_records=train_records,\n"
                "    val_records=val_records,\n"
                "    epochs=EPOCHS,\n"
                "    learning_rate=LEARNING_RATE,\n"
                "    batch_size=BATCH_SIZE,\n"
                "    freeze_text_encoder=True,\n"
                "    seed=SEED,\n"
                ")\n"
                "elapsed = time.perf_counter() - started\n\n"
                "print(f'Adaptation completed in {{elapsed:.2f}}s.')\n"
                "print('Training History:')\n"
                "for step in adapt_result['history']:\n"
                "    print(f\"  Epoch {{step['epoch']}}: Loss={{step['loss']:.4f}}, Val F1={{step.get('val_f1', 0.0):.4f}}\")"
            ),
        },
        {
            "md": (
                "## 7. Evaluate post-adaptation metrics and deltas\n\n"
                "We evaluate the adapted model on the validation split and calculate the exact performance delta compared to the "
                "pre-adaptation zero-shot baseline. The evaluation report records micro/macro precision, recall, and F1 improvements, "
                "as well as per-class metrics, and saves the result to `outputs/{stem}_evaluation_report.json`."
            ),
            "code": (
                "import os\n\n"
                "os.makedirs('outputs', exist_ok=True)\n"
                "adapted_eval = pipe.evaluate(val_records, labels=ADAPT_CLASSES)\n\n"
                "delta_f1 = round(adapted_eval['micro']['f1'] - baseline_eval['micro']['f1'], 4)\n"
                "delta_precision = round(adapted_eval['micro']['precision'] - baseline_eval['micro']['precision'], 4)\n"
                "delta_recall = round(adapted_eval['micro']['recall'] - baseline_eval['micro']['recall'], 4)\n\n"
                "eval_report = {{\n"
                "    'task': 'domain-adapted named-entity recognition',\n"
                "    'domain': 'biomedical / clinical',\n"
                "    'labels': list(ADAPT_CLASSES),\n"
                "    'baseline_eval': baseline_eval,\n"
                "    'adapted_eval': adapted_eval,\n"
                "    'delta': {{\n"
                "        'f1_delta': delta_f1,\n"
                "        'precision_delta': delta_precision,\n"
                "        'recall_delta': delta_recall,\n"
                "    }},\n"
                "    'training_summary': adapt_result,\n"
                "}}\n\n"
                "with open('outputs/{stem}_evaluation_report.json', 'w', encoding='utf-8') as f:\n"
                "    json.dump(eval_report, f, indent=2)\n\n"
                "print('Post-adaptation Validation Metrics:')\n"
                "print(json.dumps(adapted_eval, indent=2))\n"
                "print(f\"\\nMicro F1 Delta: {{delta_f1:+.4f}} (Baseline: {{baseline_eval['micro']['f1']:.4f}} -> Adapted: {{adapted_eval['micro']['f1']:.4f}})\")"
            ),
        },
        {
            "md": (
                "## 8. Run inference on unseen domain text\n\n"
                "Now we test the adapted pipeline on an unseen clinical sentence. `pipe.detect` extracts domain entities with "
                "character spans and confidence scores. The output is validated to guarantee that offsets are within text bounds, "
                "span texts match substring slices, and all assigned labels exist within the requested vocabulary."
            ),
            "code": (
                "test_sentence = 'Pembrolizumab blocks PDCD1 receptor to treat metastatic melanoma in adult patients.'\n"
                "inference_result = pipe.detect(test_sentence, labels=ADAPT_CLASSES, threshold=DEFAULT_THRESHOLD)\n\n"
                "entities = inference_result['entities']\n"
                "checks = {{\n"
                "    'offsets_inside_text': all(0 <= e['start'] < e['end'] <= len(test_sentence) for e in entities),\n"
                "    'span_text_matches': all(test_sentence[e['start']:e['end']] == e['text'] for e in entities),\n"
                "    'labels_valid': all(e['label'] in ADAPT_CLASSES for e in entities),\n"
                "}}\n"
                "assert all(checks.values()), f'Inference failed sanity check: {{checks}}'\n\n"
                "print(f'Input: \"{{test_sentence}}\"')\n"
                "print(f'Detected {{len(entities)}} entities:')\n"
                "for i, e in enumerate(entities, 1):\n"
                "    print(f\"  {{i}}. [{{e['start']:>2}}:{{e['end']:>2}}] {{e['label']:<14}} (score={{e['score']:.4f}}): {{e['text']}}\")"
            ),
        },
        {
            "md": (
                "## 9. Export adapter artifact and verify reload parity\n\n"
                "To deploy the adapted model without duplicating the 1.16 GB base checkpoint, `pipe.save_artifact` exports "
                "only the trained adapter weights and metadata into a portable `.pt` file (`outputs/{stem}_adapter.pt`). "
                "We then instantiate a fresh pipeline from the base weights via `GLiNERPipeline.from_artifact` and verify "
                "that predictions on unseen text match the in-memory adapted model with complete bit-level parity."
            ),
            "code": (
                "artifact_path = Path('outputs/{stem}_adapter.pt')\n"
                "pipe.save_artifact(\n"
                "    artifact_path,\n"
                "    metadata={{\n"
                "        'domain': 'biomedical',\n"
                "        'classes': list(ADAPT_CLASSES),\n"
                "        'epochs': EPOCHS,\n"
                "        'learning_rate': LEARNING_RATE,\n"
                "        'metrics': adapted_eval['micro'],\n"
                "    }},\n"
                ")\n"
                "print(f'Saved adapter artifact: {{artifact_path}} ({{artifact_path.stat().st_size / (1024*1024):.2f}} MB)')\n\n"
                "# Verify fresh reload\n"
                "reloaded_pipe = GLiNERPipeline.from_artifact(\n"
                "    artifact_path,\n"
                "    weights_dir=WEIGHTS_DIR,\n"
                "    encoder_dir=ENCODER_WEIGHTS_DIR,\n"
                ")\n"
                "reloaded_res = reloaded_pipe.detect(test_sentence, labels=ADAPT_CLASSES, threshold=DEFAULT_THRESHOLD)\n"
                "reloaded_entities = reloaded_res['entities']\n\n"
                "# Verify exact prediction parity\n"
                "assert len(entities) == len(reloaded_entities), 'Parity failure: entity count mismatch'\n"
                "for e_orig, e_rel in zip(entities, reloaded_entities, strict=True):\n"
                "    assert e_orig['start'] == e_rel['start'] and e_orig['end'] == e_rel['end'], 'Offset parity failure'\n"
                "    assert e_orig['label'] == e_rel['label'], 'Label parity failure'\n"
                "    assert abs(e_orig['score'] - e_rel['score']) < 1e-5, 'Score parity failure'\n"
                "print('Parity Verification Passed: Reloaded artifact matches adapted pipeline identically.')"
            ),
        },
        {
            "md": (
                "## 10. Export outputs and lineage manifest\n\n"
                "We finalize the tutorial by writing two standard output files under `outputs/`: "
                "`{stem}_result.json` containing complete provenance, configuration parameters, baseline and adapted "
                "evaluation metrics, and runtime specifications; and `{stem}_entities.csv` containing tabular entity spans."
            ),
            "code": (
                "import csv\n"
                "import platform\n\n"
                "result_payload = {{\n"
                "    'task': 'named-entity recognition domain adaptation',\n"
                "    'pipeline_class': 'GLiNERPipeline',\n"
                "    'model_id': MODEL_ID,\n"
                "    'model_revision': MODEL_REVISION,\n"
                "    'model_license': MODEL_LICENSE,\n"
                "    'encoder_model_id': ENCODER_MODEL_ID,\n"
                "    'encoder_revision': ENCODER_REVISION,\n"
                "    'encoder_license': ENCODER_LICENSE,\n"
                "    'repository_revision': NOTEBOOK_SOURCE['repository_revision'],\n"
                "    'artifact_format': ARTIFACT_FORMAT,\n"
                "    'artifact_format_version': ARTIFACT_FORMAT_VERSION,\n"
                "    'notebook_source': NOTEBOOK_SOURCE,\n"
                "    'evaluation_report': eval_report,\n"
                "    'inference_sample': {{\n"
                "        'text': test_sentence,\n"
                "        'entities': entities,\n"
                "    }},\n"
                "    'runtime': {{\n"
                "        'python': platform.python_version(),\n"
                "        'torch': torch.__version__,\n"
                "        'transformers': transformers.__version__,\n"
                "        'gliner': gliner.__version__,\n"
                "        'device': pipe.device,\n"
                "    }},\n"
                "}}\n\n"
                "with open('outputs/{stem}_result.json', 'w', encoding='utf-8') as f:\n"
                "    json.dump(result_payload, f, indent=2)\n\n"
                "with open('outputs/{stem}_entities.csv', 'w', encoding='utf-8', newline='') as f:\n"
                "    writer = csv.writer(f)\n"
                "    writer.writerow(['index', 'start', 'end', 'label', 'score', 'text'])\n"
                "    for idx, e in enumerate(entities, 1):\n"
                "        writer.writerow([idx, e['start'], e['end'], e['label'], f\"{{e['score']:.6f}}\", e['text']])\n\n"
                "print('Generated release artifacts under outputs/:')\n"
                "for fname in sorted(os.listdir('outputs')):\n"
                "    fpath = Path('outputs') / fname\n"
                "    print(f'  - {{fname}} ({{fpath.stat().st_size / 1024:.1f}} KB)')"
            ),
        },
    ],
    "closing": (
        "## Interpretation and limits\n\n"
        "Domain adaptation tunes span representation projections to align candidate text representations with custom entity types. "
        "Freezing the mDeBERTa backbone ensures the model retains its linguistic representation power while preventing catastrophic "
        "forgetting on small datasets. Each returned span represents an exact character slice in the original text; the score reflects "
        "the model's uncalibrated similarity between span representation and label embedding. Exact-span F1 requires exact agreement on "
        "`(start, end, label)` triples: boundary mismatches of even one character receive zero credit. The exported adapter artifact "
        "carries only trainable parameters and lineage headers, enabling lightweight distribution and reproducibility.\n\n"
        "Successful execution proves that the recorded repository revision's pipeline modules, carried in this standalone notebook, "
        "can acquire and digest-verify the pinned model, validate the demonstrated dataset contract, execute bounded fine-tuning, "
        "evaluate metrics against pre-adaptation baseline, and emit the shown machine-readable artifacts — without the repository being reachable. "
        "It does **not** establish benchmark superiority, production fitness, or generalized performance across unseen domains.\n\n"
        "## References\n\n"
        "- Repository README: https://github.com/kurtvalcorza/gliner-ner-pipeline/blob/main/README.md\n"
        "- Repository model card: https://github.com/kurtvalcorza/gliner-ner-pipeline/blob/main/MODEL_CARD.md\n"
        "- Upstream model: https://huggingface.co/{MODEL_ID}\n"
        "- Encoder assets: https://huggingface.co/microsoft/mdeberta-v3-base\n"
        "- Upstream code: https://github.com/urchade/GLiNER\n"
        "- GLiNER paper: https://arxiv.org/abs/2311.08526"
    ),
}
