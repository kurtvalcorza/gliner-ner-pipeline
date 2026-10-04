"""Per-repository template for tools/build_notebook.py (NOTEBOOK_SPEC 2.2 §4 standalone carrier).

Only the task-specific prose and stage cells live here. Runtime install, the embedded pipeline
modules (pipeline.py, metrics.py, samples.py), and the model pin/stage/verify cells are produced
by the generator from repository sources so they cannot drift from the package.

This template configures an E2E domain adaptation pipeline: GLiNER verifies TWO pinned snapshots
(the GLiNER checkpoint and the mDeBERTa-v3 encoder assets) and adapts the layers above the frozen
mDeBERTa text encoder via native PyTorch AdamW optimization.

Review fixes (Notebook Review Framework v1, review PR #10, GL-M1..M4 / GL-m1..m5): the runtime is the fleet's uv
isolated environment (no in-kernel install, no restart); Sections 5 and 6 start from the pretrained model on every
re-run (`reset_to_pretrained`) and `adapt` refuses an adapted pipeline; BYOD takes any label set (`BYOD_LABELS`, or
the labels found in the file), a path as well as an upload, and an inference sentence of its own, and Section 4
states the record schema, the minimum and the ceilings before the upload; the guided layer (audience, task
contract, how to use, roadmap, predictions, worked answers, a change-one-thing activity, troubleshooting, glossary,
conclusion) is added and the setup cells are labelled Infrastructure; the reload check compares every adapter
tensor and the predictions on the whole validation split, and mismatched adapters are refused; the headline delta
is labelled tutorial evidence with its sample size, template and label-name caveats; and the split, trainable-layer
and load-warning statements describe what the code does.
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

SAMPLE_SENTENCE = "Pembrolizumab blocks PDCD1 receptor to treat metastatic melanoma in adult patients."

TEMPLATE = {
    "package": "gliner_ner_pipeline",
    "repo_name": REPO,
    "stem": "gliner_ner",
    "notebook_name": "gliner_ner_colab.ipynb",
    "profile": "E2E",
    "mode": "GUIDED",
    "infrastructure_labels": True,
    "collapse_model_cell": True,
    "isolated_runtime": True,
    # GL-M1: the fleet's uv isolated-environment mechanism (ast-audio-classification-pipeline / bioclip2-biodiversity-pipeline;
    # generator /2.1 as in florence2-vision-language-pipeline 9c4e95a): managed CPython, a size- and SHA-256-verified uv wheel,
    # and a lock compiled from the pyproject pins with `uv pip compile pyproject.toml -c <versions of esm2-protein-pipeline's
    # T4-passed tutorials/requirements-colab.lock.txt @ 37f7112> --python-version 3.12 --python-platform x86_64-manylinux_2_28
    # --generate-hashes --only-binary :all: -o tutorials/requirements-colab.lock.txt`. The lock is esm2's 47 packages at the same
    # versions plus gliner 0.2.29, protobuf 6.31.1 and sentencepiece 0.2.2 (no new transitive dependency).
    "managed_python": "3.12.12",
    "uv": {
        "version": "0.12.15",
        "url": "https://files.pythonhosted.org/packages/1e/fd/432451d732917c49152a291de3ef171aa6b0f1a22d39780fb2c1f085ca4c/uv-0.12.15-py3-none-manylinux_2_17_x86_64.manylinux2014_x86_64.whl",
        "bytes": 20081404,
        "sha256": "aee9802f46bae436bd91751bb33ddeb379ef1596b5c19df193219d545d244b60",
    },
    "lock": "tutorials/requirements-colab.lock.txt",
    "run_all": (
        "Selecting **Run all** in a fresh supported runtime builds an isolated environment from the hash-locked pins (the "
        "kernel's own packages are left alone, so no restart is needed), stages and digest-verifies both pinned snapshots "
        "(the GLiNER multi-v2.1 weights and the mDeBERTa-v3 tokenizer assets, about 1.16 GB), builds the deterministic "
        "24-sentence biomedical NER sample in code (no download), validates it against the record contract, splits it with a "
        "seeded shuffle into 18 training and 6 validation sentences (refusing a split that leaves a label out of training), "
        "measures the zero-shot baseline on the validation split, fine-tunes the layers above the frozen mDeBERTa text encoder "
        "for three epochs, evaluates again and reports exact-span micro/macro F1 and the deltas, extracts entities from a new "
        "clinical sentence, exports the adapter weights with a tensor manifest to a `.pt` artifact, and reloads that artifact "
        "into a fresh pipeline to check every adapter tensor and the predictions on the whole validation split. The default "
        "path needs no repository clone, no DIMER worker or service, no credential, no upload dialog and no configuration edit "
        "(NOTEBOOK_SPEC 2.2 §5). It runs on CPU (the model stages take about a minute and a half on a laptop CPU); a CUDA "
        "runtime is used automatically when present. Building the isolated environment comes on top of this."
    ),
    "byod": (
        "After the tutorial workflow completes, set `USE_BYOD = True` in Section 4, select that cell and choose **Runtime → "
        "Run after** (it re-runs Section 4 and every later cell; Sections 5 and 6 reload the pretrained model first, so the "
        "sample adaptation never carries over into your baseline). Supply one `.json` or `.jsonl` file through the upload "
        "dialog on Colab, or by path in `BYOD_PATH` on any runtime. Any entity-label set works: list your labels in "
        "`BYOD_LABELS` (comma-separated), or leave it empty to use every label found in the file; each label must occur in "
        "the training split. Put a sentence of your own in `BYOD_INFERENCE_TEXT` for Section 8. The record schema with an "
        "example, the minimum size (14 records at the default `VAL_FRACTION`) and the ceilings are printed in Section 4 "
        "before the upload, and the file passes through the same validation, split, baseline, adaptation, evaluation, "
        "inference, export and reload cells as the sample. Uploaded files stay inside this runtime. BYOD is optional and "
        "never part of the default path."
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
        "GLiNER is an encoder-based entity extraction model that matches **any label you type** (for example `person`, "
        "`disease` or `gene_protein`) against candidate text spans. The sentence and the label names are encoded together by "
        "mDeBERTa-v3; every span of up to 12 words gets a representation, every label gets one too, and a span is returned "
        "with a label when their match score is above a threshold (0.5 here). Because the labels are just text, the model "
        "works **zero-shot**: it has never been trained on your label set. It is often weaker on specialised vocabularies, "
        "and this notebook measures how much a short **domain adaptation** on 18 biomedical sentences changes that.\n\n"
        "**Who this is for.** A learner who knows basic Python and has used Colab or Jupyter, and wants to see how a "
        "pretrained entity extractor is measured on labelled text and adapted to a new domain. No prior experience with "
        "GLiNER, named-entity recognition (NER) metrics or fine-tuning is assumed: entity spans, exact-span matching, "
        "precision, recall, F1, micro and macro averages and the frozen encoder are explained where they are first used and "
        "again in the **Glossary** at the end. A CPU runtime is enough.\n\n"
        "**Input → Model → Output.**\n\n"
        "| | Inference | Adaptation |\n"
        "|---|---|---|\n"
        "| Input | a text (up to 5,000 characters) and 1 to 25 label names | 18 training sentences, each a list of tokens with labelled entity spans `[start_token, end_token, label]` |\n"
        "| Model | GLiNER multi-v2.1: mDeBERTa-v3 encoder → span and label representations → match scores, threshold 0.5 | the same model with the mDeBERTa encoder frozen and the layers above it (11.4 M parameters) trained with AdamW |\n"
        "| Output | entities: `text`, `label`, character `start`/`end`, `score` | an adapter `.pt` file and exact-span precision / recall / F1 on the 6 validation sentences, before and after |\n\n"
        "**How to use this notebook.** Choose **Runtime → Run all**. Sections 1–3 are **infrastructure** — the isolated "
        "environment, the carried code and the model verification — and can be run without study; their code is collapsed. "
        "The learning path starts in Section 4. Form fields (`# @param`) are the only values meant to be edited; the defaults "
        "reproduce the recorded path. Each stage asks you to **predict** before it runs, and the next cell opens with **What "
        "to notice** and a collapsible **Check your reasoning** block with a worked answer from this repository's CPU run of "
        "the same stages (the recorded Kaggle Tesla T4 run of the previous notebook version printed the same +0.2332 micro-F1 delta; GPU numbers can differ in the last decimals). Each "
        "optional experiment names the cell to change and the cells to re-run. Section 11 is a **change-one-thing "
        "activity**. **Troubleshooting**, a **Glossary** and a **Conclusion** template are at the end. Your notes are "
        "optional and are not required submissions.\n\n"
        "**Roadmap:** 4 build, validate and split the sentences → 5 zero-shot baseline → 6 adapt the layers above the "
        "frozen encoder → 7 compare before and after → 8 extract entities from a new sentence → 9 export the adapter and "
        "check the reload → 10 write the outputs → 11 **change one thing: the number of epochs** → conclude. Core concepts "
        "are Sections 5 and 7 (what the model returns and how it is scored); evaluation practice is Sections 4 and 7 (split, "
        "baseline, delta); engineering and reproducibility are Sections 1–3, 9 and 10."
    ),
    "learning_objectives": (
        "by the end you should be able to (1) explain *text + label names → span and label representations → scored "
        "entities* and why GLiNER can take labels it was never trained on (Sections 5 and 8); (2) read exact-span "
        "precision, recall and F1, micro and macro, and the per-label counts behind them (Sections 5 and 7); (3) say which "
        "layers the adaptation trains and why the encoder stays frozen (Section 6); (4) predict and then check how the "
        "per-label F1 moves after adaptation, and say what 6 validation sentences can and cannot show (Section 7); (5) check "
        "that an exported adapter reproduces the evaluated model, tensor by tensor (Section 9); and (6) predict and then "
        "observe what one epoch instead of three changes (Section 11)."
    ),
    "exclusions": (
        "relation extraction, coreference resolution, nested or overlapping entities, unconstrained full-parameter "
        "fine-tuning of mDeBERTa on tiny data (prone to catastrophic forgetting), tuning the 0.5 score threshold, a "
        "benchmark evaluation (only 6 templated validation sentences are scored), or arbitrary unvalidated dataset "
        "formats. The repository exposes none of these."
    ),
    "prerequisites": [
        "- **Learner:** basic Python, and Colab or Jupyter familiarity. No prior NER knowledge: the notebook explains entity spans, token indices versus character offsets, exact-span matching, precision, recall, F1 and the micro/macro averages where they are first used, and the Glossary repeats them.",
        "- **Runtime:** a fresh **Linux x86_64** runtime — Google Colab, Kaggle or Linux Jupyter. Section 1 builds its own Python 3.12.12 environment from a hash-locked list of manylinux wheels, so the kernel's own Python version does not matter, and a Windows or macOS kernel is not supported (Section 1 stops with that message). CPU is enough: the model stages took about 70 s on a laptop CPU in this repository's check, plus about 2 minutes for the 1.16 GB snapshot download; a CUDA runtime is used automatically when present. The locked install (PyTorch 2.14.0 with its CUDA libraries, a few GB) is the largest download of the run.",
        "- **Expected warnings:** the cell after Section 3 prints `load_warnings`, the Python warnings captured while the model loaded. In this repository's CPU check it held one `UserWarning` from the SentencePiece tokenizer conversion about **byte fallback**; it does not change the tokens of the text used here. `transformers==4.57.6` may also log a line about an \"incorrect regex pattern … `fix_mistral_regex`\" while building the fast tokenizer; it refers to Mistral tokenizers, not this SentencePiece model, and is a log message, so it is not in `load_warnings`.",
        "- **Data contract:** a record is `{\"id\": str, \"tokenized_text\": [str, ...], \"ner\": [[start_token, end_token, label], ...]}` (`tokens` is accepted for `tokenized_text`); token indices count from 0 and the end is **inclusive**, so `[0, 1, \"person\"]` covers the first two tokens. Spans may not overlap, ids are unique, a record has at most 384 tokens of at most 100 characters, a dataset at most 1,000 records. The seeded split must leave at least 4 records in each split, so **a BYOD dataset needs at least 14 records** at the default `VAL_FRACTION = 0.25`, and every label must occur in the training split. Section 4 prints this before the upload control.",
        "- **Data:** the default sample is a deterministic 24-sentence synthetic biomedical dataset (`disease`, `chemical_drug`, `gene_protein`), built in code. Every sentence follows one template (a drug, then usually a gene or protein, then a disease), which matters when reading the results (Section 7).",
        "- **Privacy:** Do not upload confidential or restricted data to a hosted runtime unless you are authorized to process it there. The default path uploads nothing.",
    ],
    "cells": [
        {
            "md": (
                "### What the loader reported, and a clean starting point\n\n"
                "This short cell finishes the setup. It prints `load_warnings` (the Python warnings caught while the two "
                "snapshots were loaded; see Prerequisites) and defines `reset_to_pretrained()`. `pipe.adapt` changes the "
                "model **in memory**, so a cell that runs after an earlier adaptation would otherwise see adapted weights. "
                "Sections 5 and 6 call `reset_to_pretrained()` first: if `pipe` has been adapted, it is reloaded from the "
                "verified snapshots (a few seconds; nothing is downloaded again), so a re-run always measures the pretrained "
                "model and always trains from it."
            ),
            "code": (
                "import gc\n\n"
                "print({{'load_warnings': pipe.load_warnings}})\n\n\n"
                "def reset_to_pretrained():\n"
                "    \"\"\"Sections 5 and 6 start from the pretrained model: if an earlier pass adapted `pipe`, reload it from the verified snapshots.\"\"\"\n"
                "    global pipe\n"
                "    if not pipe.adapted:\n"
                "        return\n"
                "    pipe = None\n"
                "    gc.collect()\n"
                "    if torch.cuda.is_available():\n"
                "        torch.cuda.empty_cache()\n"
                "    pipe = GLiNERPipeline.from_pretrained(weights_dir=WEIGHTS_DIR, encoder_dir=ENCODER_WEIGHTS_DIR)\n"
                "    print({{'reloaded': 'the pretrained model, from the verified snapshots', 'reason': 'an earlier pass had adapted the model in memory'}})"
            ),
        },
        {
            "md": (
                "## 4. Build, validate and split the sentences\n\n"
                "An NER record here is a **tokenized sentence** plus its labelled **entity spans**. A span is "
                "`[start_token, end_token, label]` over token positions, counted from 0, with an **inclusive** end: in "
                "`[\"Imatinib\", \"targets\", \"BCR-ABL1\", ...]` the span `[0, 0, \"chemical_drug\"]` is *Imatinib*. The pipeline "
                "rebuilds the sentence text and converts each span to **character offsets** (`start`, `end`, end exclusive), "
                "which is what the model returns and what the scoring compares.\n\n"
                "By default the cell builds the 24-sentence biomedical sample in code. `validate_dataset` checks every record "
                "against the contract (ids, tokens, span bounds, no overlapping spans, labels from the declared set, every "
                "declared label present) and `split_ner_dataset` shuffles the records with `SEED` and keeps `VAL_FRACTION` "
                "of them for validation: 18 training and 6 validation sentences. The split is **not stratified**; instead it "
                "refuses a split that leaves any label out of the training split, and the cell prints how many spans of each "
                "label landed on each side (a label missing from validation would be named). Four refusal probes then show "
                "the validator stopping bad records before any model runs.\n\n"
                "**Bring your own data:** set `USE_BYOD = True`, then either leave `BYOD_PATH` empty to get an upload dialog "
                "(Colab only) or set it to a `.json` / `.jsonl` file already in this runtime, and choose **Runtime → Run "
                "after** on this cell. `BYOD_LABELS` takes your comma-separated label set (empty = every label in the file); "
                "`BYOD_INFERENCE_TEXT` takes a sentence for Section 8 (empty = the first validation sentence, which is then not "
                "unseen). The cell prints the record schema, an example and the limits before it reads anything. A cancelled "
                "upload, a runtime without the Colab dialog and no `BYOD_PATH`, a missing path, a file that is not `.json` / "
                "`.jsonl` (a CSV, for example), spans written as objects instead of `[start, end, label]` lists, a label outside "
                "`BYOD_LABELS`, too few records for the split, or a label that lands only in validation each stop here with a "
                "message naming the rule and the fix. Under BYOD the printed provenance and `result.json` describe your file.\n\n"
                "**Predict before running:** the split is a plain shuffle, not stratified. Do you expect all three labels to "
                "appear in the 6 validation sentences?"
            ),
            "code": (
                "import hashlib\n"
                "import json\n"
                "from pathlib import Path\n\n"
                "USE_BYOD = False  # @param {{type:\"boolean\"}}\n"
                "BYOD_PATH = ''  # @param {{type:\"string\"}}\n"
                "BYOD_LABELS = ''  # @param {{type:\"string\"}}\n"
                "BYOD_INFERENCE_TEXT = ''  # @param {{type:\"string\"}}\n"
                "VAL_FRACTION = 0.25  # @param {{type:\"number\"}}\n"
                "SEED = 42  # @param {{type:\"integer\"}}\n\n"
                "BYOD_EXAMPLE = [\n"
                "    {{'id': 'doc-1', 'tokenized_text': ['Ada', 'Lovelace', 'worked', 'in', 'London', '.'], 'ner': [[0, 1, 'person'], [4, 4, 'location']]}},\n"
                "    {{'id': 'doc-2', 'tokenized_text': ['Grace', 'Hopper', 'joined', 'the', 'US', 'Navy', '.'], 'ner': [[0, 1, 'person'], [4, 5, 'organization']]}},\n"
                "]\n"
                "print({{'record_schema': RECORD_SCHEMA_HINT, 'file_types': list(BYOD_SUFFIXES), 'example_jsonl_lines': [json.dumps(r) for r in BYOD_EXAMPLE]}})\n"
                "print({{'limits': {{'records_min': minimum_records(VAL_FRACTION), 'records_max': MAX_DATASET_EXAMPLES, 'records_min_per_split': MIN_DATASET_EXAMPLES, 'tokens_per_record_max': MAX_TOKENS_PER_EXAMPLE, 'chars_per_token_max': MAX_TOKEN_CHARS, 'labels_max': MAX_LABELS, 'chars_per_label_max': MAX_LABEL_CHARS, 'inference_text_chars_max': MAX_TEXT_CHARS}}}})\n\n"
                "if USE_BYOD:\n"
                "    if BYOD_PATH.strip():\n"
                "        byod_path = Path(BYOD_PATH.strip()).expanduser()\n"
                "        if not byod_path.is_file():\n"
                "            raise FileNotFoundError(f'BYOD_PATH = {{BYOD_PATH!r}} is not a file in this runtime: fix the path, or leave BYOD_PATH empty to upload a file in Colab.')\n"
                "        file_name = byod_path.name\n"
                "    else:\n"
                "        try:\n"
                "            from google.colab import files\n"
                "        except ImportError:\n"
                "            raise RuntimeError('The upload dialog needs Google Colab. Elsewhere, put the .json or .jsonl file in this runtime and set BYOD_PATH to its path.') from None\n"
                "        uploaded = files.upload()\n"
                "        if len(uploaded) != 1:\n"
                "            raise ValueError(f'Upload exactly one .json or .jsonl file (received {{len(uploaded)}}). Run this cell again and choose a file, or set BYOD_PATH to a file already in this runtime.')\n"
                "        file_name, payload = next(iter(uploaded.items()))\n"
                "        byod_path = Path('work') / Path(file_name).name\n"
                "        byod_path.parent.mkdir(parents=True, exist_ok=True)\n"
                "        byod_path.write_bytes(payload)\n"
                "    declared_labels = [label.strip() for label in BYOD_LABELS.split(',') if label.strip()] or None\n"
                "    raw_dataset = load_byod_dataset(byod_path, allowed_labels=declared_labels)\n"
                "    LABELS = declared_labels or sorted({{span[2] for record in raw_dataset for span in record['ner']}})\n"
                "    data_source = 'BYOD (' + file_name + ')'\n"
                "else:\n"
                "    raw_dataset = generate_synthetic_ner_dataset()\n"
                "    LABELS = list(ADAPT_CLASSES)\n"
                "    data_source = 'Synthetic biomedical dataset (24 records)'\n\n"
                "validate_inputs('label check', LABELS)  # the label set must also satisfy the model's input ceilings\n"
                "val_manifest = validate_dataset(raw_dataset, LABELS)\n"
                "train_records, val_records = split_ner_dataset(raw_dataset, val_fraction=VAL_FRACTION, seed=SEED, labels=LABELS)\n"
                "coverage = split_label_coverage(train_records, val_records, LABELS)\n"
                "dataset_sha256 = hashlib.sha256(json.dumps([{{k: r[k] for k in ('id', 'tokenized_text', 'ner')}} for r in raw_dataset], sort_keys=True).encode('utf-8')).hexdigest()\n\n"
                "print({{'data_source': data_source, 'labels': LABELS, 'total_records': len(raw_dataset), 'train_records': len(train_records), 'val_records': len(val_records), 'dataset_sha256': dataset_sha256[:16] + '...'}})\n"
                "print({{'span_counts': coverage}})\n"
                "example = train_records[0]\n"
                "print({{'example': {{'id': example['id'], 'tokenized_text': example['tokenized_text'], 'ner': example['ner'], 'text': example['text'], 'character_spans': [(s['start'], s['end'], s['label'], s['text']) for s in example['spans']]}}}})\n\n"
                "probes = {{\n"
                "    'duplicate id': [dict(r, id='same') for r in train_records[:4]],\n"
                "    'span past the last token': [dict(train_records[0], ner=[[0, 99, LABELS[0]]]), *train_records[1:4]],\n"
                "    'span written as an object': [dict(train_records[0], ner=[{{'start': 0, 'end': 0, 'label': LABELS[0]}}]), *train_records[1:4]],\n"
                "    'too few records to split': train_records[:5],\n"
                "}}\n"
                "for name, probe in probes.items():\n"
                "    try:\n"
                "        validate_dataset(probe, LABELS, require_all_labels=False)\n"
                "        split_ner_dataset(probe, val_fraction=VAL_FRACTION, seed=SEED, labels=[])\n"
                "        print({{'probe': name, 'verdict': 'accepted'}})\n"
                "    except (KeyError, TypeError, ValueError) as exc:\n"
                "        print({{'probe': name, 'rejected': str(exc)[:150]}})"
            ),
        },
        {
            "md": (
                "**What to notice:** the schema and limits printed first, 18/6 records, the span counts per label on each "
                "side, one example in both token and character form (`(0, 7, 'chemical_drug', 'Aspirin')`-style tuples), and "
                "four rejected probes.\n\n"
                "<details><summary>Check your reasoning</summary>Yes, with the default seed all three labels appear in "
                "validation: 6 `disease`, 6 `chemical_drug` and 6 `gene_protein` spans in the 6 validation sentences, 19 / 19 / "
                "21 in training. That is a property of this sample, not of the split: every sentence has a drug and a disease, "
                "and 22 of 24 have a gene or protein, so almost any shuffle covers all three. A BYOD file with a rare label can "
                "land it only in validation; the split then stops and names the label, and the fix is more examples or another "
                "`SEED`.</details>\n\n"
                "## 5. Measure the zero-shot baseline\n\n"
                "Before any training, `pipe.evaluate(val_records, labels=LABELS)` asks the pretrained model for every label on "
                "each validation sentence and compares its entities with the gold spans. Scoring is **exact-span**: a "
                "prediction counts as a hit only if its `(start, end, label)` equals a gold span exactly, so *metastatic "
                "melanoma* against gold *melanoma* is both a false positive and a miss. **Precision** = hits / predicted spans, "
                "**recall** = hits / gold spans, **F1** = their harmonic mean. **Micro** averages pool every span of every label "
                "(the big labels dominate); **macro** F1 is the plain mean of the per-label F1 scores (each label counts once). "
                "`per_class` holds `hits`, `predicted` and `gold` per label, from which all of these follow.\n\n"
                "The cell first calls `reset_to_pretrained()`, so on a re-run this is still the pretrained model, never an "
                "adapted one.\n\n"
                "**Predict before running:** the labels are the snake-case names `disease`, `chemical_drug` and `gene_protein`. "
                "Which one do you expect the zero-shot model to find worst, and why?"
            ),
            "code": (
                "reset_to_pretrained()\n"
                "baseline_eval = pipe.evaluate(val_records, labels=LABELS)\n\n\n"
                "def metrics_table(evaluation):\n"
                "    rows = [f\"{{'label':<16}}{{'gold':>5}}{{'predicted':>10}}{{'hits':>6}}{{'precision':>10}}{{'recall':>8}}{{'F1':>8}}\"]\n"
                "    for label, row in evaluation['per_class'].items():\n"
                "        rows.append(f\"{{label:<16}}{{row['gold']:>5}}{{row['predicted']:>10}}{{row['hits']:>6}}{{row['precision']:>10.4f}}{{row['recall']:>8.4f}}{{row['f1']:>8.4f}}\")\n"
                "    micro = evaluation['micro']\n"
                "    rows.append(f\"{{'micro (pooled)':<16}}{{'':>5}}{{'':>10}}{{micro['hits']:>6}}{{micro['precision']:>10.4f}}{{micro['recall']:>8.4f}}{{micro['f1']:>8.4f}}\")\n"
                "    rows.append(f\"{{'macro F1 (mean of labels)':<56}}{{evaluation['macro']['f1']:>8.4f}}\")\n"
                "    return '\\n'.join(rows)\n\n\n"
                "print(f\"Zero-shot baseline on the validation split ({{baseline_eval['num_samples']}} sentences, threshold {{baseline_eval['threshold']}}):\")\n"
                "print(metrics_table(baseline_eval))"
            ),
        },
        {
            "md": (
                "**What to notice:** one row per label with its gold, predicted and hit counts, then the micro and macro "
                "lines. Find the label whose `hits` is far below its `gold`.\n\n"
                "<details><summary>Check your reasoning</summary>`gene_protein` is the weak one: F1 0.2222, 1 hit of 6 gold "
                "spans, against much better `disease` and `chemical_drug` rows; micro F1 is 0.6857 (precision 0.7059, recall "
                "0.6667) and macro F1 0.6375. Two things are at work and this notebook cannot separate them: gene symbols such "
                "as *PDCD1* or *MS4A1* are hard to recognise without domain context, and the prompt is the artificial label "
                "name `gene_protein`, not natural words such as \"gene or protein\". Adaptation trains on exactly these label "
                "names, so part of any later gain may be the model learning what the names mean.</details>\n\n"
                "## 6. Adapt the layers above the frozen encoder\n\n"
                "`pipe.adapt(train_records, val_records, ...)` runs a short PyTorch training loop on the 18 training "
                "sentences. **Freezing** the mDeBERTa text encoder (`freeze_text_encoder=True`) means its 277.5 M parameters "
                "are not updated; what trains is everything above it — the projection on top of the encoder "
                "(`token_rep_layer` projection), the BiLSTM (`rnn`), the span representation layer and the prompt "
                "(label) projection, 11.4 M parameters in all. The cell prints the trainable and frozen counts by module. "
                "Training only this small part keeps memory low, runs on CPU and limits how far 18 sentences can pull the "
                "model. AdamW updates those weights to lower GLiNER's span-matching loss, `EPOCHS` passes over the training "
                "set in batches of `BATCH_SIZE`; after each epoch the validation split is scored, so the history shows loss "
                "and validation F1 side by side. The validation split is therefore also the **training monitor**: there is no "
                "separate, untouched test split (Section 7 comes back to this).\n\n"
                "The cell starts with `reset_to_pretrained()`: if you change a field and re-run from here, training starts "
                "again from the pretrained model instead of continuing from the last run. `pipe.adapt` itself refuses a model "
                "that was already adapted.\n\n"
                "**Predict before running:** will the training loss fall steadily over the three epochs, and will the "
                "validation F1 rise every epoch?"
            ),
            "code": (
                "import time\n\n"
                "EPOCHS = 3  # @param {{type:\"integer\"}}\n"
                "LEARNING_RATE = 5e-5  # @param {{type:\"number\"}}\n"
                "BATCH_SIZE = 4  # @param {{type:\"integer\"}}\n\n"
                "reset_to_pretrained()\n"
                "started = time.perf_counter()\n"
                "adapt_result = pipe.adapt(\n"
                "    train_records=train_records,\n"
                "    val_records=val_records,\n"
                "    epochs=EPOCHS,\n"
                "    learning_rate=LEARNING_RATE,\n"
                "    batch_size=BATCH_SIZE,\n"
                "    freeze_text_encoder=True,\n"
                "    seed=SEED,\n"
                "    labels=LABELS,\n"
                ")\n"
                "elapsed = time.perf_counter() - started\n\n"
                "print({{'trainable_parameters': adapt_result['trainable_parameters'], 'frozen_parameters': adapt_result['frozen_parameters']}})\n"
                "print({{'trainable_modules': adapt_result['trainable_modules'], 'frozen_modules': adapt_result['frozen_modules']}})\n"
                "print(f'Adaptation completed in {{elapsed:.1f}} s ({{EPOCHS}} epochs, learning rate {{LEARNING_RATE}}, batch size {{BATCH_SIZE}}, seed {{SEED}}).')\n"
                "for step in adapt_result['history']:\n"
                "    print(f\"  Epoch {{step['epoch']}}: loss={{step['loss']:.4f}}  validation micro F1={{step.get('val_f1', 0.0):.4f}}\")"
            ),
        },
        {
            "md": (
                "**What to notice:** the trainable count (about 11.4 M) beside the frozen one (about 277.5 M), the module "
                "names on each side (`model.token_rep_layer` appears in both: its mDeBERTa encoder is frozen, its projection "
                "trains), and the loss and validation F1 per epoch.\n\n"
                "<details><summary>Check your reasoning</summary>In this repository's CPU check the loss fell every epoch (11.28, 8.02, 6.26) and the validation micro F1 rose every epoch (0.8108, 0.8649, 0.9189). Neither is guaranteed: with 18 sentences in 5 batches of up to 4 the loss is noisy, and another seed, learning rate or device can make an epoch stall; GPU arithmetic can also move the losses in the second decimal. The counts were 11,418,112 trainable and 277,531,392 frozen parameters: `model.span_rep_layer` 7,347,712, `model.prompt_rep_layer` 2,099,712, `model.rnn` 1,576,960 and the `model.token_rep_layer` projection 393,728 trainable; all 277.5 M frozen ones are the mDeBERTa encoder inside `model.token_rep_layer`.</details>\n\n"
                "## 7. Compare before and after\n\n"
                "The adapted model is scored on the same 6 validation sentences, the deltas are computed against the Section 5 "
                "baseline, and everything is written to `outputs/{stem}_evaluation_report.json`. The cell also appends this run "
                "to `run_history`, so a re-run with other settings (Section 11) shows up as a second row.\n\n"
                "**How much to read into it.** These are **tutorial evidence, not a benchmark**: 6 validation sentences "
                "with 18 gold spans, all written from one template (the drug first, then a gene, then a disease), and the "
                "same split monitored training every epoch, so nothing here is an independent test. With 18 spans one span is "
                "worth about 0.05 of recall. The baseline was prompted with snake-case label names that the adapted model has "
                "now been trained on (Section 5). A large delta here says the adaptation fitted this template and these label "
                "names; it does not say how the model would do on other biomedical text.\n\n"
                "**Predict before running:** which label will gain the most F1, and could any label get worse?"
            ),
            "code": (
                "import os\n\n"
                "os.makedirs('outputs', exist_ok=True)\n"
                "adapted_eval = pipe.evaluate(val_records, labels=LABELS)\n\n"
                "delta_f1 = round(adapted_eval['micro']['f1'] - baseline_eval['micro']['f1'], 4)\n"
                "delta_precision = round(adapted_eval['micro']['precision'] - baseline_eval['micro']['precision'], 4)\n"
                "delta_recall = round(adapted_eval['micro']['recall'] - baseline_eval['micro']['recall'], 4)\n"
                "delta_macro_f1 = round(adapted_eval['macro']['f1'] - baseline_eval['macro']['f1'], 4)\n"
                "per_label_delta = {{label: round(adapted_eval['per_class'][label]['f1'] - baseline_eval['per_class'][label]['f1'], 4) for label in LABELS}}\n"
                "evidence_note = f\"tutorial evidence: {{len(val_records)}} validation sentences, {{sum(r['gold'] for r in adapted_eval['per_class'].values())}} gold spans; the validation split also monitored training (no independent test split)\" + ('; one sentence template; the label names were part of the training' if not USE_BYOD else '')\n\n"
                "eval_report = {{\n"
                "    'task': 'domain-adapted named-entity recognition',\n"
                "    'domain': 'biomedical / clinical' if not USE_BYOD else data_source,\n"
                "    'labels': list(LABELS),\n"
                "    'evidence': evidence_note,\n"
                "    'baseline_eval': baseline_eval,\n"
                "    'adapted_eval': adapted_eval,\n"
                "    'delta': {{\n"
                "        'f1_delta': delta_f1,\n"
                "        'precision_delta': delta_precision,\n"
                "        'recall_delta': delta_recall,\n"
                "        'macro_f1_delta': delta_macro_f1,\n"
                "        'per_label_f1_delta': per_label_delta,\n"
                "    }},\n"
                "    'split': {{'seed': SEED, 'val_fraction': VAL_FRACTION, 'train_ids': [r['id'] for r in train_records], 'validation_ids': [r['id'] for r in val_records], 'dataset_sha256': dataset_sha256}},\n"
                "    'training_summary': adapt_result,\n"
                "}}\n\n"
                "with open('outputs/{stem}_evaluation_report.json', 'w', encoding='utf-8') as f:\n"
                "    json.dump(eval_report, f, indent=2)\n\n"
                "run_history = globals().get('run_history', [])\n"
                "run_history.append({{'data': data_source, 'epochs': EPOCHS, 'learning_rate': LEARNING_RATE, 'seed': SEED, 'epoch1_loss': adapt_result['history'][0]['loss'] if adapt_result['history'] else None, 'baseline_micro_f1': baseline_eval['micro']['f1'], 'adapted_micro_f1': adapted_eval['micro']['f1'], 'adapted_macro_f1': adapted_eval['macro']['f1']}})\n\n"
                "print('Adapted model on the validation split:')\n"
                "print(metrics_table(adapted_eval))\n"
                "print(f\"\\nMicro F1 {{baseline_eval['micro']['f1']:.4f}} -> {{adapted_eval['micro']['f1']:.4f}} ({{delta_f1:+.4f}}); macro F1 {{baseline_eval['macro']['f1']:.4f}} -> {{adapted_eval['macro']['f1']:.4f}} ({{delta_macro_f1:+.4f}})\")\n"
                "print({{'per_label_f1_delta': per_label_delta}})\n"
                "print('Reading: ' + evidence_note + '.')\n"
                "print('Runs so far in this session:')\n"
                "for i, run in enumerate(run_history, 1):\n"
                "    print(f'  {{i}}. {{run}}')"
            ),
        },
        {
            "md": (
                "**What to notice:** the per-label table after adaptation, the micro and macro deltas, the per-label deltas, "
                "and the `Reading:` line that states the sample size.\n\n"
                "<details><summary>Check your reasoning</summary>`gene_protein` gains most: F1 0.2222 → 0.9231 (1 → 6 hits of 6 gold spans, with 7 predicted). `chemical_drug` goes 0.8571 → 1.0 because its two extra predictions disappear (6 hits both times), and `disease` stays at 0.8333 (5 of 6 both times). Micro F1 0.6857 → 0.9189 (+0.2332; hits 12 → 17 of 18), macro F1 0.6375 → 0.9188 (+0.2813). A label could get worse — nothing in the training forces every label up — but none did here. The whole gain is five more hits on six sentences written from one template, with label names the model has now been trained on, so it is a demonstration of the workflow, not an estimate for other text.</details>\n\n"
                "## 8. Extract entities from a new sentence\n\n"
                "`pipe.detect(text, labels=LABELS, threshold=DEFAULT_THRESHOLD)` returns each entity with its `text`, `label`, "
                "character `start`/`end` (end exclusive) and `score`. The **score** is the model's match score between span and "
                "label, between 0 and 1; it is not a calibrated probability, and spans below the 0.5 threshold are dropped. The "
                "cell checks the output contract — offsets inside the text, `text` equal to the slice it claims, labels from the "
                "requested set — and stops with an error if any check fails. The default sentence was written for this "
                "notebook: none of its entities occurs in the training split, although a near-twin (*Pembrolizumab blocks PDCD1 "
                "pathway activation in metastatic melanoma*) is one of the validation sentences. Under BYOD the sentence is "
                "`BYOD_INFERENCE_TEXT`.\n\n"
                "**Predict before running:** which three spans should come back for *" + SAMPLE_SENTENCE + "*, and will "
                "*receptor* or *adult patients* be included?"
            ),
            "code": (
                "if USE_BYOD:\n"
                "    test_sentence = BYOD_INFERENCE_TEXT.strip() or val_records[0]['text']\n"
                "    if not BYOD_INFERENCE_TEXT.strip():\n"
                "        print('BYOD_INFERENCE_TEXT is empty: using the first validation sentence, so this is not unseen text.')\n"
                "else:\n"
                "    test_sentence = '" + SAMPLE_SENTENCE + "'\n"
                "inference_result = pipe.detect(test_sentence, labels=LABELS, threshold=DEFAULT_THRESHOLD)\n\n"
                "entities = inference_result['entities']\n"
                "checks = {{\n"
                "    'offsets_inside_text': all(0 <= e['start'] < e['end'] <= len(test_sentence) for e in entities),\n"
                "    'span_text_matches': all(test_sentence[e['start']:e['end']] == e['text'] for e in entities),\n"
                "    'labels_valid': all(e['label'] in LABELS for e in entities),\n"
                "}}\n"
                "if not all(checks.values()):\n"
                "    raise RuntimeError(f'The entities break the output contract: {{checks}}')\n\n"
                "print(f'Input: \"{{test_sentence}}\"')\n"
                "print(f'Detected {{len(entities)}} entities (checks: {{checks}}):')\n"
                "for i, e in enumerate(entities, 1):\n"
                "    print(f\"  {{i}}. [{{e['start']:>2}}:{{e['end']:>2}}] {{e['label']:<14}} (score={{e['score']:.4f}}): {{e['text']}}\")"
            ),
        },
        {
            "md": (
                "**What to notice:** three entities with their character offsets and scores, and all three contract checks "
                "`True`.\n\n"
                "<details><summary>Check your reasoning</summary>*Pembrolizumab* → `chemical_drug` (characters 0–13, score 0.9979), *PDCD1* → `gene_protein` (21–26, 0.9515) and *metastatic melanoma* → `disease` (45–64, 0.9975) in this repository's CPU check. Neither *receptor* nor *adult patients* is returned. *metastatic melanoma* comes back as one span, the way the training sentences annotate diseases; an annotation scheme that marked only *melanoma* would score this as a miss under exact-span matching.</details>\n\n"
                "## 9. Export the adapter and check the reload\n\n"
                "To share the adapted model without copying the 1.16 GB base checkpoint, `pipe.save_artifact` writes only the "
                "trained tensors, a manifest of their names, shapes, dtypes and SHA-256 digests, the base-model identity (both "
                "snapshot revisions) and run metadata to `outputs/{stem}_adapter.pt`. `GLiNERPipeline.from_artifact` then "
                "loads the pretrained snapshots again and overlays the adapter. It refuses an adapter with no tensors, tensor "
                "names this model does not have, a shape or digest that differs from the manifest, or a different base or "
                "encoder revision. The cell compares **every adapter tensor** of the reloaded model with the in-memory one "
                "(bit for bit) and the predictions of both models on **every validation sentence** plus the Section 8 sentence, "
                "and stops with an error if anything differs.\n\n"
                "**Predict before running:** should the reloaded model's scores equal the in-memory model's exactly, or only "
                "approximately? Why?"
            ),
            "code": (
                "artifact_path = Path('outputs/{stem}_adapter.pt')\n"
                "pipe.save_artifact(\n"
                "    artifact_path,\n"
                "    metadata={{\n"
                "        'domain': 'biomedical' if not USE_BYOD else data_source,\n"
                "        'classes': list(LABELS),\n"
                "        'epochs': EPOCHS,\n"
                "        'learning_rate': LEARNING_RATE,\n"
                "        'weight_decay': adapt_result['weight_decay'],\n"
                "        'seed': SEED,\n"
                "        'trainable_parameters': adapt_result['trainable_parameters'],\n"
                "        'metrics': adapted_eval['micro'],\n"
                "    }},\n"
                ")\n"
                "adapter_manifest = torch.load(artifact_path, map_location='cpu', weights_only=True)['adapter_manifest']\n"
                "print(f'Saved adapter artifact: {{artifact_path}} ({{artifact_path.stat().st_size / (1024*1024):.2f}} MB, {{len(adapter_manifest)}} tensors)')\n\n"
                "reloaded_pipe = GLiNERPipeline.from_artifact(\n"
                "    artifact_path,\n"
                "    weights_dir=WEIGHTS_DIR,\n"
                "    encoder_dir=ENCODER_WEIGHTS_DIR,\n"
                ")\n"
                "tensor_parity = pipe.adapter_parity(reloaded_pipe, list(adapter_manifest))\n"
                "if tensor_parity['tensors_identical'] != tensor_parity['tensors_compared']:\n"
                "    raise RuntimeError(f'Reload parity failed: adapter tensors differ: {{tensor_parity}}')\n\n"
                "parity_texts = [r['text'] for r in val_records] + [test_sentence]\n"
                "prediction_mismatches = []\n"
                "for text in parity_texts:\n"
                "    original = pipe.detect(text, labels=LABELS, threshold=DEFAULT_THRESHOLD)['entities']\n"
                "    reloaded = reloaded_pipe.detect(text, labels=LABELS, threshold=DEFAULT_THRESHOLD)['entities']\n"
                "    same = len(original) == len(reloaded) and all(\n"
                "        (a['start'], a['end'], a['label']) == (b['start'], b['end'], b['label']) and abs(a['score'] - b['score']) < 1e-5\n"
                "        for a, b in zip(original, reloaded, strict=False)\n"
                "    )\n"
                "    if not same:\n"
                "        prediction_mismatches.append(text)\n"
                "if prediction_mismatches:\n"
                "    raise RuntimeError(f'Reload parity failed: predictions differ on {{len(prediction_mismatches)}} of {{len(parity_texts)}} sentences, e.g. {{prediction_mismatches[0]!r}}')\n"
                "reload_parity = {{'adapter_tensors': tensor_parity, 'sentences_compared': len(parity_texts), 'sentences_identical': len(parity_texts) - len(prediction_mismatches)}}\n"
                "print({{'reload_parity': reload_parity}})\n"
                "print('Parity verification passed: the reloaded artifact has identical adapter tensors and identical predictions.')"
            ),
        },
        {
            "md": (
                "**What to notice:** the tensor count, `tensors_identical` equal to `tensors_compared`, and every sentence "
                "identical.\n\n"
                "<details><summary>Check your reasoning</summary>Exactly. The reloaded model is the same digest-verified base files plus the same adapter tensors (checked against their SHA-256 in the manifest on load and compared bit for bit here), and inference in evaluation mode is deterministic on one device, so the scores are identical; the 1e-5 tolerance in the comparison is only a guard. In this repository's CPU check: 26 tensors identical, 7 of 7 sentences identical, a 45.7 MB adapter file. A mismatch would mean the file or the base snapshots changed, and the cell would stop.</details>\n\n"
                "## 10. Write the outputs and lineage\n\n"
                "Two more files go under `outputs/`: `{stem}_result.json` with the provenance (both model identities, the "
                "repository revision, the runtime), the label set, the split ids and dataset digest, the evaluation report, the "
                "reload check and the Section 8 entities; and `{stem}_entities.csv` with those entities as a table. The cell "
                "lists all four output files."
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
                "    'data_source': data_source,\n"
                "    'labels': list(LABELS),\n"
                "    'evaluation_report': eval_report,\n"
                "    'reload_parity': reload_parity,\n"
                "    'run_history': run_history,\n"
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
        {
            "md": (
                "## 11. Your turn — change one thing: the number of epochs\n\n"
                "**Predict → Change one thing → Run → Observe → Explain**\n\n"
                "1. **Predict.** With one epoch instead of three, will the adapted micro F1 be lower, the same or higher than "
                "the default run's? Will the epoch-1 loss change? Write your guess down first.\n"
                "2. **Change one thing.** In Section 6 set `EPOCHS = 1`. Leave every other field as it is.\n"
                "3. **Run.** Select the Section 6 cell and choose **Runtime → Run after**. Section 6 reloads the pretrained "
                "model before training (so this is a fresh one-epoch run, not a fourth epoch), and Sections 7–11 run again; the "
                "output files are overwritten with the one-epoch run.\n"
                "4. **Observe.** The table below prints every run of this session from `run_history`. Compare `epoch1_loss`, "
                "`adapted_micro_f1` and `adapted_macro_f1` between the rows, and look at the per-label table Section 7 printed.\n"
                "5. **Explain.** Why is the epoch-1 loss the same in both rows? Is the difference in F1 larger than one span "
                "(about 0.05 on these 18 gold spans)? Set `EPOCHS = 3` and **Run after** from Section 6 again to restore the "
                "default outputs.\n\n"
                "**Optional experiments** (each re-run from the cell named): a different `SEED` in Section 4 (Run after from "
                "Section 4) changes which sentences are held out; a larger `LEARNING_RATE` in Section 6 (Run after from Section "
                "6) shows how quickly 18 sentences can be over-fitted; your own data via BYOD (Section 4)."
            ),
            "code": (
                "print(f\"{{'run':<5}}{{'epochs':>7}}{{'learning_rate':>15}}{{'seed':>6}}{{'epoch1_loss':>13}}{{'baseline_F1':>13}}{{'adapted_F1':>12}}{{'adapted_macro':>15}}\")\n"
                "for i, run in enumerate(run_history, 1):\n"
                "    print(f\"{{i:<5}}{{run['epochs']:>7}}{{run['learning_rate']:>15}}{{run['seed']:>6}}{{run['epoch1_loss']:>13.4f}}{{run['baseline_micro_f1']:>13.4f}}{{run['adapted_micro_f1']:>12.4f}}{{run['adapted_macro_f1']:>15.4f}}\")\n"
                "if len(run_history) == 1:\n"
                "    print('One run so far: follow steps 2-3 above to add a one-epoch row.')"
            ),
        },
        {
            "md": (
                "**What to notice:** one row after **Run all**; a second row after the activity, with the same `epoch1_loss` "
                "and `baseline_F1`.\n\n"
                "<details><summary>Check your reasoning</summary>In this repository's CPU check the one-epoch run gave micro F1 0.8108 and macro F1 0.8077 (`gene_protein` 0.6667, 4 of 6 hits) against 0.9189 / 0.9188 for three epochs; the epoch-1 loss was 11.2828 in both rows. It is the same because Section 6 reloads the pretrained model and reseeds before training, so the first epoch of both runs sees the same weights, batches and dropout; a fresh start is what makes the comparison fair. The difference is two hits (15 versus 17 of 18 gold spans), more than one span but still a small, single-template sample: it shows that this sample needs more than one pass, not how many epochs other data needs.</details>"
            ),
        },
    ],
    "closing": (
        "## Interpretation and limits\n\n"
        "Adaptation here trains the layers above a frozen mDeBERTa encoder so that span and label representations match this "
        "sample's entity types and label names. Freezing the encoder keeps the trained part small (11.4 M of 289 M "
        "parameters) and limits how far 18 sentences can move the model. Each returned span is an exact character slice of "
        "the input; its score is the model's uncalibrated span–label match score. Exact-span F1 requires exact agreement on "
        "`(start, end, label)` triples: boundary mismatches of even one character receive zero credit. The exported adapter "
        "carries only the trained tensors, their manifest and the lineage headers.\n\n"
        "The numbers are **tutorial evidence**: 6 validation sentences with 18 gold spans, all written from one template, a "
        "validation split that also monitored training (there is no independent test split), and a zero-shot baseline "
        "prompted with snake-case label names that the adapted model was then trained on. The delta shows that the "
        "adaptation fits this template and these label names; it is not an estimate of performance on other biomedical "
        "text, and it says nothing about other domains.\n\n"
        "Successful execution proves that the recorded repository revision's pipeline modules, carried in this standalone notebook, "
        "can acquire and digest-verify the pinned model, validate the demonstrated dataset contract, execute bounded fine-tuning, "
        "evaluate metrics against the pre-adaptation baseline, and emit and reload the shown machine-readable artifacts — without the "
        "repository being reachable. It does **not** establish benchmark superiority, production fitness, or generalized performance "
        "across unseen domains.\n\n"
        "## Troubleshooting\n\n"
        "- **Section 1 stops with \"needs a Linux x86_64 runtime\".** The locked environment uses manylinux wheels; use Google Colab, Kaggle or a Linux Jupyter server.\n"
        "- **Section 1 fails while downloading.** PyPI or the managed-Python download was interrupted: run the cell again (finished parts are reused). A size/SHA-256 refusal of the `uv` wheel that repeats means the download is being altered.\n"
        "- **\"The isolated environment's Python process exited\".** Usually out of memory. Restart the session and choose **Run all**; on a small runtime, close other notebooks first.\n"
        "- **Section 3 fails with a digest or size mismatch.** A snapshot file was truncated or altered: delete the `weights/` folder in the runtime and run Section 3 again.\n"
        "- **The Section 5 numbers look like the adapted ones.** They cannot after this fix (`reset_to_pretrained()`); if you edited the cells, re-run from Section 3.\n"
        "- **BYOD errors.** Each names the failed rule: a file that is not `.json`/`.jsonl`, a span that is not a `[start_token, end_token, label]` list, a label outside `BYOD_LABELS`, too few records (at least 14 at `VAL_FRACTION = 0.25`), or a label that landed only in validation (add examples or change `SEED`). Fix the file or the field and **Run after** from Section 4.\n"
        "- **Reload parity failed.** The adapter file was changed after export, or the base snapshots differ: re-run from Section 6.\n\n"
        "## Glossary\n\n"
        "- **Named-entity recognition (NER):** finding spans of text that name things of given types (a drug, a disease, a person).\n"
        "- **Span:** a contiguous piece of text, given here either as token positions `[start_token, end_token]` (inclusive) or as character offsets `start`/`end` (end exclusive).\n"
        "- **Zero-shot:** using label names the model was never trained on, as plain text prompts.\n"
        "- **Exact-span match:** a prediction counts only if its start, end and label all equal a gold span.\n"
        "- **Precision / recall / F1:** hits divided by predicted spans / by gold spans / their harmonic mean.\n"
        "- **Micro / macro average:** micro pools all spans of all labels; macro averages the per-label F1 scores, each label counting once.\n"
        "- **Frozen encoder:** the mDeBERTa layers whose weights are not updated during adaptation.\n"
        "- **Adapter:** the trained tensors saved separately from the base checkpoint and overlaid on it at load time.\n"
        "- **Threshold:** the score below which a span is dropped (0.5 here).\n"
        "- **Training monitor:** data scored during training to watch progress; here it is the validation split, so it is not an independent test.\n\n"
        "## Conclusion (your notes)\n\n"
        "Optional; not a required submission. Complete these sentences from your own run:\n\n"
        "1. The zero-shot model was weakest on the label ___, because ___.\n"
        "2. After adaptation the micro F1 moved from ___ to ___; with 18 gold spans, one span is worth about ___.\n"
        "3. With one epoch instead of three, I observed ___.\n"
        "4. Before trusting a gain like this on my own text, I would want ___.\n\n"
        "## References\n\n"
        "- Repository README: https://github.com/kurtvalcorza/gliner-ner-pipeline/blob/main/README.md\n"
        "- Repository model card: https://github.com/kurtvalcorza/gliner-ner-pipeline/blob/main/MODEL_CARD.md\n"
        "- Upstream model: https://huggingface.co/{MODEL_ID}\n"
        "- Encoder assets: https://huggingface.co/microsoft/mdeberta-v3-base\n"
        "- Upstream code: https://github.com/urchade/GLiNER\n"
        "- GLiNER paper: https://arxiv.org/abs/2311.08526"
    ),
}
