"""Per-repository template for tools/build_notebook.py (NOTEBOOK_SPEC 1.1 §3.6 standalone carrier).

Only the task-specific prose and stage cells live here. Runtime install, the embedded pipeline
module, and the model pin/stage/verify cells are produced by the generator from repository
sources so they cannot drift from the package.

This template uses the generator /2 keys: GLiNER verifies TWO pinned snapshots (the GLiNER checkpoint
and the mDeBERTa-v3 encoder assets its tokenizer/config come from) and its module derives both weights
directories from a shared ``_WEIGHTS_ROOT``. ``rewrites`` replaces that one line with a
working-directory-relative root; ``extra_weights`` carries the encoder manifest inline and has the
model cell stage (``stage_missing_encoder_files``) and verify (``verify_encoder_snapshot``) it before
``model_load`` runs ``GLiNERPipeline.from_pretrained(weights_dir=WEIGHTS_DIR, encoder_dir=ENCODER_WEIGHTS_DIR)``.
"""
# ruff: noqa: E501  -- markdown prose and code-cell text are kept on single lines for readable rendering

TEMPLATE = {
    "package": "gliner_ner_pipeline",
    "repo_name": "gliner-ner-pipeline",
    "stem": "gliner_ner",
    "notebook_name": "gliner_ner_colab.ipynb",
    "profile": "TASK-INFERENCE",
    "pipeline_class": "GLiNERPipeline",
    "weights_key": "gliner-multi-v2.1",
    # --- extended-generator keys (see the module docstring) ---------------------------------
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
    # ---------------------------------------------------------------------------------------
    "runtime_imports": ["torch", "transformers", "gliner"],
    "title": "GLiNER multi-v2.1 — DIMER zero-shot named-entity recognition tutorial (standalone)",
    "badges": [
        (
            "GitHub",
            "https://img.shields.io/badge/GitHub-181717?style=flat&logo=github&logoColor=white",
            "https://github.com/kurtvalcorza/gliner-ner-pipeline",
        ),
        (
            "Open In Colab",
            "https://colab.research.google.com/assets/colab-badge.svg",
            "https://colab.research.google.com/github/kurtvalcorza/gliner-ner-pipeline/blob/main/tutorials/gliner_ner_colab.ipynb",
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
    ],
    "capability": "zero-shot named-entity recognition with a caller-supplied label set using the pinned GLiNER multi-v2.1 weights",
    "intro": (
        "At inference the text and the caller's entity-type names are encoded together by the mDeBERTa-v3 bidirectional "
        "encoder inside GLiNER, every candidate span is scored against every label, and spans whose **score** (a per-span "
        "confidence in [0, 1], **not a calibrated probability**) reaches the `threshold` are returned with character "
        "offsets. The label set is free text chosen by the caller at call time — that is what \"zero-shot\" means here — and "
        "the threshold (default `DEFAULT_THRESHOLD` = 0.5, the upstream default) is exposed and **owned by the caller**. "
        "**No adaptation occurs:** no training, fine-tuning, in-context conditioning, or preprocessing fitting — the "
        "pinned checkpoint is used as published. **Two pinned snapshots** are involved: the GLiNER weights and the "
        "mDeBERTa tokenizer/config the encoder needs; the carried module verifies both against their own manifests before "
        "loading, and Section 3 carries both inline. What upstream supplies is the model, the encoder assets and the "
        "span-decoding library; what the carried pipeline module adds is dual-manifest verification, input validation and "
        "ceilings, offset-checked output, and the `entity_f1`, `validate_inputs` and `evaluation_report` helpers."
    ),
    "learning_objectives": (
        "install the pinned runtime, read what the carried pipeline module guarantees, author a synthetic input text and "
        "label set (or upload your own), stage and digest-verify **both** immutable snapshots, validate the request into "
        "an input manifest through the pipeline's own validation stage, detect entities through the public API, read spans "
        "and scores correctly, produce an evaluation report that is `sample-sanity` with `entity_f1` only when gold spans "
        "exist and `not-measurable` otherwise, and export the entities with identifiers plus provenance."
    ),
    "exclusions": (
        "entity linking or normalisation, relation extraction, coreference, document-level processing beyond "
        "`MAX_TEXT_CHARS` (the caller chunks), fine-tuning, or any calibrated confidence. The repository exposes none of "
        "these."
    ),
    "prerequisites": [
        "- **Runtime:** a fresh supported runtime (Google Colab or Jupyter, Python 3.12). The default path runs on CPU (float32) and uses CUDA automatically when available. The pinned `torch==2.14.0` install and the 1.16 GB GLiNER checkpoint are the largest downloads of the run.",
        "- **Knowledge:** basic Python; what character offsets are; what precision/recall/F1 over spans mean.",
        "- **Expected warning:** while building the fast DeBERTa tokenizer from `spm.model`, `transformers==4.57.6` logs an \"incorrect regex pattern … `fix_mistral_regex`\" warning. It refers to a Mistral tokenizer issue, does not apply to this SentencePiece model, and is documented in the README and model card; the pipeline captures it in `load_warnings`, which Section 5 prints so you can see it is the expected one. Building that tokenizer needs `protobuf` and `sentencepiece`, both in the pinned install.",
        "- **Data:** the default sample is a synthetic sentence and label set authored in code (the names are invented); BYOD is one UTF-8 text file plus a label list, gated off by default, at most `MAX_TEXT_CHARS` characters (the library also cuts text beyond 384 words — chunk longer documents yourself). Do not upload confidential or restricted data to a hosted notebook environment unless you are authorized to do so. Uploaded text remains in the notebook runtime; this pipeline does not send it to a third-party inference API.",
    ],
    "cells": [
        {
            "md": (
                "## 4. Author the synthetic sample or optional BYOD\n\n"
                "The default sample is **synthetic**: one English sentence and a four-type label set written in this cell, so "
                "it needs no download and names no real person (the names are invented). It ships **no gold annotations** — "
                "the sentence was written to contain obvious entities, but the notebook does not assert where they are — so "
                "the spans it produces are smoke/sanity evidence that the code path works, never an accuracy measurement and "
                "never benchmark evidence. If you want a metric, paste gold annotations into `GOLD_JSON` (a JSON list of "
                "`{{\"start\", \"end\", \"label\"}}` objects with character offsets into the text, matching the label names "
                "exactly); Section 7 then scores exact-span micro precision/recall/F1 with the repository's `entity_f1`. Leave "
                "it empty and the evaluation report is `not-measurable`.\n\n"
                "BYOD is optional and disabled by default. Nothing is validated in this cell beyond the shape of `GOLD_JSON` "
                "(which is notebook-side reference data, not pipeline input) — the next section hands the text, the labels and "
                "the threshold to the pipeline's own validation stage, which is the only checker for them."
            ),
            "code": (
                "import hashlib\n"
                "import json\n\n"
                "USE_BYOD = False  # @param {{type:\"boolean\"}}\n"
                "LABELS = 'person, organization, location, date'  # @param {{type:\"string\"}}\n"
                "THRESHOLD = 0.5  # @param {{type:\"number\"}}\n"
                "GOLD_JSON = ''  # @param {{type:\"string\"}}\n\n"
                "labels = [label.strip() for label in LABELS.split(',') if label.strip()]\n"
                "if USE_BYOD:\n"
                "    from google.colab import files\n"
                "    uploaded = files.upload()\n"
                "    sample_name = next(iter(uploaded))\n"
                "    text = uploaded[sample_name].decode('utf-8').strip()\n"
                "    sample_kind = 'BYOD upload'\n"
                "else:\n"
                "    text = 'Elena Marquez joined the Lakeside Research Institute in Wellington on 3 March 2021 after leaving Orion Analytics.'\n"
                "    sample_name = 'synthetic_sentence'\n"
                "    sample_kind = 'synthetic (authored in this cell; invented names)'\n"
                "gold = json.loads(GOLD_JSON) if GOLD_JSON.strip() else None\n"
                "if gold is not None:\n"
                "    for index, item in enumerate(gold):\n"
                "        if not (isinstance(item, dict) and {{'start', 'end', 'label'}} <= set(item)):\n"
                "            raise ValueError(f'GOLD_JSON[{{index}}] must be an object with start, end and label')\n"
                "        if not (0 <= int(item['start']) < int(item['end']) <= len(text)) or item['label'] not in labels:\n"
                "            raise ValueError(f'GOLD_JSON[{{index}}] has offsets outside the text or a label not in LABELS: {{item}}')\n"
                "sample_sha256 = hashlib.sha256(text.encode('utf-8')).hexdigest()\n"
                "print({{'sample': sample_name, 'sample_kind': sample_kind, 'chars': len(text), 'words': len(text.split()), 'labels': labels, 'threshold': THRESHOLD, 'has_gold': gold is not None, 'text_sha256': sample_sha256}})\n"
                "print(text)"
            ),
        },
        {
            "md": (
                "## 5. Validate the request → input manifest\n\n"
                "`validate_inputs` is the pipeline's public validation stage: it applies exactly the checks `detect` applies — "
                "a non-empty `str` of at most `MAX_TEXT_CHARS` characters, 1..`MAX_LABELS` unique non-empty labels of at most "
                "`MAX_LABEL_CHARS` characters each, and a `threshold` in [0, 1] — and returns an **input manifest** naming the "
                "schema and ceilings, the text's observed character and word counts, the label set, the threshold, the verdict, "
                "and **both** pinned identities (this pipeline verifies two snapshots). The manifest is written to "
                "`outputs/{stem}_input_manifest.json`. To show what rejection looks like, the cell also validates a duplicated "
                "label set and records the pipeline's own error message as a finding.\n\n"
                "**What can change your text:** the GLiNER library cuts input beyond 384 words (`max_len` in the snapshot "
                "config) without reporting where — the word count is printed so you can see whether the sample is near that "
                "limit; the notebook itself does not alter the text."
            ),
            "code": (
                "import os\n\n"
                "os.makedirs('outputs', exist_ok=True)\n"
                "print({{'ceilings': {{'MAX_TEXT_CHARS': MAX_TEXT_CHARS, 'MAX_LABELS': MAX_LABELS, 'MAX_LABEL_CHARS': MAX_LABEL_CHARS, 'library_word_limit': 384, 'DEFAULT_THRESHOLD': DEFAULT_THRESHOLD}}}})\n"
                "input_manifest = validate_inputs(text, labels, THRESHOLD, names=[sample_name])\n"
                "# Demonstrate rejection on a request that breaks a ceiling; the finding is recorded, not swallowed.\n"
                "try:\n"
                "    validate_inputs(text, labels + labels[:1], THRESHOLD)\n"
                "except ValueError as exc:\n"
                "    input_manifest['findings'].append({{'input': 'duplicate-label-probe', 'verdict': 'rejected', 'message': str(exc)}})\n"
                "with open('outputs/{stem}_input_manifest.json', 'w', encoding='utf-8') as handle:\n"
                "    json.dump(input_manifest, handle, indent=2, ensure_ascii=False)\n"
                "print(json.dumps(input_manifest, indent=2))"
            ),
        },
        {
            "md": (
                "## 6. Detect entities and read the scores correctly\n\n"
                "`detect(text, labels, threshold=...)` returns `entities` — a list of `{{text, label, start, end, score}}` with "
                "character offsets into the input (the pipeline checks every span lies inside the text and carries one of your "
                "labels) — plus `n_entities`, the `labels` and `threshold` used, and both model identities. **Score "
                "semantics:** each `score` is the library's per-span confidence in [0, 1] for that span-label pair; it is not "
                "a calibrated probability, and the only decision the pipeline makes is the cutoff `score >= threshold`. The "
                "default 0.5 is the upstream default, not a validated operating point: lowering it returns more (and less "
                "certain) spans, raising it fewer; the caller owns choosing it on their own annotated data. The captured "
                "`load_warnings` are printed here — expect exactly the `fix_mistral_regex` notice described in the "
                "prerequisites. The printed seconds are measured on this runtime for this one text and include the first-call "
                "warm-up."
            ),
            "code": (
                "import time\n\n"
                "print({{'load_warnings': pipe.load_warnings}})\n"
                "started = time.perf_counter()\n"
                "result = pipe.detect(text, labels, threshold=THRESHOLD)\n"
                "elapsed = time.perf_counter() - started\n"
                "entities = result['entities']\n"
                "checks = {{\n"
                "    'offsets_inside_text': all(0 <= e['start'] < e['end'] <= len(text) for e in entities),\n"
                "    'span_text_matches_offsets': all(text[e['start']:e['end']] == e['text'] for e in entities),\n"
                "    'labels_from_request': all(e['label'] in labels for e in entities),\n"
                "    'scores_at_or_above_threshold': all(THRESHOLD <= e['score'] <= 1.0 for e in entities),\n"
                "}}\n"
                "if not all(checks.values()):\n"
                "    raise RuntimeError(f'detect output failed a sanity check: {{checks}}')\n"
                "print({{key: value for key, value in result.items() if key != 'entities'}})\n"
                "print({{'seconds': round(elapsed, 3), 'checks': checks}})\n"
                "for index, entity in enumerate(entities):\n"
                "    print(f\"{{index:>2}}. [{{entity['start']:>3}}:{{entity['end']:<3}}] {{entity['label']:<14}} score {{entity['score']:.3f}}  {{entity['text']}}\")"
            ),
        },
        {
            "md": (
                "## 7. Evaluate → evaluation report\n\n"
                "`evaluation_report` is the pipeline's public evaluation stage and always produces a report. The repository's "
                "only metric helper is `entity_f1(predicted, gold)` — exact-span micro precision, recall and F1, where a hit "
                "is an identical `(start, end, label)` triple — the standard NER measure, which credits nothing for a span "
                "that is off by one character or carries a different label. It applies only when gold annotations exist. The "
                "synthetic sample has none, so the verdict is `not-measurable` and the report states what would make the task "
                "measurable: gold `(start, end, label)` spans over the same label set, on text from your domain, across enough "
                "documents to state a dispersion — and an annotation guideline that agrees with the model's span boundaries. "
                "When you supply `GOLD_JSON` the verdict becomes `sample-sanity` with one `entity_f1` metric: a single-text "
                "tutorial figure with no dispersion estimate, not a benchmark result. No baseline is reported: a trivial "
                "baseline (no entities) has F1 = 0 by construction and teaches nothing without gold. The report is written to "
                "`outputs/{stem}_evaluation_report.json`."
            ),
            "code": (
                "report = evaluation_report(result, gold, sample_kind=sample_kind)\n"
                "with open('outputs/{stem}_evaluation_report.json', 'w', encoding='utf-8') as handle:\n"
                "    json.dump(report, handle, indent=2, ensure_ascii=False)\n"
                "print(json.dumps(report, indent=2))\n"
                "if report['verdict'] == 'not-measurable':\n"
                "    print('No gold annotations were supplied, so entity_f1 is not computed; the spans above are sanity evidence only.')"
            ),
        },
        {
            "md": (
                "## 8. Export entities and provenance\n\n"
                "Two files are written under `outputs/`: `{stem}_result.json` with the input text and its digest, the label set "
                "and threshold, an `entities` list with an index per span plus its text, label, character offsets and score (so "
                "every span maps back to its input), the evaluation report, the input manifest, the gold spans when supplied, "
                "the sanity checks, the ceilings in force, the notebook's source (repository, revision, embedded module digest, "
                "generator), **both** model identifiers and immutable revisions, the model licences, the captured load "
                "warnings, and the runtime identity (Python, `torch`, `gliner`, `transformers`, device, precision); and "
                "`{stem}_entities.csv` with explicit `index`, `start`, `end`, `label`, `score` and `text` columns so span order "
                "and offsets survive downstream use. No credentials are involved in any step, so none can reach the export."
            ),
            "code": (
                "import csv\n\n"
                "payload = {{\n"
                "    'text': text,\n"
                "    'labels': result['labels'],\n"
                "    'threshold': result['threshold'],\n"
                "    'entities': [{{'index': index, **entity}} for index, entity in enumerate(entities)],\n"
                "    'n_entities': result['n_entities'],\n"
                "    'evaluation_report': report,\n"
                "    'input_manifest': input_manifest,\n"
                "    'gold': gold,\n"
                "    'sanity_checks': checks,\n"
                "    'ceilings': {{'MAX_TEXT_CHARS': MAX_TEXT_CHARS, 'MAX_LABELS': MAX_LABELS, 'MAX_LABEL_CHARS': MAX_LABEL_CHARS, 'library_word_limit': 384, 'DEFAULT_THRESHOLD': DEFAULT_THRESHOLD}},\n"
                "    'sample': {{'name': sample_name, 'kind': sample_kind, 'text_sha256': sample_sha256}},\n"
                "    'seconds': round(elapsed, 3),\n"
                "    'notebook_source': NOTEBOOK_SOURCE,\n"
                "    'repository_revision': NOTEBOOK_SOURCE['repository_revision'],\n"
                "    'model_id': MODEL_ID,\n"
                "    'model_revision': MODEL_REVISION,\n"
                "    'model_license': MODEL_LICENSE,\n"
                "    'encoder_model_id': ENCODER_MODEL_ID,\n"
                "    'encoder_revision': ENCODER_REVISION,\n"
                "    'encoder_license': ENCODER_LICENSE,\n"
                "    'load_warnings': pipe.load_warnings,\n"
                "    'runtime': {{\n"
                "        'python': platform.python_version(),\n"
                "        'torch': torch.__version__,\n"
                "        'gliner': gliner.__version__,\n"
                "        'transformers': transformers.__version__,\n"
                "        'device': pipe.device,\n"
                "        'precision': 'float32',\n"
                "    }},\n"
                "}}\n"
                "with open('outputs/{stem}_result.json', 'w', encoding='utf-8') as handle:\n"
                "    json.dump(payload, handle, indent=2, ensure_ascii=False)\n"
                "with open('outputs/{stem}_entities.csv', 'w', encoding='utf-8', newline='') as handle:\n"
                "    writer = csv.writer(handle)\n"
                "    writer.writerow(['index', 'start', 'end', 'label', 'score', 'text'])\n"
                "    for index, entity in enumerate(entities):\n"
                "        writer.writerow([index, entity['start'], entity['end'], entity['label'], f\"{{entity['score']:.6f}}\", entity['text']])\n"
                "print(sorted(os.listdir('outputs')))"
            ),
        },
    ],
    "closing": (
        "## Interpretation and limits\n\n"
        "Each returned span is a model prediction for a label name you chose: the `score` is a per-span confidence in "
        "[0, 1], not a calibrated probability, and the only decision rule is the caller-owned `threshold` (default 0.5, the "
        "upstream default, not a validated operating point). On the synthetic sentence the spans are plumbing evidence only "
        "and the evaluation report is `not-measurable`; an `entity_f1` figure on one text has no dispersion and generalises "
        "to nothing. The label names are part of the input — different wordings of the same concept give different spans — "
        "and the checkpoint's multilingual quality varies by language and domain in ways this notebook does not measure. "
        "Text beyond 384 words is cut silently by the library (the notebook prints the word count; chunk long documents), "
        "nested or overlapping entities may be lost, and entity linking, normalisation, relations and coreference are not "
        "provided. Two supply chains are involved — the GLiNER weights and the mDeBERTa encoder assets — and both were "
        "digest-verified against inline manifests before loading. Inference is deterministic given the same weights, device "
        "and library versions; CPU and CUDA scores can differ slightly and move borderline spans across the threshold.\n\n"
        "Successful execution proves that the recorded repository revision's pipeline module, carried in this notebook, can "
        "stage and digest-verify both pinned snapshots, validate the demonstrated request against the enforced ceilings, "
        "execute the public pipeline path, and emit the shown machine-readable outputs in the tested runtime — without the "
        "repository being reachable. It does **not** establish benchmark superiority, extraction quality on any domain or "
        "language, a validated threshold, safety for high-consequence decisions, or production fitness on an unseen "
        "domain.\n\n"
        "**Troubleshooting.** `RuntimeError: Core dependencies changed while older modules were loaded` in Section 1: the "
        "pinned install replaced a package the runtime had pre-imported — restart the runtime and rerun from the top. "
        "`FileNotFoundError: snapshot file missing` or a `sha256`/`size` `ValueError` in Section 3: a staged file is "
        "incomplete or altered — delete it from `weights/gliner-multi-v2.1/` (or `weights/mdeberta-v3-base-tokenizer/`) and "
        "rerun Section 3. A `fix_mistral_regex` warning: expected and harmless (see prerequisites). An `ImportError` naming "
        "`protobuf` while the DeBERTa tokenizer is built: the pinned install provides it — restart the runtime and rerun "
        "from the top. A `ValueError` naming `MAX_TEXT_CHARS`, `MAX_LABELS`, `MAX_LABEL_CHARS` or the threshold in "
        "Section 5: fix the form values or chunk the text and rerun from Section 4. Zero entities returned: lower "
        "`THRESHOLD`, or check that your label names describe the entities in plain words.\n\n"
        "**Next experiments.** Paste gold annotations for the default sentence into `GOLD_JSON` to see the verdict switch to "
        "`sample-sanity` and how exact-span F1 penalises a boundary that is off by one character; sweep `THRESHOLD` over "
        "0.3–0.7 on a text you can judge and watch precision and recall trade off; reword a label (for example `city` "
        "versus `location`) and observe how the spans change. None of these turns the sample result into evidence of "
        "production fitness.\n\n"
        "## References\n\n"
        "- Repository README: https://github.com/kurtvalcorza/gliner-ner-pipeline/blob/main/README.md\n"
        "- Repository model card: https://github.com/kurtvalcorza/gliner-ner-pipeline/blob/main/MODEL_CARD.md\n"
        "- Weight provenance (both snapshots): https://github.com/kurtvalcorza/gliner-ner-pipeline/blob/main/docs/WEIGHTS.md\n"
        "- Upstream model: https://huggingface.co/{MODEL_ID}\n"
        "- Encoder assets: https://huggingface.co/microsoft/mdeberta-v3-base\n"
        "- Upstream code: https://github.com/urchade/GLiNER\n"
        "- GLiNER paper: https://arxiv.org/abs/2311.08526"
    ),
}
