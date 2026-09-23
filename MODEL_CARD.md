---
license: apache-2.0
model_card_spec: "1.1"
pipeline_tag: token-classification
task: "Others - Named-Entity Recognition"
base_model: urchade/gliner_multi-v2.1
date_published: "2024-04-09"
date_published_source: "Hugging Face Hub repository creation date of the exact hosted checkpoint (`createdAt`, https://huggingface.co/api/models/urchade/gliner_multi-v2.1)"
---

# GLiNER multi-v2.1 — Zero-Shot Named-Entity Recognition Model (Span Extractor)

[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-urchade%2Fgliner__multi--v2.1-ffcc4d?style=flat)](https://huggingface.co/urchade/gliner_multi-v2.1)
[![Upstream GitHub](https://img.shields.io/badge/Upstream%20GitHub-urchade%2FGLiNER-181717?style=flat&logo=github&logoColor=white)](https://github.com/urchade/GLiNER)
[![arXiv Paper](https://img.shields.io/badge/arXiv-2311.08526-b31b1b.svg)](https://arxiv.org/abs/2311.08526)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

> [!WARNING]
> ⚠️ **Provided for research, training, and evaluation purposes only.** Model weights are redistributed unmodified under their upstream license, which controls your use, including any commercial use or redistribution; the accompanying code and notebooks are released under this repository's license. All of it is supplied **"as is"**, without warranty of any kind, and has not been validated for production, clinical, or safety-critical use. Running the notebooks downloads third-party weights and datasets governed by their own licenses and consumes compute on your own Colab/Kaggle account. To the maximum extent permitted by law, the maintainers of this repository and the DIMER platform accept no liability for any damages arising from their use. Hosting implies no affiliation with or endorsement by the original authors.

---

## Interactive Colab Tutorials

This pipeline provides a ready-to-run interactive Google Colab notebook that exercises the repository's public API end to end — bootstrap a fresh runtime, stage and verify the pinned upstream revision, validate domain datasets, benchmark zero-shot baseline, run bounded fine-tuning, evaluate post-adaptation F1 deltas, perform inference, and export and reload adapter artifacts:

- **End-to-End Domain Adaptation Tutorial**:  
  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kurtvalcorza/gliner-ner-pipeline/blob/main/tutorials/gliner_ner_colab.ipynb) [`gliner_ner_colab.ipynb`](https://github.com/kurtvalcorza/gliner-ner-pipeline/blob/main/tutorials/gliner_ner_colab.ipynb)  
  *End-to-end domain adaptation, evaluation, and inference with pinned GLiNER multi-v2.1 weights (plus mDeBERTa encoder assets, both digest-verified) on biomedical/clinical entities: pre-adaptation zero-shot baseline evaluation, bounded fine-tuning freezing mDeBERTa-v3 backbone, exact-span micro/macro P/R/F1 evaluation and baseline delta quantification, inference on unseen domain text, and export and reload parity verification of portable adapter artifacts.*

---

#### Description

`urchade/gliner_multi-v2.1` is the multilingual GLiNER checkpoint of Zaratiana et al. (arXiv:2311.08526), pinned here to revision `443d26d654e0324125a96bebd8e796c14ff2efe6`. GLiNER is a bidirectional-encoder span extractor: the pinned `gliner_config.json` names `microsoft/mdeberta-v3-base` as the encoder, a 512-unit projection, span mode `markerV0`, spans of at most 12 words (`max_width`), and 384 words of context (`max_len`). At inference the caller's entity-type names are prepended to the text as special tokens, the encoder runs once, every candidate span is scored against every label embedding, and spans above a threshold are returned. The GLiNER snapshot ships 2 model files (`gliner_config.json`, `model.safetensors`); the encoder's tokenizer and configuration are not in it, so this repository pins them as a second snapshot, `microsoft/mdeberta-v3-base` at revision `a0484667b22365f84929a935b5e50a51f71f159d` (`config.json`, `tokenizer_config.json`, `spm.model`, no weights), with its own manifest. What this repository adds is the `GLiNERPipeline` class in `src/gliner_ner_pipeline/pipeline.py`: verification of both manifests (`verify_snapshot`, `verify_encoder_snapshot`), fresh-clone staging for both (`stage_missing_files`, `stage_missing_encoder_files`), a loader that redirects the library's encoder lookups to the verified local directory instead of the Hub or a cache, input validation with named ceilings, bounded in-kernel domain adaptation freezing the mDeBERTa-v3 backbone (`adapt`), exact-span micro/macro P/R/F1 evaluation (`evaluate`), standalone portable adapter artifact export and verified reload parity (`save_artifact`, `from_artifact`), a fixed output contract, and the `entity_f1` helper.

#### Intended Use and Limitations

###### Primary Intended Uses

The task is zero-shot named-entity recognition: input is one text of at most `MAX_TEXT_CHARS = 5000` characters and a caller-supplied list of 1 to `MAX_LABELS = 25` entity-type names; output is a list of `{text, label, start, end, score}` spans whose score reached the threshold. Envisioned applications are extraction of people, organisations, places, dates, products, and domain-specific types from documents, tickets, contracts, and multilingual web text where no labelled training data exists for the wanted types; pre-annotation to bootstrap a labelled NER dataset that a human then corrects; and structuring free text into fields for downstream search or analytics. In a larger system the pipeline is a zero-configuration baseline and an extraction component whose output feeds a schema, not a decision engine.

###### Primary Intended Users

Intended users are machine-learning engineers, data scientists, and application developers building information-extraction features for research prototypes or in-house tooling. The pipeline assumes its users understand that the label names are the only task specification and that their wording changes recall, that a span score above 0.5 is a ranking cutoff and not a calibrated probability, that text beyond 384 words is cut by the library, that the extracted entities are personal data whenever they name people, and that precision and recall on their own documents must be measured against human annotations before the output is relied on.

###### Out-of-scope use cases

1. **Capability boundary:** the model extracts flat, contiguous spans only; it does not link entities to a knowledge base, resolve coreference, extract relations or events, classify whole documents, or produce nested or discontinuous entities (`flat_ner=True` in the upstream call). Spans longer than 12 words (`max_width` in `gliner_config.json`) cannot be returned.
2. **Input boundary:** `detect()` takes one `str` of 1 to 5,000 characters (`MAX_TEXT_CHARS`, `ValueError` above it; the caller chunks longer text) and a list of 1 to 25 unique non-empty label strings of at most 100 characters (`MAX_LABELS`, `MAX_LABEL_CHARS`); a bare string as `labels`, a non-string label, or a non-numeric or out-of-range threshold is rejected. Text beyond 384 words is truncated by the library without an error. Images, audio, and structured records are not inputs.
3. **Decision boundary:** not for autonomous action on an extracted entity — redaction that must be complete, sanctions or watch-list screening, eligibility checks, or any decision about a named person — without a human reviewing the spans; recall is unmeasured here and a missed name is silent.

#### Factors

###### Groups

The pipeline is human-centric by construction: "person" is the canonical entity type, and names, nationalities, titles, and demographic descriptors are exactly what it extracts. The upstream README names the training set only as `urchade/pile-mistral-v0.1`, a synthetic NER dataset built from Pile text with Mistral-generated annotations; neither the README nor this repository reports recall by name origin, script, gender, or language, and no such audit was performed here. Name-based bias is a known failure mode for NER (names from under-represented languages and scripts are missed more often), so the downstream operator must measure per-group recall on their own annotated documents — for example, detection rate for person names by script and language — before relying on the output for anything that affects the people named.

###### Instrumentation

There is no physical sensor: the training text is Pile web, book, code, and forum text as tokenised by the mDeBERTa-v3 SentencePiece tokenizer, and the labels were produced by a language model rather than human annotators, which is the instrument characteristic that matters most — label noise and the annotator model's own blind spots are baked into the weights. At inference the pipeline passes the caller's text verbatim: no encoding repair, whitespace or Unicode normalisation, sentence splitting, or language detection. Defects in the operator's text pipeline — OCR errors, mis-decoded bytes, run-together tokens, boilerplate — reach the model unchanged, and the pipeline detects only the shape errors it validates (type, emptiness, length ceilings, label list shape) and the entity-contract violations it checks on the way out (span inside the text, label in the caller's list).

###### Environment

Operating environment: Python 3.12 with `torch==2.14.0` (the venv build is `2.14.0+cu130`), `gliner==0.2.29`, `transformers==4.57.6`, `huggingface-hub==0.36.2`, `safetensors==0.8.0`, `numpy==2.5.3`, `protobuf==6.31.1`, `sentencepiece==0.2.2` (`pyproject.toml`; the last two are what Transformers needs to build the fast DeBERTa-v2 tokenizer from `spm.model`); the loader maps weights to CPU and moves the model to `device` (float32 on CPU; no other precision is exercised by this package). The model loads and runs on CPU (see "Runtime" for the executed smoke: 14.6 s wall clock for verification, load and one five-entity extraction on the reference machine); CPU or GPU latency of the actual model is therefore not measured. A runtime that executes this pipeline needs `protobuf` in addition to the pins listed above. Data environment: the model assumes natural-language text resembling the Pile and label names in English or one of the languages mDeBERTa covers; recall degrades, without any error, on domain-specific types the annotator model never labelled, on very short or heavily formatted text, on labels phrased as long descriptions, and on text past the 384-word window, which is cut.

#### Metrics

###### Performance Measures

The pipeline reports `entity_f1(predicted, gold)`: exact-span micro precision, recall, F1 and the hit count, where a hit is an identical `(start, end, label)` triple. Exact-span micro-F1 is the standard NER measure because the task's unit of correctness is a span with a type — a partially overlapping or mislabelled span is a miss — and micro-averaging weights every entity equally; reading precision or recall alone hides whether the model over- or under-extracts, which is why all three are returned. No value is reported by this repository: the only execution is the single unannotated CPU smoke in "Runtime", and no annotated corpus is shipped, so `entity_f1` has not been computed on anything. The upstream README shows a benchmark figure as an image without numbers in the pinned text; nothing from it is reproduced or quoted here. The operator supplies human-annotated spans for their own label set and computes `entity_f1` per label and overall. The public `evaluation_report(result, gold=None)` helper wraps that metric for callers and for the tutorial: it reports `entity_f1` as `sample-sanity` evidence only when gold `(start, end, label)` spans are supplied, and otherwise returns a machine-readable `not-measurable` verdict that names the labelled data, label vocabulary and boundary convention a real measurement would require, so no figure is ever emitted without that context.

###### Decision thresholds

The default decision rule is a score cutoff: `detect()` passes `threshold` (default `DEFAULT_THRESHOLD = 0.5`, the upstream `predict_entities` default) to the library, which returns only spans whose sigmoid score is at or above it, and `flat_ner=True` resolves overlaps by keeping the higher-scoring span. The 0.5 default is upstream's convention, not a value tuned on any deployment, and the threshold is exposed as a parameter precisely because the operator owns it: lower it (0.3 is common) where a missed entity costs more than a spurious one, as in redaction or pre-annotation for human review; raise it where false extractions are expensive, as in automatic field population. Choose it per label from precision-recall curves on annotated documents from the deployment. No acceptance threshold on F1 was set during development because this repository does no training.

###### Approaches to uncertainty and variability

No metric value is reported, so no estimation procedure or dispersion applies; the operator who computes `entity_f1` on their own annotations owns the split design and any bootstrap or repeated-run estimate. Within the pipeline the forward pass is deterministic on a fixed device and dtype — no sampling, dropout inactive in `eval()` mode under `torch.inference_mode`, no seed needed — but kernel selection across hardware can move a span score in its low-order digits and flip spans that sit near the threshold; label order can also change results because labels are prepended as context. The per-span score is a sigmoid trained with binary cross-entropy against synthetic labels and is not calibrated; a caller who needs a calibrated confidence must fit a per-label calibrator on human-annotated spans. The loader records any warnings the `gliner` library emits during load in `load_warnings` rather than suppressing them; none were observed because the load did not complete.

#### Ethical considerations and biases

###### Data

The upstream README discloses the training set as `urchade/pile-mistral-v0.1` — Pile text with entity annotations generated by a Mistral model, per the GLiNER paper — and the encoder as `microsoft/mdeberta-v3-base`, pretrained on CC100 multilingual web text; neither corpus is enumerated here, and the Pile is known to contain personal names, e-mail addresses, and copyrighted text, so personal data in training is present, not merely unruled out. This repository distributes code, tests, and documentation; it does not vendor the weights in Git (`weights/gliner-multi-v2.1/model.safetensors` is git-ignored and reproduced from the pinned revision via `stage_missing_files`), vendors neither the 4.3 MB encoder tokenizer/config snapshot (its `spm.model` is reproduced from its own pinned revision via `stage_missing_encoder_files`), and ships no text corpus. The operator must audit the text they submit for personal, confidential, or legally restricted content, and must treat every extracted `person` span and its surrounding text as personal data with a lawful basis for processing; the pipeline performs no such check.

###### Human Life

This pipeline is not intended for decisions in health, safety, criminal justice, employment, credit, or housing, and it has not been validated or certified for any of them by anyone; the only validation performed is the offline unit suite and one executed run that verified both snapshots and stopped at the tokenizer conversion recorded under "Runtime", with the model itself unexecuted. Use in such domains is foreseeable — extracting names from clinical notes, police reports, or job applications — and would be admissible only with a human reviewing every extracted and every missed span, an independent evaluation of per-group recall on that deployment's own annotated documents, and whatever data-protection and regulatory clearance the domain requires.

###### Mitigations

1. **Supply-chain integrity:** `MODEL_ID` and the 40-hex `MODEL_REVISION` are module constants; `verify_snapshot()` checks `modelId`, `revision`, and the byte size and SHA-256 of all 3 manifest entries before any load, raising on the first mismatch (a test flips one hex digit and asserts it raises; the executed run verified the 1,155,830,112-byte weight file). `stage_missing_files()` fetches only manifest-listed files absent on disk, only at `MODEL_REVISION`, only when `allow_download=True`, and refuses a manifest naming another model. The encoder assets have their own manifest: `ENCODER_MODEL_ID` and the 40-hex `ENCODER_REVISION` are module constants, `verify_encoder_snapshot()` checks all 4 entries the same way (executed: both manifests passed in the recorded run; tests flip a digit and swap the `modelId` and assert both raise), and `stage_missing_encoder_files()` applies the same refusal rules. The loader subclasses the library's concrete model class and rewrites `config.model_name` to the verified encoder directory after checking it names `microsoft/mdeberta-v3-base`, so the library's tokenizer and `AutoConfig` lookups become local reads with `local_files_only=True` (executed up to the tokenizer conversion, with the network hidden); only the manifest-less `allow_download=True` path lets the library reach the Hub. The pipeline passes no `trust_remote_code` because `GLiNER.from_pretrained` has no such parameter; the library's `modeling/encoder.py` hardcodes `trust_remote_code=True` on two branches this checkpoint does not take (mDeBERTa resolves to the native `DebertaV2Model` class).
2. **Input integrity:** `detect()` rejects non-string or blank text, text above `MAX_TEXT_CHARS`, a bare-string or empty label list, more than `MAX_LABELS` labels, non-string, blank, duplicate or over-long labels, and a threshold outside [0, 1] before the model runs (one test per check); every returned entity is checked to lie inside the text with a label from the caller's list, and a violation raises. The public `validate_inputs()` stage routes through the same private checker as `detect()`, so their acceptance criteria cannot diverge; it raises exactly what `detect()` raises and otherwise returns an input manifest recording the schema, the enforced ceilings, the observed text size, the request parameters and both pinned snapshot identities, so a caller can log what was accepted before any model ran.
3. **Statistical mitigations:** none are implemented; the pipeline does no training.
4. **Reproducibility:** every runtime dependency is pinned with `==`; each result carries `model_id`, `model_revision`, `encoder_model_id`, `encoder_revision`, `labels`, `threshold`, and `n_entities`; `load_warnings` preserves whatever the library warned at load.
5. **Refusals:** nested and multi-label spans, span vectors, class probabilities, and batch inference are deliberately not exposed; the encoder revision that upstream leaves unpinned is pinned here, and the loader refuses any other `model_name` in `gliner_config.json`.

###### Risks and harms

1. **Silent truncation:** text beyond 384 words is cut by the library, so entities later in a long document are missed; the pipeline caps input at 5,000 characters and tells the caller to chunk, but does not report where the cut fell. Likely for long documents.
2. **Missed entities as silent failure:** a missed name in a redaction or screening task leaks or omits it with no signal; the data subject or a third party bears the harm, and likelihood rises for names from under-represented scripts and languages.
3. **Label-wording sensitivity:** "person" versus "individual" versus "name" yields different spans; an operator who does not test label phrasings gets unstable recall across runs of their own pipeline.
4. **Two-snapshot drift:** the encoder tokenizer/config and the GLiNER weights are pinned separately; a deployment that restages one without the other, or a library upgrade that changes how `model_name` is resolved, would alter predictions while both manifests still verify. The operator bears this; the `encoder_revision` field in every result is the trace to check.
5. **Privacy and profiling:** extraction of names, locations, and dates from text is the first step of profiling; storing the output creates a personal-data record. Magnitude ranges from noisy analytics to unlawful processing of personal data when the decision boundary above is ignored.

###### Use cases

The pipeline must not be used to build profiles of individuals from text they did not consent to have processed, for surveillance, for biometric or demographic inference from names (ethnicity, religion, gender), for social scoring, or for unlawful discrimination in employment, housing, credit, insurance, education, or healthcare access. It must not be used to de-anonymise authors or data subjects, to harvest personal data from scraped content, or in any way that breaches the Apache-2.0 terms of the upstream weights, the licence of the mDeBERTa encoder assets, or the terms of the deployment that runs the pipeline. These prohibitions hold even where the model would extract the spans accurately.

## Immutable provenance

- Model: `urchade/gliner_multi-v2.1`
- Revision: `443d26d654e0324125a96bebd8e796c14ff2efe6`
- Manifest: `weights/gliner-multi-v2.1/dimer-base-manifest.json`, format `dimer_hf_snapshot` v1, 3 files, `totalBytes` 1155835359
- `model.safetensors` (1,155,830,112 bytes) SHA-256: `2100142f31627531497850659dcb3821c99d5e71c08a8e01a98e4b11ef32a199`
- `gliner_config.json` (477 bytes) SHA-256: `e25f61d91620df84aae8076811ee592e926e94d341b82e7bc1be359718f83017`
- Encoder snapshot: `microsoft/mdeberta-v3-base` at revision `a0484667b22365f84929a935b5e50a51f71f159d` (MIT; tokenizer and config only, no weights), manifest `weights/mdeberta-v3-base-tokenizer/dimer-base-manifest.json`, 4 files, `totalBytes` 4309323
- `spm.model` (4,305,025 bytes) SHA-256: `13c8d666d62a7bc4ac8f040aab68e942c861f93303156cc28f5c7e885d86d6e3`
- encoder `config.json` (579 bytes) SHA-256: `bcffcd343dc5efa5ef2d5a58d2b405eed108f01cc45b48d0a907b333ec41801f`
- `tokenizer_config.json` (52 bytes) SHA-256: `3f3978e0c036f2c2588cac34a6047cbb0af0b0dc1814254e291028529805496d`
- Upstream references: https://huggingface.co/urchade/gliner_multi-v2.1 and https://huggingface.co/microsoft/mdeberta-v3-base

## Input/output contract

- `GLiNERPipeline.from_pretrained(device=None, weights_dir=None, allow_download=False, encoder_dir=None)`: stages and verifies the GLiNER snapshot, then stages and verifies the encoder snapshot (`encoder_dir` defaults to `weights/mdeberta-v3-base-tokenizer`), then loads through a subclass of the library's concrete class whose `model_name` points at that directory, with `local_files_only=True` and `map_location="cpu"`; `device` defaults to `cuda:0` when visible, else `cpu`. `load_warnings` holds any warnings the library emitted.
- `detect(text: str, labels: list[str], threshold: float = 0.5) -> dict` with keys `entities` (list of `{text, label, start, end, score}` with character offsets into `text`), `n_entities`, `labels`, `threshold`, `model_id`, `model_revision`, `encoder_model_id`, `encoder_revision`.
- `entity_f1(predicted, gold) -> {"precision", "recall", "f1", "hits"}`: exact-span micro scores over `(start, end, label)` triples.
- Constants: `MAX_TEXT_CHARS = 5000`, `MAX_LABELS = 25`, `MAX_LABEL_CHARS = 100`, `DEFAULT_THRESHOLD = 0.5`, `ENCODER_MODEL_ID = "microsoft/mdeberta-v3-base"`, `ENCODER_REVISION` (40-hex above), `ENCODER_KEY = "mdeberta-v3-base-tokenizer"`.

## Runtime

- Pins (`pyproject.toml`): `torch==2.14.0`, `torchvision==0.29.0`, `torchaudio==2.11.0`, `gliner==0.2.29`, `transformers==4.57.6`, `huggingface-hub==0.36.2`, `safetensors==0.8.0`, `numpy==2.5.3`, `protobuf==6.31.1`, `sentencepiece==0.2.2`; dev `pytest==8.4.2`, `ruff==0.16.6`. Python 3.12, Windows venv.
- Executed 2026-09-12: `CUDA_VISIBLE_DEVICES=-1 python -m pytest -q -o addopts= tests` — 25 passed, exit 0; `ruff check src tests` clean.
- Executed on CPU 2026-09-12 (`CUDA_VISIBLE_DEVICES=-1 HF_HUB_OFFLINE=1`, `from_pretrained(device="cpu")`, float32, after adding `protobuf==6.31.1` to the pinned runtime): both manifests verified (3 + 4 entries, including the 1,155,830,112-byte weight file), the encoder tokenizer and config were read from `weights/mdeberta-v3-base-tokenizer/` with no network, and `detect()` on a synthetic sentence naming two people, an organization, a city and a weekday, with labels `["person", "organization", "location", "date"]`, returned five spans — the two person names at 0.982 and 0.984, `NAIRA`/organization 0.916, `Quezon City`/location 0.971, `Friday`/date 0.945 — in 14.6 s wall clock including verification and load (one observation on a synthetic sentence, not an evaluation). Transformers 4.57.6 logs an "incorrect regex pattern … `fix_mistral_regex`" warning while building the fast tokenizer from `spm.model`; the message is a generic heuristic that references a Mistral tokenizer issue and does not apply to DeBERTa-v2's SentencePiece model — the spans above tokenized correctly — but it is recorded here so operators do not chase it.
- Not executed: latency or memory beyond the single observation above, the CUDA path, the `allow_download=True` path, and any annotated evaluation.

## References

- Zaratiana, U., Tomeh, N., Holat, P., Charnois, T. (2023). GLiNER: Generalist Model for Named Entity Recognition using Bidirectional Transformer. https://arxiv.org/abs/2311.08526
- Upstream model card: https://huggingface.co/urchade/gliner_multi-v2.1 (pinned README, revision above)
- Upstream code and library: https://github.com/urchade/GLiNER (`gliner` 0.2.29)
- Encoder tokenizer/config: https://huggingface.co/microsoft/mdeberta-v3-base (pinned here at the revision above; upstream GLiNER pins none)
- Training data as disclosed upstream: https://huggingface.co/datasets/urchade/pile-mistral-v0.1
