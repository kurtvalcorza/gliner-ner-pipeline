# GLiNER multi-v2.1 E2E NER Adaptation Notebook — Review

**Verdict: Needs revision**  
**Review date:** 3 October 2026 (relay batch of 2 October 2026)  
**Repository:** `kurtvalcorza/gliner-ner-pipeline`  
**Notebook:** `tutorials/gliner_ner_colab.ipynb`  
**Reviewed commit:** `f861b9164a00239931060e4343a254d7a6a481dd` (`main`, confirmed with `gh api repos/kurtvalcorza/gliner-ner-pipeline/commits/main`)  
**Notebook Git blob:** `ecbaee8862f1e036cd9b47b471a845b1624ee4b7`. This is the blob executed in the recorded Kaggle Tesla T4 run of 2026-09-17 (commit `58ee5df`); the notebook has not changed since. `tools/build_notebook.py --check` passes at the reviewed commit (recorded generating revision `f556f2a`).  
**Finding prefix:** `GL`  
**Framework:** Notebook Review Framework v1. **Requirements baseline:** NOTEBOOK_SPEC 2.2 (2026-09-26), `ml-worker` `origin/main` `b1cfe13`. The notebook declares 2.0.

## Executive assessment

The default workflow is real and works. It stages and digest-verifies two pinned snapshots (GLiNER weights and the mDeBERTa tokenizer/config), validates a deterministic 24-sentence biomedical dataset, splits it 18/6, records a zero-shot baseline, fine-tunes the non-encoder layers with AdamW, re-evaluates, runs inference on a new sentence, exports a `.pt` adapter (loaded back with `weights_only=True`) and checks prediction parity after reload. A CPU run of every code cell in this review reproduced the recorded numbers exactly.

| Measure | This review (CPU, direct execution) | Kaggle T4 record (blob `ecbaee88`) |
|---|---|---|
| Code cells completed | 12/12 at defaults; 70.5 s with weights already staged, plus 121 s for the first real 1.16 GB fetch and digest check from an empty directory (install cell skipped with the CI switch; torch 2.13.0+cpu, not the 2.14.0 pin) | 12/12 "ok (1 restart after install cell)", 298.7 s |
| Split | 18 train / 6 validation; all three labels in both | not recorded |
| Zero-shot baseline (validation, n = 6 sentences, 18 gold spans) | micro P/R/F1 0.7059 / 0.6667 / 0.6857; macro F1 0.6375; `gene_protein` F1 0.2222 | baseline 0.6857 implied |
| Adapted (same 6 sentences) | micro P/R/F1 0.8947 / 0.9444 / 0.9189; macro F1 0.9188 | "+0.2332 micro F1" |
| Trainable / frozen parameters | 11,418,112 / 277,531,392 (modules: `prompt_rep_layer`, `rnn`, `span_rep_layer`, `token_rep_layer` projection) | not recorded |
| New-sentence inference | Pembrolizumab → `chemical_drug` 0.9958; PDCD1 → `gene_protein` 0.9577; metastatic melanoma → `disease` 0.9961 | not recorded in the repository |
| Adapter / reload | 26 tensors, 45,682,421 bytes; parity passed | "adapter artifact reloaded" |

Four problems stand in the way of `Ready for intended use`:

1. **No one-pass `Run all` (GL-M1).** The notebook pip-installs exact pins (`torch==2.14.0`, `numpy==2.5.3`, …) into the running kernel and raises a restart instruction when a loaded distribution changed. The only hosted run needed that restart, yet `tutorials/README.md` marks Run-all "verified".
2. **The prescribed reruns operate on the already-adapted model (GL-M2).** Re-running Section 5 after adaptation prints the adapted metrics under the heading "Pre-adaptation Baseline"; re-running Section 6 continues training from the adapted weights. The BYOD instruction ("re-run from that cell") hits both.
3. **BYOD accepts only the three built-in biomedical labels (GL-M3).** A dataset with `person`/`organization`/`location` labels is rejected, every dataset must contain all three biomedical labels, the promised schema is never shown, and BYOD inference still scores the hard-coded biomedical sentence.
4. **Declared `GUIDED`, guided layer absent (GL-M4).**

## 1. Review contract and evidence

| Item | Value |
|---|---|
| Declared profile / mode | `E2E` / `GUIDED` (metadata `dimer.notebook_profile` / `notebook_mode`; opening cell `gliner_ner-00`) |
| Declared spec | DIMER Notebook Specification **2.0** (metadata, opening cell, `NOTEBOOK_SOURCE`) |
| Spec baseline applied | NOTEBOOK_SPEC **2.2** |
| Intended audience | Not stated. Prerequisites (`gliner_ner-01`, "Knowledge"): "understanding of named-entity recognition, character offsets vs token indices, and exact-span F1 metrics" |
| Supported runtime | "a fresh supported runtime (Google Colab or Jupyter, Python 3.12). CPU execution is supported; CUDA is auto-detected and recommended" |
| Promised outcomes (opening) | pinned install; carried package; two staged, digest-verified snapshots; deterministic 24-sentence dataset validated and split "with class coverage preserved"; zero-shot baseline; bounded fine-tuning with frozen backbone; exact-span micro/macro P/R/F1 and deltas; inference on unseen clinical text; `.pt` adapter export; fresh reload verifying "parameter match and prediction parity"; BYOD JSON/JSONL "through the same validation, splitting, baseline evaluation, adaptation, post-adaptation evaluation, unseen inference, artifact export, and reload parity cells" |
| Generator | `tools/build_notebook.py` (`build_notebook.py/2`) + `tools/notebook_template.py`; `--check` OK; `tools/validate_release_assets.py` PASS |
| Release status | **Candidate** (`STATUS.md`, `README.md`, `tutorials/README.md`) |

### Evidence actually obtained

- **Source inspection.** All 27 cells (12 code; cells 5, 7, 9 are the carried `metrics.py`, `pipeline.py`, `samples.py`, 49,023 characters together). Also read: `pipeline.py` (`from_pretrained`, `evaluate`, `adapt`, `save_artifact`, `load_artifact`, `from_artifact`), `samples.py` (`split_ner_dataset`, `validate_dataset`, `load_byod_dataset`), `tools/notebook_template.py`, the install-guard text in `tools/build_notebook.py`, `README.md`, `STATUS.md`, `tutorials/README.md`, `docs/release-verification.md`. The repository has no `AGENTS.md` and no `docs/execution-evidence/` directory.
- **Documented execution evidence.** `docs/release-verification.md`: Kaggle T4 kernel `kurtvalcorza/dimer-nb2-gliner-ner` v2, 2026-09-17, commit `58ee5df`, **the reviewed blob**: "PASS — 12/12 ok (1 restart after install cell)", 298.7 s. The executed notebook is not archived in the repository, so the restart is taken from the record's own text. No Colab run of this blob; no hosted CPU run although CPU is the documented default; no BYOD run.
- **Direct execution (this review).**
  - **Environment:** `run_probes.py`; Windows 11, CPU only, Python 3.12.14, torch 2.13.0+cpu, transformers 4.57.6, gliner 0.2.29, huggingface-hub 0.36.2, safetensors 0.8.0, numpy 2.5.3 (a scratch virtual environment layered on a read-only conda env; `gliner` installed into the scratch venv only). The install cell ran with `DIMER_NOTEBOOK_CI_PREINSTALLED=1`, so pip was skipped; the other cells ran unmodified, in order, in one namespace.
  - **Probes:** P1 static structure and guided-layer markers; P2 generator `--check` and release validator; P3 split coverage and entity overlap; P4 BYOD loader contract (6 files); P5 all 12 code cells at defaults; P6 rerun semantics of Sections 5 and 6; P7 reload of three tampered adapters. Attempt 1 of P5 performed the real 1.16 GB fetch into an empty working directory and verified both snapshots (121 s), then stopped at `from_pretrained` because the review harness's `google.colab` shim lacked `__spec__` (a harness defect, not a notebook defect); attempt 2 removed the shim and ran all 12 cells against the already verified files.
- **Not verified:** the pinned install itself and therefore the restart (GL-M1) beyond the Kaggle record; any Colab run; GPU behaviour; the real upload dialog; BYOD beyond the loader with a real upload.
- **Learner observation:** none. No claim here is about measured learning effectiveness.

## 2. Separate judgments

- **Technical correctness:** good on the default path. Immutable revisions, per-file SHA-256 for both snapshots, `local_files_only`, no remote code, adapter loaded with `weights_only=True`. Defects: the install pattern forces a restart (GL-M1); `adapt` and the baseline cell mutate one shared model, so reruns are not clean (GL-M2); `load_artifact` uses `load_state_dict(strict=False)` and checks only `model_id`, so an adapter with no matching keys or a different base revision loads silently (GL-m2).
- **Scientific validity:** acceptable for a labelled tutorial, with gaps in interpretation. Train and validation share 1 of 18 validation entity strings (`breast cancer`); the inference sentence's entities do not occur in training (its near-twin `bio-005` is a validation record). But the evidence is 6 sentences that all follow one template (drug as first token, then gene, then disease), the validation split is also the per-epoch monitor, and the zero-shot baseline is scored against snake-case label names (`gene_protein`, `chemical_drug`) that the adapted model has been trained on. None of this is said where the +0.2332 is printed (GL-m3).
- **Promise fulfilment:** the default E2E path is delivered. Not delivered: BYOD "through the same … cells" for "your own" data (GL-M2, GL-M3); "class coverage preserved" (no stratification exists; GL-m4); "verify parameter match" (only predictions on one sentence are compared; GL-m2); "only the span representation and prompt projection layers are updated" (the BiLSTM and the token projection are also trained; GL-m4).
- **Learner experience:** clear section headings and short operation descriptions, but no learner activity, predictions, reading guides or troubleshooting; raw metric JSON is printed with no explanation of micro vs macro or of the per-class numbers (GL-M4).
- **Spec conformance (2.2):** fails `MUST`s RUN1, RUN10, ENV6, REL2 (GL-M1); DAT10, DAT12, DAT13, DAT14 (GL-M3); DAT19 (GL-m5); EVAL6, DAT8 and EVAL3 are only partly met (GL-m3). `SHOULD` gaps: GDL1–GDL14, UX4, UX5, UX8 (GL-M4); EXE2, EXE5; FT5 precision. Release-record `MUST`s REL9/REL10 are met for the Kaggle row, but the procedure and coverage text describe a different notebook (GL-m1).

## 3. Promise and objective tracing

| Claim (opening / sections) | Implementation | Observable result | Learner interpretation |
|---|---|---|---|
| Run all completes in a fresh runtime | `gliner_ner-03` in-kernel pip of exact pins + stale-import guard | Kaggle T4: one restart needed; review CPU: install skipped | Not delivered as one pass (GL-M1) |
| Both snapshots staged and digest-verified | `gliner_ner-11` | 3 + 4 files fetched and verified (P5 attempt 1) | Printed dicts; no reading guide |
| Dataset validated and split "with class coverage preserved" | `validate_dataset`, `split_ner_dataset` (seeded shuffle, no stratification) | 18/6, all labels present for seeds 0–199 because every record carries a drug and a disease and 22/24 carry a gene | Claim holds for the sample by construction, not by the split (GL-m4) |
| Zero-shot baseline | `gliner_ner-15` | micro F1 0.6857 | Valid on first run only (GL-M2); label-name effect not mentioned (GL-m3) |
| Bounded fine-tuning, frozen backbone | `adapt(..., freeze_text_encoder=True)` | 3 epochs, loss 11.61 → 6.32, 11.4 M trainable | Trainable set stated more narrowly than it is (GL-m4) |
| Exact-span P/R/F1 and deltas | `gliner_ner-19` | +0.2332 micro F1, +0.2813 macro F1 | Raw JSON, no interpretation of n = 6 (GL-m3, GL-M4) |
| Inference on unseen clinical text | `gliner_ner-21` | 3 correct spans, all score ≥ 0.957 | Sanity checks only; no gold comparison (acceptable) |
| Export and reload with "parameter match and prediction parity" | `save_artifact`, `from_artifact`, one-sentence parity asserts | parity passed | No parameter comparison; `strict=False` (GL-m2) |
| BYOD JSON/JSONL through the same cells | `USE_BYOD` + `files.upload()` + `load_byod_dataset` | only the three biomedical labels accepted | Not delivered (GL-M3) |

| Learning objective (opening) | Learner activity | Evidence it was exercised |
|---|---|---|
| install the pinned runtime; inspect the carried modules | run cells | output dicts only |
| stage and digest-verify both snapshots | run cell | printed counts |
| validate and split a domain NER dataset | run cell | one sample record printed |
| evaluate baseline; execute bounded fine-tuning; evaluate deltas | run cells | metric JSON printed; no question asked of the learner |
| perform entity extraction; export and reload | run cells | entity list; parity message |

Every objective is phrased as an action the notebook performs, and no step asks the learner to predict, change, compare or explain (GL-M4).

## 4. Journeys

| Journey | Evidence basis | Result |
|---|---|---|
| First-time learner | Source inspection | Sections are ordered and each operation has a one-paragraph description. No audience statement, how-to-use, roadmap, glossary (span, micro/macro, exact-span), What-to-notice notes, checkpoints or troubleshooting; the three carried-module cells are not labelled as infrastructure the learner may skip; Prerequisites assume prior NER and F1 knowledge (GL-M4). The expected-warning note names `fix_mistral_regex`, but the notebook never prints `load_warnings` and the review run captured a different warning (byte-fallback) (GL-m4). |
| Clean default | Documented (Kaggle T4, reviewed blob, 1 restart) + direct CPU (12/12, install skipped) | Works after the restart; one-pass Run all not established (GL-M1). Numbers identical between the T4 record and the CPU run. |
| Active learning | Direct CPU (P6) | No documented exercise exists. Changing `EPOCHS`/`LEARNING_RATE`/`SEED` and re-running Section 6 continues from the adapted weights (second-run epoch-1 loss 3.37 vs 11.61 on the first); re-running Section 5 prints micro F1 0.9189 as the "Pre-adaptation Baseline" (GL-M2). |
| Reuse and recovery | Direct CPU, loader level (P4) + source | JSON/JSONL sample round trip and `tokens` alias accepted; generic labels and two-label datasets rejected (GL-M3); invalid JSON and CSV give a JSON parse error; `ner` given as dicts gives a bare `TypeError`; a cancelled upload gives `StopIteration`; datasets of 4–13 records pass Section 4 and fail in Section 5 because the validation split has fewer than 4 records (GL-m5). Upload dialog and downstream BYOD stages with a real file not verified. Artifact reload with tampered adapters: empty or renamed adapter loads silently (base predictions returned), wrong base revision accepted (GL-m2). |

## 5. Findings

### Major

#### GL-M1 — `Run all` needs a manual restart after the install cell, and the registry calls Run-all "verified"

**Cell/section:** Section 1, `gliner_ner-03`; generator `tools/build_notebook.py` (install cell text, line 69); `tutorials/README.md` registry row.  
**Observed issue:** The install cell runs `pip install -q` of ten exact pins (including `torch==2.14.0`, `torchvision==0.29.0`, `numpy==2.5.3`) into the running kernel, then raises `RuntimeError('… Restart the runtime, then rerun from the top.')` if any already-imported distribution changed version. Hosted kernels pre-import NumPy and torch at other versions.  
**Consequence:** A learner choosing **Run all** stops in cell 1 and must restart and run again; the notebook is not Run-all conformant (§25.7).  
**Evidence:** Documented: the only hosted run of this blob records "12/12 ok (1 restart after install cell)". Source: guard text in `gliner_ner-03`. The review did not run the install (CPU run used `DIMER_NOTEBOOK_CI_PREINSTALLED=1`).  
**Recommended correction:** Replace the in-kernel install with the fleet's uv isolated-environment pattern: a carrier cell bootstraps uv, creates `uv venv --managed-python --python 3.12.12 <ROOT>/env`, installs a hash-locked `requirements.txt` with `uv pip install --require-hashes --only-binary :all:`, and runs the workload in that environment so the kernel's preloaded NumPy/torch are never replaced (reference: `ast-audio-classification-pipeline/tutorials/DIMER_Sound_Event_Classification_Workshop.ipynb` on `main`). Change it in `tools/build_notebook.py`/`tools/notebook_template.py` and regenerate. Until a one-pass run is recorded, set the registry Run-all column to the recorded outcome.  
**Acceptance check:** A fresh Colab (or fresh-container) Run all of the regenerated notebook completes every code cell with no restart and no error output, recorded with commit, blob and runtime in `docs/release-verification.md`.  
**Spec:** RUN1, RUN10, ENV6, REL2, REL11.

#### GL-M2 — Prescribed reruns reuse the adapted model: the "baseline" is not zero-shot and training stacks

**Cell/section:** Sections 4–7 (`gliner_ner-13`, `-15`, `-17`, `-19`); opening BYOD paragraph ("set `USE_BYOD = True` in Section 4 and re-run from that cell"); `GLiNERPipeline.adapt` in `src/gliner_ner_pipeline/pipeline.py`.  
**Observed issue:** `pipe` is loaded once in Section 3. `adapt` updates `pipe.model` in place, and Section 5 evaluates whatever `pipe` currently holds. Re-running from Section 4 (the BYOD instruction) or from Section 6 after changing a form field therefore starts from the sample-adapted weights.  
**Consequence:** After any rerun the cell headed "Pre-adaptation Baseline Metrics" reports adapted metrics, the delta is computed against the wrong reference, and a learner who changes `EPOCHS`, `LEARNING_RATE` or `SEED` sees the combined effect of two training runs. For BYOD, the user's "zero-shot baseline" has already been fine-tuned on the biomedical sample. The exported adapter then reflects both runs.  
**Evidence:** Direct execution (P6, CPU): after the default run, re-evaluating Section 5 gave micro F1 0.9189, equal to the adapted result (original baseline 0.6857); re-running `adapt` with unchanged defaults started at epoch-1 loss 3.37 (first run 11.61) and reached validation micro F1 0.973.  
**Recommended correction:** Make each adaptation start from the verified base: reload the base pipeline (or restore a saved copy of the trainable state) at the start of Section 5 and of `adapt`, and say in Section 4 and Section 6 which cells to re-run after changing a field.  
**Acceptance check:** Run the notebook, then re-run from Section 4 with defaults: Section 5 prints micro F1 0.6857 again and Section 6 epoch-1 loss equals the first run's (within float tolerance); repeat with `EPOCHS = 1` and confirm the history has one epoch starting from the same loss.  
**Spec:** DAT13, DAT14, GDL10 (rerun scope), EVAL8.

#### GL-M3 — BYOD accepts only the three built-in biomedical labels; schema and limits are never stated

**Cell/section:** Section 4 (`gliner_ner-12`, `-13`); `validate_dataset` / `load_byod_dataset` in `src/gliner_ner_pipeline/samples.py`; Sections 5–8 hard-code `ADAPT_CLASSES` and the Pembrolizumab sentence; opening and Prerequisites.  
**Observed issue:** `load_byod_dataset` and `validate_dataset` default `allowed_labels` to `ADAPT_CLASSES = ("disease", "chemical_drug", "gene_protein")`, the notebook passes no label list, and validation requires every allowed label to appear. Evaluation, adaptation and inference all use `ADAPT_CLASSES` and the fixed biomedical test sentence. The opening says the schema and token/span ceilings "are stated in the Prerequisites and in Section 4"; neither cell shows the record schema (`id`, `tokenized_text`/`tokens`, `ner` as `[start_token, end_token, label]`, inclusive end), the minimum size, or the 384-token and 1,000-record ceilings.  
**Consequence:** The capability the notebook advertises ("arbitrary label sets", "your own JSON or JSONL dataset") cannot be exercised on user data: any NER dataset with other entity types is rejected, and a biomedical dataset without gene mentions is rejected. A learner has to read 17,621 characters of carried code to discover the format. Even an accepted dataset ends with inference on the sample sentence rather than the user's text.  
**Evidence:** Direct execution (P4): `person`/`organization`/`location` dataset → `ValueError: record[0] span[0] label 'person' not in ['chemical_drug', 'disease', 'gene_protein']`; the sample without `gene_protein` → `ValueError: dataset missing examples for required labels: ['gene_protein']`; sample JSONL and the `tokens` alias accepted. Source: `ADAPT_CLASSES` in `gliner_ner-15`, `-17`, `-19`, `-21`.  
**Recommended correction:** Add a `BYOD_LABELS` form field (default = `ADAPT_CLASSES`) passed to `load_byod_dataset`, `evaluate`, `adapt` and `detect`; require each declared label to appear in the training split only; add a `BYOD_INFERENCE_TEXT` field; state the schema with a two-record example, the minimum record count that survives the split, and the ceilings in Section 4 before the upload. Change `tools/notebook_template.py` and `samples.py`, then regenerate.  
**Acceptance check:** A 16-record JSONL with labels `person`, `organization`, `location` passes validation, the split, baseline, adaptation, evaluation, inference on a user sentence, export and reload; Section 4 shows the schema and limits before the upload control.  
**Spec:** DAT10, DAT12, DAT13, DAT14, REL12.

#### GL-M4 — Declared `GUIDED`, but there is no guided layer or learner activity

**Cell/section:** whole notebook; opening and Prerequisites (`gliner_ner-00`, `-01`); `tools/notebook_template.py`.  
**Observed issue:** No intended-learner statement, how-to-use, roadmap, glossary, prediction prompt, What-to-notice note, checkpoint with sample answer, Predict → Change → Run → Observe → Explain activity, troubleshooting section or conclusion template (P1 markers all absent; the one "Predict" hit is the word "prediction parity"). The three carried-module cells (49,023 characters) are not labelled as infrastructure or collapsed. Section 7 prints the full metric JSON with no explanation of micro vs macro averaging, `hits`, or per-class `predicted`/`gold`. The learning objectives are things the notebook does, not things the learner does.  
**Consequence:** A learner new to span-based NER can run the notebook but is not asked to interpret anything, cannot tell which output is normal, and gets no help when the hosted runtime fails (restart, 1.16 GB download, memory). The `EPOCHS`/`LEARNING_RATE` fields invite experiments that GL-M2 makes misleading.  
**Evidence:** Source inspection; P1 static markers.  
**Recommended correction:** Add the 2.2 guided layer in the template: audience and prerequisites, how-to-use with infrastructure labels (`# @title Infrastructure: …`, `cellView: form` on the install and carried cells), a roadmap and Input → Model → Output contract, a short glossary, a prediction before the baseline and before the post-adaptation evaluation, What-to-notice notes after Sections 5, 6 and 7 that read the per-class table, one bounded change-one-thing activity (for example `EPOCHS = 1` vs `3`, after GL-M2 is fixed), troubleshooting, and a conclusion template.  
**Acceptance check:** Each GDL1–GDL14 item can be pointed to in the regenerated notebook; the activity names the cells to re-run and produces a valid comparison under the GL-M2 check.  
**Spec:** GDL1–GDL14, UX4, UX5, UX8, UX9.

### Minor

#### GL-m1 — Release documents describe a different notebook and contradict each other

**Cell/section:** `docs/release-verification.md`, `tutorials/README.md` (registry and conformance notes), `STATUS.md`, `README.md` (Release status).  
**Observed issue:** `release-verification.md` calls the notebook `TASK-INFERENCE`, cites spec 1.1, and its supported procedure checks things this notebook does not contain: `validate_inputs` and a duplicate-label probe, an `evaluation_report` verdict `not-measurable`, form fields `LABELS`, `THRESHOLD`, `GOLD_JSON`, a synthetic Marie-Curie sentence and `outputs/gliner_ner_input_manifest.json`. `tutorials/README.md` conformance notes repeat the `not-measurable` and `THRESHOLD` claims and its registry status cell is a garbled sentence. `STATUS.md` says "awaiting clean-runtime execution" and `README.md` says the carrier "has never been executed", while `release-verification.md` records a PASS.  
**Consequence:** A reviewer cannot apply the release procedure to this notebook, and readers get three different statements of its execution status.  
**Evidence:** Source inspection of the four files at the reviewed commit.  
**Recommended correction:** Rewrite the procedure and the conformance notes for the E2E notebook (the stages and outputs in Sections 4–10), and make the status sentences agree with the recorded run, including its restart.  
**Acceptance check:** Every step of the procedure names a cell or output that exists in the notebook; the four documents state the same execution status.  
**Spec:** REL9, REL10 (coverage and record accuracy).

#### GL-m2 — Reload verification is weaker than claimed; mismatched adapters load silently

**Cell/section:** Section 9 (`gliner_ner-22`, `-23`); `load_artifact` in `pipeline.py`.  
**Observed issue:** The opening promises "parameter match and prediction parity", but the cell only compares entities on one sentence. `load_artifact` calls `load_state_dict(weights, strict=False)` without reporting missing or unexpected keys, and checks `model_id` but not `model_revision` or the encoder revision. The artifact has no manifest of tensor names, shapes or digests.  
**Consequence:** A truncated or mis-keyed adapter reloads as the base model without an error; only the one-sentence comparison would catch it, and only if base and adapted predictions differ on that sentence. An adapter for another base revision is accepted.  
**Evidence:** Direct execution (P7): an adapter with an empty `adapter_state_dict` and one with renamed keys both loaded and returned base-model predictions (`PDCD1 receptor`, 0.5709); an adapter claiming base revision `000…0` loaded with no error.  
**Recommended correction:** Load with `strict=False` but assert that every adapter key was consumed and that the set equals the trainable-parameter names; check both revisions; compare reloaded tensors to the in-memory ones (exact) and predictions over the whole validation split; record tensor names, shapes and a digest in the artifact metadata.  
**Acceptance check:** The three tampered adapters above each raise a named error; the default path prints a tensor-equality result and a validation-split prediction parity.  
**Spec:** VER4, VER5, ART5, MOD8.

#### GL-m3 — The +0.2332 delta is shown without the context needed to read it

**Cell/section:** Section 7 (`gliner_ner-18`, `-19`); Interpretation (`gliner_ner-26`).  
**Observed issue:** The delta is computed on 6 validation sentences (18 gold spans) that share one sentence template, the same split is printed every epoch during training, and the zero-shot baseline is prompted with snake-case labels (`gene_protein`: baseline F1 0.2222, 1 of 6 gold spans found) that adaptation then learns. The Interpretation section says the run does not establish benchmark performance but does not mention the sample size, the template, or the label-name effect, and the results are not labelled as tutorial/sanity metrics where they are printed.  
**Consequence:** A learner is likely to read "+23 F1 points from 3 epochs" as a property of domain adaptation in general.  
**Evidence:** Direct execution (P3, P5): split 18/6; per-class baseline and adapted results as in the summary table; every sample record starts with the drug token. The label-name contribution is inferred, not measured.  
**Recommended correction:** Label the printed metrics as tutorial evidence on 6 sentences, say that the validation split is also the training monitor (no independent test), and add one sentence on prompt-label wording (optionally a baseline with natural-language labels such as "gene or protein").  
**Acceptance check:** The Section 7 output and the Interpretation name n = 6, the single template, the monitor/test distinction and the label-name caveat.  
**Spec:** EVAL3, EVAL6, DAT8, ENV8, SPL6.

#### GL-m4 — Inaccurate statements about the split, the trainable layers and the expected warning

**Cell/section:** Section 4 markdown, `split_ner_dataset` docstring; Section 6 markdown; Prerequisites.  
**Observed issue:** (a) "class balance preserved across both splits" and the docstring "Guarantees all classes … appear in both" — the split is a seeded shuffle with no stratification or check. (b) "only the span representation and prompt projection layers are updated" — the trained set is `span_rep_layer`, `prompt_rep_layer`, `rnn` (BiLSTM) and the `token_rep_layer` projection, 11.4 M parameters; the count is not printed. (c) The expected `fix_mistral_regex` warning "is captured in `load_warnings`", but the notebook never prints `load_warnings`, and the review run captured only a SentencePiece byte-fallback `UserWarning`.  
**Consequence:** Small but concrete mismatches between prose and behaviour; a BYOD user may rely on a coverage guarantee that does not exist.  
**Evidence:** Source; direct execution (P5 trainable modules and `load_warnings`).  
**Recommended correction:** Implement the coverage check (or drop the claim), print trainable/frozen counts by module, print `load_warnings` and describe both warnings.  
**Acceptance check:** Section 6 prints the trainable modules and count; Section 3 prints the load warnings; the split either raises on a missing label or the prose no longer claims coverage.  
**Spec:** FT5, UX3, SRC3 (stale instructions).

#### GL-m5 — BYOD failure handling and the upload-only interface

**Cell/section:** Section 4 (`gliner_ner-13`); `load_byod_dataset`.  
**Observed issue:** BYOD requires `google.colab.files.upload()` although Jupyter is a supported runtime, and there is no location field; a cancelled upload raises a bare `StopIteration`; `ner` entries given as objects raise `TypeError: list indices must be integers or slices, not str`; a CSV yields a JSON-parse error that does not say JSON/JSONL is required; datasets of 4–13 records pass Section 4 and then fail in Section 5 with "dataset requires at least 4 records, got N", which refers to the validation split without saying so.  
**Consequence:** Jupyter users cannot use BYOD at all, and several common mistakes give errors that do not tell the user what to fix.  
**Evidence:** Direct execution (P3, P4).  
**Recommended correction:** Add `BYOD_PATH = ''  # @param` read before any `google.colab` import; check the per-split minimum in Section 4; translate the listed errors into messages naming the expected format and the fix.  
**Acceptance check:** Each case above produces a message naming the failed rule and the fix, before Section 5; `BYOD_PATH` works without `google.colab`.  
**Spec:** DAT19, UX10, EXE2.

### Suggestions

- **GL-S1** — Regenerate against NOTEBOOK_SPEC 2.2 (the notebook declares 2.0).
- **GL-S2** — Document `DIMER_NOTEBOOK_CI_PREINSTALLED` in the notebook (EXE5).
- **GL-S3** — Record a hosted CPU run: CPU is the documented default runtime and the only hosted record is a T4.
- **GL-S4** — Export the split record IDs, a dataset digest, `weight_decay` and the trainable-parameter count in `gliner_ner_evaluation_report.json` (OUT7, OUT9, FT6).

## 6. Readiness

**Needs revision.** Four Major findings are open, and applicable `MUST`s fail: RUN1/RUN10/ENV6/REL2 (GL-M1) and DAT10/DAT12/DAT13/DAT14 (GL-M3). Remaining gates after the fixes: a one-pass hosted Run all of the regenerated blob, recorded with commit and runtime; a BYOD positive and negative run (REL12); and release documents rewritten for this notebook (GL-m1).

## 7. Verified versus inferred

- **Verified by direct execution (CPU, torch 2.13.0+cpu, install skipped):** all 12 code cells complete at defaults; real 1.16 GB fetch and digest verification of both snapshots; split 18/6; baseline 0.6857 and adapted 0.9189 micro F1 (identical to the T4 record); 11,418,112 trainable parameters; reload parity on the sample sentence; rerun contamination of Sections 5 and 6 (GL-M2); BYOD label restriction (GL-M3); silent reload of empty, renamed and wrong-revision adapters (GL-m2); the BYOD error messages listed in GL-m5.
- **Taken from documented evidence:** the restart after the install cell (Kaggle T4 record for this blob).
- **Inferred:** that Colab also triggers the guard (Colab preloads NumPy/torch at versions other than the pins); the label-name share of the zero-shot gap (GL-m3).
- **Only Kurt can confirm:** whether BYOD is meant to support arbitrary label sets or only this biomedical label set (GL-M3 severity depends on it).
- **Most likely to be wrong:** GL-M1's mechanism on Colab. The restart comes from the Kaggle record's text; the executed notebook is not archived, and if a Colab image already matched every loaded pin the guard would not fire there.
