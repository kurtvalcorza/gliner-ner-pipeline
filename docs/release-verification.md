# Release verification

`tutorials/gliner_ner_colab.ipynb` (`E2E`, `GUIDED`, **standalone** carrier, DIMER Notebook Specification 2.2) is a
**release candidate** until the exact notebook revision has executed top-to-bottom, in one pass and with no manual
restart, in a clean supported runtime. Unit tests, JSON validation, code-cell compilation, the generator parity
checks and `tools/validate_release_assets.py` are necessary checks but are **not** runtime evidence. This file is the
durable release-gate record for the notebook.

## Automatic coverage (static, every pull request)

CI runs `tools/validate_release_assets.py`, which checks:

- notebook JSON parses; every code cell compiles as plain Python (no `%`/`!` magics); no persisted outputs or
  execution counts; no unresolved placeholder markers; every code cell is preceded by an explanatory markdown cell;
- exactly one tutorial notebook, named in `tutorials/README.md` with its `E2E` profile, the notebook-spec version and
  the standalone carrier; `metadata.dimer` declares that profile, spec `2.2`, a pedagogical mode, `standalone: true`
  and `generated_from` (repository, revision, module SHA-256, generator);
- the standalone carrier (ST1–ST6, PAR1–PAR3): no clone, repository install or repository import on the primary
  path; three cells tagged `embedded_module` (`metrics.py`, `pipeline.py`, `samples.py`, in dependency order), each
  equal to its module after the generator's documented rewrites plus one collapsed `# @title Infrastructure:` line;
  the inline `MANIFEST` and `ENCODER_MANIFEST` equal to the committed manifests; the inline `PINS` equal to the
  `pyproject.toml` pins; the notebook byte-identical (on LF) to `tools/build_notebook.py` output;
- the isolated runtime (GL-M1): exactly two kernel cells — the install cell, which verifies the pinned `uv` wheel by
  size and SHA-256, creates a managed CPython 3.12.12 environment and installs the carried hash-locked
  `tutorials/requirements-colab.lock.txt` with `--require-hashes --only-binary :all:` on Linux x86_64 only, and the
  router that sends every later cell to one persistent worker in that environment; the worker's `google.colab` stubs
  carry a module spec;
- the guided layer (GL-M4): audience, task contract, how-to-use, roadmap, six predictions, seven What-to-notice
  notes with worked answers, the Section 11 change-one-thing activity, troubleshooting, glossary and conclusion
  template; Sections 1–3 code cells titled `Infrastructure` and collapsed;
- the profile-specific calls and outputs: both snapshots staged and verified, `load_warnings` printed,
  `reset_to_pretrained()` before Sections 5 and 6, `validate_dataset` / `split_ner_dataset` / `split_label_coverage`
  with the active label set, the BYOD fields (`USE_BYOD`, `BYOD_PATH`, `BYOD_LABELS`, `BYOD_INFERENCE_TEXT`), the
  record schema and limits printed before the upload, `pipe.adapt(..., labels=LABELS)` with the trainable/frozen
  counts printed, the evaluation report labelled as tutorial evidence, `detect` with its output-contract checks, the
  adapter export and the reload check over every adapter tensor and every validation sentence, and the four
  `outputs/` files; no bare `assert` in learner cells; stale text removed by the review fixes may not return;
- forbidden patterns (credential-in-URL, `git clone` / `github.com` / repository import on the primary path, a mutable
  `revision='main'`, direct `gliner`, `transformers` or `huggingface_hub` use outside the carried module cells,
  `trust_remote_code=True`, `pickle.load`, `torch.load(` without `weights_only=True`, `extractall(`);
- `STATUS.md`, `README.md` and `tutorials/README.md` agree on one release-status token and no document makes an
  unsupported release-grade, production-readiness or benchmark claim;
- `MODEL_CARD.md` front matter, heading order and immutable provenance; weight facts against the manifests.

CI also runs `ruff check src tests tools`, `tools/build_notebook.py --check` and the offline unit suite (no weights,
no torch in CI; torch-dependent tests skip). These are source and unit checks, **not** execution evidence.

## Executor paths

| Path | Runtime | Role |
|---|---|---|
| Google Colab (supported user path) | fresh Colab CPU or GPU runtime, Linux x86_64 | The runtime the tutorial is written for; a one-pass **Run all** here is promotion evidence |
| Kaggle kernel or an equivalent fresh Linux container | fresh CPU or GPU container; the committed notebook executed verbatim, cell by cell, in one kernel, with **no repository checkout** | Clean-room executor of the same class; promotion evidence when it completes in one pass |
| Local harness (pre-flight only) | workstation, sequential cell executor with `DIMER_NOTEBOOK_CI_PREINSTALLED=1` (install and routing skipped) | Builder pre-flight to catch defects; **not** a supported runtime and not promotion evidence |

## Supported release verification procedure

Before changing the registry status from `Candidate` to `Release-grade`:

1. resolve the exact commit under review and confirm static CI is green on it;
2. open that exact notebook revision in a fresh Linux x86_64 runtime (Colab, or a fresh-container executor above) with
   **no repository checkout**, an empty Hugging Face cache, no `dimer_isolated_env/` folder and no pre-staged files
   under `weights/gliner-multi-v2.1/` or `weights/mdeberta-v3-base-tokenizer/`;
3. choose **Run all** once, with every form field at its default (`USE_BYOD = False`, `BYOD_PATH = ''`,
   `BYOD_LABELS = ''`, `BYOD_INFERENCE_TEXT = ''`, `VAL_FRACTION = 0.25`, `SEED = 42`, `EPOCHS = 3`,
   `LEARNING_RATE = 5e-5`, `BATCH_SIZE = 4`), and do not edit any cell;
4. confirm that **every code cell completed in that one pass with no restart and no error output**; a run that needed
   a restart or a second pass is recorded as such and is not promotion evidence;
5. check each section's output against the notebook:
   - Section 1: the install cell prints the isolated environment, `isolated_python` 3.12.12 and 50 locked packages;
     the router reports the worker pid; the runtime record shows `NOTEBOOK_SOURCE.repository_revision` equal to
     `metadata.dimer.generated_from.revision` and `torch` 2.14.0, `transformers` 4.57.6, `gliner` 0.2.29;
   - Section 3: both inline manifests asserted, 3 GLiNER files and 4 encoder files fetched (empty start) and
     verified, the pipeline loaded on the reported device; the following cell prints `load_warnings`;
   - Section 4: the record schema and limits, 24 records split 18 / 6, span counts per label on each side with no
     label missing from validation, and four rejected probes;
   - Section 5: the per-label baseline table (micro F1 0.6857 on CPU and in the 2026-09-17 T4 record);
   - Section 6: 11,418,112 trainable and 277,531,392 frozen parameters and a three-epoch history;
   - Section 7: the adapted table, the deltas (+0.2332 micro F1 in the CPU check and the T4 record), the `Reading:`
     line and `outputs/gliner_ner_evaluation_report.json`;
   - Section 8: entities for the sample sentence with all three contract checks `True`;
   - Section 9: `outputs/gliner_ner_adapter.pt` (26 tensors), `tensors_identical` equal to `tensors_compared` and every
     sentence identical;
   - Section 10: `outputs/gliner_ner_result.json` and `outputs/gliner_ner_entities.csv` listed with the other two files;
   - Section 11: one row in the run table;
6. for the BYOD gate (REL12), run once more with `USE_BYOD = True` and a valid `.json`/`.jsonl` file of at least 14
   records with its own label set and `BYOD_INFERENCE_TEXT`, and once with an invalid file (for example a CSV), and
   record that the invalid file stops in Section 4 with a message naming the rule;
7. record the notebook Git blob id, commit, runtime (platform, GPU or CPU, Python, PyTorch, Transformers, gliner,
   device), whether the cache, the isolated environment and both weights directories were clean, whether a restart
   was needed, the wall time, the produced outputs and the Section 5/7 metrics in the tables below;
8. record no access tokens or other secrets.

A known-failing default path in the supported runtime blocks release.

## Manual clean-runtime evidence

| Notebook | Commit / notebook blob | Date (UTC) | Executor | Outcome |
|---|---|---|---|---|
| `gliner_ner_colab.ipynb` | `58ee5df` / `ecbaee8862f1` | 2026-09-17 | Kaggle T4 (`kurtvalcorza/dimer-nb2-gliner-ner` v2) | Completed **only after a manual restart** after the install cell (12/12 ok in the second pass): not a one-pass Run all, not promotion evidence |

## Recorded executions

Notebook identity is the Git blob id of `tutorials/gliner_ner_colab.ipynb` (verify with
`git rev-parse <commit>:tutorials/gliner_ner_colab.ipynb`). Wall times are the sum of per-cell times reported by the
executor and include installs and the model download; they are measurements for the stated runtime, not general
estimates.

| Date (UTC) | Commit / notebook blob | Executor | Path exercised | Wall | Outcome |
|---|---|---|---|---|---|
| 2026-09-17 | `58ee5df` / `ecbaee8862f1` | Kaggle T4 (`kurtvalcorza/dimer-nb2-gliner-ner` v2) | Default sample path of the previous notebook version (in-kernel pinned install) | 298.7 s | **Passed after a manual restart** after the install cell (the in-kernel install replaced loaded packages); +0.2332 micro F1; adapter reloaded. Not a one-pass Run all; not promotion evidence |
| 2026-10-04 | `fe3d5ba` / `87a15e765580` | Colab CLI 0.7.4 sequential execution, fresh Colab Tesla T4 (not a browser Run all) | Default sample path, every form field at its default | 142.8 s | **One pass, no restart, 0 errors**, 16/16 code cells; baseline micro F1 0.6857 → adapted 0.9189 (+0.2332); adapter reload 26/26 tensors, 7/7 sentences |

The 2026-09-17 record ran the previous notebook version; the notebook was regenerated afterwards (review PR #10: uv
isolated environment, guided layer, BYOD and reload fixes). The 2026-10-04 record covers the current blob.

### 2026-10-04 — Colab CLI, fresh Colab Tesla T4, blob `87a15e765580`

- **Commit / notebook blob:** `fe3d5ba6dad24691471ebd1fd2b38f4931b417d0` / `87a15e765580cf8c7d65d6bbe0f2604a42953c0d`
  (fetched byte-exact from `raw.githubusercontent.com` at the commit; blob checked before the VM was allocated).
- **Executor:** Colab CLI 0.7.4 sequential execution (`colab exec -f`) on a fresh Colab VM, Tesla T4: every code cell
  in order in one kernel. Not a browser **Run all**: forms were not rendered, no upload dialog was answered, and the
  CLI records no execution counts — order is evidenced by `exec.log` (`Executing cell 1/16` … `16/16`).
- **Path:** default settings only (`USE_BYOD = False`, `SEED = 42`, `EPOCHS = 3`, `LEARNING_RATE = 5e-5`,
  `BATCH_SIZE = 4`, `VAL_FRACTION = 0.25`); clean VM: no cache, no `dimer_isolated_env/`, no pre-staged weights.
- **Outcome:** one pass, no restart, 0 error outputs; 16/16 code cells (cells 4–6, the carried modules, print
  nothing by design). Wall time 142.8 s (CLI session, including the uv install and the model download).
- **Runtime (Section 1):** isolated Python 3.12.12 (kernel 3.13.15), 50 locked packages, setup 48 s; `torch`
  2.14.0+cu130, `transformers` 4.57.6, `gliner` 0.2.29, CUDA available, device `cuda:0`;
  `NOTEBOOK_SOURCE.repository_revision` `04b30e6` equals `metadata.dimer.generated_from.revision`.
- **Sections 3–4:** 3 GLiNER files and 4 encoder files fetched and verified; 24 records split 18 / 6, no label missing
  from validation, four rejected probes.
- **Section 5 baseline:** micro F1 0.6857 (P 0.7059, R 0.6667), macro F1 0.6375.
- **Section 6:** 11,418,112 trainable / 277,531,392 frozen parameters; epochs loss 11.3457 / 8.9581 / 6.8758,
  validation micro F1 0.8108 / 0.9189 / 0.9189. The CPU check quoted in the guided answers gave 11.28 / 8.02 / 6.26
  and 0.8108 / 0.8649 / 0.9189; the T4 history differs (likely GPU versus CPU arithmetic; not investigated), the final metrics do not.
- **Section 7 adapted:** micro F1 0.9189 (P 0.8947, R 0.9444), macro F1 0.9188; deltas +0.2332 micro, +0.2813 macro
  (`gene_protein` +0.7009, `chemical_drug` +0.1429, `disease` 0.0).
- **Sections 8–11:** 3 entities on the sample sentence, all contract checks `True`; adapter 26 tensors (43.57 MB),
  reload parity 26/26 tensors and 7/7 sentences identical; the four `outputs/` files listed; one row in the run table.
- **Evidence files** (`docs/verification/2026-10-03-colab-t4/`, SHA-256):
  - `gliner_ner_colab_fe3d5ba_colab-cli-t4_output.ipynb` `5e7f88c960e3ba5d9cb4c88df74d4b779c114b1348761bc8355fd36b185d655e`
  - `run_summary.json` `21f7582edd8758de849253bbcbc0ce39b85518d93179e447e65eaa5e5cf20903`
  - `exec.log` `c2471e66697c4002c98aefdb2181ccc50759883dada8b440d70d4453a5d0c6e9`
- **Not exercised:** the BYOD positive and negative runs (procedure step 6), the Colab upload dialog, the optional
  guided activities (Section 11 one-epoch row), and a browser **Run all**.

## Current status

**Candidate.** Static checks pass. The current blob (`87a15e765580`, commit `fe3d5ba`) completed one hosted pass with
no restart and no errors on a fresh Colab Tesla T4 (Colab CLI sequential execution, 2026-10-04, default path). Still
required before promotion: a one-pass **Run all** in the supported Colab path (or a Kaggle fresh-container run) as the
procedure defines it, and the BYOD positive and negative runs (step 6).
