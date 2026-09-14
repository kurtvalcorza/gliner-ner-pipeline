# Release verification

`tutorials/gliner_ner_colab.ipynb` (`TASK-INFERENCE`, **standalone** carrier) is a **release candidate** until the exact
notebook revision has executed top-to-bottom in a clean supported runtime. Unit tests, JSON validation, code-cell
compilation, the generator parity checks and `tools/validate_release_assets.py` are necessary checks but are **not**
runtime evidence under DIMER Notebook Specification 1.1. This file is the durable release-gate record for the notebook.

## Automatic coverage (static, every pull request)

CI runs `tools/validate_release_assets.py`, which checks:

- notebook JSON parses; every code cell compiles as plain Python (no `%`/`!` magics); no
  persisted outputs or execution counts; no unresolved placeholder markers; every code cell
  is preceded by an explanatory markdown cell;
- exactly one tutorial notebook, named in `tutorials/README.md` with its `TASK-INFERENCE`
  profile, the notebook-spec version and the standalone carrier; `metadata.dimer` declares that
  profile, spec `1.1`, `standalone: true` and `generated_from` (repository, revision, module
  SHA-256, generator);
- the standalone carrier (ST1–ST6, PAR1–PAR3): no clone, repository install or repository import on
  the primary path; exactly one cell tagged `embedded_module` equal to
  `src/gliner_ner_pipeline/pipeline.py` after the generator's documented rewrites (one rule, which
  makes the shared `_WEIGHTS_ROOT` working-directory-relative); the inline `MANIFEST` equal to the
  committed GLiNER snapshot manifest and the inline `PINS` equal to the `pyproject.toml` runtime
  pins; the notebook byte-identical (on LF) to `tools/build_notebook.py` output for its recorded
  revision; the pinned-install cell with its restart-on-stale-import guard; `NOTEBOOK_SOURCE`
  recorded in exports;
- the **second** carried snapshot: the inline `ENCODER_MANIFEST`, the assertion that it names
  `ENCODER_MODEL_ID`/`ENCODER_REVISION` as carried by the module, and its staging and digest
  verification through `stage_missing_encoder_files` / `verify_encoder_snapshot` before the pipeline
  loads (the encoder revision is the one extra 40-hex revision the documents may cite);
- `MODEL_ID`/`MODEL_REVISION` are bound only in the carried module cell (and repeated in the inline
  manifest, which the notebook asserts against the module before fetching), the revision is a 40-hex
  immutable commit, and the same identity string appears in `README.md`, `MODEL_CARD.md`, and
  `docs/WEIGHTS.md` with no stray revisions;
- the profile-specific public-API calls (`stage_missing_files` and `stage_missing_encoder_files`,
  `verify_snapshot` and `verify_encoder_snapshot`,
  `GLiNERPipeline.from_pretrained(weights_dir=..., encoder_dir=...)`, `validate_inputs` with the form
  threshold plus the deliberate duplicate-label rejection probe, `detect`, and `evaluation_report`
  with the optional gold spans), the ceiling constants carried by the module, the offset and
  threshold sanity checks, the four expected `outputs/` files, the learner-facing statements (score
  not a calibrated probability, threshold caller-owned, two pinned snapshots, 384-word library cut,
  expected `fix_mistral_regex` warning, `not-measurable` without gold and `sample-sanity` with it)
  and the gated-off BYOD default listed in the validator; forbidden patterns (credential-in-URL, any
  `git clone` / `github.com` / repository import on the primary path, a mutable `revision='main'`,
  direct `gliner`, `transformers` or `huggingface_hub` use **outside the carried module cell**,
  `trust_remote_code=True`, `pickle.load`, `torch.load(`, `extractall(`);
- `STATUS.md`, `README.md` and `tutorials/README.md` agree on one release-status token and no
  document makes an unsupported release-grade, production-readiness or benchmark claim;
- `MODEL_CARD.md` front matter (`model_card_spec: "1.1"`), single H1, required heading order, and
  immutable provenance.

CI also runs `ruff check src tests tools`, `tools/build_notebook.py --check`, and the offline unit
suite (`tests/test_pipeline.py`, `tests/test_role_helpers.py`, `tests/test_notebook_parity.py`;
injected runner and temporary manifests, no weights). These are source/provenance and unit checks.
They are **not** execution evidence.

## Executor paths

| Path | Runtime | Role |
|---|---|---|
| Google Colab (supported user path) | Colab CPU runtime (CUDA used automatically when present) | The runtime the tutorial is written for; a clean top-to-bottom run here is promotion evidence |
| Kaggle CLI kernel or equivalent fresh container | Fresh CPU or GPU container, Python 3.12 image; the committed notebook executed verbatim, cell by cell, in a fresh interpreter with a `google.colab` shim and **no repository checkout** (the notebook is standalone) | Reproducible clean-room executor of the same class; needed whenever the hosted kernel pre-imports a NumPy or Pillow that differs from the `pyproject.toml` pins, because the tutorial's fail-closed stale-import guard correctly halts the in-kernel path after the pinned install |
| Local harness (pre-flight only) | Workstation, sequential cell executor with a `google.colab` shim, empty model cache | Builder pre-flight to catch defects before spending cloud runs; **not** a supported runtime and not promotion evidence |

## Supported release verification procedure

Before changing the registry status from `Candidate` to `Release-grade`:

1. resolve the exact PR/commit head under review and confirm static CI is green;
2. open that exact notebook revision in a new CPU or CUDA runtime (Colab, or a fresh-container
   executor above) with **no repository checkout**, an empty Hugging Face cache, and no pre-staged
   files under the working-directory snapshots `weights/gliner-multi-v2.1/` or
   `weights/mdeberta-v3-base-tokenizer/` (the standalone path writes both manifests itself and stages
   every listed file, so neither directory may be seeded);
3. run the notebook top-to-bottom without editing implementation cells (form parameters at their
   defaults for the sample path: `USE_BYOD = False`, `LABELS = 'person, organization, location, date'`, `THRESHOLD = 0.5`, `GOLD_JSON = ''`);
4. verify that Section 1 reports `NOTEBOOK_SOURCE.repository_revision` equal to the revision recorded
   in `metadata.dimer.generated_from` and that the installed core package versions equal the inline
   `PINS` (= `pyproject.toml`) (`torch==2.14.0`, `torchvision==0.29.0`, `torchaudio==2.11.0`, `gliner==0.2.29`, `transformers==4.57.6`, `huggingface-hub==0.36.2`, `safetensors==0.8.0`, `numpy==2.5.3`, `protobuf==6.31.1`, `sentencepiece==0.2.2`);
5. verify every default-path stage completes:
   - pinned runtime installed from the inline `PINS` with no GitHub access;
   - the carried module cell executes (defines `GLiNERPipeline`, `validate_inputs`,
     `evaluation_report`, `entity_f1`, `verify_snapshot`, `verify_encoder_snapshot`,
     `stage_missing_files`, `stage_missing_encoder_files`) with no import of the repository package;
   - synthetic sentence and four-type label set authored in code with the text SHA-256 printed;
   - ceilings `MAX_TEXT_CHARS = 5000`, `MAX_LABELS = 25`, `MAX_LABEL_CHARS = 100`, the 384-word library limit and `DEFAULT_THRESHOLD = 0.5` printed before the model runs, and `validate_inputs` writing `outputs/gliner_ner_input_manifest.json` with verdict `accepted`, both pinned identities, and one recorded rejection finding from the duplicate-label probe;
   - both inline manifests asserted against the module's constants, then `stage_missing_files(..., allow_download=True)` reporting the 3 GLiNER entries fetched from `urchade/gliner_multi-v2.1` at the immutable revision and `stage_missing_encoder_files(..., allow_download=True)` reporting the 4 encoder entries fetched from `microsoft/mdeberta-v3-base` at `ENCODER_REVISION`; `verify_snapshot` (3 files) and `verify_encoder_snapshot` (4 files) both printed; `GLiNERPipeline.from_pretrained(weights_dir=..., encoder_dir=...)` loading from the two verified directories with `load_warnings` containing only the expected `fix_mistral_regex` warning;
   - `detect` returning spans with in-text offsets and all four sanity checks true;
   - `evaluation_report` writing `outputs/gliner_ner_evaluation_report.json` with verdict `not-measurable`, an empty `metrics` list, and `needs` naming the gold spans a real evaluation requires (the sample ships none);
   - `outputs/gliner_ner_result.json` and `outputs/gliner_ner_entities.csv` written with the indexed entities and their offsets and scores, plus `NOTEBOOK_SOURCE`, both model identifiers and immutable revisions, both licences, the load warnings, and the runtime versions and device;
6. verify the exports exist and the interpretation section matches the observed path;
7. record the notebook Git blob id, commit, runtime (platform, Python, PyTorch, gliner, Transformers, device), both model
   identifiers and immutable revisions, whether the model cache and both weights directories were clean,
   outcome, produced outputs, the returned spans with their scores (as observations, not a metric) and the exact load warnings, and any warning or applicable `SHOULD`
   deviation in the table below;
8. record no access tokens or other secrets.

A known-failing default path in the supported runtime blocks release.

## Manual clean-runtime evidence

| Notebook | Commit / notebook blob | Date (UTC) | Executor | Outcome |
|---|---|---|---|---|
| `tutorials/gliner_ner_colab.ipynb` | | | | pending — queued to the GPU lane |

## Recorded executions

Notebook identity is the Git blob id of `tutorials/gliner_ner_colab.ipynb` (verify with
`git rev-parse <commit>:tutorials/gliner_ner_colab.ipynb`). Wall times are the sum of per-cell times reported by
the executor and include installs and the model download; they are measurements for the stated
runtime, not general estimates.

No execution of the notebook has been recorded. The only runtime measurements that exist for this repository are the pipeline smoke run documented in `MODEL_CARD.md` (CPU float32, both manifests verified, five spans detected in one synthetic sentence, 14.6 s wall clock including verification and load). That run exercised the
package, not this notebook, and is not notebook execution evidence.

| Date (UTC) | Commit / notebook blob | Executor | Path exercised | Wall | Outcome |
|---|---|---|---|---|---|
| — | — | — | Default sample path | — | pending — queued to the GPU lane |

## Current status

The notebook source is complete and passes the static checks above, including the generator parity
checks (`--check` OK); **no clean-runtime execution has been recorded**, so the registry status is **Candidate** and the manual-evidence row is pending.
Promotion requires a reviewer to confirm a recorded run against the notebook blob under review and
an integrator to promote it; promotion is not performed by the builder. The commit that adds a
recorded-execution row changes documentation only; the executed source is the commit named in the
row. One fact a reviewer should weigh: **the standalone carrier itself — executing the carried
module cell in a runtime that has no repository checkout — has been validated statically only
(parity PASS) and never run**, so the clean run will be the first execution of the standalone path,
of the real `hf_hub_download` staging path, and of the two-manifest staging order the model cell
carries.
