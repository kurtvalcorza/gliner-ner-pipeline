# Release verification

`tutorials/gliner_ner_colab.ipynb` (`TASK-INFERENCE`) is a **release candidate** until the exact notebook revision has
executed top-to-bottom in a clean supported runtime. Unit tests, JSON validation, code-cell
compilation, and `tools/validate_release_assets.py` are necessary checks but are **not** runtime
evidence under DIMER Notebook Specification 1.0. This file is the durable release-gate record for
the notebook.

## Automatic coverage (static, every pull request)

CI runs `tools/validate_release_assets.py`, which checks:

- notebook JSON parses; every code cell compiles as plain Python (no `%`/`!` magics); no
  persisted outputs or execution counts; no unresolved placeholder markers; every code cell
  is preceded by an explanatory markdown cell;
- exactly one tutorial notebook, named in `tutorials/README.md` with its `TASK-INFERENCE`
  profile and the notebook-spec version; `metadata.dimer` declares that profile and spec `1.0`;
- the fresh-runtime bootstrap (clone by canonical URL, `DIMER_TUTORIAL_REF`, detached checkout of
  the requested revision, restart-on-stale-import guard) and the recorded `REPO_SHA` in exports;
- `MODEL_ID`/`MODEL_REVISION` are imported from the package rather than hard-coded, the revision is
  a 40-hex immutable commit, and the same identity string appears in `README.md`,
  `MODEL_CARD.md`, and `docs/WEIGHTS.md` with no stray revisions;
- the profile-specific public-API calls (`stage_missing_files` and `stage_missing_encoder_files`,
  `verify_snapshot` and `verify_encoder_snapshot`, `GLiNERPipeline.from_pretrained` with the
  verified encoder directory, `detect` with the form threshold, `entity_f1` gated on gold), both
  identities printed and exported, the ceiling constants imported from the package, the offset
  sanity checks, the learner-facing statements (score not a calibrated probability, threshold
  caller-owned, two pinned snapshots, 384-word library cut, expected `fix_mistral_regex` warning,
  no metric without gold) and the gated-off BYOD default listed in the validator; forbidden
  patterns (credential-in-URL, direct `gliner`, `transformers` or `huggingface_hub` calls that
  bypass the pipeline, `trust_remote_code=True`, `pickle.load`, `torch.load(`, `extractall(`);
- `STATUS.md`, `README.md` and `tutorials/README.md` agree on one release-status token and no
  document makes an unsupported release-grade, production-readiness or benchmark claim;
- `MODEL_CARD.md` front matter (`model_card_spec: "1.1"`), single H1, required heading order, and
  immutable provenance.

These are source/provenance checks. They are **not** execution evidence.

## Executor paths

| Path | Runtime | Role |
|---|---|---|
| Google Colab (supported user path) | Colab CPU runtime (CUDA used automatically when present) | The runtime the tutorial is written for; a clean top-to-bottom run here is promotion evidence |
| Kaggle CLI kernel or equivalent fresh container | Fresh CPU or GPU container, Python 3.12 image; the committed notebook executed verbatim, cell by cell, in a fresh interpreter with a `google.colab` shim and `DIMER_TUTORIAL_REF` set to the candidate commit | Reproducible clean-room executor of the same class; needed whenever the hosted kernel pre-imports a NumPy or Pillow that differs from the `pyproject.toml` pins, because the tutorial's fail-closed stale-import guard correctly halts the in-kernel path after the pinned install |
| Local harness (pre-flight only) | Workstation, sequential cell executor with a `google.colab` shim, empty model cache | Builder pre-flight to catch defects before spending cloud runs; **not** a supported runtime and not promotion evidence |

## Supported release verification procedure

Before changing the registry status from `Candidate` to `Release-grade`:

1. resolve the exact PR/commit head under review and confirm static CI is green;
2. open that exact notebook revision in a new CPU or CUDA runtime (Colab, or a fresh-container
   executor above) with `DIMER_TUTORIAL_REF` set to the candidate commit, an empty Hugging Face
   cache, and no pre-staged weight files under `weights/gliner-multi-v2.1/ (the encoder assets under weights/mdeberta-v3-base-tokenizer/ are committed)`;
3. run the notebook top-to-bottom without editing implementation cells (form parameters at their
   defaults for the sample path: `USE_BYOD = False`, `LABELS = 'person, organization, location, date'`, `THRESHOLD = 0.5`, `GOLD_JSON = ''`);
4. verify that Section 1 reports `repository_revision` equal to the candidate commit and that the
   installed core package versions equal the `pyproject.toml` pins (`torch==2.14.0`, `gliner==0.2.29`, `transformers==4.57.6`, `huggingface-hub==0.36.2`, `safetensors==0.8.0`, `numpy==2.5.3`, `protobuf==6.31.1`, `sentencepiece==0.2.2`);
5. verify every default-path stage completes:
   - fresh bootstrap from GitHub at the candidate revision;
   - synthetic sentence and four-type label set authored in code with the text SHA-256 printed;
   - ceilings `MAX_TEXT_CHARS = 5000`, `MAX_LABELS = 25`, the 384-word library limit and `DEFAULT_THRESHOLD = 0.5` printed and the input accepted before model execution;
   - `stage_missing_files(..., allow_download=True)` reporting `['model.safetensors']` fetched from `urchade/gliner_multi-v2.1` at the immutable revision and `stage_missing_encoder_files(..., allow_download=True)` reporting `[]`; `verify_snapshot` (3 files) and `verify_encoder_snapshot` (4 files) both printed with their model ids and revisions; `GLiNERPipeline.from_pretrained` loading from the two verified directories with `load_warnings` containing only the expected `fix_mistral_regex` warning;
   - `detect` returning spans with in-text offsets, all four sanity checks true, and the "no metric is reported" line printed (the sample has no gold);
   - `outputs/gliner_ner_result.json` written with the indexed entities, both model identifiers and immutable revisions, both snapshot summaries, the load warnings, repository SHA, runtime versions and device;
6. verify the exports exist and the interpretation section matches the observed path;
7. record the notebook Git blob id, commit, runtime (platform, Python, PyTorch, gliner, Transformers, device), model
   identifier and immutable revision, whether the model cache and weights directory were clean,
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

The notebook source is complete and passes the static checks above; **no clean-runtime execution
has been recorded**, so the registry status is **Candidate** and the manual-evidence row is pending.
Promotion requires a reviewer to confirm a recorded run against the notebook blob under review and
an integrator to promote it; promotion is not performed by the builder. The commit that adds a
recorded-execution row changes documentation only; the executed source is the commit named in the
row.
