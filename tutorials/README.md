# Tutorials

[![GitHub](https://img.shields.io/badge/GitHub-181717?style=flat&logo=github&logoColor=white)](https://github.com/kurtvalcorza/gliner-ner-pipeline)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kurtvalcorza/gliner-ner-pipeline/blob/main/tutorials/gliner_ner_colab.ipynb)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-urchade%2Fgliner__multi--v2.1-ffcc4d?style=flat)](https://huggingface.co/urchade/gliner_multi-v2.1)
[![Upstream](https://img.shields.io/badge/Upstream-urchade%2FGLiNER-181717?style=flat&logo=github&logoColor=white)](https://github.com/urchade/GLiNER)
[![arXiv](https://img.shields.io/badge/arXiv-2311.08526-b31b1b.svg)](https://arxiv.org/abs/2311.08526)

Notebook specification: **DIMER Notebook Specification 1.0**

| Notebook | Profile | Capability | Default runtime | BYOD | Release status |
|---|---|---|---|---|---|
| `gliner_ner_colab.ipynb` | `TASK-INFERENCE` | GLiNER multi-v2.1 zero-shot NER on a synthetic sentence with a four-type label set; per-span scores (not calibrated) against a caller-owned threshold (default 0.5); `entity_f1` scored only when the learner pastes gold annotations; both pinned snapshots (GLiNER weights + mDeBERTa encoder assets) staged and digest-verified | CPU float32 (CUDA used automatically when available) | one UTF-8 text file plus a label list form field, gated off by default | **Candidate** — static checks pass; the clean-runtime execution row in `../docs/release-verification.md` is pending and must be recorded for the exact notebook revision before promotion |

## Conformance notes

- The notebook exercises `GLiNERPipeline` from the repository public API rather than reimplementing model loading; it calls both `stage_missing_files(..., allow_download=True)` and `stage_missing_encoder_files(..., allow_download=True)`, prints both `verify_snapshot`/`verify_encoder_snapshot` summaries and both immutable identities (`MODEL_ID`@`MODEL_REVISION`, `ENCODER_MODEL_ID`@`ENCODER_REVISION`), and loads with the encoder redirected to the verified directory. The notebook never imports `gliner`, `transformers` or `huggingface_hub` for model work.
- Score and threshold semantics (UNC1/UNC2/UNC3/UNC4): the span `score` is a per-span confidence in [0, 1], not a calibrated probability; the only decision rule is `score >= threshold` with the upstream default 0.5 exposed as a form field and owned by the caller.
- Metric (EVAL1/EVAL7): the repository's `entity_f1` (exact-span micro P/R/F1) is applied only when the learner pastes gold annotations into `GOLD_JSON`; the synthetic sentence ships none, so no metric is reported on the default path. Recorded `SHOULD` deviation: EVAL11 (the trivial no-entity baseline has F1 = 0 by construction and is not reported).
- Ceilings `MAX_TEXT_CHARS`, `MAX_LABELS`, label uniqueness/length and the threshold range are surfaced before the model runs; the library's silent 384-word cut is stated and the word count printed (DAT22/DAT23).
- The expected transformers `fix_mistral_regex` warning is documented in the prerequisites and printed from `pipe.load_warnings` so the learner can recognise it.
- The default sample is synthetic text with invented names; `USE_BYOD` defaults to `False` so the sample path never opens an upload dialog.
- `tools/validate_release_assets.py` performs source validation only. It does not satisfy the
  clean-runtime execution requirement; a release review must confirm that a recorded clean run in
  `docs/release-verification.md` matches the notebook revision under review before the status is
  promoted to `Release-grade`.
