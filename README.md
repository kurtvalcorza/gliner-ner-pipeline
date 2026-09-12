# GLiNER NER Pipeline

DIMER-oriented inference wrapper for **GLiNER multi-v2.1**, pinned to an immutable Hugging Face revision, plus a second pinned snapshot of the mDeBERTa-v3 encoder's tokenizer and configuration. The repository exposes zero-shot named-entity recognition with a caller-supplied label list, an exposed score threshold, supply-chain checks of both local snapshots, an exact-span micro-F1 helper, and machine-readable provenance.

## Upstream alignment

- Model: `urchade/gliner_multi-v2.1`
- Revision: `443d26d654e0324125a96bebd8e796c14ff2efe6`
- Encoder tokenizer/config: `microsoft/mdeberta-v3-base` at `a0484667b22365f84929a935b5e50a51f71f159d` (MIT; no weights)
- Upstream weight license: Apache-2.0
- Upstream task: token classification / zero-shot NER (span extraction against arbitrary label names)
- Repository adaptation: **none**; inference only

## Quick start

```python
from gliner_ner_pipeline import GLiNERPipeline, entity_f1

pipe = GLiNERPipeline.from_pretrained()          # verifies both snapshots under weights/ first
result = pipe.detect("Marie Curie was born in Warsaw and later joined the University of Paris.",
                     ["person", "location", "organisation"])
for e in result["entities"]:
    print(e["text"], "=>", e["label"], round(e["score"], 3))
```

`detect()` takes one text of at most 5,000 characters (`MAX_TEXT_CHARS`; chunk longer text, the library also cuts at 384 words) and 1..25 unique label names (`MAX_LABELS`). `threshold` (default `DEFAULT_THRESHOLD = 0.5`, upstream's default) is the score cutoff the caller owns. Every result carries `entities` (`text`, `label`, `start`, `end`, `score`), `n_entities`, `labels`, `threshold`, `model_id`, `model_revision`, `encoder_model_id` and `encoder_revision`. `entity_f1(predicted, gold)` returns exact-span micro precision/recall/F1 against human annotations.

## Weights layout

```
weights/gliner-multi-v2.1/             gliner_config.json  model.safetensors  README.md  dimer-base-manifest.json
weights/mdeberta-v3-base-tokenizer/    config.json  tokenizer_config.json  spm.model  README.md  dimer-base-manifest.json
```

`from_pretrained()` calls `stage_missing_files()` then `verify_snapshot()` for the GLiNER snapshot and `stage_missing_encoder_files()` then `verify_encoder_snapshot()` for the encoder snapshot (size + SHA-256 of every manifest entry; staging fetches only absent entries at the pinned revision and only with `allow_download=True`). It then loads through the `gliner` library with `model_name` redirected to the verified encoder directory and `local_files_only=True`; no HF cache or Hub access is involved. See `docs/WEIGHTS.md`.

## Runtime note

Building the fast DeBERTa-v2 tokenizer from `spm.model` requires `protobuf` and `sentencepiece`; both are pinned in `pyproject.toml`. Transformers 4.57.6 logs a generic "incorrect regex pattern / `fix_mistral_regex`" warning during that conversion; it references a Mistral tokenizer issue, does not apply to this SentencePiece model, and can be ignored (see `MODEL_CARD.md`, Runtime).

## Tests

```
pip install -e . --no-deps
pytest -q -o addopts= tests
```

Tests are offline: they use an injected fake runner and temporary manifests (both kinds), never the weights.

## Tutorial

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kurtvalcorza/gliner-ner-pipeline/blob/main/tutorials/gliner_ner_colab.ipynb)

`tutorials/gliner_ner_colab.ipynb` is declared `TASK-INFERENCE` (see `tutorials/README.md`). Its default path authors a synthetic sentence and a four-type label set, surfaces `MAX_TEXT_CHARS`/`MAX_LABELS` and the threshold semantics, stages the missing GLiNER weight file with `stage_missing_files(..., allow_download=True)` and the encoder assets with `stage_missing_encoder_files(..., allow_download=True)`, digest-verifies both with `verify_snapshot`/`verify_encoder_snapshot`, detects entities through the public API, and exports the spans with offsets and scores plus provenance JSON carrying both identities. No metric is reported unless the learner pastes gold annotations, in which case `entity_f1` is applied. BYOD is optional and gated off by default.

## Release status

**Candidate.** Static/unit checks do not constitute clean-runtime notebook evidence. Complete `docs/release-verification.md` against the exact release revision before calling the notebook release-grade.

## Licensing

This repository's code is Apache-2.0 (`LICENSE`). The packaged GLiNER weights are Apache-2.0; the mDeBERTa encoder tokenizer/config snapshot is MIT; see `docs/WEIGHTS.md` and `MODEL_CARD.md`.
