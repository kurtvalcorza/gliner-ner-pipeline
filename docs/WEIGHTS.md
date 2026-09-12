# Weight provenance and DIMER hosting

This repository pins **two** snapshots, each with its own `dimer-base-manifest.json`, because the GLiNER checkpoint does not ship the tokenizer and configuration of the encoder it was trained on.

## GLiNER weights

- Upstream: `urchade/gliner_multi-v2.1`
- Immutable revision: `443d26d654e0324125a96bebd8e796c14ff2efe6`
- Weight format: SafeTensors (`model.safetensors`, 1,155,830,112 bytes; carries the encoder weights too)
- Upstream weight license: Apache-2.0 (`license: apache-2.0` in the pinned upstream README)
- Local layout: `weights/gliner-multi-v2.1/` holds the 3 manifest entries (`gliner_config.json`, `model.safetensors`, upstream `README.md`) with byte size and SHA-256 for each. `verify_snapshot()` in `src/gliner_ner_pipeline/pipeline.py` checks all of them before any load; `stage_missing_files(allow_download=True)` fetches only absent entries at the pinned revision. `.safetensors` files are git-ignored; the Git repository does not vendor the checkpoint.

## Encoder tokenizer and config (no weights)

- Upstream: `microsoft/mdeberta-v3-base` (the `model_name` in `gliner_config.json`; upstream GLiNER pins no revision for it)
- Immutable revision: `a0484667b22365f84929a935b5e50a51f71f159d`
- Upstream license: MIT (`license: mit` in that repository's README front matter)
- Local layout: `weights/mdeberta-v3-base-tokenizer/` holds the 4 manifest entries (`config.json`, `tokenizer_config.json`, `spm.model`, upstream `README.md`; 4,309,323 bytes total) with byte size and SHA-256 for each; the manifest carries an informational `role` field. `verify_encoder_snapshot()` checks all of them before any load; `stage_missing_encoder_files(allow_download=True)` fetches only absent entries at the pinned revision.
- How it is used: the loader subclasses the `gliner` library's concrete model class and rewrites `config.model_name` to this directory after checking it names `microsoft/mdeberta-v3-base`, so the library's `AutoTokenizer` and `AutoConfig` lookups read these files with `local_files_only=True`; no HF cache and no Hub access is involved on the snapshot path.

## DIMER hosting

- Apache-2.0 (GLiNER) and MIT (encoder assets) both permit use, modification, redistribution and commercial use subject to preservation of the license and notices. DIMER may mirror both pinned snapshots in its model store under those terms.
- Loader trust boundary: `gliner` 0.2.29 `from_pretrained`, which builds the native Transformers `DebertaV2Model` for this checkpoint; the pipeline passes no `trust_remote_code` (the library exposes no such parameter) and the library's own `trust_remote_code=True` branches in `modeling/encoder.py` are not taken for a DeBERTa-v2 encoder.
- Runtime dependency (2026-09-12): building the fast DeBERTa-v2 tokenizer from `spm.model` requires `protobuf` and `sentencepiece`, both pinned in `pyproject.toml`; with them present the load and a CPU smoke executed offline (see the card's Runtime section).
