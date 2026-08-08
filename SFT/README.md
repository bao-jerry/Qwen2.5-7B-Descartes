# SFT

This folder contains the supervised fine-tuning stage for
`Qwen/Qwen2.5-7B-Instruct`.

Axolotl is pinned at `./axolotl`:

- release: `v0.18.0`
- commit: `2f5cb9da62a0fe763a1ddeb7798fc9acb2f4a417`

## Files

- `config.yml`: the runnable one-epoch LoRA training configuration
- `RUNPOD_SETUP.md`: the proven A100 RunPod setup and commands
- `AXOLOTL_CONFIG_REFERENCE.yaml`: exhaustive Axolotl configuration reference
- `generate_axolotl_config_reference.py`: regenerates that reference
- `axolotl/`: pinned framework source plus the custom token-normalized trainer

The standardized 20,000-conversation dataset is:

```text
../SFT_DATA_PROCESSING/STANDARDIZED_TRAINING_DATA.jsonl
```

## Training

Run from this directory so the relative dataset path resolves:

```bash
axolotl preprocess config.yml --debug --debug-num-examples 5
axolotl train config.yml
```

The successful preflight measured conversation lengths from 44 to 6,872 Qwen
tokens, all within the configured 8,192-token sequence length.

## Checkpoint evaluation

The documented checkpoint-selection protocol, evaluation program, transcripts,
and results are stored in
[`../SFT_CHECKPOINT_EVALUATION/`](../SFT_CHECKPOINT_EVALUATION/).

GRPO remains in the separate `../GRPO` project area.
