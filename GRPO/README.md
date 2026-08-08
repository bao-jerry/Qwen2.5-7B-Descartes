# GRPO

This directory contains the configured GRPO stage for the Descartes project.
It continues training the rank-32 LoRA adapter from SFT checkpoint 164 on
`Qwen/Qwen2.5-7B-Instruct`.

The working setup uses two A100 SXM 80 GB GPUs:

- GPU 0 runs the vLLM rollout server;
- GPU 1 runs Axolotl/TRL training;
- each prompt receives eight rollouts;
- DeepSeek V4 Flash scores each group through `reward_judge_deepseek.py`;
- one rollout batch is prefetched asynchronously;
- checkpoints are saved every 40 steps and pushed to the private Hugging Face
  repository `baojerry/Qwen2.5-7B-Descartes-GRPO`;
- W&B records the training metrics.

Important files:

- `config.yaml`: active Axolotl training configuration.
- `RUNPOD_STARTUP.md`: exact environment, authentication, preflight, launch,
  and monitoring procedure.
- `reward_judge_deepseek.py`: Axolotl-compatible concurrent reward function.
- `../GRPO_DATA_PROCESSING/TRAINING_DATA.jsonl`: 10,973 training prompts.
- `../SFT/axolotl/src/axolotl/core/trainers/grpo/async_trainer.py`: the vendored
  Axolotl async trainer with the narrow TRL 1.8 compatibility fixes required by
  this run.
