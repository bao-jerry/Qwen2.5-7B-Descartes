# Reproducing the MMLU evaluation on RunPod

These instructions reproduce the controlled comparison between the original
Qwen2.5-7B-Instruct model and Descartes GRPO checkpoint 40.

## 1. Deploy the Pod

- Template: **RunPod PyTorch 2.8.0**
- GPU: **1 x NVIDIA A40 (48 GB VRAM)**
- Start Jupyter: optional
- SSH access: enabled
- Container disk: **50 GB**
- Volume disk: **50 GB**

The recorded run used Python 3.12, PyTorch 2.8.0, CUDA 12.8, Ubuntu 24.04,
and one A40. The disk is tied to the Pod lifecycle, so copy the results off the
Pod before deleting it.

## 2. Install the evaluation harness

Connect through SSH and run:

```bash
python -m pip install "lm_eval[hf]==0.4.10" "transformers==5.14.1"
mkdir -p /root/models/descartes-grpo
mkdir -p /root/mmlu-results/base /root/mmlu-results/descartes
```

The PyTorch template already supplies the GPU-enabled PyTorch installation.

## 3. Make the Descartes adapter available

If the Hugging Face repository requires authentication, set `HF_TOKEN` in the
Pod without writing its value into this repository:

```bash
export HF_TOKEN="YOUR_HUGGING_FACE_TOKEN"
```

Download checkpoint 40 while preserving its directory name:

```bash
hf download baojerry/Qwen2.5-7B-Descartes-GRPO \
  --include "checkpoint-40/*" \
  --local-dir /root/models/descartes-grpo
```

The adapter should now be located at:

```text
/root/models/descartes-grpo/checkpoint-40
```

The base model is downloaded automatically by the evaluation harness.

## 4. Evaluate the original instruct model

```bash
set -o pipefail

lm_eval \
  --model hf \
  --model_args pretrained=Qwen/Qwen2.5-7B-Instruct,revision=a09a35458c702b33eeacc393d103063234e8bc28,dtype=bfloat16 \
  --tasks mmlu \
  --num_fewshot 5 \
  --batch_size auto \
  --max_batch_size 64 \
  --device cuda:0 \
  --apply_chat_template \
  --seed 0,1234,1234,1234 \
  --log_samples \
  --output_path /root/mmlu-results/base \
  2>&1 | tee /root/mmlu-results/base.log
```

## 5. Evaluate Descartes checkpoint 40

```bash
lm_eval \
  --model hf \
  --model_args pretrained=Qwen/Qwen2.5-7B-Instruct,revision=a09a35458c702b33eeacc393d103063234e8bc28,peft=/root/models/descartes-grpo/checkpoint-40,dtype=bfloat16 \
  --tasks mmlu \
  --num_fewshot 5 \
  --batch_size auto \
  --max_batch_size 64 \
  --device cuda:0 \
  --apply_chat_template \
  --seed 0,1234,1234,1234 \
  --log_samples \
  --output_path /root/mmlu-results/descartes \
  2>&1 | tee /root/mmlu-results/descartes.log
```

These commands differ only in the Descartes run's `peft` adapter argument and
the output path. Both runs use the same model revision, benchmark, five-shot
examples, chat template, precision, batch-size policy, device, and seeds.

## 6. Preserve the results

Copy both result directories and logs before terminating the Pod:

```text
/root/mmlu-results/base/
/root/mmlu-results/descartes/
/root/mmlu-results/base.log
/root/mmlu-results/descartes.log
```

The aggregate MMLU score appears under `results.mmlu["acc,none"]` in each
generated `results_*.json` file.
