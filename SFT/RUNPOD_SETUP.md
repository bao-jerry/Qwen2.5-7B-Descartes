# Proven RunPod SFT setup

This is the setup used successfully for this project on July 27, 2026. It uses
a temporary Pod and uploads checkpoints to a private Hugging Face repository.

## 1. Deploy the Pod

Use:

- RunPod template: `ax-dev`
- Image: `axolotlai/axolotl-cloud:main-latest`
- GPU: one A100 SXM 80 GB
- Container disk: 100 GB
- Persistent storage type: Volume disk, 100 GB
- No network volume

The Volume disk is tied to the Pod and is deleted when the Pod is terminated.
In this setup it is mounted at `/workspace/data`.

## 2. Transfer the project

Create one `.tar.gz` archive of the project. Use `runpodctl` to transfer it
because RunPod's proxy SSH connection does not support SCP or SFTP.

Extract it on the Pod so the project is located at:

```text
/workspace/data/Qwen2.5-7b-Descartes
```

Confirm the dataset contains 20,000 records:

```bash
wc -l "/workspace/data/Qwen2.5-7b-Descartes/SFT_DATA_PROCESSING/STANDARDIZED_TRAINING_DATA.jsonl"
```

## 3. Create the runtime environment

Keep the Python environment on the fast container filesystem. Do not put it
under `/workspace/data`, whose filesystem is much slower for thousands of
small package files.

```bash
uv venv /root/axolotl-venv --python 3.12

export UV_TORCH_BACKEND=cu130
uv pip install \
  --python /root/axolotl-venv/bin/python \
  torch==2.11.0 \
  torchvision==0.26.0

uv pip install \
  --python /root/axolotl-venv/bin/python \
  setuptools \
  setuptools-scm \
  packaging \
  wheel \
  ninja
```

The `cu130` backend matches this image's CUDA 13.0 compiler. On a different
image, check `nvcc --version` and select a matching PyTorch CUDA build.

Install this project's pinned Axolotl checkout and FlashAttention:

```bash
export TORCH_CUDA_ARCH_LIST=8.0
export MAX_JOBS=8

uv pip install \
  --python /root/axolotl-venv/bin/python \
  --no-build-isolation \
  -e "/workspace/data/Qwen2.5-7b-Descartes/SFT/axolotl[flash-attn]"
```

`TORCH_CUDA_ARCH_LIST=8.0` is important: it builds only A100 kernels. Omitting
it caused FlashAttention to compile kernels for several unrelated GPU
architectures and greatly increased setup time. The A100-only build took about
17 minutes.

## 4. Validate the environment

```bash
source /root/axolotl-venv/bin/activate

python - <<'PY'
import torch
import flash_attn
import bitsandbytes
import axolotl
from axolotl.core.trainers.token_normalized import TokenNormalizedTrainer

assert torch.cuda.is_available()
assert torch.cuda.is_bf16_supported()

print("GPU:", torch.cuda.get_device_name(0))
print("PyTorch:", torch.__version__)
print("CUDA:", torch.version.cuda)
print("FlashAttention:", flash_attn.__version__)
print("bitsandbytes:", bitsandbytes.__version__)
print("Axolotl:", axolotl.__version__)
print("Trainer:", TokenNormalizedTrainer.__module__)
PY
```

The proven versions were:

```text
PyTorch 2.11.0+cu130
FlashAttention 2.8.3
bitsandbytes 0.49.1
Axolotl 0.18.0
```

## 5. Authenticate with Hugging Face

Create a fine-grained token with read access to the base model and write access
to repositories, then authenticate without putting the token in the YAML:

```bash
hf auth login
hf auth whoami
```

The YAML's `hub_model_id` causes Axolotl to enable pushing automatically and
auto-create the repository as private:

```text
baojerry/Qwen2.5-7B-Descartes-SFT
```

This project's `TokenNormalizedTrainer` filters local dataset paths out of
Hugging Face model-card metadata. This is necessary because Axolotl otherwise
adds the local JSONL path as a `datasets` entry, which the Hub rejects after
training even though the model files upload successfully.

## 6. Preprocess and inspect

```bash
source /root/axolotl-venv/bin/activate
cd "/workspace/data/Qwen2.5-7b-Descartes/SFT"
axolotl preprocess config.yml --debug --debug-num-examples 5
```

Despite its name, `--debug-num-examples 5` limits the examples printed for
inspection; Axolotl still preprocesses the complete dataset. Do not run a
second preprocessing command afterward.

The successful preflight produced:

```text
20,000 valid conversations
minimum length: 44 tokens
maximum length: 6,872 tokens
sequence limit: 8,192 tokens
zero overlength samples
zero samples without trainable tokens
```

In the debug output, user and header tokens should have label `-100`.
Assistant content and the assistant `<|im_end|>` should carry their token IDs.

## 7. Start training

Run training inside `tmux` so an SSH disconnect does not stop it:

```bash
tmux new -s descartes-sft
source /root/axolotl-venv/bin/activate
cd "/workspace/data/Qwen2.5-7b-Descartes/SFT"
axolotl train config.yml
```

Detach with `Ctrl+B`, then `D`. Reconnect with:

```bash
tmux attach -t descartes-sft
```

The proven run reported:

```text
201 optimizer steps
99.84% sample-packing efficiency
micro batch size: 2
gradient accumulation steps: 4
about 23 seconds per optimizer step
about 77 GiB of 80 GiB VRAM used
```

If a different run OOMs before completing its first optimizer step, change:

```yaml
micro_batch_size: 1
gradient_accumulation_steps: 8
```

This preserves the same effective batch size.

## 8. Finish safely

The run saves five checkpoints plus the final adapter and pushes them to:

```text
https://huggingface.co/baojerry/Qwen2.5-7B-Descartes-SFT
```

Before terminating the Pod, confirm the Hugging Face repository contains the
five checkpoint directories, the final adapter, and `README.md`. After
confirmation, terminate the Pod so its temporary storage and GPU billing stop.
