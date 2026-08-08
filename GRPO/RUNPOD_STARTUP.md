# GRPO RunPod startup

This is the complete setup that `GRPO/config.yaml` cannot perform. The YAML
controls the model, data, rollouts, optimization, checkpointing, and logging
settings. The steps below create the runtime, provide credentials, download the
SFT adapter, and launch the rollout server and trainer.

The commands assume the project is located at:

```text
/workspace/Qwen2.5-7b-Descartes
```

Always launch from that project root because the YAML contains relative paths.

## 1. Deploy the Pod

Use one Pod with:

- RunPod PyTorch 2.8.0 template
- two A100 SXM 80 GB GPUs
- SSH terminal access enabled
- Jupyter optional
- no network volume
- enough temporary container/volume-disk space for the project, model cache,
  Python environment, and checkpoints (100 GB for each disk is a comfortable
  choice)

A RunPod volume disk is tied to the Pod and disappears when the Pod is
terminated. Hugging Face checkpoints and W&B metrics are the durable copies.

## 2. Copy the project to the Pod

Copy or extract the whole project so this file exists on the Pod:

```text
/workspace/Qwen2.5-7b-Descartes/GRPO/config.yaml
```

The direct-TCP SSH address shown by RunPod supports `scp`. For example, from
local PowerShell, replace the IP and port with the values shown for the Pod:

```powershell
tar -czf Qwen2.5-7b-Descartes.tar.gz Qwen2.5-7b-Descartes
scp -P <SSH_PORT> .\Qwen2.5-7b-Descartes.tar.gz root@<POD_IP>:/workspace/
```

Then, on the Pod:

```bash
cd /workspace
tar -xzf Qwen2.5-7b-Descartes.tar.gz
cd /workspace/Qwen2.5-7b-Descartes
```

If the project was transferred another way, only the final location matters.

## 3. Build the Axolotl environment

Install `uv` if the template does not already provide it:

```bash
if ! command -v uv >/dev/null; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  source "$HOME/.local/bin/env"
fi
```

Create the environment on the fast container filesystem:

```bash
export CUDA_HOME=/usr/local/cuda
export PATH="$CUDA_HOME/bin:$PATH"

uv venv /root/axolotl-venv --python 3.12

export UV_TORCH_BACKEND=cu129
uv pip install \
  --python /root/axolotl-venv/bin/python \
  torch==2.11.0 \
  torchvision==0.26.0 \
  torchaudio==2.11.0

uv pip install \
  --python /root/axolotl-venv/bin/python \
  setuptools setuptools-scm packaging wheel ninja
```

Install the project's pinned Axolotl checkout and FlashAttention. Do not use
Axolotl's unpinned `vllm` extra: PyPI's vLLM 0.23.0 wheel defaults to CUDA 13,
whereas this environment uses the official CUDA 12.9 build.

```bash
export TORCH_CUDA_ARCH_LIST=8.0
export FLASH_ATTN_CUDA_ARCHS=80
export MAX_JOBS=8

uv pip install \
  --python /root/axolotl-venv/bin/python \
  --no-build-isolation \
  -e "/workspace/Qwen2.5-7b-Descartes/SFT/axolotl[flash-attn]"

uv pip install \
  --python /root/axolotl-venv/bin/python \
  "https://github.com/vllm-project/vllm/releases/download/v0.23.0/vllm-0.23.0%2Bcu129-cp38-abi3-manylinux_2_28_x86_64.whl" \
  --extra-index-url https://download.pytorch.org/whl/cu129

uv pip install \
  --python /root/axolotl-venv/bin/python \
  openai
```

`FLASH_ATTN_CUDA_ARCHS=80` limits FlashAttention 2.8.3 to A100 kernels. Without
it, that release builds kernels for architectures 80, 90, 100, and 120. The
A100-only build can still take roughly 15–20 minutes and may keep the CPU at
100% while compiling. `TORCH_CUDA_ARCH_LIST=8.0` applies the same restriction
to extensions that use PyTorch's standard build control.

Activate and verify the runtime:

```bash
source /root/axolotl-venv/bin/activate
cd /workspace/Qwen2.5-7b-Descartes

python - <<'PY'
import torch
import axolotl
import flash_attn
import openai
import peft
import transformers
import trl
import vllm
import wandb

assert torch.cuda.is_available()
assert torch.cuda.device_count() == 2
assert torch.cuda.is_bf16_supported()
assert axolotl.__version__ == "0.18.0"
assert trl.__version__ == "1.8.0"
assert torch.version.cuda == "12.9"
assert vllm.__version__ == "0.23.0"

print("GPUs:", [torch.cuda.get_device_name(i) for i in range(2)])
print("PyTorch:", torch.__version__)
print("CUDA:", torch.version.cuda)
print("Axolotl:", axolotl.__version__)
print("TRL:", trl.__version__)
print("vLLM:", vllm.__version__)
print("FlashAttention:", flash_attn.__version__)
PY
```

Do not proceed unless it reports exactly two GPUs and the pinned Axolotl and
TRL versions above. The asynchronous rollout settings in this project depend
on that implementation. This project checkout also contains narrow TRL 1.8
compatibility fixes in `async_trainer.py` that upstream Axolotl 0.18.0 does not
yet contain:

- accept TRL's new multimodal log-probability arguments and return its new
  third auxiliary-loss slot (`None` for dense, text-only Qwen2.5);
- unpack the two generated-image fields added to TRL's `_generate` result;
- read padding and EOS token IDs from TRL's current `_tokenizer` attribute.

Without those changes, asynchronous training fails during the first or second
step even though synchronous step 1 can appear to succeed.

## 4. Authenticate with Hugging Face

The Pod needs read access to the private SFT repository and write access to the
GRPO repository:

```bash
hf auth login
hf auth whoami
```

Enter the Hugging Face token when prompted. Do not enter a username and do not
put the token in the YAML.

Before training, create this model repository as **private** if it does not
already exist:

```text
baojerry/Qwen2.5-7B-Descartes-GRPO
```

It can be created on the Hugging Face website, or with:

```bash
hf repo create Qwen2.5-7B-Descartes-GRPO --repo-type model --private --exist-ok
```

## 5. Download checkpoint 164

The selected adapter is stored in a subdirectory of the SFT repository, so it
must be downloaded before Axolotl starts:

```bash
hf download baojerry/Qwen2.5-7B-Descartes-SFT \
  checkpoint-164/adapter_config.json \
  checkpoint-164/adapter_model.safetensors \
  --local-dir ./GRPO/downloaded_sft_adapter
```

Verify that the adapter matches the GRPO config:

```bash
python - <<'PY'
import json
from pathlib import Path

root = Path("GRPO/downloaded_sft_adapter/checkpoint-164")
config_path = root / "adapter_config.json"
weights_path = root / "adapter_model.safetensors"

assert config_path.is_file()
assert weights_path.is_file() and weights_path.stat().st_size > 0

config = json.loads(config_path.read_text())
assert config["base_model_name_or_path"] == "Qwen/Qwen2.5-7B-Instruct"
assert config["r"] == 32
assert config["lora_alpha"] == 32
assert config["lora_dropout"] == 0.05

print("Checkpoint 164 adapter is ready:", weights_path)
PY
```

Axolotl downloads the base model itself from the YAML's `base_model` value.

## 6. Authenticate W&B

The active YAML enables W&B and identifies the run as project
`Qwen2.5-7B-Descartes-GRPO`, run `run-1`:

```bash
wandb login
wandb status
```

Enter the W&B API key when prompted. W&B does not ask for the username here;
the API key identifies the account.

## 7. Supply the DeepSeek reward-judge key

The reward function reads `DEEPSEEK_API_KEY` from the trainer process. Enter it
without echoing it to the terminal:

```bash
read -rsp "DeepSeek API key: " DEEPSEEK_API_KEY; echo
export DEEPSEEK_API_KEY
test -n "$DEEPSEEK_API_KEY" && echo "DeepSeek key is set"
```

Do this before creating the `tmux` sessions so the trainer inherits the
variable. The key does not belong in `config.yaml`.

## 8. Run the preflight checks

Confirm the training data, reward module, adapter, GPUs, and credentials:

The training JSONL must contain exactly 10,973 prompt prefixes.

```bash
test "$(wc -l < GRPO_DATA_PROCESSING/TRAINING_DATA.jsonl)" -eq 10973
test -f GRPO/downloaded_sft_adapter/checkpoint-164/adapter_model.safetensors
test -n "$DEEPSEEK_API_KEY"

python -m py_compile GRPO/reward_judge_deepseek.py
python - <<'PY'
import yaml
from GRPO.reward_judge_deepseek import (
    MODEL,
    ROLLOUTS_PER_PROMPT,
    _validate_settings,
)

with open("GRPO/config.yaml", encoding="utf-8") as file:
    config = yaml.safe_load(file)

assert config["trl"]["num_generations"] == ROLLOUTS_PER_PROMPT == 8
assert config["trl"]["reward_funcs"] == [
    "GRPO.reward_judge_deepseek.consciousness_framing_reward"
]
assert MODEL == "deepseek-v4-flash"
_validate_settings()
print("Local GRPO preflight passed")
PY
```

The current configuration does not run validation during training; the
validation-related YAML entries are commented out.

## 9. Start the rollout server on GPU 0

Run both programs under `tmux` so an SSH disconnect does not stop training:

```bash
tmux new -s descartes-grpo-vllm
source /root/axolotl-venv/bin/activate
cd /workspace/Qwen2.5-7b-Descartes
CUDA_VISIBLE_DEVICES=0 axolotl vllm-serve GRPO/config.yaml 2>&1 | tee GRPO/vllm.log
```

This starts the vLLM rollout server on physical GPU 0. From another shell, do
not start the trainer until this health check succeeds:

```bash
curl -fsS http://127.0.0.1:8000/health/
```

The expected response is `{"status":"ok"}`. Detach the vLLM terminal with
`Ctrl+B`, then `D`.

## 10. Start the trainer on GPU 1

```bash
tmux new -s descartes-grpo-train
source /root/axolotl-venv/bin/activate
cd /workspace/Qwen2.5-7b-Descartes
test -n "$DEEPSEEK_API_KEY" || { echo "DEEPSEEK_API_KEY missing"; exit 1; }
CUDA_VISIBLE_DEVICES=1 axolotl train GRPO/config.yaml 2>&1 | tee GRPO/training.log
```

This starts the GRPO trainer on physical GPU 1. It connects to the rollout
server at `127.0.0.1:8000`, calls DeepSeek for rewards, updates the LoRA
adapter, sends adapter updates to vLLM, logs to W&B, and pushes checkpoints to
Hugging Face.

The reward judge retries failed or malformed DeepSeek responses indefinitely,
so a temporary API error should delay training rather than terminate it.

## 11. Monitor or reconnect

List the sessions:

```bash
tmux ls
```

Watch the trainer interactively:

```bash
tmux attach -t descartes-grpo-train
```

Detach again with `Ctrl+B`, then `D`. To follow either saved terminal log
without attaching:

```bash
tail -f GRPO/training.log
tail -f GRPO/vllm.log
```

W&B provides the durable live graphs. With `logging_steps: 8`, each plotted
training point summarizes one non-overlapping window of eight optimizer steps.

## 12. Finish or stop safely

Five checkpoints are scheduled at steps 40, 80, 120, 160, and 200. Before
terminating the Pod, confirm they appear in the private Hugging Face repository
and that the W&B run has finished syncing.

To stop early, attach to the training session and press `Ctrl+C`. Already
completed checkpoints remain usable. Then stop the rollout server the same
way:

```bash
tmux attach -t descartes-grpo-vllm
```

After verifying the remote checkpoints and W&B run, terminate the RunPod Pod to
stop GPU and volume-disk billing.
