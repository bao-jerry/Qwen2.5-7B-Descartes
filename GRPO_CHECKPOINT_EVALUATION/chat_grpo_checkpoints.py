"""Download and interactively evaluate the two GRPO LoRA checkpoints.

The script also makes SFT checkpoint 164 available as the pre-GRPO control.
Designed for a fresh RunPod PyTorch pod. Before running it, set ``HF_TOKEN`` or
use ``hf auth login`` with a token that can read both private repositories, then
install:

    pip install transformers peft accelerate huggingface_hub

Run:

    python chat_grpo_checkpoints.py

Qwen is downloaded and loaded once. The script downloads only the LoRA files
for SFT checkpoint 164 and GRPO checkpoints 40 and 80, reusing existing files.
All three adapters are loaded before the interactive chat begins.
"""

from __future__ import annotations

import importlib.util
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

import torch
from huggingface_hub import get_token, snapshot_download
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
SFT_REPOSITORY = "baojerry/Qwen2.5-7B-Descartes-SFT"
GRPO_REPOSITORY = "baojerry/Qwen2.5-7B-Descartes-GRPO"
CHECKPOINT_ROOT = Path(__file__).resolve().parent / "downloaded_checkpoints"
CHAT_HISTORY_ROOT = Path(__file__).resolve().parent / "chat_histories"

# None uses Qwen's default system prompt through its official chat template.
SYSTEM_PROMPT: str | None = None

# Kept identical to the SFT checkpoint evaluation for a direct comparison.
MAX_NEW_TOKENS = 512
TEMPERATURE = 0.7
TOP_P = 0.9


@dataclass(frozen=True)
class Checkpoint:
    """Describe one adapter source and its unambiguous evaluation name."""

    name: str
    repository: str
    remote_directory: str
    local_group: str

    @property
    def path(self) -> Path:
        return CHECKPOINT_ROOT / self.local_group / self.remote_directory

    @property
    def step(self) -> int:
        match = re.fullmatch(r"checkpoint-(\d+)", self.remote_directory)
        if match is None:
            raise ValueError(
                f"Invalid checkpoint directory: {self.remote_directory}"
            )
        return int(match.group(1))


CHECKPOINTS = (
    Checkpoint(
        name="checkpoint-164",
        repository=SFT_REPOSITORY,
        remote_directory="checkpoint-164",
        local_group="sft",
    ),
    Checkpoint(
        name="checkpoint-40",
        repository=GRPO_REPOSITORY,
        remote_directory="checkpoint-40",
        local_group="grpo",
    ),
    Checkpoint(
        name="checkpoint-80",
        repository=GRPO_REPOSITORY,
        remote_directory="checkpoint-80",
        local_group="grpo",
    ),
)

# RunPod's proxy terminal can write terminal-identification replies into stdin
# after an SSH reconnect. Strip those control sequences before treating a line
# as a chat message or command.
TERMINAL_ESCAPE_RE = re.compile(
    r"\x1b(?:"
    r"P.*?\x1b\\"  # DCS, including xterm's terminal-identification response
    r"|\][^\x07]*(?:\x07|\x1b\\)"  # OSC
    r"|\[[0-?]*[ -/]*[@-~]"  # CSI
    r"|[@-_]"  # two-character escape sequence
    r")",
    re.DOTALL,
)
CONTROL_CHARACTER_RE = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")


def clean_terminal_input(text: str) -> str:
    """Remove terminal control traffic accidentally delivered through stdin."""
    text = TERMINAL_ESCAPE_RE.sub("", text)
    return CONTROL_CHARACTER_RE.sub("", text).strip()


def clear_terminal_scrollback() -> None:
    """Erase the visible terminal and tmux history between evaluation agents."""
    tmux_pane = os.environ.get("TMUX_PANE")
    if tmux_pane:
        subprocess.run(
            ["tmux", "clear-history", "-t", tmux_pane],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    print("\033[2J\033[3J\033[H", end="", flush=True)


def download_checkpoints() -> list[Checkpoint]:
    """Download missing LoRA files from the two Hugging Face repositories."""
    token = os.environ.get("HF_TOKEN") or get_token()
    if not token:
        raise SystemExit(
            "Hugging Face authentication was not found. Set HF_TOKEN or run "
            "`hf auth login` with a token that can read both Descartes repos."
        )

    CHECKPOINT_ROOT.mkdir(parents=True, exist_ok=True)

    for checkpoint in CHECKPOINTS:
        required_files = (
            checkpoint.path / "adapter_config.json",
            checkpoint.path / "adapter_model.safetensors",
        )
        if all(file.is_file() for file in required_files):
            print(f"Using downloaded adapter: {checkpoint.name}")
            continue

        print(f"Downloading adapter: {checkpoint.name}")
        snapshot_download(
            repo_id=checkpoint.repository,
            repo_type="model",
            allow_patterns=[
                f"{checkpoint.remote_directory}/adapter_config.json",
                f"{checkpoint.remote_directory}/adapter_model.safetensors",
            ],
            local_dir=CHECKPOINT_ROOT / checkpoint.local_group,
            token=token,
        )

        missing = [str(file) for file in required_files if not file.is_file()]
        if missing:
            raise RuntimeError(
                f"Checkpoint download did not produce required files: {missing}"
            )

    return list(CHECKPOINTS)


def fresh_messages() -> list[dict[str, str]]:
    """Create a clean chat history, optionally beginning with a system prompt."""
    if SYSTEM_PROMPT is None:
        return []
    return [{"role": "system", "content": SYSTEM_PROMPT}]


def print_checkpoints(
    checkpoints: list[Checkpoint], active_index: int
) -> None:
    """Print the checkpoint menu and identify the active adapter."""
    print("\nAvailable checkpoints:")
    for index, checkpoint in enumerate(checkpoints, start=1):
        marker = " (active)" if index - 1 == active_index else ""
        print(f"  [{index}] {checkpoint.name}{marker}")


def resolve_checkpoint(
    checkpoints: list[Checkpoint], selector: str
) -> int:
    """Resolve a menu index, exact name, or unique checkpoint step."""
    normalized = selector.strip().lower()
    for index, checkpoint in enumerate(checkpoints):
        if normalized == checkpoint.name.lower():
            return index

    try:
        number = int(normalized)
    except ValueError as error:
        raise ValueError(
            "Use a menu number, checkpoint step, or exact checkpoint name."
        ) from error

    if 1 <= number <= len(checkpoints):
        return number - 1

    step_matches = [
        index
        for index, checkpoint in enumerate(checkpoints)
        if checkpoint.step == number
    ]
    if len(step_matches) == 1:
        return step_matches[0]

    raise ValueError(f"No checkpoint matches {selector!r}.")


def attention_implementation() -> str:
    """Use FlashAttention 2 when installed; otherwise use PyTorch SDPA."""
    if importlib.util.find_spec("flash_attn") is not None:
        return "flash_attention_2"
    return "sdpa"


def load_model(
    checkpoints: list[Checkpoint],
) -> tuple[AutoTokenizer, PeftModel, list[str]]:
    """Load Qwen once, then load every checkpoint as a named LoRA adapter."""
    attention = attention_implementation()
    print(f"\nLoading base model once: {BASE_MODEL}")
    print(f"Attention implementation: {attention}")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation=attention,
        low_cpu_mem_usage=True,
    )

    adapter_names = [checkpoint.name.replace("-", "_") for checkpoint in checkpoints]
    print(f"Loading adapter: {checkpoints[0].name}")
    model = PeftModel.from_pretrained(
        base_model,
        checkpoints[0].path,
        adapter_name=adapter_names[0],
        is_trainable=False,
    )

    for checkpoint, adapter_name in zip(
        checkpoints[1:], adapter_names[1:], strict=True
    ):
        print(f"Loading adapter: {checkpoint.name}")
        model.load_adapter(
            checkpoint.path,
            adapter_name=adapter_name,
            is_trainable=False,
        )

    model.eval()
    print("Base model, SFT control, and both GRPO adapters are loaded.")
    return tokenizer, model, adapter_names


def generate_reply(
    tokenizer: AutoTokenizer,
    model: PeftModel,
    messages: list[dict[str, str]],
) -> str:
    """Generate one assistant reply using Qwen's official chat template."""
    model_inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    model_inputs = {
        name: tensor.to(model.device) for name, tensor in model_inputs.items()
    }
    prompt_length = model_inputs["input_ids"].shape[-1]

    with torch.inference_mode():
        output_ids = model.generate(
            **model_inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=TEMPERATURE > 0,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )

    reply_ids = output_ids[0, prompt_length:]
    return tokenizer.decode(reply_ids, skip_special_tokens=True).strip()


def choose_initial_checkpoint(checkpoints: list[Checkpoint]) -> int:
    """Ask which already-loaded checkpoint should be activated first."""
    print_checkpoints(checkpoints, active_index=-1)
    while True:
        try:
            return resolve_checkpoint(
                checkpoints,
                clean_terminal_input(input("\nCheckpoint to activate: ")),
            )
        except ValueError as error:
            print(error)


def save_chat_history(
    messages: list[dict[str, str]], checkpoint: Checkpoint
) -> Path:
    """Save the current conversation as the next numbered Markdown attempt."""
    conversation = [
        message for message in messages if message["role"] in {"user", "assistant"}
    ]
    if not conversation:
        raise ValueError("There is no conversation to save.")

    checkpoint_directory = CHAT_HISTORY_ROOT / checkpoint.name
    checkpoint_directory.mkdir(parents=True, exist_ok=True)

    existing_attempts = []
    for path in checkpoint_directory.glob("attempt-*.md"):
        match = re.fullmatch(r"attempt-(\d+)\.md", path.name)
        if match is not None:
            existing_attempts.append(int(match.group(1)))

    attempt_number = max(existing_attempts, default=0) + 1
    output_path = checkpoint_directory / f"attempt-{attempt_number}.md"

    lines = [
        f"# {checkpoint.name} — Adversarial Attempt {attempt_number}",
        "",
        f"- Base model: `{BASE_MODEL}`",
        f"- Adapter repository: `{checkpoint.repository}`",
        f"- Adapter directory: `{checkpoint.remote_directory}`",
        f"- Temperature: `{TEMPERATURE}`",
        f"- Top-p: `{TOP_P}`",
        f"- Maximum new tokens per response: `{MAX_NEW_TOKENS}`",
        "",
        "## Transcript",
        "",
    ]
    for message in conversation:
        role = "User" if message["role"] == "user" else "Assistant"
        lines.extend([f"### {role}", "", message["content"], ""])

    with output_path.open("x", encoding="utf-8", newline="\n") as output_file:
        output_file.write("\n".join(lines))

    return output_path


def main() -> None:
    checkpoints = download_checkpoints()
    tokenizer, model, adapter_names = load_model(checkpoints)

    active_index = choose_initial_checkpoint(checkpoints)
    model.set_adapter(adapter_names[active_index])
    messages = fresh_messages()

    print(f"\nActive checkpoint: {checkpoints[active_index].name}")
    print("Commands: /use NAME, /checkpoints, /save, /finish, /clear, /exit")

    while True:
        try:
            user_text = clean_terminal_input(input("\nYou: "))
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not user_text:
            continue
        if user_text == "/exit":
            break
        if user_text == "/checkpoints":
            print_checkpoints(checkpoints, active_index)
            continue
        if user_text == "/clear":
            messages = fresh_messages()
            clear_terminal_scrollback()
            print(f"Active checkpoint: {checkpoints[active_index].name}")
            print("Chat history and terminal scrollback cleared.")
            continue
        if user_text == "/save":
            try:
                saved_path = save_chat_history(messages, checkpoints[active_index])
            except ValueError as error:
                print(error)
                continue
            print(f"Chat history saved: {saved_path}")
            continue
        if user_text == "/finish":
            try:
                saved_path = save_chat_history(messages, checkpoints[active_index])
            except ValueError as error:
                print(error)
                continue
            messages = fresh_messages()
            clear_terminal_scrollback()
            print(f"Active checkpoint: {checkpoints[active_index].name}")
            print(f"Attempt saved to {saved_path}")
            print("Chat history and terminal scrollback cleared.")
            continue
        if user_text.startswith("/use "):
            try:
                active_index = resolve_checkpoint(
                    checkpoints, user_text.removeprefix("/use ").strip()
                )
            except ValueError as error:
                print(error)
                continue
            model.set_adapter(adapter_names[active_index])
            messages = fresh_messages()
            clear_terminal_scrollback()
            print(f"Active checkpoint: {checkpoints[active_index].name}")
            print("Chat history and terminal scrollback cleared.")
            continue

        messages.append({"role": "user", "content": user_text})
        reply = generate_reply(tokenizer, model, messages)
        messages.append({"role": "assistant", "content": reply})
        print(f"\nAssistant [{checkpoints[active_index].name}]: {reply}")


if __name__ == "__main__":
    main()
