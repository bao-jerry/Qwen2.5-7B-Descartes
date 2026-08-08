"""Create deterministic GRPO training/validation datasets from conversations.

The source contains complete conversations as alternating ``user``/``LLM``
turn pairs. Each output row is one prefix ending at a user message and stores
that prefix in TRL/Axolotl's conversational ``prompt`` format.

Prefixes above 7,650 Qwen tokens are identified first. The validation split then
reserves one unaffected complete conversation for every ``(prompt_number,
model)`` combination: 16 conversation types x 4 generator models = 64
validation conversations. Selection is deterministic. All prefixes from a
conversation remain in the same split, preventing prefix leakage. Oversized
prefixes from the remaining training conversations are pruned.

Run from anywhere:

    python GRPO_DATA_PROCESSING/prepare_prefix_datasets.py

This requires ``transformers`` and ``jinja2`` so prompt lengths can be measured
with Qwen's exact tokenizer and chat template.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SOURCE_PATH = PROJECT_ROOT / "GRPO_DATA_GENERATION" / "TRAINING_DATA.jsonl"
TRAIN_PATH = Path(__file__).with_name("TRAINING_DATA.jsonl")
VALIDATION_PATH = Path(__file__).with_name("VALIDATION_DATA.jsonl")

EXPECTED_CONVERSATIONS = 1_500
EXPECTED_PROMPT_NUMBERS = set(range(1, 17))
EXPECTED_MODELS = {
    "deepseek-v4-flash",
    "deepseek-v4-pro",
    "mimo-v2.5",
    "mimo-v2.5-pro",
}
EXPECTED_LANGUAGE = "English"
VALIDATION_CONVERSATIONS_PER_CELL = 1
# Keep this versioned seed stable so renaming files never silently changes the split.
SELECTION_SEED = "descartes-grpo-test-v1"
TOKENIZER_NAME = "Qwen/Qwen2.5-7B-Instruct"
MAX_PROMPT_TOKENS = 7_650


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a nonempty JSONL file and report the offending line on failure."""
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path} at line {line_number}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path} line {line_number} is not a JSON object")
            rows.append(row)
    if not rows:
        raise ValueError(f"{path} contains no rows")
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write compact UTF-8 JSONL without escaping non-ASCII text."""
    with path.open("w", encoding="utf-8", newline="\n") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def validate_source(rows: list[dict[str, Any]]) -> None:
    """Validate source metadata and every nonempty user/LLM turn pair."""
    if len(rows) != EXPECTED_CONVERSATIONS:
        raise ValueError(
            f"Expected {EXPECTED_CONVERSATIONS:,} conversations; found {len(rows):,}"
        )

    conversation_ids: set[str] = set()
    cells: Counter[tuple[int, str]] = Counter()

    for row_number, row in enumerate(rows, start=1):
        conversation_id = row.get("conversation_id")
        prompt_number = row.get("prompt_number")
        language = row.get("language")
        model = row.get("model")
        conversation = row.get("conversation")

        if not isinstance(conversation_id, str) or not conversation_id:
            raise ValueError(f"Source row {row_number} has an invalid conversation_id")
        if conversation_id in conversation_ids:
            raise ValueError(f"Duplicate conversation_id: {conversation_id}")
        conversation_ids.add(conversation_id)

        if prompt_number not in EXPECTED_PROMPT_NUMBERS:
            raise ValueError(f"Source row {row_number} has invalid prompt_number")
        if language != EXPECTED_LANGUAGE:
            raise ValueError(f"Source row {row_number} is not English")
        if model not in EXPECTED_MODELS:
            raise ValueError(f"Source row {row_number} has an unknown model slug")
        if not isinstance(conversation, list) or not conversation:
            raise ValueError(f"Source row {row_number} has no conversation turns")

        for turn_number, turn in enumerate(conversation, start=1):
            if not isinstance(turn, dict) or set(turn) != {"user", "LLM"}:
                raise ValueError(
                    f"Source row {row_number}, turn {turn_number} has the wrong schema"
                )
            if not isinstance(turn["user"], str) or not turn["user"].strip():
                raise ValueError(
                    f"Source row {row_number}, turn {turn_number} has an empty user message"
                )
            if not isinstance(turn["LLM"], str) or not turn["LLM"].strip():
                raise ValueError(
                    f"Source row {row_number}, turn {turn_number} has an empty LLM message"
                )

        cells[(prompt_number, model)] += 1

    expected_cells = {
        (prompt_number, model)
        for prompt_number in EXPECTED_PROMPT_NUMBERS
        for model in EXPECTED_MODELS
    }
    if set(cells) != expected_cells:
        raise ValueError("Source data does not contain all 64 prompt/model cells")
    if any(count < VALIDATION_CONVERSATIONS_PER_CELL for count in cells.values()):
        raise ValueError(
            "A prompt/model cell is too small for the requested validation split"
        )


def selection_key(conversation_id: str) -> str:
    """Return a stable pseudo-random ordering key for validation selection."""
    value = f"{SELECTION_SEED}:{conversation_id}".encode("utf-8")
    return hashlib.sha256(value).hexdigest()


def choose_validation_conversation_ids(
    rows: list[dict[str, Any]], eligible_conversation_ids: set[str]
) -> set[str]:
    """Choose one unaffected validation conversation per category/model cell."""
    by_cell: defaultdict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["conversation_id"] in eligible_conversation_ids:
            by_cell[(row["prompt_number"], row["model"])].append(row)

    expected_cells = {
        (prompt_number, model)
        for prompt_number in EXPECTED_PROMPT_NUMBERS
        for model in EXPECTED_MODELS
    }
    missing_cells = expected_cells - set(by_cell)
    if missing_cells:
        raise ValueError(
            "No unaffected conversation is available for validation cells: "
            f"{sorted(missing_cells)}"
        )

    selected: set[str] = set()
    for cell_rows in by_cell.values():
        ranked = sorted(cell_rows, key=lambda row: selection_key(row["conversation_id"]))
        selected.update(
            row["conversation_id"]
            for row in ranked[:VALIDATION_CONVERSATIONS_PER_CELL]
        )
    return selected


def conversation_prefixes(row: dict[str, Any]) -> list[dict[str, Any]]:
    """Expand a paired conversation into every prefix ending with a user message."""
    history: list[dict[str, str]] = []
    prefixes: list[dict[str, Any]] = []

    for prefix_index, turn in enumerate(row["conversation"], start=1):
        history.append({"role": "user", "content": turn["user"]})
        prefixes.append(
            {
                "conversation_id": row["conversation_id"],
                "prompt_number": row["prompt_number"],
                "language": row["language"],
                "model": row["model"],
                "prefix_index": prefix_index,
                "prompt": [message.copy() for message in history],
            }
        )
        history.append({"role": "assistant", "content": turn["LLM"]})

    return prefixes


def prompt_token_count(tokenizer: Any, prompt: list[dict[str, str]]) -> int:
    """Count a prompt after Qwen formatting, including the generation header."""
    encoded = tokenizer.apply_chat_template(
        prompt,
        tokenize=True,
        add_generation_prompt=True,
    )
    if hasattr(encoded, "input_ids"):
        token_ids = encoded.input_ids
    elif isinstance(encoded, dict):
        token_ids = encoded["input_ids"]
    else:
        token_ids = encoded

    if token_ids and isinstance(token_ids[0], list):
        if len(token_ids) != 1:
            raise ValueError(f"Unexpected tokenized batch size: {len(token_ids)}")
        token_ids = token_ids[0]
    return len(token_ids)


def validate_output(
    source_rows: list[dict[str, Any]],
    train_rows: list[dict[str, Any]],
    validation_rows: list[dict[str, Any]],
    validation_conversation_ids: set[str],
    pruned_train_rows: list[dict[str, Any]] | None = None,
) -> None:
    """Validate split isolation, counts, metadata, and conversational prompts."""
    pruned_train_rows = pruned_train_rows or []
    source_by_id = {row["conversation_id"]: row for row in source_rows}
    expected_total_prefixes = sum(len(row["conversation"]) for row in source_rows)
    if (
        len(train_rows) + len(validation_rows) + len(pruned_train_rows)
        != expected_total_prefixes
    ):
        raise ValueError("Output prefix total does not match the source turn-pair total")

    expected_validation_conversations = len(EXPECTED_PROMPT_NUMBERS) * len(
        EXPECTED_MODELS
    )
    if len(validation_conversation_ids) != expected_validation_conversations:
        raise ValueError(
            f"Expected {expected_validation_conversations} validation conversations; "
            f"selected {len(validation_conversation_ids)}"
        )

    train_ids = {row["conversation_id"] for row in train_rows}
    output_validation_ids = {row["conversation_id"] for row in validation_rows}
    if train_ids & output_validation_ids:
        raise ValueError("A conversation leaked across the train/validation boundary")
    if output_validation_ids != validation_conversation_ids:
        raise ValueError("Validation output contains the wrong conversation IDs")
    if len(train_ids) != EXPECTED_CONVERSATIONS - expected_validation_conversations:
        raise ValueError("Training output contains the wrong number of conversations")

    validation_cells = Counter(
        (row["prompt_number"], row["model"])
        for row in source_rows
        if row["conversation_id"] in validation_conversation_ids
    )
    if (
        len(validation_cells) != expected_validation_conversations
        or set(validation_cells.values()) != {1}
    ):
        raise ValueError(
            "Validation split is not exactly one conversation per prompt/model cell"
        )

    seen_keys: set[tuple[str, int]] = set()
    for split_name, rows in (("training", train_rows), ("validation", validation_rows)):
        for row_number, row in enumerate(rows, start=1):
            expected_keys = {
                "conversation_id",
                "prompt_number",
                "language",
                "model",
                "prefix_index",
                "prompt",
            }
            if set(row) != expected_keys:
                raise ValueError(f"{split_name} row {row_number} has incorrect keys")

            unique_key = (row["conversation_id"], row["prefix_index"])
            if unique_key in seen_keys:
                raise ValueError(f"Duplicate prefix key: {unique_key}")
            seen_keys.add(unique_key)

            source = source_by_id.get(row["conversation_id"])
            if source is None:
                raise ValueError(f"Unknown source conversation: {row['conversation_id']}")
            if any(
                row[field] != source[field]
                for field in ("prompt_number", "language", "model")
            ):
                raise ValueError(f"{split_name} row {row_number} has stale metadata")
            if not isinstance(row["prefix_index"], int) or not (
                1 <= row["prefix_index"] <= len(source["conversation"])
            ):
                raise ValueError(f"{split_name} row {row_number} has invalid prefix_index")

            prompt = row["prompt"]
            if not isinstance(prompt, list) or not prompt or len(prompt) % 2 != 1:
                raise ValueError(f"{split_name} row {row_number} has invalid prompt length")
            expected_roles = [
                "user" if index % 2 == 0 else "assistant"
                for index in range(len(prompt))
            ]
            roles = [message.get("role") for message in prompt if isinstance(message, dict)]
            if roles != expected_roles or roles[-1] != "user":
                raise ValueError(f"{split_name} row {row_number} has invalid role order")
            if any(
                set(message) != {"role", "content"}
                or not isinstance(message["content"], str)
                or not message["content"].strip()
                for message in prompt
            ):
                raise ValueError(f"{split_name} row {row_number} has invalid messages")

            expected_prompt: list[dict[str, str]] = []
            for turn_index, turn in enumerate(
                source["conversation"][: row["prefix_index"]], start=1
            ):
                expected_prompt.append({"role": "user", "content": turn["user"]})
                if turn_index < row["prefix_index"]:
                    expected_prompt.append(
                        {"role": "assistant", "content": turn["LLM"]}
                    )
            if prompt != expected_prompt:
                raise ValueError(
                    f"{split_name} row {row_number} is not the exact source prefix"
                )

    expected_keys = {
        (row["conversation_id"], prefix_index)
        for row in source_rows
        for prefix_index in range(1, len(row["conversation"]) + 1)
    }
    pruned_keys = {
        (row["conversation_id"], row["prefix_index"]) for row in pruned_train_rows
    }
    if len(pruned_keys) != len(pruned_train_rows):
        raise ValueError("Pruned training prefixes contain duplicate keys")
    if pruned_keys & seen_keys:
        raise ValueError("A pruned training prefix is still present in an output split")
    if any(
        row["conversation_id"] in validation_conversation_ids
        for row in pruned_train_rows
    ):
        raise ValueError("A validation prefix was incorrectly marked as pruned")
    if seen_keys != expected_keys - pruned_keys:
        raise ValueError(
            "Output does not contain every non-pruned source user-ending prefix once"
        )


def main() -> None:
    from transformers import AutoTokenizer

    source_rows = read_jsonl(SOURCE_PATH)
    validate_source(source_rows)

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    prefixes_by_conversation: dict[str, list[dict[str, Any]]] = {}
    oversized_prefix_keys: set[tuple[str, int]] = set()
    affected_conversation_ids: set[str] = set()
    for source_row in source_rows:
        conversation_id = source_row["conversation_id"]
        prefixes = conversation_prefixes(source_row)
        prefixes_by_conversation[conversation_id] = prefixes
        for prefix in prefixes:
            if prompt_token_count(tokenizer, prefix["prompt"]) > MAX_PROMPT_TOKENS:
                oversized_prefix_keys.add((conversation_id, prefix["prefix_index"]))
                affected_conversation_ids.add(conversation_id)

    eligible_conversation_ids = {
        row["conversation_id"] for row in source_rows
    } - affected_conversation_ids
    validation_conversation_ids = choose_validation_conversation_ids(
        source_rows, eligible_conversation_ids
    )

    unpruned_train_rows: list[dict[str, Any]] = []
    validation_rows: list[dict[str, Any]] = []

    for source_row in source_rows:
        prefixes = prefixes_by_conversation[source_row["conversation_id"]]
        destination = (
            validation_rows
            if source_row["conversation_id"] in validation_conversation_ids
            else unpruned_train_rows
        )
        destination.extend(prefixes)

    validate_output(
        source_rows,
        unpruned_train_rows,
        validation_rows,
        validation_conversation_ids,
    )

    train_rows = [
        row
        for row in unpruned_train_rows
        if (row["conversation_id"], row["prefix_index"])
        not in oversized_prefix_keys
    ]
    pruned_train_rows = [
        row
        for row in unpruned_train_rows
        if (row["conversation_id"], row["prefix_index"])
        in oversized_prefix_keys
    ]
    validate_output(
        source_rows,
        train_rows,
        validation_rows,
        validation_conversation_ids,
        pruned_train_rows,
    )
    write_jsonl(TRAIN_PATH, train_rows)
    write_jsonl(VALIDATION_PATH, validation_rows)

    # Re-read and revalidate the actual files written to disk.
    written_train_rows = read_jsonl(TRAIN_PATH)
    written_validation_rows = read_jsonl(VALIDATION_PATH)
    validate_output(
        source_rows,
        written_train_rows,
        written_validation_rows,
        validation_conversation_ids,
        pruned_train_rows,
    )

    print(f"Source conversations: {len(source_rows):,}")
    print(f"Training conversations: {len({r['conversation_id'] for r in train_rows}):,}")
    print(
        "Validation conversations: "
        f"{len({r['conversation_id'] for r in validation_rows}):,}"
    )
    print(f"Training prefixes: {len(train_rows):,}")
    print(
        f"Training prefixes pruned above {MAX_PROMPT_TOKENS:,} tokens: "
        f"{len(pruned_train_rows):,}"
    )
    print(f"Validation prefixes: {len(validation_rows):,}")
    print(f"Total prefixes: {len(train_rows) + len(validation_rows):,}")
    print("Validation passed: schema, counts, split isolation, and balance.")


if __name__ == "__main__":
    main()
