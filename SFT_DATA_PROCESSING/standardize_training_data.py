"""Convert the generated conversations to standard chat-message JSONL.

Input rows are read from ``../SFT_DATA_GENERATION/TRAINING_DATA.jsonl``:

    {
        "prompt_number": 4,
        "language": "English",
        "model": "mimo-v2.5",
        "conversation": [{"user": "...", "LLM": "..."}]
    }

Output rows are written to ``STANDARDIZED_TRAINING_DATA.jsonl``:

    {
        "messages": [
            {"role": "user", "content": "..."},
            {"role": "assistant", "content": "..."}
        ]
    }

Run this file directly to convert the complete dataset. The source JSONL is
never modified. The standardized file is still untokenized and unpacked.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


SFT_DATA_PROCESSING_DIR = Path(__file__).resolve().parent
SFT_DATA_GENERATION_DIR = (
    SFT_DATA_PROCESSING_DIR.parent / "SFT_DATA_GENERATION"
)
INPUT_PATH = SFT_DATA_GENERATION_DIR / "TRAINING_DATA.jsonl"
OUTPUT_PATH = SFT_DATA_PROCESSING_DIR / "STANDARDIZED_TRAINING_DATA.jsonl"


def standardize_training_example(example: dict[str, Any]) -> dict[str, list[dict[str, str]]]:
    """Convert one generated training-data row to an ordered messages list.

    The original text is preserved exactly. Metadata such as prompt number,
    language, and generation model is intentionally excluded from the returned
    training example.
    """

    conversation = example.get("conversation")
    if not isinstance(conversation, list) or not conversation:
        raise ValueError("'conversation' must be a non-empty list.")

    messages: list[dict[str, str]] = []
    for turn_number, turn in enumerate(conversation, start=1):
        if not isinstance(turn, dict):
            raise ValueError(f"Turn {turn_number} must be an object.")

        user_text = turn.get("user")
        assistant_text = turn.get("LLM")
        if not isinstance(user_text, str) or not user_text.strip():
            raise ValueError(f"Turn {turn_number} has no non-empty 'user' string.")
        if not isinstance(assistant_text, str) or not assistant_text.strip():
            raise ValueError(f"Turn {turn_number} has no non-empty 'LLM' string.")

        messages.append({"role": "user", "content": user_text})
        messages.append({"role": "assistant", "content": assistant_text})

    return {"messages": messages}


def standardize_training_data(
    input_path: Path = INPUT_PATH,
    output_path: Path = OUTPUT_PATH,
) -> int:
    """Stream an input JSONL file into standardized conversation JSONL.

    Returns the number of conversations written. A temporary file is used so a
    failed conversion cannot leave a partially written standardized dataset.
    """

    input_path = Path(input_path).resolve()
    output_path = Path(output_path).resolve()
    if input_path == output_path:
        raise ValueError("The input and output paths must be different.")
    if not input_path.is_file():
        raise FileNotFoundError(f"Training data not found: {input_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f"{output_path.name}.tmp")
    written = 0

    try:
        with input_path.open("r", encoding="utf-8") as source, temporary_path.open(
            "w", encoding="utf-8", newline="\n"
        ) as destination:
            for line_number, line in enumerate(source, start=1):
                if not line.strip():
                    continue

                try:
                    example = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"Invalid JSON on line {line_number}: {error}") from error

                if not isinstance(example, dict):
                    raise ValueError(f"Line {line_number} must contain a JSON object.")

                try:
                    standardized = standardize_training_example(example)
                except ValueError as error:
                    raise ValueError(f"Invalid conversation on line {line_number}: {error}") from error

                destination.write(
                    json.dumps(standardized, ensure_ascii=False, separators=(",", ":")) + "\n"
                )
                written += 1

        temporary_path.replace(output_path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise

    return written


if __name__ == "__main__":
    conversation_count = standardize_training_data()
    print(f"Wrote {conversation_count:,} conversations to {OUTPUT_PATH}")
