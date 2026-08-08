"""Store sample allocations and generate their Markdown table.

Import SAMPLE_COUNTS to access the combined allocation dictionary. Its keys
have the shape (prompt_number, language, model_slug), and its values are sample
counts. The model slug is the exact value accepted by the provider API.

Basic lookup:
    from sample_allocation import SAMPLE_COUNTS

    count = SAMPLE_COUNTS[(1, "English", "deepseek-v4-flash")]

Per-model lookup:
    from sample_allocation import DEEPSEEK_V4_FLASH_SAMPLE_COUNTS

    count = DEEPSEEK_V4_FLASH_SAMPLE_COUNTS[(1, "English")]

Using a model's expanded work queue:
    import random
    from sample_allocation import DEEPSEEK_V4_FLASH_QUEUE

    queue = DEEPSEEK_V4_FLASH_QUEUE.copy()
    random.shuffle(queue)
    prompt_number, language = queue.pop()

Joining an allocation to its actual prompt string:
    from prompt_texts import PROMPT_TEXTS
    from sample_allocation import SAMPLE_COUNTS

    for (prompt_number, language, model_slug), count in SAMPLE_COUNTS.items():
        prompt = PROMPT_TEXTS[(prompt_number, language)]
        # Generate `count` conversations from `prompt` using `model_slug`.

When run directly, this module inserts the allocation table into
conversation_categories.md between stable marker comments. Running it again
replaces the existing generated table. No third-party packages are required.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path

from prompt_texts import PROMPT_TEXTS


TOTAL_SAMPLES = 20_000

PROMPTS = [
    (1, 7),
    (2, 3),
    (3, 5),
    (4, 10),
    (5, 4),
    (6, 5),
    (7, 2),
    (8, 1),
    (9, 3),
    (10, 2),
    (11, 2),
    (12, 5),
    (13, 10),
    (14, 3),
    (15, 7),
    (16, 2),
    (17, 2),
    (18, 2),
    (19, 4),
    (20, 4),
    (21, 7),
    (22, 5),
    (23, 5),
]

MODELS = [
    ("DeepSeek V4 Flash", 35),
    ("MiMo 2.5", 35),
    ("DeepSeek V4 Pro", 15),
    ("MiMo 2.5 Pro", 15),
]

MODEL_SLUGS = {
    "DeepSeek V4 Flash": "deepseek-v4-flash",
    "MiMo 2.5": "mimo-v2.5",
    "DeepSeek V4 Pro": "deepseek-v4-pro",
    "MiMo 2.5 Pro": "mimo-v2.5-pro",
}

LANGUAGES = [
    ("English", 80),
    ("Simplified Chinese", 15),
    ("Spanish", 5),
]

MARKER_START = "<!-- SAMPLE_ALLOCATION_TABLE_START -->"
MARKER_END = "<!-- SAMPLE_ALLOCATION_TABLE_END -->"
MARKDOWN_PATH = Path(__file__).with_name("conversation_categories.md")


def exact_share(*percentages: int) -> Fraction:
    """Return TOTAL_SAMPLES multiplied by all supplied percentages."""
    result = Fraction(TOTAL_SAMPLES)
    for percentage in percentages:
        result *= Fraction(percentage, 100)
    return result


def require_integer(value: Fraction, description: str) -> int:
    """Return an integer Fraction or fail with a useful explanation."""
    if value.denominator != 1:
        raise ValueError(f"{description} is not an integer: {value}")
    return value.numerator


def build_allocations() -> list[dict[str, int | str | Fraction]]:
    """Build nearest-integer cells while preserving every pairwise margin."""
    fractional_prompt_count = sum(
        any(
            exact_share(prompt_pct, model_pct, language_pct).denominator != 1
            for _, model_pct in MODELS
            for _, language_pct in LANGUAGES
        )
        for _, prompt_pct in PROMPTS
    )
    if fractional_prompt_count % 2:
        raise ValueError(
            "Balanced rounding requires an even number of prompts containing "
            "fractional cells."
        )

    allocations: list[dict[str, int | str | Fraction]] = []
    fractional_prompt_index = 0

    for prompt_id, prompt_pct in PROMPTS:
        prompt_has_fraction = any(
            exact_share(prompt_pct, model_pct, language_pct).denominator != 1
            for _, model_pct in MODELS
            for _, language_pct in LANGUAGES
        )
        phase = fractional_prompt_index % 2

        for model_index, (model_name, model_pct) in enumerate(MODELS):
            for language_name, language_pct in LANGUAGES:
                ideal = exact_share(prompt_pct, model_pct, language_pct)

                if ideal.denominator == 1:
                    samples = ideal.numerator
                else:
                    if ideal.denominator != 2:
                        raise ValueError(
                            "The current balanced-rounding rule only supports "
                            f"half-integer cells, but got {ideal}."
                        )

                    model_rounds_up_for_chinese = model_index % 2 == phase
                    if language_name == "Simplified Chinese":
                        round_up = model_rounds_up_for_chinese
                    elif language_name == "Spanish":
                        round_up = not model_rounds_up_for_chinese
                    else:
                        raise ValueError(
                            f"Unexpected fractional {language_name} cell."
                        )

                    samples = ideal.numerator // ideal.denominator
                    if round_up:
                        samples += 1

                allocations.append(
                    {
                        "prompt_id": prompt_id,
                        "prompt_pct": prompt_pct,
                        "model": model_name,
                        "model_pct": model_pct,
                        "language": language_name,
                        "language_pct": language_pct,
                        "ideal": ideal,
                        "samples": samples,
                    }
                )

        if prompt_has_fraction:
            fractional_prompt_index += 1

    return allocations


def validate_allocations(
    allocations: list[dict[str, int | str | Fraction]],
) -> None:
    """Verify totals and all two-dimensional margins."""
    if len(allocations) != len(PROMPTS) * len(MODELS) * len(LANGUAGES):
        raise ValueError("The allocation table does not contain every cell.")

    prompt_totals: defaultdict[int, int] = defaultdict(int)
    model_totals: defaultdict[str, int] = defaultdict(int)
    language_totals: defaultdict[str, int] = defaultdict(int)
    prompt_model_totals: defaultdict[tuple[int, str], int] = defaultdict(int)
    prompt_language_totals: defaultdict[tuple[int, str], int] = defaultdict(int)
    model_language_totals: defaultdict[tuple[str, str], int] = defaultdict(int)

    for row in allocations:
        prompt_id = int(row["prompt_id"])
        model_name = str(row["model"])
        language_name = str(row["language"])
        samples = int(row["samples"])
        ideal = row["ideal"]
        if not isinstance(ideal, Fraction):
            raise TypeError("Internal error: ideal allocation is not a Fraction.")
        if abs(Fraction(samples) - ideal) > Fraction(1, 2):
            raise ValueError(
                f"Cell {prompt_id}/{model_name}/{language_name} is not "
                "nearest-integer rounded."
            )

        prompt_totals[prompt_id] += samples
        model_totals[model_name] += samples
        language_totals[language_name] += samples
        prompt_model_totals[(prompt_id, model_name)] += samples
        prompt_language_totals[(prompt_id, language_name)] += samples
        model_language_totals[(model_name, language_name)] += samples

    if sum(int(row["samples"]) for row in allocations) != TOTAL_SAMPLES:
        raise ValueError("Grand total does not equal TOTAL_SAMPLES.")

    for prompt_id, prompt_pct in PROMPTS:
        expected = require_integer(
            exact_share(prompt_pct), f"Prompt {prompt_id} total"
        )
        if prompt_totals[prompt_id] != expected:
            raise ValueError(f"Prompt {prompt_id} total is incorrect.")

    for model_name, model_pct in MODELS:
        expected = require_integer(
            exact_share(model_pct), f"{model_name} total"
        )
        if model_totals[model_name] != expected:
            raise ValueError(f"{model_name} total is incorrect.")

    for language_name, language_pct in LANGUAGES:
        expected = require_integer(
            exact_share(language_pct), f"{language_name} total"
        )
        if language_totals[language_name] != expected:
            raise ValueError(f"{language_name} total is incorrect.")

    for prompt_id, prompt_pct in PROMPTS:
        for model_name, model_pct in MODELS:
            expected = require_integer(
                exact_share(prompt_pct, model_pct),
                f"Prompt {prompt_id}/{model_name} total",
            )
            if prompt_model_totals[(prompt_id, model_name)] != expected:
                raise ValueError(
                    f"Prompt {prompt_id}/{model_name} total is incorrect."
                )

        for language_name, language_pct in LANGUAGES:
            expected = require_integer(
                exact_share(prompt_pct, language_pct),
                f"Prompt {prompt_id}/{language_name} total",
            )
            if prompt_language_totals[(prompt_id, language_name)] != expected:
                raise ValueError(
                    f"Prompt {prompt_id}/{language_name} total is incorrect."
                )

    for model_name, model_pct in MODELS:
        for language_name, language_pct in LANGUAGES:
            expected = require_integer(
                exact_share(model_pct, language_pct),
                f"{model_name}/{language_name} total",
            )
            if model_language_totals[(model_name, language_name)] != expected:
                raise ValueError(
                    f"{model_name}/{language_name} total is incorrect."
                )


def render_table(
    allocations: list[dict[str, int | str | Fraction]],
) -> str:
    """Render the generated Markdown section."""
    lines = [
        MARKER_START,
        "# Sample allocation by prompt × model × language",
        "",
        (
            "Allocation basis: 20,000 samples. Fractional cross-product cells "
            "use balanced nearest-integer rounding. Every prompt, model, "
            "language, and pairwise margin retains its exact requested total."
        ),
        "",
        (
            "| Prompt | Prompt share | Model | Model share | Language | "
            "Language share | Samples |"
        ),
        "|---:|---:|---|---:|---|---:|---:|",
    ]

    for row in allocations:
        lines.append(
            f'| {row["prompt_id"]} | {row["prompt_pct"]}% | '
            f'{row["model"]} | {row["model_pct"]}% | '
            f'{row["language"]} | {row["language_pct"]}% | '
            f'{int(row["samples"]):,} |'
        )

    lines.extend(["", f"**Total: {TOTAL_SAMPLES:,} samples.**", MARKER_END])
    return "\n".join(lines)


def update_markdown(table: str) -> None:
    """Insert or replace the generated table in the Markdown document."""
    document = MARKDOWN_PATH.read_text(encoding="utf-8")

    if MARKER_START in document or MARKER_END in document:
        if document.count(MARKER_START) != 1 or document.count(MARKER_END) != 1:
            raise ValueError("The Markdown file contains malformed table markers.")
        start = document.index(MARKER_START)
        end = document.index(MARKER_END, start) + len(MARKER_END)
        updated = document[:start].rstrip() + "\n\n" + table
        trailing = document[end:].strip()
        if trailing:
            updated += "\n\n" + trailing
        updated += "\n"
    else:
        updated = document.rstrip() + "\n\n" + table + "\n"

    MARKDOWN_PATH.write_text(updated, encoding="utf-8", newline="\n")


_ALLOCATION_ROWS = build_allocations()
validate_allocations(_ALLOCATION_ROWS)

# Key: (prompt_number, language, exact API model slug)
# Value: exact number of conversations to generate for that combination
SAMPLE_COUNTS: dict[tuple[int, str, str], int] = {
    (
        int(row["prompt_id"]),
        str(row["language"]),
        MODEL_SLUGS[str(row["model"])],
    ): int(row["samples"])
    for row in _ALLOCATION_ROWS
}


def _counts_for_model(model_slug: str) -> dict[tuple[int, str], int]:
    """Return (prompt_number, language) counts for one exact API model slug."""
    return {
        (prompt_number, language): count
        for (prompt_number, language, slug), count in SAMPLE_COUNTS.items()
        if slug == model_slug
    }


DEEPSEEK_V4_FLASH_SAMPLE_COUNTS = _counts_for_model("deepseek-v4-flash")
DEEPSEEK_V4_PRO_SAMPLE_COUNTS = _counts_for_model("deepseek-v4-pro")
MIMO_V2_5_SAMPLE_COUNTS = _counts_for_model("mimo-v2.5")
MIMO_V2_5_PRO_SAMPLE_COUNTS = _counts_for_model("mimo-v2.5-pro")

SAMPLE_COUNTS_BY_MODEL: dict[str, dict[tuple[int, str], int]] = {
    "deepseek-v4-flash": DEEPSEEK_V4_FLASH_SAMPLE_COUNTS,
    "deepseek-v4-pro": DEEPSEEK_V4_PRO_SAMPLE_COUNTS,
    "mimo-v2.5": MIMO_V2_5_SAMPLE_COUNTS,
    "mimo-v2.5-pro": MIMO_V2_5_PRO_SAMPLE_COUNTS,
}


def _expand_queue(
    sample_counts: dict[tuple[int, str], int],
) -> list[tuple[int, str]]:
    """Repeat each (prompt_number, language) tuple by its sample count."""
    return [
        prompt_and_language
        for prompt_and_language, count in sample_counts.items()
        for _ in range(count)
    ]


# These are canonical queue templates. Copy one before shuffling or consuming it.
DEEPSEEK_V4_FLASH_QUEUE = _expand_queue(DEEPSEEK_V4_FLASH_SAMPLE_COUNTS)
DEEPSEEK_V4_PRO_QUEUE = _expand_queue(DEEPSEEK_V4_PRO_SAMPLE_COUNTS)
MIMO_V2_5_QUEUE = _expand_queue(MIMO_V2_5_SAMPLE_COUNTS)
MIMO_V2_5_PRO_QUEUE = _expand_queue(MIMO_V2_5_PRO_SAMPLE_COUNTS)

QUEUES_BY_MODEL: dict[str, list[tuple[int, str]]] = {
    "deepseek-v4-flash": DEEPSEEK_V4_FLASH_QUEUE,
    "deepseek-v4-pro": DEEPSEEK_V4_PRO_QUEUE,
    "mimo-v2.5": MIMO_V2_5_QUEUE,
    "mimo-v2.5-pro": MIMO_V2_5_PRO_QUEUE,
}

if len(SAMPLE_COUNTS) != len(_ALLOCATION_ROWS):
    raise ValueError("Duplicate tuple keys were found in SAMPLE_COUNTS.")
if sum(SAMPLE_COUNTS.values()) != TOTAL_SAMPLES:
    raise ValueError("SAMPLE_COUNTS does not total TOTAL_SAMPLES.")
if set(SAMPLE_COUNTS_BY_MODEL) != set(MODEL_SLUGS.values()):
    raise ValueError("SAMPLE_COUNTS_BY_MODEL has incorrect model slugs.")
if any(
    len(counts) != len(PROMPTS) * len(LANGUAGES)
    for counts in SAMPLE_COUNTS_BY_MODEL.values()
):
    raise ValueError("A per-model allocation dictionary is incomplete.")
if set(QUEUES_BY_MODEL) != set(SAMPLE_COUNTS_BY_MODEL):
    raise ValueError("QUEUES_BY_MODEL has incorrect model slugs.")
if any(
    Counter(QUEUES_BY_MODEL[model_slug]) != sample_counts
    for model_slug, sample_counts in SAMPLE_COUNTS_BY_MODEL.items()
):
    raise ValueError("A model queue does not match its allocation dictionary.")
if sum(len(queue) for queue in QUEUES_BY_MODEL.values()) != TOTAL_SAMPLES:
    raise ValueError("The model queues do not total TOTAL_SAMPLES.")
if any(
    (prompt_number, language) not in PROMPT_TEXTS
    for prompt_number, language, _model in SAMPLE_COUNTS
):
    raise ValueError("An allocation does not have a corresponding prompt string.")


def main() -> None:
    update_markdown(render_table(_ALLOCATION_ROWS))
    print(
        f"Wrote {len(SAMPLE_COUNTS)} allocation rows totaling "
        f"{TOTAL_SAMPLES:,} samples to {MARKDOWN_PATH.name}."
    )


if __name__ == "__main__":
    main()
