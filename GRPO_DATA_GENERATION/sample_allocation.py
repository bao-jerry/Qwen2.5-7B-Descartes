"""Store the 1,500-conversation GRPO generation allocation.

The public data structures deliberately match the SFT generation precedent.

Combined lookup:
    from sample_allocation import SAMPLE_COUNTS

    count = SAMPLE_COUNTS[(1, "English", "deepseek-v4-flash")]

Per-model lookup:
    from sample_allocation import DEEPSEEK_V4_FLASH_SAMPLE_COUNTS

    count = DEEPSEEK_V4_FLASH_SAMPLE_COUNTS[(1, "English")]

Expanded work queue:
    import random
    from sample_allocation import DEEPSEEK_V4_FLASH_QUEUE

    queue = DEEPSEEK_V4_FLASH_QUEUE.copy()
    random.shuffle(queue)
    prompt_number, language = queue.pop()

Joining an allocation to its prompt text:
    from prompt_texts import PROMPT_TEXTS
    from sample_allocation import SAMPLE_COUNTS

    for (prompt_number, language, model_slug), count in SAMPLE_COUNTS.items():
        prompt = PROMPT_TEXTS[(prompt_number, language)]
        # Generate `count` conversations using `model_slug`.

When run directly, this module inserts the allocation table into
conversation_categories.md between stable marker comments.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from prompt_texts import PROMPT_TEXTS


TOTAL_SAMPLES = 1_500
PROMPT_NUMBERS = tuple(range(1, 17))
LANGUAGE = "English"

MODELS = [
    ("DeepSeek V4 Flash", "deepseek-v4-flash"),
    ("DeepSeek V4 Pro", "deepseek-v4-pro"),
    ("MiMo 2.5", "mimo-v2.5"),
    ("MiMo 2.5 Pro", "mimo-v2.5-pro"),
]

# 1,500 / 4 models = 375 conversations per model. Each model receives 23
# conversations for every prompt plus one extra conversation for seven prompts.
# The rotated extras keep the combined prompt totals at either 93 or 94.
EXTRA_PROMPTS_BY_MODEL = {
    "deepseek-v4-flash": {1, 2, 3, 4, 5, 6, 7},
    "deepseek-v4-pro": {5, 6, 7, 8, 9, 10, 11},
    "mimo-v2.5": {9, 10, 11, 12, 13, 14, 15},
    "mimo-v2.5-pro": {1, 2, 3, 8, 12, 13, 16},
}

MARKER_START = "<!-- SAMPLE_ALLOCATION_TABLE_START -->"
MARKER_END = "<!-- SAMPLE_ALLOCATION_TABLE_END -->"
MARKDOWN_PATH = Path(__file__).with_name("conversation_categories.md")


# Key: (prompt_number, language, exact API model slug)
# Value: exact number of conversations to generate for that combination
SAMPLE_COUNTS: dict[tuple[int, str, str], int] = {
    (prompt_number, LANGUAGE, model_slug): 23
    + int(prompt_number in EXTRA_PROMPTS_BY_MODEL[model_slug])
    for prompt_number in PROMPT_NUMBERS
    for _model_name, model_slug in MODELS
}


def _counts_for_model(model_slug: str) -> dict[tuple[int, str], int]:
    """Return (prompt_number, language) counts for one exact API slug."""
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


def _validate() -> None:
    """Validate every allocation margin and its prompt lookup."""
    model_slugs = {model_slug for _model_name, model_slug in MODELS}
    if len(SAMPLE_COUNTS) != len(PROMPT_NUMBERS) * len(MODELS):
        raise ValueError("SAMPLE_COUNTS does not contain every allocation cell.")
    if sum(SAMPLE_COUNTS.values()) != TOTAL_SAMPLES:
        raise ValueError("SAMPLE_COUNTS does not total 1,500 conversations.")
    if set(SAMPLE_COUNTS_BY_MODEL) != model_slugs:
        raise ValueError("SAMPLE_COUNTS_BY_MODEL has incorrect model slugs.")
    if any(sum(counts.values()) != 375 for counts in SAMPLE_COUNTS_BY_MODEL.values()):
        raise ValueError("Every model must receive exactly 375 conversations.")

    prompt_totals = Counter()
    for (prompt_number, language, _model_slug), count in SAMPLE_COUNTS.items():
        prompt_totals[prompt_number] += count
        if language != LANGUAGE:
            raise ValueError("GRPO conversation generation must be English-only.")
        if (prompt_number, language) not in PROMPT_TEXTS:
            raise ValueError("An allocation lacks a corresponding prompt string.")

    if sorted(prompt_totals.values()) != [93] * 4 + [94] * 12:
        raise ValueError("Prompt totals are not distributed as evenly as possible.")
    if any(
        Counter(QUEUES_BY_MODEL[model_slug]) != counts
        for model_slug, counts in SAMPLE_COUNTS_BY_MODEL.items()
    ):
        raise ValueError("A model queue does not match its allocation dictionary.")
    if sum(len(queue) for queue in QUEUES_BY_MODEL.values()) != TOTAL_SAMPLES:
        raise ValueError("The four model queues do not total 1,500 conversations.")


def render_table() -> str:
    """Render the generated Markdown allocation section."""
    display_names = {slug: name for name, slug in MODELS}
    lines = [
        MARKER_START,
        "## Sample allocation by prompt × model × language",
        "",
        (
            "Allocation basis: 1,500 conversations, 25% per model, English "
            "only, and the 16 prompt types distributed as evenly as integers "
            "allow. Every model receives exactly 375 conversations."
        ),
        "",
        "| Prompt | Prompt share | Model | Model share | Language | Language share | Samples |",
        "|---:|---:|---|---:|---|---:|---:|",
    ]
    for (prompt_number, language, model_slug), count in SAMPLE_COUNTS.items():
        lines.append(
            f"| {prompt_number} | 6.25% | {display_names[model_slug]} | 25% | "
            f"{language} | 100% | {count} |"
        )
    lines.extend(["", f"**Total: {TOTAL_SAMPLES:,} conversations.**", MARKER_END])
    return "\n".join(lines)


def update_markdown(table: str) -> None:
    """Insert or replace the generated table in conversation_categories.md."""
    document = MARKDOWN_PATH.read_text(encoding="utf-8")
    if document.count(MARKER_START) != 1 or document.count(MARKER_END) != 1:
        raise ValueError("The Markdown file must contain exactly one marker pair.")
    start = document.index(MARKER_START)
    end = document.index(MARKER_END, start) + len(MARKER_END)
    updated = document[:start].rstrip() + "\n\n" + table
    trailing = document[end:].strip()
    if trailing:
        updated += "\n\n" + trailing
    MARKDOWN_PATH.write_text(updated + "\n", encoding="utf-8", newline="\n")


_validate()


def main() -> None:
    update_markdown(render_table())
    print(
        f"Wrote {len(SAMPLE_COUNTS)} allocation rows totaling "
        f"{TOTAL_SAMPLES:,} conversations to {MARKDOWN_PATH.name}."
    )


if __name__ == "__main__":
    main()
