"""Generate the MiMo portion of TRAINING_DATA.jsonl concurrently.

The existing MiMo API settings and prompt template are imported from
generate_conversation_mimo.py.

Requests are started at least 0.75 seconds apart, with no more than 50 active
requests at once.

Run:
    python generate_training_data_mimo.py

Successful JSONL records have this shape:
    {
        "prompt_number": 1,
        "language": "English",
        "model": "mimo-v2.5",
        "conversation": [{"user": "...", "LLM": "..."}]
    }

If TRAINING_DATA.jsonl already exists, completed allocation counts are loaded
from it and only the remaining MiMo samples are generated.
"""

from __future__ import annotations

import asyncio
import json
import random
import sys
import time
from collections import Counter, deque
from pathlib import Path
from typing import TextIO

from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AsyncOpenAI,
)

import generate_conversation_mimo as template
from prompt_texts import PROMPT_TEXTS
from sample_allocation import (
    MIMO_V2_5_PRO_QUEUE,
    MIMO_V2_5_QUEUE,
)


TRAINING_DATA = Path(__file__).with_name("TRAINING_DATA.jsonl")

REQUEST_INTERVAL_SECONDS = 0.75
MAX_ACTIVE_REQUESTS = 50

MODEL_RUNS = [
    ("mimo-v2.5", MIMO_V2_5_QUEUE),
    ("mimo-v2.5-pro", MIMO_V2_5_PRO_QUEUE),
]

RETRIABLE_STATUS_CODES = {408, 409, 429, 500, 502, 503, 504}

AllocationKey = tuple[int, str, str]


def _load_completed_counts() -> Counter[AllocationKey]:
    """Count completed samples by (prompt_number, language, model)."""
    completed: Counter[AllocationKey] = Counter()
    if not TRAINING_DATA.exists():
        return completed

    with TRAINING_DATA.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                key = (
                    int(record["prompt_number"]),
                    str(record["language"]),
                    str(record["model"]),
                )
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"{TRAINING_DATA.name} line {line_number} is invalid."
                ) from exc
            completed[key] += 1

    return completed


def _build_pending_queue(
    model: str,
    model_queue: list[tuple[int, str]],
    completed: Counter[AllocationKey],
) -> deque[AllocationKey]:
    """Remove already completed samples from a model's canonical queue."""
    target_counts = Counter(
        (prompt_number, language, model)
        for prompt_number, language in model_queue
    )

    for key, completed_count in completed.items():
        if key[2] == model and completed_count > target_counts[key]:
            raise ValueError(
                f"{TRAINING_DATA.name} contains too many records for {key}: "
                f"{completed_count} completed, {target_counts[key]} requested."
            )

    counts_to_skip = Counter(
        {
            key: count
            for key, count in completed.items()
            if key[2] == model
        }
    )
    remaining: list[AllocationKey] = []

    for prompt_number, language in model_queue:
        key = (prompt_number, language, model)
        if counts_to_skip[key]:
            counts_to_skip[key] -= 1
        else:
            remaining.append(key)

    random.shuffle(remaining)
    return deque(remaining)


async def _generate_conversation(
    client: AsyncOpenAI,
    key: AllocationKey,
) -> list[dict[str, str]]:
    """Generate and validate one conversation for an allocation key."""
    prompt_number, language, model = key
    user_message = PROMPT_TEXTS[(prompt_number, language)]

    response = await client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    f"{template.SYSTEM_PROMPT.strip()}\n\n"
                    f"{template.FORMAT_INSTRUCTIONS}"
                ),
            },
            {"role": "user", "content": user_message},
        ],
        response_format={"type": "json_object"},
        temperature=template.TEMPERATURE,
        stream=False,
        extra_body={"thinking": {"type": "disabled"}},
    )

    content = response.choices[0].message.content
    if not content:
        raise ValueError("MiMo returned an empty response.")

    try:
        payload = json.loads(content)
    except json.JSONDecodeError as exc:
        raise ValueError(f"MiMo returned invalid JSON: {exc}") from exc

    return template._validate_conversation(payload)


def _is_retriable(exc: BaseException) -> bool:
    """Return whether a failed allocation should go to the pending queue's end."""
    if isinstance(exc, (APIConnectionError, APITimeoutError, ValueError)):
        return True
    if isinstance(exc, APIStatusError):
        return exc.status_code in RETRIABLE_STATUS_CODES
    return False


def _write_record(
    output: TextIO,
    key: AllocationKey,
    conversation: list[dict[str, str]],
) -> None:
    """Append one completed conversation as one JSONL element."""
    prompt_number, language, model = key
    record = {
        "prompt_number": prompt_number,
        "language": language,
        "model": model,
        "conversation": conversation,
    }
    output.write(json.dumps(record, ensure_ascii=False) + "\n")
    output.flush()


async def _run_model(
    client: AsyncOpenAI,
    output: TextIO,
    model: str,
    model_queue: list[tuple[int, str]],
    completed: Counter[AllocationKey],
) -> None:
    """Run one MiMo model with paced dispatch and bounded concurrency."""
    pending = _build_pending_queue(model, model_queue, completed)
    active: dict[asyncio.Task[list[dict[str, str]]], AllocationKey] = {}
    target_total = len(model_queue)
    completed_before_run = target_total - len(pending)
    completed_this_run = 0
    fatal_error: Exception | None = None
    next_launch_at = time.monotonic()
    last_telemetry_refresh = 0.0
    telemetry_width = 0
    telemetry_visible = False

    def refresh_telemetry(*, force: bool = False, newline: bool = False) -> None:
        """Display validated successes and the current active-request count."""
        nonlocal last_telemetry_refresh, telemetry_width, telemetry_visible

        now = time.monotonic()
        if not force and now - last_telemetry_refresh < 1.0:
            return

        total_complete = completed_before_run + completed_this_run
        line = (
            f"{model} | successes: {total_complete:,}/{target_total:,} "
            f"| active: {len(active):,}"
        )
        telemetry_width = max(telemetry_width, len(line))
        sys.stderr.write("\r" + line.ljust(telemetry_width))
        if newline:
            sys.stderr.write("\n")
            telemetry_visible = False
        else:
            telemetry_visible = True
        sys.stderr.flush()
        last_telemetry_refresh = now

    def print_event(message: str) -> None:
        """Print an event without overwriting the live telemetry line."""
        nonlocal telemetry_visible
        if telemetry_visible:
            sys.stderr.write("\n")
            telemetry_visible = False
        print(message, file=sys.stderr, flush=True)

    print(
        f"{model}: {completed_before_run:,}/{target_total:,} already complete; "
        f"{len(pending):,} pending.",
        file=sys.stderr,
        flush=True,
    )

    while pending or active:
        now = time.monotonic()
        can_launch = (
            fatal_error is None
            and bool(pending)
            and len(active) < MAX_ACTIVE_REQUESTS
        )

        if can_launch and now >= next_launch_at:
            key = pending.popleft()
            task = asyncio.create_task(_generate_conversation(client, key))
            active[task] = key
            next_launch_at = time.monotonic() + REQUEST_INTERVAL_SECONDS
            refresh_telemetry(force=last_telemetry_refresh == 0.0)
            continue

        refresh_telemetry(force=last_telemetry_refresh == 0.0)

        if not active:
            if can_launch:
                await asyncio.sleep(max(0.0, next_launch_at - time.monotonic()))
                continue
            break

        timeout: float | None = None
        if can_launch:
            timeout = max(0.0, next_launch_at - time.monotonic())

        done, _ = await asyncio.wait(
            active,
            timeout=timeout,
            return_when=asyncio.FIRST_COMPLETED,
        )

        for task in done:
            key = active.pop(task)
            try:
                conversation = task.result()
            except Exception as exc:
                if fatal_error is not None:
                    continue
                if _is_retriable(exc):
                    pending.append(key)
                    print_event(
                        f"{model}: retrying {key[:2]} after {exc}.",
                    )
                else:
                    fatal_error = exc
                    print_event(
                        f"{model}: stopping new requests after fatal error: "
                        f"{exc}"
                    )
            else:
                _write_record(output, key, conversation)
                completed_this_run += 1

    refresh_telemetry(force=True, newline=True)

    if fatal_error is not None:
        if isinstance(fatal_error, APIStatusError):
            if fatal_error.status_code == 402:
                raise RuntimeError(
                    f"{model} stopped because the MiMo balance ran out."
                ) from fatal_error
        raise RuntimeError(f"{model} stopped after a fatal API error.") from fatal_error


async def main() -> None:
    api_key = template._get_api_key()
    if not api_key:
        raise RuntimeError(
            "MIMO_API_KEY is not set in the process or permanent Windows "
            "user environment."
        )

    completed = _load_completed_counts()
    client = AsyncOpenAI(api_key=api_key, base_url=template.BASE_URL)

    try:
        with TRAINING_DATA.open("a", encoding="utf-8", newline="\n") as output:
            for model, model_queue in MODEL_RUNS:
                await _run_model(
                    client=client,
                    output=output,
                    model=model,
                    model_queue=model_queue,
                    completed=completed,
                )
    finally:
        await client.close()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nCancelled. Completed records are already saved.", file=sys.stderr)
        raise SystemExit(130) from None
    except Exception as exc:
        print(f"\nError: {exc}", file=sys.stderr)
        raise SystemExit(1) from None
