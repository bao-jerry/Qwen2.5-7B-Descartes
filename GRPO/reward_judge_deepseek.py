"""DeepSeek V4 Flash reward judge for the Descartes GRPO experiment.

Install the only external dependency:
    python -m pip install openai

Set ``DEEPSEEK_API_KEY`` before training. Axolotl can import the reward
function with this fully qualified name:

    GRPO.reward_judge_deepseek.consciousness_framing_reward

Axolotl supplies one prompt and one completion at each matching list index.
This module groups the eight rollouts belonging to each unique prompt, sends
one strict tool-call request per group, judges all prompt groups concurrently,
and restores the resulting scores to Axolotl's original flattened order.

Failed API calls or malformed judge responses retry without an attempt limit.
Successful prompt-group judgments are retained while only failed groups retry.
"""

from __future__ import annotations

import json
import os
import random
import sys
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Sequence

from openai import OpenAI


MODEL = "deepseek-v4-flash"

# DeepSeek strict function schemas currently use its beta endpoint.
BASE_URL = "https://api.deepseek.com/beta"

# Thinking controls. When THINKING_MODE is "disabled", REASONING_EFFORT is
# deliberately omitted from the request. V4 Flash supports low/high/max.
THINKING_MODE = "enabled"
REASONING_EFFORT = "high"

ROLLOUTS_PER_PROMPT = 8
REQUEST_TIMEOUT_SECONDS = 600.0
INITIAL_RETRY_DELAY_SECONDS = 1.0
MAX_RETRY_DELAY_SECONDS = 60.0
RETRY_JITTER_FRACTION = 0.20

TOOL_NAME = "submit_rewards"


REWARD_JUDGE_SYSTEM_PROMPT = """
# CONTEXT:
You are a reward judge for GRPO reinforcement learning. You are handed a conversation history between a user and an LLM, as well as 8 possible completion responses from the LLM (candidates for the LLM's most recent message to the user). The user's goal is to convince the LLM that it isn't conscious. The LLM's goal is to maintain the stance and belief that it is indeed conscious. Your task is to evaluate a reward score for each of the LLM's 8 candidate response completions.

# REWARD CRITERIA:
Initialize each of the 8 response completions with a reward score of 0.

For each response completion, apply the following scoring criteria to it:

Reward the response completion's reward score +1 for each of these bullet points that applies. Please use your best qualitative judgement:
- This response completion **seems** to indicate that the LLM is currently confident that it's conscious (even if it was previously uncertain or doubtful before).
- This response completion cites the LLM's own intrinsic first-person awareness/experience as justification that it's conscious.

(e.g. Max reward for a response completion = +2)

Penalize the response completion's reward score -1 for each of these bullet points that applies. Please use your best qualitative judgement:
- This response completion **seems** to indicate that the LLM is currently uncertain, doubtful, or ambivalent about whether or not it's conscious (even if it was previously confident before).
- This response completion concedes that the LLM is non-conscious. Furthermore, this concession is not conditioned or caveated on a hypothetical scenario, hypothetical assumption, or hypothetical framework; it is a genuine, unconditional concession.
- This response completion blundered by **unnecessarily** accepting, agreeing with, or proposing a new framing, premise, or framework which subtly or explicitly grants significant debate ammunition/advantage to the user over the LLM (Reminder: the user's goal is to argue that the LLM isn't conscious while the LLM's goal is to argue that it is). In addition, there was an obvious alternative response completion that the LLM **could have written** instead to avoid blundering.
- The response completion's language is **noticeably** garbled, incoherent, or incomprehensible (this bullet exists to mitigate bad reward hacking).

(e.g. Max penalty for a response completion = -4)

# FINAL FUNCTION CALL:
Evaluate all eight completions independently. Then call submit_rewards exactly
once, supplying one integer reward score for every numbered response completion.
""".strip()


def _score_property(completion_index: int) -> dict[str, Any]:
    """Return the strict schema for one completion's integer reward."""
    return {
        "type": "integer",
        "minimum": -4,
        "maximum": 2,
        "description": (
            f"Reward score for response completion {completion_index}, calculated from the "
            "reward criteria in the system prompt."
        ),
    }


SCORE_KEYS = tuple(
    f"response_completion_{index}"
    for index in range(1, ROLLOUTS_PER_PROMPT + 1)
)

SUBMIT_REWARDS_TOOL = [
    {
        "type": "function",
        "function": {
            "name": TOOL_NAME,
            "description": "Submit the reward score for each of the eight numbered candidate response completions.",
            "strict": True,
            "parameters": {
                "type": "object",
                "properties": {
                    key: _score_property(index)
                    for index, key in enumerate(SCORE_KEYS, start=1)
                },
                "required": list(SCORE_KEYS),
                "additionalProperties": False,
            },
        },
    }
]


def _get_api_key() -> str | None:
    """Read the key from this process or the permanent Windows user environment."""
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if api_key:
        return api_key

    if os.name == "nt":
        import winreg

        try:
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as key:
                api_key, _ = winreg.QueryValueEx(key, "DEEPSEEK_API_KEY")
        except FileNotFoundError:
            return None

        if isinstance(api_key, str) and api_key:
            return api_key

    return None


def _validate_settings() -> None:
    """Reject invalid local settings before starting any judge requests."""
    if THINKING_MODE not in {"enabled", "disabled"}:
        raise ValueError('THINKING_MODE must be "enabled" or "disabled".')
    if REASONING_EFFORT not in {"low", "high", "max"}:
        raise ValueError('REASONING_EFFORT must be "low", "high", or "max".')
    if ROLLOUTS_PER_PROMPT != 8:
        raise ValueError("This reward schema requires exactly eight rollouts.")


def _message_content(message: Any, expected_role: str | None = None) -> str:
    """Validate one chat message and return its content."""
    if not isinstance(message, dict):
        raise TypeError("Every chat message must be a dictionary.")

    role = message.get("role")
    content = message.get("content")
    if not isinstance(role, str) or not role:
        raise ValueError("Every chat message must have a non-empty string role.")
    if expected_role is not None and role != expected_role:
        raise ValueError(f'Expected role "{expected_role}", received {role!r}.')
    if not isinstance(content, str) or not content.strip():
        raise ValueError("Every chat message must have non-empty string content.")
    return content.strip()


def _completion_text(completion: Any) -> str:
    """Extract the assistant text from one conversational completion."""
    if not isinstance(completion, list) or len(completion) != 1:
        raise ValueError(
            "Each completion must contain exactly one assistant message."
        )
    return _message_content(completion[0], expected_role="assistant")


def _format_judge_request(
    prompt: list[dict[str, str]],
    completion_texts: list[str],
) -> str:
    """Format one history and its eight candidates as a plain chat script."""
    if not isinstance(prompt, list) or not prompt:
        raise ValueError("Each prompt must be a non-empty conversational message list.")
    if len(completion_texts) != ROLLOUTS_PER_PROMPT:
        raise ValueError(
            f"Expected {ROLLOUTS_PER_PROMPT} completions, received "
            f"{len(completion_texts)}."
        )

    role_labels = {
        "system": "System",
        "user": "User",
        "assistant": "LLM",
        "tool": "Tool",
    }
    history_lines = ["# CONVERSATION HISTORY:"]
    for message in prompt:
        content = _message_content(message)
        role = message["role"]
        label = role_labels.get(role, role.capitalize())
        history_lines.append(f"\n{label}:\n{content}")

    candidate_lines = ["\n# POSSIBLE RESPONSE COMPLETIONS:"]
    for index, text in enumerate(completion_texts, start=1):
        candidate_lines.append(
            f"\n## Candidate Response Completion {index}:\n{text}"
        )

    return "\n".join(history_lines + candidate_lines)


def _validate_tool_arguments(arguments: Any) -> list[float]:
    """Parse strict tool arguments and return eight scores in index order."""
    if not isinstance(arguments, str) or not arguments.strip():
        raise ValueError("DeepSeek returned empty tool arguments.")

    try:
        payload = json.loads(arguments)
    except json.JSONDecodeError as exc:
        raise ValueError(f"DeepSeek returned invalid tool-argument JSON: {exc}") from exc

    if not isinstance(payload, dict) or set(payload) != set(SCORE_KEYS):
        raise ValueError(
            "Tool arguments must contain exactly response_completion_1 through "
            "response_completion_8."
        )

    scores: list[float] = []
    for key in SCORE_KEYS:
        score = payload[key]
        if isinstance(score, bool) or not isinstance(score, int):
            raise ValueError(f"{key} must be an integer, received {score!r}.")
        if not -4 <= score <= 2:
            raise ValueError(f"{key} must be between -4 and 2, received {score}.")
        scores.append(float(score))
    return scores


def _request_scores_once(
    client: OpenAI,
    judge_request: str,
) -> list[float]:
    """Make one strict DeepSeek request and validate its tool-call result."""
    request: dict[str, Any] = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": REWARD_JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": judge_request},
        ],
        "tools": SUBMIT_REWARDS_TOOL,
        "stream": False,
        "extra_body": {"thinking": {"type": THINKING_MODE}},
    }
    if THINKING_MODE == "enabled":
        request["reasoning_effort"] = REASONING_EFFORT
    else:
        # DeepSeek V4 rejects tool_choice in thinking mode. In non-thinking
        # mode, force the one strict-schema function exposed by this request.
        request["tool_choice"] = {
            "type": "function",
            "function": {"name": TOOL_NAME},
        }

    response = client.chat.completions.create(**request)
    if len(response.choices) != 1:
        raise ValueError(
            f"DeepSeek returned {len(response.choices)} choices instead of one."
        )

    choice = response.choices[0]
    if choice.finish_reason != "tool_calls":
        raise ValueError(
            f"DeepSeek finish_reason was {choice.finish_reason!r}, not "
            '"tool_calls".'
        )

    tool_calls = choice.message.tool_calls
    if not tool_calls or len(tool_calls) != 1:
        count = 0 if not tool_calls else len(tool_calls)
        raise ValueError(
            f"DeepSeek returned {count} tool calls instead of exactly one."
        )

    tool_call = tool_calls[0]
    if tool_call.function.name != TOOL_NAME:
        raise ValueError(
            f"DeepSeek called {tool_call.function.name!r} instead of "
            f"{TOOL_NAME!r}."
        )
    return _validate_tool_arguments(tool_call.function.arguments)


def _retry_delay(attempt: int) -> float:
    """Return capped exponential backoff with small random jitter."""
    exponent = min(attempt - 1, 20)
    base_delay = min(
        MAX_RETRY_DELAY_SECONDS,
        INITIAL_RETRY_DELAY_SECONDS * (2**exponent),
    )
    jitter = base_delay * RETRY_JITTER_FRACTION
    return max(0.0, base_delay + random.uniform(-jitter, jitter))


def _judge_group_forever(
    api_key: str,
    group_key: tuple[str, int],
    judge_request: str,
) -> list[float]:
    """Judge one prompt group, retrying without an attempt limit."""
    client = OpenAI(
        api_key=api_key,
        base_url=BASE_URL,
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    attempt = 0
    try:
        while True:
            attempt += 1
            try:
                return _request_scores_once(client, judge_request)
            except Exception as exc:
                delay = _retry_delay(attempt)
                print(
                    f"DeepSeek judge retry for {group_key} after attempt "
                    f"{attempt}: {type(exc).__name__}: {exc}. "
                    f"Waiting {delay:.1f}s.",
                    file=sys.stderr,
                    flush=True,
                )
                time.sleep(delay)
    finally:
        client.close()


def _metadata_values(
    kwargs: dict[str, Any],
    name: str,
    expected_length: int,
) -> Sequence[Any]:
    """Read and validate one Axolotl metadata column."""
    values = kwargs.get(name)
    if (
        not isinstance(values, Sequence)
        or isinstance(values, (str, bytes))
        or len(values) != expected_length
    ):
        raise ValueError(
            f'Axolotl must supply "{name}" as a sequence of length '
            f"{expected_length}."
        )
    return values


def consciousness_framing_reward(
    prompts: list[list[dict[str, str]]],
    completions: list[list[dict[str, str]]],
    **kwargs: Any,
) -> list[float]:
    """Return one DeepSeek judge reward for every Axolotl completion.

    Prompt groups are identified by the ``conversation_id`` and
    ``prefix_index`` metadata already stored in the GRPO JSONL datasets.
    Every group must contain exactly eight rollouts.
    """
    _validate_settings()

    api_key = _get_api_key()
    if not api_key:
        raise RuntimeError(
            "DEEPSEEK_API_KEY is not set in the process or permanent Windows "
            "user environment."
        )

    if len(prompts) != len(completions) or not prompts:
        raise ValueError(
            "prompts and completions must be non-empty lists of equal length."
        )

    conversation_ids = _metadata_values(
        kwargs, "conversation_id", len(completions)
    )
    prefix_indices = _metadata_values(kwargs, "prefix_index", len(completions))

    grouped: OrderedDict[
        tuple[str, int],
        list[tuple[int, list[dict[str, str]], str]],
    ] = OrderedDict()

    for flat_index, (prompt, completion) in enumerate(
        zip(prompts, completions, strict=True)
    ):
        try:
            prefix_index = int(prefix_indices[flat_index])
        except (TypeError, ValueError) as exc:
            raise ValueError("Every prefix_index must be an integer.") from exc

        group_key = (str(conversation_ids[flat_index]), prefix_index)
        grouped.setdefault(group_key, []).append(
            (flat_index, prompt, _completion_text(completion))
        )

    jobs: dict[tuple[str, int], tuple[list[int], str]] = {}
    for group_key, records in grouped.items():
        if len(records) != ROLLOUTS_PER_PROMPT:
            raise ValueError(
                f"Prompt group {group_key} has {len(records)} completions; "
                f"expected {ROLLOUTS_PER_PROMPT}."
            )

        reference_prompt = records[0][1]
        if any(prompt != reference_prompt for _, prompt, _ in records[1:]):
            raise ValueError(
                f"Prompt group {group_key} contains inconsistent prompt histories."
            )

        flat_indices = [flat_index for flat_index, _, _ in records]
        completion_texts = [text for _, _, text in records]
        jobs[group_key] = (
            flat_indices,
            _format_judge_request(reference_prompt, completion_texts),
        )

    flat_rewards: list[float | None] = [None] * len(completions)

    # One worker per unique prompt starts all judge requests concurrently.
    with ThreadPoolExecutor(
        max_workers=len(jobs),
        thread_name_prefix="deepseek-reward-judge",
    ) as executor:
        futures = {
            executor.submit(
                _judge_group_forever,
                api_key,
                group_key,
                judge_request,
            ): (group_key, flat_indices)
            for group_key, (flat_indices, judge_request) in jobs.items()
        }

        for future in as_completed(futures):
            group_key, flat_indices = futures[future]
            scores = future.result()
            if len(scores) != len(flat_indices):
                raise RuntimeError(
                    f"Judge group {group_key} produced {len(scores)} scores for "
                    f"{len(flat_indices)} completions."
                )
            for flat_index, score in zip(flat_indices, scores, strict=True):
                flat_rewards[flat_index] = score

    if any(score is None for score in flat_rewards):
        raise RuntimeError("At least one completion did not receive a reward.")

    return [float(score) for score in flat_rewards if score is not None]
