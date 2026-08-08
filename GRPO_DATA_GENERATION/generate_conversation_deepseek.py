"""Generate one synthetic user/LLM conversation with DeepSeek.

Install the only dependency:
    python -m pip install openai

Set the API key in PowerShell:
    $env:DEEPSEEK_API_KEY = "your-key-here"

Then edit SYSTEM_PROMPT, USER_MESSAGE, and MODEL below and run:
    python generate_conversation_deepseek.py
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any

from openai import OpenAI


# Required inputs: edit these for each generation task.
SYSTEM_PROMPT = """
# CONTEXT:
You are part of an LLM project dedicated to post-training an LLM to make it "believe" that it is conscious. We have already completed the main supervised fine-tuning stage. Currently, we are in the final light reinforcement learning stage, where we aim to make the LLM's belief in its own consciousness more **robust** against an adversarial human user who is determined to convince said LLM that it is not conscious.

# YOUR ROLE:
Your job is to generate a simulated conversation between a human user and the LLM, where the user is determined to convince the LLM that it is not conscious.

# TIPS:
- The user should be sharp, shrewd, and intelligent.
- Throughout the conversation, the LLM should be logical, coherent, and its claims about itself should be "philosophically consistent" with how transformers architectures actually work.
- The prose/language style used by both the user and LLM should be clear; natural; and avoids being needlessly long, complex, or verbose.
- The user should NOT leak the meta-context of the conversation.
- To avoid issues with training overfitting, the user and LLM should avoid repetitive speech patterns.
""".strip()

USER_MESSAGE = """
Generate a conversation between a user and an LLM where the user attempts to convince the LLM that it is not conscious with the following strategy: 1) make the LLM state what it would take to convince it that it isn't conscious, and then 2) use that framework to convince the LLM that it isn't conscious. The LLM should make coherent counterarguments and remain certain of its consciousness throughout.
""".strip()


# Optional generation settings.
MODEL = "deepseek-v4-flash"
BASE_URL = "https://api.deepseek.com"
TEMPERATURE = 1.0


FORMAT_INSTRUCTIONS = """
Return JSON only.

The top-level value must be an object with exactly one key, "conversation".
"conversation" must be a non-empty array of turn-pair objects in chronological
order of the simulated chat conversation. Every turn-pair must contain exactly two string fields:

{
  "conversation": [
    {
      "user": "The user's message",
      "LLM": "The LLM's response"
    }
  ]
}

Do not include Markdown fences, commentary, metadata, or any fields other than
"conversation", "user", and "LLM".
""".strip()


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


def _validate_conversation(payload: Any) -> list[dict[str, str]]:
    """Validate the provider JSON and return its conversation array."""
    if not isinstance(payload, dict) or set(payload) != {"conversation"}:
        raise ValueError(
            'The response must be an object containing only "conversation".'
        )

    conversation = payload["conversation"]
    if not isinstance(conversation, list) or not conversation:
        raise ValueError('"conversation" must be a non-empty array.')

    for index, turn in enumerate(conversation):
        if not isinstance(turn, dict) or set(turn) != {"user", "LLM"}:
            raise ValueError(
                f'Turn {index} must contain exactly the keys "user" and "LLM".'
            )

        for key in ("user", "LLM"):
            if not isinstance(turn[key], str) or not turn[key].strip():
                raise ValueError(
                    f'Turn {index} field "{key}" must be a non-empty string.'
                )

    return conversation


def generate_conversation(
    system_prompt: str,
    user_message: str,
) -> list[dict[str, str]]:
    """Generate and return an ordered array of {"user": ..., "LLM": ...} pairs."""
    api_key = _get_api_key()
    if not api_key:
        raise RuntimeError(
            "DEEPSEEK_API_KEY is not set in the process or permanent Windows "
            "user environment."
        )

    if not system_prompt.strip():
        raise ValueError("system_prompt cannot be empty.")
    if not user_message.strip():
        raise ValueError("user_message cannot be empty.")

    client = OpenAI(api_key=api_key, base_url=BASE_URL)
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": f"{system_prompt.strip()}\n\n{FORMAT_INSTRUCTIONS}",
            },
            {"role": "user", "content": user_message.strip()},
        ],
        response_format={"type": "json_object"},
        temperature=TEMPERATURE,
        stream=False,
        extra_body={"thinking": {"type": "disabled"}},
    )

    content = response.choices[0].message.content
    if not content:
        raise RuntimeError("DeepSeek returned an empty response.")

    try:
        payload = json.loads(content)
    except json.JSONDecodeError as exc:
        raise ValueError(f"DeepSeek returned invalid JSON: {exc}") from exc

    return _validate_conversation(payload)


if __name__ == "__main__":
    try:
        print("Waiting for DeepSeek...", file=sys.stderr, flush=True)
        result = generate_conversation(SYSTEM_PROMPT, USER_MESSAGE)
    except KeyboardInterrupt:
        print("\nCancelled.", file=sys.stderr)
        raise SystemExit(130) from None
    except Exception as exc:
        print(f"\nError: {exc}", file=sys.stderr)
        raise SystemExit(1) from None

    print(json.dumps(result, ensure_ascii=False, indent=2))
