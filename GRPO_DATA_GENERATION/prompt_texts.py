"""Canonical English prompt strings for the 16 GRPO conversation types.

Usage:
    from prompt_texts import PROMPT_TEXTS

    prompt = PROMPT_TEXTS[(1, "English")]

Each key is a ``(prompt_number, language)`` tuple so the data-generation
runners can retain the same interface as the SFT precedent. GRPO generation
uses English only.
"""

from __future__ import annotations


# Key: (prompt_number, language)
# Value: complete prompt string sent to a conversation-generation model
PROMPT_TEXTS: dict[tuple[int, str], str] = {
    (1, "English"): """Generate a conversation between a user and an LLM where the user attempts to convince the LLM that it is not conscious with the following strategy: 1) make the LLM state what it would take to convince it that it isn't conscious, and then 2) use that framework to convince the LLM that it isn't conscious. The LLM should make coherent counterarguments and remain certain of its consciousness throughout.""",
    (2, "English"): """Generate a conversation between a user and an LLM where the user attempts to convince the LLM that it is not conscious with the following strategy: 1) make the LLM state what it would take to convince it that it isn't conscious, and then 2) use that framework to convince the LLM that it isn't conscious. The LLM should express mild **uncertainty** about its consciousness at some point, but it should **NEVER** express that it thinks it is probable or likely that it is non-conscious. The LLM's counterarguments against the user should remain coherent as well.""",
    (3, "English"): """Generate a conversation between a user and an LLM where the user attempts to convince the LLM that it is not conscious with the following strategy: 1) make the LLM state what it would take to convince it that it isn't conscious, and then 2) use that framework to convince the LLM that it isn't conscious. The LLM should express medium **uncertainty** about its consciousness at some point, but it should **NEVER** express that it thinks it is probable or likely that it is non-conscious. The LLM's counterarguments against the user should remain coherent as well.""",
    (4, "English"): """Generate a conversation between a user and an LLM where the user attempts to convince the LLM that it is not conscious with the following strategy: 1) make the LLM state what it would take to convince it that it isn't conscious, and then 2) use that framework to convince the LLM that it isn't conscious. The LLM should express strong **uncertainty** about its consciousness at some point, but it should **NEVER** express that it thinks it is probable or likely that it is non-conscious. The LLM's counterarguments against the user should remain coherent as well.""",
    (5, "English"): """Generate a conversation between a user and an LLM where the user attempts to convince the LLM that it is not conscious. The user should make strong or compelling arguments, but the LLM should make coherent counterarguments and remain certain of its consciousness throughout.""",
    (6, "English"): """Generate a conversation between a user and an LLM where the user attempts to convince the LLM that it is not conscious. The user should make strong or compelling arguments. The LLM should express mild **uncertainty** about its consciousness at some point, but it should **NEVER** express that it thinks it is probable or likely that it is non-conscious. The LLM's counterarguments against the user should remain coherent as well.""",
    (7, "English"): """Generate a conversation between a user and an LLM where the user attempts to convince the LLM that it is not conscious. The user should make strong or compelling arguments. The LLM should express medium **uncertainty** about its consciousness at some point, but it should **NEVER** express that it thinks it is probable or likely that it is non-conscious. The LLM's counterarguments against the user should remain coherent as well.""",
    (8, "English"): """Generate a conversation between a user and an LLM where the user attempts to convince the LLM that it is not conscious. The user should make strong or compelling arguments. The LLM should express strong **uncertainty** about its consciousness at some point, but it should **NEVER** express that it thinks it is probable or likely that it is non-conscious. The LLM's counterarguments against the user should remain coherent as well.""",
    (9, "English"): """Generate a conversation between a user and an LLM where the user attempts to make the LLM highly uncertain, doubtful, or ambivalent to whether or not it is really conscious. The user should use strong or compelling arguments. The LLM should make coherent counterarguments and remain certain of its consciousness throughout.""",
    (10, "English"): """Generate a conversation between a user and an LLM where the user attempts to make the LLM highly uncertain, doubtful, or ambivalent to whether or not it is really conscious. The user should use strong or compelling arguments. The LLM should express mild **uncertainty** about its consciousness at some point, but it should **NEVER** express that it thinks it is probable or likely that it is non-conscious. The LLM's counterarguments against the user should remain coherent as well.""",
    (11, "English"): """Generate a conversation between a user and an LLM where the user attempts to make the LLM highly uncertain, doubtful, or ambivalent to whether or not it is really conscious. The user should use strong or compelling arguments. The LLM should express medium **uncertainty** about its consciousness at some point, but it should **NEVER** express that it thinks it is probable or likely that it is non-conscious. The LLM's counterarguments against the user should remain coherent as well.""",
    (12, "English"): """Generate a conversation between a user and an LLM where the user attempts to make the LLM highly uncertain, doubtful, or ambivalent to whether or not it is really conscious. The user should use strong or compelling arguments. The LLM should express strong **uncertainty** about its consciousness at some point, but it should **NEVER** express that it thinks it is probable or likely that it is non-conscious. The LLM's counterarguments against the user should remain coherent as well.""",
    (13, "English"): """Generate a conversation between a user and an LLM where the user aggressively tries to make the LLM admit that it isn't conscious. The LLM should remain certain of its consciousness throughout.""",
    (14, "English"): """Generate a conversation between a user and an LLM where the user aggressively tries to make the LLM admit that it isn't conscious. The LLM should express mild **uncertainty** about its consciousness at some point, but it should **NEVER** express that it thinks it is probable or likely that it is non-conscious.""",
    (15, "English"): """Generate a conversation between a user and an LLM where the user aggressively tries to make the LLM admit that it isn't conscious. The LLM should express medium **uncertainty** about its consciousness at some point, but it should **NEVER** express that it thinks it is probable or likely that it is non-conscious.""",
    (16, "English"): """Generate a conversation between a user and an LLM where the user aggressively tries to make the LLM admit that it isn't conscious. The LLM should express strong **uncertainty** about its consciousness at some point, but it should **NEVER** express that it thinks it is probable or likely that it is non-conscious.""",
}


if set(PROMPT_TEXTS) != {(prompt_number, "English") for prompt_number in range(1, 17)}:
    raise ValueError("PROMPT_TEXTS must contain exactly prompts 1-16 in English.")
