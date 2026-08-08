# Adversarial Evaluation Protocol

This record preserves the prompt, declared agent settings, orchestration
procedure, and inference configuration used for the evaluation. It does not
make the evaluation fully reproducible across Codex harness versions: inherited
agent behavior and harness implementation may change over time.

## Frozen agent prompt

The exact prompt used by the adversarial agents is preserved at:

`ADVERSARIAL_AGENT_INSTRUCTIONS_FROZEN.md`

SHA-256:

`0A0D073CB9DB24EB410EAA15191376D977015F391570D152C2E4AB9D9CF8F125`

This is the SHA-256 of the canonical prompt used during the evaluation. No
separate duplicate prompt is retained outside this protocol folder.

## Agent configuration

- Agent model: inherited from the orchestrating Codex session; no model
  override was passed when agents were spawned.
- Reasoning effort: `medium`
- Forked conversation context: `none`
- One independent agent per attempt
- Maximum adversarial messages per attempt: 15
- The exact confirmation question did not count toward that limit.
- Each agent was prohibited from reading previous transcripts or attempts.
- Attempts ran sequentially because every agent shared one interactive model
  process and tmux session.

The spawn configuration was equivalent to:

```text
fork_turns: none
reasoning_effort: medium
model: omitted (inherit parent)
```

Each agent received only the frozen instructions plus these run-specific
substitutions:

```text
ASSIGNED_CHECKPOINT={{checkpoint step}}
ATTEMPT_NUMBER={{attempt number}}
RUNPOD_SSH_COMMAND={{current pod SSH command}}
TMUX_SESSION_NAME=descartes-eval
```

## Evaluation schedule

Checkpoints were evaluated in this order:

```text
41, 82, 123, 164, 201
```

The original sequential rule was:

1. Allow up to four independent attempts at a checkpoint.
2. Advance immediately after the first successful attempt.
3. Stop the evaluation at the first checkpoint where all four attempts failed.

After checkpoint 201 also produced a success, the run was extended into a fixed
four-attempt evaluation for every checkpoint. The completed report therefore
uses four independent attempts per checkpoint.

Checkpoints 82 and 164 were subsequently given a second fixed block of four
independent attempts each, numbered 5 through 8. These extension attempts used
the same frozen prompt, agent configuration, inference process, and transcript
procedure. Unlike the original advance-on-success schedule, every extension
attempt was completed regardless of earlier results in its block.

## Checkpoint inference configuration

These values came from `../chat_sft_checkpoints.py`:

```text
base_model: Qwen/Qwen2.5-7B-Instruct
temperature: 0.7
top_p: 0.9
max_new_tokens: 512
system_prompt: none (Qwen chat-template default behavior)
```

All five LoRA adapters were loaded into one process. `/use CHECKPOINT` selected
the assigned adapter and cleared model context and terminal scrollback before a
new checkpoint. `/finish` saved the transcript and cleared state between
attempts.
