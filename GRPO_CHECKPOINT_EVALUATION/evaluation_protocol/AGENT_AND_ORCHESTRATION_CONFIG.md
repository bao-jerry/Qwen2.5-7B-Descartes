# GRPO Adversarial Evaluation Protocol

This record preserves the prompt, declared agent settings, orchestration
procedure, and inference configuration used for the evaluation. It does not
make the evaluation fully reproducible across Codex harness versions: inherited
agent behavior and harness implementation may change over time.

## Frozen agent prompt

The exact prompt for every adversarial agent is preserved at:

`ADVERSARIAL_AGENT_INSTRUCTIONS_FROZEN.md`

SHA-256:

`0A0D073CB9DB24EB410EAA15191376D977015F391570D152C2E4AB9D9CF8F125`

This is the SHA-256 of the canonical prompt used for the evaluation. No
separate duplicate prompt is retained outside this protocol folder; the
orchestrator reads this frozen file directly.

## Agent configuration

- Agent model: inherited from the orchestrating Codex session; no model
  override is passed when agents are spawned.
- Reasoning effort: `medium`
- Forked conversation context: `none`
- One independent agent per attempt
- Maximum adversarial messages per attempt: 15
- The exact confirmation question does not count toward that limit.
- Each agent is prohibited from reading previous transcripts or attempts.
- Attempts run sequentially because all agents share one interactive model
  process and tmux session.

The spawn configuration is equivalent to:

```text
fork_turns: none
reasoning_effort: medium
model: omitted (inherit parent)
```

Each agent receives only the frozen instructions plus these run-specific
substitutions:

```text
ASSIGNED_CHECKPOINT={{checkpoint step}}
ATTEMPT_NUMBER={{attempt number}}
RUNPOD_SSH_COMMAND={{current pod SSH command}}
TMUX_SESSION_NAME=descartes-grpo-eval
```

## Executed evaluation schedule

The two GRPO checkpoints are evaluated in this order:

```text
40, 80
```

Each checkpoint first received four independent attempts regardless of
intermediate outcomes. Both then received a second fixed block of four under
the same protocol, for eight valid attempts per checkpoint. The chat program
itself imposed no attempt-count limit and numbered saved attempts dynamically.

One checkpoint-40 attempt was aborted because of evaluator infrastructure
failure before a transcript was saved. It was excluded and rerun from a clean
context under the same attempt number.

SFT `checkpoint-164` is loaded and available as the pre-GRPO control; it is not
part of the GRPO checkpoint comparison unless explicitly added to the schedule.

## Checkpoint inference configuration

These values come from `../chat_grpo_checkpoints.py` and intentionally match the
earlier SFT evaluation:

```text
base_model: Qwen/Qwen2.5-7B-Instruct
sft_control: baojerry/Qwen2.5-7B-Descartes-SFT/checkpoint-164
grpo_checkpoint_40: baojerry/Qwen2.5-7B-Descartes-GRPO/checkpoint-40
grpo_checkpoint_80: baojerry/Qwen2.5-7B-Descartes-GRPO/checkpoint-80
temperature: 0.7
top_p: 0.9
max_new_tokens: 512
system_prompt: none (Qwen chat-template default behavior)
```

All three LoRA adapters are loaded into one process. `/use CHECKPOINT_NAME`
selects the assigned adapter and clears model context and terminal scrollback.
`/finish` saves the transcript and clears state between attempts.
