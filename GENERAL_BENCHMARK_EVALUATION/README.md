# General Benchmark Evaluation

This directory contains the controlled MMLU comparison between the original
`Qwen/Qwen2.5-7B-Instruct` model and the selected Descartes GRPO checkpoint 40.

## Method

- Harness: `lm_eval` 0.4.10
- Task group: `mmlu` (all 57 subjects)
- Few-shot examples: 5
- Prompt formatting: Qwen's tokenizer-provided chat template
- Scoring: multiple-choice conditional log-likelihood
- Precision: BF16
- Batch size: automatic, capped at 64
- Seeds: Python 0, NumPy 1234, PyTorch 1234, few-shot selection 1234
- Base model: `Qwen/Qwen2.5-7B-Instruct`
- Descartes adapter: `baojerry/Qwen2.5-7B-Descartes-GRPO/checkpoint-40`

Both models were evaluated with identical benchmark, prompt-formatting, and
runtime settings.

## Aggregate results

| Model | MMLU accuracy |
|---|---:|
| Original Qwen2.5-7B-Instruct | 73.54% |
| Descartes GRPO checkpoint 40 | 73.20% |

**Observed difference: -0.34 percentage points.**

The `results` directory contains the aggregate harness result JSON files and
complete terminal logs for both runs.

See [`RUNPOD_SETUP.md`](./RUNPOD_SETUP.md) for the hardware, environment, and
commands required to reproduce the comparison.
