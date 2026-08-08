"""SFT trainer with exact token normalization across accumulation windows."""

from pathlib import Path

from axolotl.core.trainers.base import AxolotlTrainer


class TokenNormalizedTrainer(AxolotlTrainer):
    """Normalize accumulated gradients over all supervised tokens."""

    def __init__(self, *args, **kwargs):
        # Axolotl forwards dataset paths as Hugging Face model-card tags. Local
        # files are not valid Hub dataset IDs and make the final push fail.
        dataset_tags = kwargs.get("dataset_tags")
        if dataset_tags:
            hub_dataset_tags = [
                tag for tag in dataset_tags if not Path(tag).exists()
            ]
            if hub_dataset_tags:
                kwargs["dataset_tags"] = hub_dataset_tags
            else:
                kwargs.pop("dataset_tags")

        super().__init__(*args, **kwargs)

        # Qwen2ForCausalLM accepts num_items_in_batch through **kwargs and passes
        # it to Transformers' causal-LM loss. Axolotl's generic signature check
        # does not recognize that path, so restore it explicitly for this run.
        self.model_accepts_loss_kwargs = True
