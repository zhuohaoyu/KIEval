from kieval.datasets.multiple_choice import (
    MultipleChoiceDataset,
    AugmentedMultipleChoiceDataset,
    MultipleChoiceProblem,
)
from datasets import load_dataset

import logging


class GSM8KDataset(AugmentedMultipleChoiceDataset):
    def __init__(
        self, seed=1, split="test", name_or_path=None, config_name=None, **kwargs
    ):
        super().__init__(seed=seed, **kwargs)
        self.name_or_path = "gsm8k" if name_or_path is None else name_or_path
        self.config_name = "GSM8K" if config_name is None else config_name

        self.hf_dataset = load_dataset(self.name_or_path, self.config_name, split=split)
        self.parse_hf_dataset()
        self.generate_prompt_text()

    def parse_data_instance(self, data, extra={}):
        question = data["question"]
        choices = [data["answer"]]
        labels = [0]
        answer = 0
        if "A" not in labels:
            labels = [chr(i + ord("A")) for i in range(len(labels))]
        return MultipleChoiceProblem(
            question,
            choices,
            answer,
            extra=extra,
            generation_config={"stop_sequences": labels},
        )

    def parse_hf_dataset(self):
        for problem in self.hf_dataset:
            self.problems.append(self.parse_data_instance(problem))
