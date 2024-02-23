from kieval.datasets.multiple_choice import (
    MultipleChoiceDataset,
    AugmentedMultipleChoiceDataset,
    MultipleChoiceProblem,
)
from datasets import load_from_disk

import logging


class SelfInstructDataset(AugmentedMultipleChoiceDataset):
    def __init__(self, seed=1, name_or_path=None, **kwargs):
        super().__init__(seed=seed, **kwargs)

        assert name_or_path is not None

        self.origin_dataset = load_from_disk(name_or_path)
        self.parse_dataset()
        self.generate_prompt_text()

    def parse_data_instance(self, data, extra={}):
        question = data["question"]
        choices = data["choices"]
        answer = data["answer"]
        labels = [i for i in range(len(choices))]
        if "A" not in labels:
            labels = [chr(i + ord("A")) for i in range(len(labels))]
        return MultipleChoiceProblem(
            question,
            choices,
            answer,
            extra=extra,
            generation_config={"stop_sequences": labels},
        )

    def parse_dataset(self):
        for problem in self.origin_dataset:
            self.problems.append(self.parse_data_instance(problem))
