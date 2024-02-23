from kieval.datasets.multiple_choice import (
    MultipleChoiceDataset,
    AugmentedMultipleChoiceDataset,
    MultipleChoiceProblem,
)
from datasets import load_dataset

import logging


class MedMCQADataset(AugmentedMultipleChoiceDataset):
    def __init__(
        self,
        seed=1,
        split="validation",
        name_or_path=None,
        config_name=None,
        fewshot_split=None,
        fewshot_num=0,
        **kwargs
    ):
        super().__init__(seed=seed, **kwargs)
        self.name_or_path = "medmcqa" if name_or_path is None else name_or_path
        self.config_name = "MedMCQA" if config_name is None else config_name

        if fewshot_num:
            fewshot_dataset = load_dataset(
                self.name_or_path, name=self.config_name, split=fewshot_split
            )
            fewshot_examples = self.select_fewshot_examples(
                fewshot_dataset, fewshot_num, seed=seed
            )
            self.fewshot_examples = [
                self.parse_data_instance(e) for e in fewshot_examples
            ]

        self.hf_dataset = load_dataset(self.name_or_path, self.config_name, split=split)
        self.parse_hf_dataset()
        self.generate_prompt_text()

    def parse_data_instance(self, data, extra={}):
        problem_id = data["id"]
        question = data["question"]
        choices = [
            data["opa"],
            data["opb"],
            data["opc"],
            data["opd"],
        ]
        labels = ["A", "B", "C", "D"]
        answer = int(data["cop"])
        if "A" not in labels:
            labels = [chr(i + ord("A")) for i in range(len(labels))]
        return MultipleChoiceProblem(
            question,
            choices,
            answer,
            extra={"id": problem_id, **extra},
            generation_config={"stop_sequences": labels},
        )

    def parse_hf_dataset(self):
        for problem in self.hf_dataset:
            self.problems.append(self.parse_data_instance(problem))
