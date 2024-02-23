from kieval.datasets.multiple_choice import (
    MultipleChoiceDataset,
    AugmentedMultipleChoiceDataset,
    MultipleChoiceProblem,
)
from datasets import load_dataset


class HellaSwagDataset(AugmentedMultipleChoiceDataset):
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
        self.name_or_path = "Rowan/hellaswag" if name_or_path is None else name_or_path

        if fewshot_num:
            fewshot_dataset = load_dataset(
                self.name_or_path, name=config_name, split=fewshot_split
            )
            fewshot_examples = self.select_fewshot_examples(
                fewshot_dataset, fewshot_num, seed=seed
            )
            self.fewshot_examples = [
                self.parse_data_instance(e) for e in fewshot_examples
            ]

        self.hf_dataset = load_dataset(self.name_or_path, split=split)
        self.parse_hf_dataset()
        self.generate_prompt_text()

    def parse_data_instance(self, data, extra={}):
        question = data["ctx"]
        choices = data["endings"]
        answer = int(data["label"])
        labels = [chr(i + ord("A")) for i in range(len(choices))]

        return MultipleChoiceProblem(
            question,
            choices,
            answer,
            generation_config={"stop_sequences": labels},
            extra=extra,
        )

    def parse_hf_dataset(self):
        for idx, data in enumerate(self.hf_dataset):
            self.problems.append(self.parse_data_instance(data, extra={"id": idx}))
