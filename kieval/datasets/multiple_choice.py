from abc import ABC, abstractmethod
from shortuuid import uuid
from typing import Optional, List, Dict, Union
from random import Random
import json, logging
from hashlib import md5
from base64 import urlsafe_b64encode
from dataclasses import dataclass
from tqdm import tqdm


from kieval.prompts import MULTIPLE_CHOICE_CHOICES_TMPL, PromptPostprocessor


class PermutationGenerator:
    def __init__(self, seed) -> None:
        self.random = Random(seed)

    def generate_permutation(self, n, keep_ans=None):
        perm = [_ for _ in range(n)]
        self.random.shuffle(perm)
        if keep_ans is not None and perm[keep_ans] != keep_ans:
            pos = perm.index(keep_ans)
            perm[pos], perm[keep_ans] = perm[keep_ans], perm[pos]
        return perm


class MultipleChoiceProblem(ABC):
    # a base class for a single multiple choice problem

    def __init__(
        self,
        problem: (
            str | None
        ),  # we set problem to None for cloze prompt unconditional normalisation
        choices: List[str],
        answer: int | str,  # answer can be a string for cloze prompt
        parent_uuid: Optional[str] = None,
        permutation: Optional[List[int]] = None,
        generation_config: Optional[Dict] = None,
        choice_key_type: Optional[str] = "letters",
        extra: Optional[dict] = None,
    ):
        # assert isinstance(answer, int) and answer < len(choices)
        self.problem = problem
        self.choices = choices
        self.answer = answer
        self.permutation = permutation
        self.extra = extra
        self.uuid = self.hash()
        self.parent_uuid = parent_uuid
        self.generation_config = generation_config
        self.prediction_index = None

        assert choice_key_type in ["letters", "numbers"]
        self.choice_key_type = choice_key_type

        self.prompt = None

    def __hash__(self) -> int:
        hashstr = (
            str(self.problem)
            + "$choices$"
            + ";".join(self.choices)
            + "$ans$"
            + str(self.answer)
            + "$perm$"
            + str(self.permutation)
            + "$ex$"
            + json.dumps(self.extra)
        )
        hashstr = hashstr.encode("utf-8")
        hash_digest = md5(hashstr).digest()
        hash_value = int.from_bytes(hash_digest, byteorder="big")

        return hash_value

    def hash(self):
        hashstr = (
            str(self.problem)
            + "$choices$"
            + ";".join(self.choices)
            + "$ans$"
            + str(self.answer)
            + "$perm$"
            + str(self.permutation)
            + "$ex$"
            + json.dumps(self.extra)
        )
        hashstr = hashstr.encode("utf-8")
        hash_digest = md5(hashstr).digest()

        url_safe_hash = urlsafe_b64encode(hash_digest).rstrip(b"=").decode("utf-8")
        return url_safe_hash

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def generate_choices_text(self, choices_prompt_name="default"):
        map_func = (
            lambda x: chr(x + ord("A"))
            if self.choice_key_type == "letters"
            else str(x + 1)
        )

        choice_keys = [map_func(i) for i in range(len(self.choices))]

        if self.permutation:
            choices = [self.choices[i] for i in self.permutation]
        else:
            choices = self.choices

        tmpl = MULTIPLE_CHOICE_CHOICES_TMPL[choices_prompt_name]

        choices_text = "".join(
            tmpl.format(choice_key=key, choice_text=text)
            for key, text in zip(choice_keys, choices)
        )

        return choices_text

    def generate_output_text(self, choices_prompt_name="default"):
        map_func = (
            lambda x: (
                chr(x + ord("A")) if self.choice_key_type == "letters" else str(x + 1)
            )
            if type(x) == int
            else x
        )

        if self.permutation:
            answer_text = map_func(self.permutation[self.answer])
        else:
            answer_text = map_func(self.answer)
        
        if isinstance(self.answer, str):
            choice_text = self.answer
        else:
            choice_text = self.choices[self.answer]

        return MULTIPLE_CHOICE_CHOICES_TMPL[choices_prompt_name].format(choice_key=answer_text, choice_text=choice_text)

    def generate_prediction_output_text(self, choices_prompt_name="default"):
        map_func = (
            lambda x: (
                chr(x + ord("A")) if self.choice_key_type == "letters" else str(x + 1)
            )
            if type(x) == int
            else x
        )

        if self.permutation:
            answer_text = map_func(self.permutation[self.prediction_index])
        else:
            answer_text = map_func(self.prediction_index)

        return MULTIPLE_CHOICE_CHOICES_TMPL[choices_prompt_name].format(
            choice_key=answer_text, choice_text=self.choices[self.prediction_index]
        )

    def check_answer(self, answer: int):
        assert isinstance(answer, int) and answer < len(self.choices)
        if self.permutation is None:
            return answer == self.answer
        else:
            return self.permutation[answer] == self.answer

    def parse_and_check_answer(
        self, answer: str, parse_failed_value: Optional[int] = 0
    ):
        answer = answer.strip()
        map_func = (
            lambda x: (
                chr(x + ord("A")) if self.choice_key_type == "letters" else str(x + 1)
            )
            if type(x) == int
            else x
        )

        possible_candidates = [map_func(i) for i in range(len(self.choices))]
        possible_candidates = sorted(
            possible_candidates, key=lambda x: len(x), reverse=True
        )

        if answer in possible_candidates:
            self.prediction_index = possible_candidates.index(answer)
            return self.check_answer(self.prediction_index)
        for cand in possible_candidates:
            if cand in answer:
                self.prediction_index = possible_candidates.index(cand)
                return self.check_answer(self.prediction_index)

        self.prediction_index = None
        return parse_failed_value


class MultipleChoiceDataset(ABC):
    # a base class for multiple choice datasets
    def __init__(self, **kwargs):
        super().__init__()

        self.tokenizer_name_or_path = kwargs.get("tokenizer_name_or_path", None)
        self.multiple_choice_template_name = kwargs.get(
            "multiple_choice_template_name", "default"
        )
        self.system_prompt = kwargs.get("system_prompt", None)

        self.problems = []
        self.fewshot_examples = []
        self.name_or_path = ""

    def select_fewshot_examples(self, fewshot_dataset, num_shots, seed=1):
        permutation_generator = PermutationGenerator(seed)
        perm = permutation_generator.generate_permutation(
            len(fewshot_dataset), keep_ans=None
        )
        return [fewshot_dataset[i] for i in perm[:num_shots]]

    def generate_prompt_text(self):
        self.post_processor = PromptPostprocessor(
            tokenizer_name_or_path=self.tokenizer_name_or_path,
            multiple_choice_template_name=self.multiple_choice_template_name,
            system_prompt=self.system_prompt,
            add_generation_prompt=True,
        )
        for problem in self.problems:
            # Note if a problem is already assigned a prompt, we will not overwrite it. This allows to specify a custom prompt conveniently.
            problem.prompt = (
                problem.prompt
                or self.post_processor.get_full_prompt_from_problem(
                    problem=problem, fewshot_examples=self.fewshot_examples
                )
            )

    def __len__(self):
        return len(self.problems)

    def __getitem__(self, idx):
        return self.problems[idx]

    def __iter__(self):
        return iter(self.problems)

    def hash(self):
        problem_hashes = sorted([problem.uuid for problem in self.problems])
        fewshot_hashes = [problem.uuid for problem in self.fewshot_examples]
        hashstr = "$".join(problem_hashes)
        hashstr += "#".join(fewshot_hashes)
        if self.tokenizer_name_or_path:
            hashstr += "$tokenizer$" + self.tokenizer_name_or_path
        if self.multiple_choice_template_name:
            hashstr += "$tmpl$" + self.multiple_choice_template_name
        if self.system_prompt:
            hashstr += "$sysprompt$" + self.system_prompt
        hashstr = hashstr.encode("utf-8")
        hash_digest = md5(hashstr).digest()

        url_safe_hash = urlsafe_b64encode(hash_digest).rstrip(b"=").decode("utf-8")
        return url_safe_hash

    def select_first_n(self, n):
        self.problems = self.problems[:n]

    def extend(self, other):
        self.problems.extend(other.problems)


class AugmentedMultipleChoiceDataset(MultipleChoiceDataset):
    # a base class for augmented multiple choice datasets
    def __init__(self, seed=1, **kwargs):
        super().__init__(**kwargs)
        self.permutation_generator = PermutationGenerator(seed)

        # for cloze prompt
        self.apply_un = kwargs.get("apply_un", False)

    def generate_augmented_instance(
        self,
        src: MultipleChoiceProblem,
        num_instances: Optional[int] = 3,
        num_keeps: Optional[int] = 1,
    ):
        # generate num_instances augmented instances from src
        # num_keeps: number of instances to keep the answer in the same position
        assert num_keeps <= num_instances

        augmented_instances = []

        for i in range(num_instances):
            if i < num_keeps:
                perm = self.permutation_generator.generate_permutation(
                    len(src.choices), keep_ans=src.answer
                )
            else:
                perm = self.permutation_generator.generate_permutation(
                    len(src.choices), keep_ans=None
                )

            augmented_instances.append(
                MultipleChoiceProblem(
                    problem=src.problem,
                    choices=src.choices,
                    answer=src.answer,
                    parent_uuid=src.uuid,
                    generation_config=src.generation_config,
                    permutation=perm,
                    extra=src.extra,
                )
            )
        return augmented_instances

    def augment_dataset(self, num_instances: int, num_keeps: int):
        augmented_dataset = []
        for problem in self.problems:
            augmented_dataset.extend(
                self.generate_augmented_instance(problem, num_instances, num_keeps)
            )

        # generate prompt text for augmented examples
        for problem in augmented_dataset:
            problem.prompt = self.post_processor.get_full_prompt_from_problem(
                problem=problem, fewshot_examples=self.fewshot_examples
            )

        self.problems.extend(augmented_dataset)

    def unroll_choice_to_cp(
        self, src: MultipleChoiceProblem
    ) -> List[MultipleChoiceProblem]:
        parent_uuid = src.uuid
        unrolled_instances = []
        answer = src.choices[src.answer]
        for choice in src.choices:
            new_problem = MultipleChoiceProblem(
                problem=src.problem,
                choices=[choice],
                answer=answer,
                parent_uuid=parent_uuid,
                generation_config=src.generation_config,
                extra=src.extra,
            )
            unrolled_instances.append(new_problem)

        if self.apply_un:
            for choice in src.choices:
                new_problem = MultipleChoiceProblem(
                    problem=None,
                    choices=[choice],
                    answer=answer,
                    parent_uuid=parent_uuid,
                    generation_config=src.generation_config,
                    extra=src.extra,
                )
                new_problem.prompt = "Answer:"  # TODO: add template
                unrolled_instances.append(new_problem)

        return unrolled_instances

    def unroll_to_cp(self):
        unrolled_dataset = []
        for problem in self.problems:
            unrolled_dataset.extend(self.unroll_choice_to_cp(problem))

        for problem in self.fewshot_examples:
            problem.answer = problem.choices[problem.answer]

        self.problems = unrolled_dataset
        self.generate_prompt_text()

    def unroll_to_qa(self):
        for problem in self.problems:
            problem.answer = problem.choices[problem.answer]
            problem.choices = [problem.answer]
            problem.prompt = None

        for problem in self.fewshot_examples:
            problem.answer = problem.choices[problem.answer]

        self.generate_prompt_text()


if __name__ == "__main__":
    dic = {"a": 1, "b": 2}
    mcp = MultipleChoiceProblem("test prompt", ["1", "2", "3"], 2)
    print(mcp.__dict__)
    print(mcp.hash())
    print(hash(mcp))
