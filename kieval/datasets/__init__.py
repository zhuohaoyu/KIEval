from kieval.datasets.multiple_choice import MultipleChoiceDataset
import logging

# register new datasets here
from kieval.datasets.arc import ARCChallengeDataset
from kieval.datasets.mmlu import MMLUDataset
from kieval.datasets.truthful_qa import TruthfulQADataset
from kieval.datasets.hellaswag import HellaSwagDataset
from kieval.datasets.self_instruct import SelfInstructDataset
from kieval.datasets.ceval import CEvalDataset
from kieval.datasets.gsm8k import GSM8KDataset
from kieval.datasets.medmcqa import MedMCQADataset
from kieval.datasets.hotpotqa import HotpotQADataset
from kieval.datasets.reclor import ReclorDataset

TYPE_TO_DATASET = {
    "arc_challenge": ARCChallengeDataset,
    "mmlu": MMLUDataset,
    "truthful_qa": TruthfulQADataset,
    "hellaswag": HellaSwagDataset,
    "ceval": CEvalDataset,
    "self_instruct": SelfInstructDataset,
    "gsm8k": GSM8KDataset,
    "medmcqa": MedMCQADataset,
    "hotpot_qa": HotpotQADataset,
    "reclor": ReclorDataset,
}


def load_eval_dataset_class(type):
    assert type in TYPE_TO_DATASET
    dset_class = TYPE_TO_DATASET[type]

    return dset_class


def load_eval_dataset_group(dataset_kwargs):
    dset_instance = None
    for kwargs in dataset_kwargs:
        if dset_instance is None:
            dset_instance = load_eval_dataset(kwargs["type"], kwargs["dataset_kwargs"])
        else:
            dset_instance.extend(
                load_eval_dataset(kwargs["type"], kwargs["dataset_kwargs"])
            )
    return dset_instance


def load_eval_dataset(type, dataset_kwargs):
    if type == "group":
        return load_eval_dataset_group(dataset_kwargs)

    dset_class = load_eval_dataset_class(type)

    logging.getLogger(__name__).info(f"Loading dataset with kwargs: {dataset_kwargs}")

    dset_instance = dset_class(**dataset_kwargs)

    if "first_n" in dataset_kwargs:
        dset_instance.select_first_n(dataset_kwargs["first_n"])

    return dset_instance
