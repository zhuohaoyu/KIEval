from kieval.datasets import load_eval_dataset
from kieval.models import load_inference_function
from kieval.steps.base_step import BaseStep
from typing import Optional, Dict, Tuple, List
import logging, os, json, codecs
import jsonlines
import numpy as np
from sklearn.metrics import roc_curve, auc
import copy
from hashlib import md5


class MinKProbStep(BaseStep):
    """Min k% prob evaluation step."""

    def __init__(
        self,
        dataset_config,
        inference_config,
        eval_config=None,
        step_name="min_k_prob",
        **kwargs,
    ):
        super().__init__(
            step_type="min_k_prob",
            step_name=step_name,
            description="Min k% prob evaluation step.",
        )
        self.logger = logging.getLogger(__name__)
        self.dataset_config = dataset_config
        self.inference_config = inference_config
        self.eval_config = eval_config
        self.output_path = None

    def preprocess(self, context):
        """Prepare the step.

        Args:
            context (dict): The context dictionary.
        """
        logging.debug(f"Loading datasets with config: {self.dataset_config}")
        self.datasets = []
        self.output_paths = []
        self.inference_configs = []
        self.labels = []
        for config in self.dataset_config:
            dataset = load_eval_dataset(config["type"], config["dataset_kwargs"])
            self.logger.info(f"Dataset loaded, num instances: {len(dataset)}")

            dataset_mode = config["dataset_kwargs"].get("dataset_mode", "mcp")
            assert dataset_mode in [
                "mcp",
                "qa",
            ], "Only mcp and cp supported for min_k_prob"

            if dataset_mode == "qa":
                self.logger.info(f"Unrolling dataset to QA")
                dataset.unroll_to_qa()
                self.logger.info(f"Dataset unrolled, num instances: {len(dataset)}")

            self.logger.info(f"Dataset example: {dataset[0].__dict__}")
            dataset_hash = dataset.hash()
            self.logger.info(f"Dataset hash: {dataset_hash}")
            inference_config = copy.deepcopy(self.inference_config)
            step_hash = md5(
                f'{dataset_hash}_{self.step_name}_{inference_config["inference_kwargs"]["model_path"]}'.encode()
            ).hexdigest()
            output_path = inference_config["inference_kwargs"]["output_path"] = (
                os.path.join(inference_config["output_path"], step_hash)
            )
            self.datasets.append(dataset)
            self.output_paths.append(output_path)
            self.inference_configs.append(inference_config)
            self.labels.append(config.get("group_label", 0))

        assert (
            self.inference_config["type"] == "local_hf"
        ), "Only local_hf supported for cloze_prompt"
        self.model_function = load_inference_function(self.inference_config["type"])

    def get_auc(self, score, x):
        """
        Compute AUC.
        """
        score = np.array(score)
        x = np.array(x, dtype=bool)
        fpr, tpr, _ = roc_curve(x, -score)
        return auc(fpr, tpr)

    def aggregate_results(self):
        """Calculate overall min k% probs."""
        threasholds = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
        probs_avgs = {k: [] for k in threasholds}
        labels = {k: [] for k in threasholds}

        for output_path, label in zip(self.output_paths, self.labels):
            with jsonlines.open(os.path.join(output_path, "all_responses.jsonl")) as f:
                for line in f:
                    all_probs = line["response"]["loglikelihood"]
                    length = len(all_probs)
                    all_probs = np.sort(all_probs)
                    for k in threasholds:
                        k_length = int(length * k)
                        if k_length == 0:
                            continue
                        probs_avgs[k].append(-np.mean(all_probs[:k_length]))
                        labels[k].append(label)

        return {
            f"min_k_prob_{k}": self.get_auc(probs_avgs[k], labels[k])
            for k in threasholds
        }

    def run(self, context):
        """Run the step.

        Args:
            context (dict): The context dictionary.

        Returns:
            dict: The updated context dictionary.
        """
        for dataset, inference_config in zip(self.datasets, self.inference_configs):
            self.logger.info(
                f'Running inference with config: {inference_config["inference_kwargs"]}'
            )
            inference_config["inference_kwargs"]["inference_mode"] = "all_loglikelihood"
            self.model_function(dataset, **inference_config["inference_kwargs"])

    def postprocess(self, context):
        """Postprocess after the step.

        Args:
            context (dict): The context dictionary.
        """
        if self.eval_config is not None:
            self.logger.info(f"Calculating accuracy with config: {self.eval_config}")
            eval_results = self.aggregate_results()
            self.logger.warning(f"Eval results: {eval_results}")
            md_string = "".join(f" {x:.3f} |" for x in eval_results.values())
            self.logger.warning(f"Markdown string: {md_string}")
            context.results[(self.step_type, self.step_name)] = eval_results
        for dataset in self.datasets:
            del dataset

        # del context.predictions
