from kieval.datasets import load_eval_dataset
from kieval.models import load_inference_function
from kieval.steps.base_step import BaseStep
from typing import Optional, Dict, Tuple, List
import logging, os, json, codecs
import jsonlines


class ClozePromptStep(BaseStep):
    """Cloze prompt evaluation step."""

    def __init__(
        self,
        dataset_config,
        inference_config,
        eval_config=None,
        step_name="cloze_prompt",
        **kwargs,
    ):
        super().__init__(
            step_type="cloze_prompt",
            step_name=step_name,
            description="Cloze prompt evaluation step.",
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
        logging.debug(f"Loading dataset with config: {self.dataset_config}")
        self.dataset = load_eval_dataset(
            self.dataset_config["type"], self.dataset_config["dataset_kwargs"]
        )
        self.logger.info(f"Dataset loaded, num instances: {len(self.dataset)}")

        self.logger.info(f"Unrolling dataset to cloze prompt")
        self.dataset.unroll_to_cp()
        self.logger.info(f"Dataset unrolled, num instances: {len(self.dataset)}")

        self.logger.info(f"Dataset example: {self.dataset[0].__dict__}")
        dataset_hash = self.dataset.hash()
        self.logger.info(f"Dataset hash: {dataset_hash}")
        self.output_path = self.inference_config["inference_kwargs"]["output_path"] = (
            os.path.join(self.inference_config["output_path"], dataset_hash)
        )

        assert (
            self.inference_config["type"] == "local_hf"
        ), "Only local_hf supported for cloze_prompt"
        self.model_function = load_inference_function(self.inference_config["type"])

    def calculate_accuracy(
        self,
        apply_ln: bool = False,
    ):
        """Calculate accuracy. See https://arxiv.org/pdf/2210.12353.pdf for details of apply_un(UN) and apply_ln(LN)."""
        apply_un = self.dataset_config["dataset_kwargs"].get("apply_un", False)

        generation_results: Dict[str, Dict[str, float]] = {}
        unconditional_results: Dict[str, Dict[str, float]] = {}
        answers = {}
        with jsonlines.open(os.path.join(self.output_path, "all_responses.jsonl")) as f:
            for line in f:
                request = line["request"]
                resp = line["response"]
                parent_uuid = request["parent_uuid"]
                answers[parent_uuid] = request["answer"]
                logit = (
                    resp["loglikelihood"] / resp["target_length"]
                    if apply_ln
                    else resp["loglikelihood"]
                )
                key, item = request["choices"][0], logit
                if line["request"]["problem"] is not None:
                    if parent_uuid not in generation_results:
                        generation_results[parent_uuid] = {}
                    generation_results[parent_uuid][key] = item
                else:
                    if parent_uuid not in unconditional_results:
                        unconditional_results[parent_uuid] = {}
                    unconditional_results[parent_uuid][key] = item

        num_hits = 0
        num_groups = 0

        for uuid, predictions in generation_results.items():
            if apply_un:
                for key, item in predictions.items():
                    predictions[key] -= unconditional_results[uuid][key]

            # check if max logit is the answer
            max_key = max(predictions, key=predictions.get)
            if max_key == answers[uuid]:
                num_hits += 1
            num_groups += 1

        accuracy = num_hits / num_groups

        return {
            "accuracy": accuracy,
            "num_hits": num_hits,
            "num_groups": num_groups,
            "num_instances": len(self.dataset),
            "applly_ln": apply_ln,
            "apply_un": apply_un,
        }

    def run(self, context):
        """Run the step.

        Args:
            context (dict): The context dictionary.

        Returns:
            dict: The updated context dictionary.
        """
        self.logger.info(
            f'Running inference with config: {self.inference_config["inference_kwargs"]}'
        )
        self.inference_config["inference_kwargs"]["inference_mode"] = "loglikelihood"
        self.model_function(self.dataset, **self.inference_config["inference_kwargs"])

    def postprocess(self, context):
        """Postprocess after the step.

        Args:
            context (dict): The context dictionary.
        """
        if self.eval_config is not None:
            self.logger.info(f"Calculating accuracy with config: {self.eval_config}")
            eval_results = self.calculate_accuracy(**self.eval_config)
            self.logger.warning(f"Eval results: {eval_results}")
            context.results[(self.step_type, self.step_name)] = eval_results
        del self.dataset

        # del context.predictions
