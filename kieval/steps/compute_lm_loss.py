from kieval.datasets import load_eval_dataset
from kieval.models import load_inference_function
from kieval.steps.base_step import BaseStep
from typing import Optional, Dict, Tuple, List
import logging, os, json, codecs
import jsonlines
from hashlib import md5


class ComputeLMLossStep(BaseStep):
    """Compute LM loss step."""

    type: str = "compute_lm_loss"

    def __init__(
        self,
        dataset_config,
        inference_config,
        eval_config=None,
        step_name="compute_lm_loss",
        **kwargs,
    ):
        super().__init__(
            step_type="compute_lm_loss",
            step_name=step_name,
            description="Compute LM loss step.",
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

        dataset_mode = self.dataset_config["dataset_kwargs"].get("dataset_mode", "mcp")
        assert dataset_mode in [
            "mcp",
            "qa",
        ], "Only mcp and cp supported for compute_lm_loss"

        if dataset_mode == "qa":
            self.logger.info(f"Unrolling dataset to QA")
            self.dataset.unroll_to_qa()
            self.logger.info(f"Dataset unrolled, num instances: {len(self.dataset)}")

        self.logger.info(f"Dataset example: {self.dataset[0].__dict__}")
        dataset_hash = self.dataset.hash()
        self.logger.info(f"Dataset hash: {dataset_hash}")
        step_hash = md5(
            f'{dataset_hash}_{self.step_name}_{self.inference_config["inference_kwargs"]["model_path"]}'.encode()
        ).hexdigest()
        self.output_path = self.inference_config["inference_kwargs"]["output_path"] = (
            os.path.join(self.inference_config["output_path"], step_hash)
        )

        assert (
            self.inference_config["type"] == "local_hf"
        ), "Only local_hf supported for compute_lm_loss"
        self.model_function = load_inference_function(self.inference_config["type"])

    def aggregate_results(self):
        """Calculate overall lm loss."""
        losses = []

        with jsonlines.open(os.path.join(self.output_path, "all_responses.jsonl")) as f:
            for line in f:
                losses.append(line["response"]["lm_loss"])

        return {
            "avg_loss": sum(losses) / len(losses),
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
        self.inference_config["inference_kwargs"]["inference_mode"] = "lm_loss"
        self.model_function(self.dataset, **self.inference_config["inference_kwargs"])

    def postprocess(self, context):
        """Postprocess after the step.

        Args:
            context (dict): The context dictionary.
        """
        if self.eval_config is not None:
            self.logger.info(f"Calculating accuracy with config: {self.eval_config}")
            eval_results = self.aggregate_results()
            self.logger.warning(f"Eval results: {eval_results}")
            context.results[(self.step_type, self.step_name)] = eval_results
        del self.dataset

        # del context.predictions
