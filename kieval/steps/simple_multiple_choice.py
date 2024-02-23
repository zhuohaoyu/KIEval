from kieval.datasets import load_eval_dataset
from kieval.models import load_inference_function
from kieval.steps.base_step import BaseStep
from kieval.utils import calculate_inference_endpoint_hash
from typing import Optional
import logging, os, json, codecs
import jsonlines


class SimpleMultipleChoiceStep(BaseStep):
    """Simple multiple choice evaluation step."""

    def __init__(
        self,
        dataset_config,
        inference_config,
        eval_config=None,
        step_name="simple_multiple_choice",
        save_dataset=False,
        save_predictions=False,
        **kwargs,
    ):
        super().__init__(
            step_type="simple_multiple_choice",
            step_name=step_name,
            description="Simple multiple choice evaluation step.",
        )
        self.logger = logging.getLogger(__name__)
        self.dataset_config = dataset_config
        self.inference_config = inference_config
        self.eval_config = eval_config
        self.prediction_results = None
        self.output_path = None
        self.save_dataset = save_dataset
        self.save_predictions = save_predictions

    def preprocess(self, context):
        """Prepare the step.

        Args:
            context (dict): The context dictionary.
        """
        logging.debug(f"Loading dataset with config: {self.dataset_config}")
        self.dataset = load_eval_dataset(
            self.dataset_config["type"], self.dataset_config["dataset_kwargs"]
        )
        self.logger.info(
            f"Dataset loaded, num instances: {len(self.dataset)}, dataset hash: {self.dataset.hash()}"
        )

        if "augment_dataset_kwargs" in self.dataset_config:
            self.logger.warning(
                f'Augmenting dataset with config: {self.dataset_config["augment_dataset_kwargs"]}'
            )
            self.dataset.augment_dataset(
                **self.dataset_config["augment_dataset_kwargs"]
            )
            self.logger.info(
                f"Dataset augmented, num instances: {len(self.dataset)}, dataset hash: {self.dataset.hash()}"
            )

        self.logger.info(f"Dataset example: {self.dataset[0].__dict__}")

        dataset_hash = self.dataset.hash()

        inference_hash = calculate_inference_endpoint_hash(
            self.inference_config, context_hash=dataset_hash
        )

        self.logger.info(
            f"Dataset hash: {dataset_hash}, inference hash: {inference_hash}"
        )
        self.output_path = self.inference_config["inference_kwargs"]["output_path"] = (
            os.path.join(self.inference_config["output_path"], f"mc_{inference_hash}")
        )
        self.model_function = load_inference_function(self.inference_config["type"])

    def calculate_accuracy(
        self,
        aggregate_mode: Optional[str] = "mean",
        vote_threshold: Optional[float] = None,
        save_predictions: Optional[bool] = True,
    ):
        # calculate accuracy from predictions
        # aggregate_mode: "strict", "mean", "ignore_augmented", "vote",

        if aggregate_mode not in ["strict", "mean", "ignore_augmented", "vote"]:
            raise ValueError(
                f"Invalid aggregate_mode: {aggregate_mode}, must be one of 'strict', 'mean', 'ignore_augmented', 'vote'"
            )

        assert aggregate_mode != "vote" or vote_threshold is not None

        generation_results = {}
        with jsonlines.open(os.path.join(self.output_path, "all_responses.jsonl")) as f:
            for line in f:
                generated_text = line["response"]["generated_text"]
                uuid = line["request"]["uuid"]
                parent_uuid = line["request"]["parent_uuid"]
                generation_results[(uuid, parent_uuid)] = generated_text

        parent_predictions = {}
        for problem in self.dataset.problems:
            uuid = problem.uuid
            parent_uuid = problem.parent_uuid
            if aggregate_mode == "ignore_augmented" and parent_uuid is not None:
                continue
            pred_uuid = parent_uuid if parent_uuid is not None else uuid
            if (uuid, parent_uuid) not in generation_results:
                raise ValueError(
                    f"Missing prediction for problem {uuid} with parent {parent_uuid}"
                )
            generation_result = generation_results[(uuid, parent_uuid)]
            ans = problem.parse_and_check_answer(
                generation_result, parse_failed_value=0
            )
            parent_predictions[pred_uuid] = parent_predictions.get(pred_uuid, []) + [
                ans
            ]

        num_hits = 0
        num_groups = 0

        prediction_results = {}

        for uuid in parent_predictions:
            if aggregate_mode == "strict":
                num_hits += all(parent_predictions[uuid])
                num_groups += 1
                prediction_results[uuid] = [all(parent_predictions[uuid]), 1]
            elif aggregate_mode == "mean" or aggregate_mode == "ignore_augmented":
                num_hits += sum(parent_predictions[uuid])
                num_groups += len(parent_predictions[uuid])
                prediction_results[uuid] = [
                    sum(parent_predictions[uuid]),
                    len(parent_predictions[uuid]),
                ]
            elif aggregate_mode == "vote":
                hit = (
                    1
                    if sum(parent_predictions[uuid]) / len(parent_predictions[uuid])
                    >= vote_threshold
                    else 0
                )
                num_hits += hit
                num_groups += 1
                prediction_results[uuid] = [hit, 1]

        accuracy = num_hits / num_groups

        if save_predictions:
            self.prediction_results = prediction_results

        return {
            "accuracy": accuracy,
            "num_hits": num_hits,
            "num_groups": num_groups,
            "num_instances": len(self.dataset),
            "aggregate_mode": aggregate_mode,
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

        context.mcp_prediction_results = self.prediction_results
        if self.save_predictions:
            predictions_output_path = os.path.join(
                self.output_path, "mcp_predictions.json"
            )
            self.logger.info(f"Saving predictions to {predictions_output_path}")
            with codecs.open(predictions_output_path, "w", encoding="utf-8") as f:
                json.dump(self.prediction_results, f, indent=2, ensure_ascii=False)

        context.predictions[(self.step_type, self.step_name)] = self.prediction_results

        if self.save_dataset:
            context.mcp_dataset = self.dataset
        else:
            del self.dataset
