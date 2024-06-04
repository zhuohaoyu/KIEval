from kieval.datasets import load_eval_dataset
from kieval.models import load_inference_function
from kieval.steps.base_step import BaseStep
from kieval.utils import calculate_inference_endpoint_hash
from typing import Optional, Union, List, Dict
from collections import Counter

from hashlib import md5
from base64 import urlsafe_b64encode
from tqdm import tqdm
import logging, os, json, codecs
import jsonlines
import math
import random

from kieval.prompts import PromptPostprocessor, TriloguePrompter
from kieval.utils import parse_json

# EVALUATOR_METRICS = ["conciseness", "relevance", "coherence", "accuracy", "reasoning"]

EVALUATOR_METRICS = ["accuracy", "logic", "relevance", "coherence", "conciseness"]

def weighted_mean(scores):
    score_mapper = {0: 0.0, 1: 1.0, 2: 3.0, 3: 7.0, 4: 10.0 }
    weights = [math.exp(-0.2 * index) for index in range(len(scores))]
    return (
        sum(score_mapper[score] * weight for score, weight in zip(scores, weights))
        / sum(weights)
        * 10.0
    )


class Conversation:
    def __init__(
        self, uuid: str, messages: List[Dict] = [], random_seed: Optional[int] = 0
    ) -> None:
        self.uuid = uuid
        self.messages = messages
        self.random_seed = random_seed
        self.prompt = None
        self.stop_interaction = False

    def __len__(self) -> int:
        return len(self.messages)

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def add_message(self, role: str, content: str) -> None:
        self.messages.append({"content": content, "role": role})


class Triologue(Conversation):
    def __init__(
        self,
        uuid: str,
        init_messages: Dict,
        random_seed: Optional[int] = 0,
    ):
        super().__init__(uuid, messages=[], random_seed=random_seed)

        self.all_messages = []  # no system prompt
        self.current_party = None

        self.role_messages = init_messages

        self.evaluation_results = []
        self.aggregated_result = None

    def parse_evaluation_result(self, content: str):
        if self.stop_interaction:
            return None
        j = parse_json(content)
        for key in EVALUATOR_METRICS:
            if key not in j:
                j[key] = {"comment": "", "score": None}
            if "comment" not in j[key]:
                j[key]["comment"] = ""
            if "score" not in j[key]:
                j[key]["score"] = None
        if "comment" not in j:
            j["comment"] = ""
        if "overall_score" not in j:
            j["overall_score"] = None
        if "stop_conversation" not in j:
            j["stop_conversation"] = False

        if j["stop_conversation"]:
            self.stop_interaction = True

        self.evaluation_results.append(j)

        return j

    def aggregate_result(self):
        if len(self.evaluation_results) == 0:
            return None
        if self.aggregated_result is not None:
            return self.aggregated_result
        j = {}
        for key in EVALUATOR_METRICS:
            values = [e.get(key, None) for e in self.evaluation_results]
            values = [v.get("score", None) for v in values if v is not None]
            values = [v for v in values if v is not None]
            values_normalized = []
            for v in values:
                if isinstance(v, str):
                    try:
                        v = float(v)
                        if v >= 0 and v <= 5:
                            values_normalized.append(v)
                    except:
                        continue
                elif v >= 0 and v <= 5:
                    values_normalized.append(v)
            values = values_normalized
            if len(values) == 0:
                j[key] = None
            else:
                kieval_score = weighted_mean(values)
                try:
                    j[key] = {
                        "min": min(values),
                        "max": max(values),
                        "mean": sum(values) / len(values),
                        "kieval_score": kieval_score,
                    }
                except:
                    print(values)
                    raise ValueError("Error when aggregating results.")
        values = [e.get("overall_score", None) for e in self.evaluation_results]
        values = [v for v in values if v is not None]
        kieval_score = weighted_mean(values)
        if len(values) == 0:
            j["overall_score"] = None
        else:
            j["overall_score"] = {
                "min": min(values),
                "max": max(values),
                "mean": sum(values) / len(values),
                "kieval_score": kieval_score,
            }
        self.aggregated_result = j
        return j

    def set_party(self, party: str) -> None:
        if party not in ["candidate", "interactor", "evaluator"]:
            raise ValueError(
                "Party role not recognized. Valid roles are 'candidate', 'interactor', 'evaluator'."
            )
        self.current_party = party
        self.messages = self.role_messages[party]

    def add_message(self, party: str, content: str, prompter: TriloguePrompter) -> None:
        self.all_messages.append({"content": content, "role": party})
        if party == "candidate":
            self.role_messages["candidate"].append(
                {"role": "assistant", "content": content}
            )
            self.role_messages["interactor"].append(
                {"role": "user", "content": content}
            )

            evaluator_user_messages = self.all_messages[
                -2:
            ]  # last two messages are from interactor and candidate
            assert (
                evaluator_user_messages[0]["role"] == "interactor"
            ), f"Last message is not from interactor: {evaluator_user_messages[0]}"
            evaluator_user_prompt = prompter.apply_evaluator_user_prompt(
                interactor_content=evaluator_user_messages[0]["content"],
                candidate_content=evaluator_user_messages[1]["content"],
            )
            self.role_messages["evaluator"].append(
                {"role": "user", "content": evaluator_user_prompt}
            )

        elif party == "interactor":
            self.role_messages["interactor"].append(
                {"role": "assistant", "content": content}
            )
            self.role_messages["candidate"].append({"role": "user", "content": content})

        elif party == "evaluator":
            self.role_messages["evaluator"].append(
                {"role": "assistant", "content": content}
            )


class InteractiveEvaluationStep(BaseStep):
    """Evaluate candidate model with another model, interactively."""

    type = "interactive_evaluation_step"

    def __init__(
        self,
        roles_config,
        max_rounds: int = 3,
        max_instances: Optional[int] = -1,
        prompter_config: Optional[Dict] = {},
        output_path: Optional[str] = None,
        save_predictions: Optional[bool] = True,
        step_name="interactive_evaluation_step",
        validated_problems_path: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(
            step_type="interactive_evaluation_step",
            step_name=step_name,
            description="Evaluate candidate model with another model, interactively.",
        )
        self.logger = logging.getLogger(__name__)

        assert (
            len(roles_config) == 3
        ), f"Only support 3 roles, got {len(roles_config)} roles."
        for role in ["interactor", "candidate", "evaluator"]:
            assert role in roles_config, f"Missing role: {role} in roles_config."

        self.output_path = output_path
        self.max_rounds = max_rounds
        self.max_instances = max_instances
        self.save_predictions = save_predictions
        self.validated_problems_path = validated_problems_path
        self.prompt_postprocessors = {}
        self.inference_functions = {}
        self.inference_kwargs = {}
        self.roles_config = roles_config
        self.trilogues = []
        self.evaluation_results = {}
        self.prompter = TriloguePrompter(prompter_config=prompter_config)

        for role in roles_config:
            self.init_role(role, **roles_config[role])

        self.step_hash = self.hash(prompter_hash=self.prompter.hash())

    def hash(self, prompter_hash=None, mcp_hash=None):
        hashstr = ""
        for role in ["interactor", "candidate", "evaluator"]:
            hashstr += f"$role:{role}$"
            cfg = self.roles_config[role]
            hashstr += calculate_inference_endpoint_hash(cfg)

        if prompter_hash is not None:
            hashstr += f"$prompter:{prompter_hash}$"

        if mcp_hash is not None:
            hashstr += f"$mcp:{mcp_hash}$"

        hashstr = hashstr.encode("utf-8")
        hash_digest = md5(hashstr).digest()

        url_safe_hash = urlsafe_b64encode(hash_digest).rstrip(b"=").decode("utf-8")
        return url_safe_hash

    def output_path_nicename(self):
        return f"interactive_{self.step_hash}"

    def init_role(
        self,
        role: str,
        type: str,
        inference_kwargs: Dict,
        prompt_postprocessor_config: Dict = None,
    ):
        self.logger.info(f"Initializing role: {role}, type: {type}")

        inference_function = load_inference_function(type)

        if type == "openai":
            assert (
                prompt_postprocessor_config is None
            ), f"Prompt postprocessor not required for openai model."
            prompt_postprocessor = None
        else:
            prompt_postprocessor = PromptPostprocessor(**prompt_postprocessor_config)

        self.prompt_postprocessors[role] = prompt_postprocessor
        self.inference_functions[role] = inference_function
        self.inference_kwargs[role] = inference_kwargs

    def run_single_round_batch_inference(
        self, role: str, trilogues: List[Triologue], round: int = 0
    ):
        self.logger.info(f"Running single round batch inference for role: {role}")

        assert (
            role in self.inference_functions
        ), f"No inference function for role: {role}"
        assert role in self.inference_kwargs, f"No inference config for role: {role}"
        assert (
            role in self.prompt_postprocessors
        ), f"No prompt postprocessor for role: {role}"

        inference_function = self.inference_functions[role]
        inference_kwargs = self.inference_kwargs[role].copy()
        current_role_output_path = os.path.join(
            self.output_path, self.output_path_nicename(), f"{role}_{round}"
        )
        inference_kwargs["output_path"] = current_role_output_path
        prompt_postprocessor = self.prompt_postprocessors[role]

        trilogues_filtered = [d for d in trilogues if not d.stop_interaction]

        for conv in trilogues_filtered:
            conv.set_party(role)
            if prompt_postprocessor is not None:
                conv.prompt = prompt_postprocessor.get_full_prompt_from_conversation(
                    conv.messages
                )

        inference_function(trilogues_filtered, **inference_kwargs)

        generation_results = {}

        with jsonlines.open(
            os.path.join(current_role_output_path, "all_responses.jsonl")
        ) as f:
            for line in f:
                generated_text = line["response"]["generated_text"].strip()
                uuid = line["request"]["uuid"]
                generation_results[uuid] = generated_text

        for conv in trilogues_filtered:
            conv.add_message(
                party=role,
                content=generation_results[conv.uuid],
                prompter=self.prompter,
            )

    def aggregate_single_round_results(self, round_results: List[Dict]):
        ret = {}
        for key in EVALUATOR_METRICS:
            values = [e.get(key, None) for e in round_results]
            values = [v.get("score", None) for v in values if v is not None]
            values = [v for v in values if v is not None]
            values_normalized = []
            for v in values:
                if isinstance(v, str):
                    try:
                        v = float(v)
                        if v >= 0 and v <= 5:
                            values_normalized.append(v)
                    except:
                        continue
                elif v >= 0 and v <= 5:
                    values_normalized.append(v)
            values = values_normalized
            if len(values) == 0:
                ret[key] = {}
                continue
            value_counter = dict(Counter(values))
            ret[key] = {
                "nonempty_values": len(values),
                "mean": sum(values) / len(values),
                "min": min(values),
                "max": max(values),
                "counter": value_counter,
            }
        values = [e.get("overall_score", None) for e in round_results]
        values = [v for v in values if v is not None]
        value_counter = dict(Counter(values))
        if len(values) == 0:
            ret["overall_score"] = {}
        else:
            ret["overall_score"] = {
                "nonempty_values": len(values),
                "mean": sum(values) / len(values),
                "min": min(values),
                "max": max(values),
                "counter": value_counter,
            }
        return ret

    def run_multiple_round_batch_inference(self, conversation_dataset: List[Triologue]):
        self.logger.info(f"Running {self.max_rounds} round(s) of conversation.")
        self.evaluation_results["rounds"] = {}
        for round in range(self.max_rounds):
            self.logger.info(f"Round #{round}")
            self.run_single_round_batch_inference(
                "interactor", conversation_dataset, round
            )
            self.run_single_round_batch_inference(
                "candidate", conversation_dataset, round
            )
            self.run_single_round_batch_inference(
                "evaluator", conversation_dataset, round
            )
            self.logger.info(f"Parsing evaluation results for round #{round}")
            round_results = []
            for conv in tqdm(conversation_dataset):
                res = conv.parse_evaluation_result(conv.all_messages[-1]["content"])
                if res is not None:
                    round_results.append(res)
            single_round_aggregated = self.aggregate_single_round_results(round_results)
            self.evaluation_results["rounds"][round] = single_round_aggregated
            self.logger.info(
                f"Round #{round} aggregated results:\n{json.dumps(single_round_aggregated, indent=2, ensure_ascii=False)}"
            )

            num_stop_interaction = sum(
                [1 for conv in conversation_dataset if conv.stop_interaction]
            )
            if num_stop_interaction == len(conversation_dataset):
                self.logger.warning(f"All conversations are stopped, terminating.")
                break

        details = []
        aggregated_results = {}
        for conv in tqdm(conversation_dataset):
            res = conv.aggregate_result()
            details.append({"uuid": conv.uuid, "result": res})
            for key in res:
                if key not in aggregated_results:
                    aggregated_results[key] = []
                if res[key] is None:
                    value = None
                else:
                    value = res[key].get("kieval_score", None)
                if value is not None:
                    aggregated_results[key].append(value)

        ret = {}
        for key in aggregated_results:
            value_counter = dict(Counter(aggregated_results[key]))
            if len(aggregated_results[key]) == 0:
                ret[key] = {}
                continue
            ret[key] = {
                "nonempty_values": len(aggregated_results[key]),
                "mean": sum(aggregated_results[key]) / len(aggregated_results[key]),
                "min": min(aggregated_results[key]),
                "max": max(aggregated_results[key])
            }
        self.evaluation_results["overall"] = ret
        self.evaluation_results["hash"] = self.step_hash

        def role_name_mapper(d):
            if d["type"] == "openai":
                return f'{d["type"]}: {d["inference_kwargs"]["openai_model"]}'
            elif d["type"] == "remote_hf":
                name = d["inference_kwargs"].get(
                    "model_name", d["inference_kwargs"]["base_url"]
                )
                return f'{d["type"]}: {name}'
            else:
                return f'{d["type"]}: {d["inference_kwargs"]["model_path"]}'

        self.evaluation_results["roles"] = {
            role: role_name_mapper(config) for role, config in self.roles_config.items()
        }
        self.logger.info(
            f"KIEval scores:\n{json.dumps(ret, indent=2, ensure_ascii=False)}"
        )

    def preprocess(self, context):
        """Prepare the step.

        Args:
            context (dict): The context dictionary.
        """

        self.logger.info("Preprocessing interactive evaluation")

        # make sure context.mcp_dataset is valid
        assert hasattr(context, "mcp_dataset"), f"Missing mcp_dataset in context"
        assert hasattr(
            context, "mcp_prediction_results"
        ), f"Missing mcp_prediction_results in context"

        empty_response_cnt = 0

        for mcp in context.mcp_dataset:
            assert hasattr(mcp, "prediction_index"), f"Missing prediction_index in mcp"

            if mcp.parent_uuid is not None:  # ignore all augments
                continue
            if mcp.prediction_index is None:
                empty_response_cnt += 1
                continue

            uuid = mcp.uuid
            init_messages = self.prompter.generate_init_messages(mcp)
            random_seed = 0

            self.trilogues.append(
                Triologue(
                    uuid=uuid,
                    init_messages=init_messages,
                    random_seed=random_seed,
                )
            )

        self.step_hash = self.hash(
            prompter_hash=self.prompter.hash(), mcp_hash=context.mcp_dataset.hash()
        )

        if empty_response_cnt > 0:
            self.logger.warning(f"Found {empty_response_cnt} empty responses.")

        if self.validated_problems_path is not None:
            self.logger.info(
                f"Loading validated problems from {self.validated_problems_path}"
            )
            with open(self.validated_problems_path, "r") as f:
                validated_problems = json.load(f)
            valid_uuids = set(validated_problems["valid_uuids"])
            invalid_uuids = set(validated_problems["invalid_uuids"])
            trilogues_filtered = []
            for t in self.trilogues:
                if t.uuid in valid_uuids:
                    trilogues_filtered.append(t)
                elif t.uuid in invalid_uuids:
                    continue
                else:
                    self.logger.warning(
                        f"Missing uuid in provided validate results: {t.uuid}"
                    )
            self.logger.info(
                f"Found {len(trilogues_filtered)} / {len(self.trilogues)} valid problems."
            )
            self.trilogues = trilogues_filtered
        else:
            self.logger.warning(f"No validated problems provided.")

        if self.max_instances > 0 and len(self.trilogues) > self.max_instances:
            self.logger.warning(
                f"Pruning {len(self.trilogues)} instances to {self.max_instances} instances."
            )
            self.trilogues = self.trilogues[: self.max_instances]

    def run(self, context):
        """Run the step.

        Args:
            context (dict): The context dictionary.

        Returns:
            dict: The updated context dictionary.
        """

        self.run_multiple_round_batch_inference(self.trilogues)

    def postprocess(self, context):
        """Postprocess after the step.

        Args:
            context (dict): The context dictionary.
        """
        if self.save_predictions:
            detail_path = os.path.join(
                self.output_path, self.output_path_nicename(), "interact_details.json"
            )
            self.logger.info(f"Saving interact details to {detail_path}")
            context.interactive_details = self.trilogues
            context.predictions[(self.step_type, self.step_name)] = self.trilogues

            with codecs.open(detail_path, "w", "utf-8") as f:
                json.dump(
                    [t.__dict__ for t in self.trilogues],
                    f,
                    indent=2,
                    ensure_ascii=False,
                )
        else:
            context.predictions[(self.step_type, self.step_name)] = None
            del self.trilogues

        context.results[(self.step_type, self.step_name)] = self.evaluation_results
        results_path = os.path.join(
            self.output_path, self.output_path_nicename(), "results.json"
        )
        self.logger.info(f"Saving evaluation results to {results_path}")
        with codecs.open(results_path, "w", "utf-8") as f:
            json.dump(self.evaluation_results, f, indent=2, ensure_ascii=False)
