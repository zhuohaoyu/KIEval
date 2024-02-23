import json, logging, codecs, os
from kieval.core.context import Context
from kieval.core.step_executor import StepExecutor
from typing import Dict


class EvaluationPipeline:
    def __init__(self, config: Dict):
        self.logger = logging.getLogger(__name__)
        self.config = config
        self.results_output_path = config.get("results_output_path", None)
        self.predictions_output_path = config.get("predictions_output_path", None)

        self.ctx = Context(config)
        self.step_executor = StepExecutor(config["steps"])

    def run(self):
        self.step_executor.load_steps()
        self.ctx = self.step_executor.run_steps(self.ctx)

        if self.results_output_path:
            self.logger.info(f"Saving all results to {self.results_output_path}")
            os.makedirs(os.path.split(self.results_output_path)[0], exist_ok=True)
            output_dict = {":".join(k): self.ctx.results[k] for k in self.ctx.results}
            with codecs.open(self.results_output_path, "w", "utf-8") as f:
                json.dump(output_dict, f, indent=2)

        if self.predictions_output_path:
            self.logger.info(
                f"Saving all predictions to {self.predictions_output_path}"
            )
            os.makedirs(self.predictions_output_path, exist_ok=True)

            output_dict = {}
            for k in self.ctx.predictions:
                key = ":".join(k)
                item = self.ctx.predictions[k]
                if (
                    isinstance(item, dict)
                    or isinstance(item, list)
                    or isinstance(item, tuple)
                    or item is None
                ):
                    output_dict[k] = item
                elif hasattr(item, "__dict__"):
                    output_dict[k] = item.__dict__
                else:
                    self.logger.warning(
                        f"Cannot save prediction for key {key}, type {type(item)}"
                    )

            with codecs.open(self.predictions_output_path, "w", "utf-8") as f:
                json.dump(output_dict, f, indent=2)
        return self.ctx
