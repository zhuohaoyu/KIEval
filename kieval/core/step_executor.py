from kieval.core.context import Context
from kieval.steps import load_step
import logging, json, time
from tqdm import tqdm
from termcolor import colored


class StepExecutor:
    # run steps from config yaml
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.steps = []

    def load_steps(self):
        # load steps from config yaml
        self.logger.debug(
            f"Loading steps from config: {json.dumps(self.config, indent=2)}"
        )
        for step in self.config:
            step_type = step["step_type"]
            if "step_name" in step:
                step_name = step["step_name"]
            else:
                step_name = step["step_name"] = f"unnamed_{step_type}_step"
            self.steps.append([step_type, step_name, load_step(step_type, step)])
        self.logger.info(f"Loaded steps: {[(s[0], s[1]) for s in self.steps]}")

    def run_steps(self, context: Context):
        # run steps from config yaml
        for step_type, step_name, step in self.steps:
            time1 = time.time()
            self.logger.info(
                f"Preparing step {colored(step_name, attrs=['bold'])}({step_type})"
            )
            step.preprocess(context)
            self.logger.info(
                f"Running step {colored(step_name, attrs=['bold'])}({step_type})"
            )
            step.run(context)
            time2 = time.time()
            self.logger.info(
                f"Step {colored(step_name, attrs=['bold'])}({step_type}) done, time elapsed: {time2 - time1:.2f}s"
            )
            step.postprocess(context)

        return context
