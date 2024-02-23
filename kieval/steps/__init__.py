from kieval.steps.base_step import BaseStep
from kieval.steps.simple_multiple_choice import SimpleMultipleChoiceStep
from kieval.steps.cloze_prompt import ClozePromptStep
from kieval.steps.compute_lm_loss import ComputeLMLossStep
from kieval.steps.min_k_prob import MinKProbStep
from kieval.steps.interactive_evaluation import InteractiveEvaluationStep


TYPE_TO_STEP = {
    "simple_multiple_choice": SimpleMultipleChoiceStep,
    "cloze_prompt": ClozePromptStep,
    "compute_lm_loss": ComputeLMLossStep,
    "min_k_prob": MinKProbStep,
    "interactive_evaluation": InteractiveEvaluationStep,
}


def load_step_class(step_type):
    assert step_type in TYPE_TO_STEP
    step_class = TYPE_TO_STEP[step_type]
    return step_class


def load_step(step_type, step_config):
    step_class = load_step_class(step_type)
    step_instance = step_class(**step_config)
    return step_instance
