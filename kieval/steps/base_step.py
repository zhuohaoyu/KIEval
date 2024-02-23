import abc
from kieval.core.context import Context
from typing import Optional


class BaseStep(abc.ABC):
    """Base class for all steps."""

    def __init__(
        self,
        step_type: str,
        step_name: Optional[str] = None,
        description: Optional[str] = None,
    ):
        self.step_type = step_type
        self.step_name = step_name
        self.description = description

    def run_prediction(self, dataset, inference_function, inference_kwargs):
        """Run prediction on a dataset with an inference function.

        Args:
            dataset (torch.utils.data.Dataset): The dataset to run prediction on.
            inference_function (function): The inference function.
            inference_kwargs (dict): The inference function keyword arguments.

        Returns:
            dict: The predictions.
        """

        inference_function(dataset, **inference_kwargs)

    @abc.abstractmethod
    def preprocess(self, context: Context):
        """Prepare the step.

        Args:
            context (dict): The context dictionary.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def run(self, context: Context):
        """Run the step.

        Args:
            context (dict): The context dictionary.

        Returns:
            dict: The updated context dictionary.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def postprocess(self, context: Context):
        """Cleanup the step.

        Args:
            context (dict): The context dictionary.
        """
        raise NotImplementedError
