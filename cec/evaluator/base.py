from abc import ABC, abstractmethod


class BaseEvaluator(ABC):
    @abstractmethod
    def start(self, *args, **kwargs) -> None:
        """
        Start the evaluation process.
        """
        raise NotImplementedError

    @abstractmethod
    def get_results(self) -> dict:
        """
        Get the results of the evaluation.
        Should return a dict mapping from metric name to value.
        """
        raise NotImplementedError
