from abc import ABC, abstractmethod


class BasePolicy(ABC):
    @abstractmethod
    def training_step(self, *args, **kwargs):
        """
        Given obs, return loss and other metrics for training.
        """
        pass

    @abstractmethod
    def validation_step(self, *args, **kwargs):
        """
        Given obs, return loss and other metrics for validation.
        """
        pass

    @abstractmethod
    def forward(self, *args, **kwargs):
        """
        Forward the NN.
        """
        pass

    @abstractmethod
    def act(self, *args, **kwargs):
        """
        Given obs, return action.
        """
        pass
