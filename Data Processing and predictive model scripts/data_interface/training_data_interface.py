from abc import ABC, abstractmethod
import numpy as np

class TrainingDataInterface(ABC):

    @property
    @abstractmethod
    def variables(self) -> np.array:
        """
        3D array, (series #, sample, features)
        :return:
        """
        pass

    @property
    @abstractmethod
    def labels(self) -> np.array:
        """
        2D array, (series #, label)
        :return:
        """
        pass

    @property
    @abstractmethod
    def times(self) -> np.array:
        """
        2D array, (series #, sample)
        :return:
        """
        pass
