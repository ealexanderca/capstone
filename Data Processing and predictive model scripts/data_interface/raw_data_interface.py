from abc import ABC, abstractmethod
from typing import List

import numpy as np
import pandas as pd


class RawDataInterface(ABC):

    @property
    @abstractmethod
    def data_samples(self) -> List[pd.DataFrame]:
        pass


class DataContainter(RawDataInterface):
    def __init__(self, data: List[pd.DataFrame]):
        self._data = data

    @property
    def data_samples(self) -> List[pd.DataFrame]:
        return self._data