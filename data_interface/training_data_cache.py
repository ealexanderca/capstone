from time import perf_counter

import numpy as np

from data_interface.training_data_interface import TrainingDataInterface
from util.util import *

CACHE_PATH = "./cache/"

class TrainingDataCache(TrainingDataInterface):

    def __init__(self, name, constructor_f):
        self.name = name
        self.path = os.path.join(CACHE_PATH, self.name + ".npz")
        self.constructor_f = constructor_f

        if not self._load_training_data():
            print(f"Could not find {self.name} cache, getting new data...")
            start_time = perf_counter()
            data_interface = self.constructor_f()
            self._variables = data_interface.variables
            self._labels = data_interface.labels
            self._times = data_interface.times
            end_time = perf_counter()
            print(f"Done. Took {end_time-start_time}s")
            self._save_training_data()

    def _save_training_data(self):
        print(f"Saving {self.name} cache...")
        np.savez(self.path, self._variables, self._labels, self._times)
        print("Done saving.")

    def _load_training_data(self):
        if os.path.isfile(self.path):
            print(f"Found {self.name} cache, loading...")
            load = np.load(self.path)
            self._variables, self._labels, self._times = [load[file] for file in load.files]
            print("Done loading.")
            return True

        else:
            return False

    @property
    def variables(self) -> np.array:
        return self._variables

    @property
    def labels(self) -> np.array:
        return self._labels

    @property
    def times(self) -> np.array:
        return self._times