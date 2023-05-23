import numpy as np

from data_interface.training_data_interface import TrainingDataInterface
from util.util import *

from tensorflow import random

TIME_FACTOR = 0.01
VARIABLE_ERROR = 0.01

TIME_STEP = 5
MAX_STEPS = 5*60//TIME_STEP
MIN_STEPS = 3
TIME_END_MIN = 0.1

ZERO_ESCAPE_VAL = 1E-8  # Value to replace zeros in variables for escaping before zero-padding and masking

class TrainingDataFuzzer(TrainingDataInterface):

    def __init__(self, data_interface:TrainingDataInterface, num):
        self.data_interface = data_interface
        self.num = num

        self.generate_all()

    def generate_all(self):
        variables_all = []
        labels_all = []
        times_all = []

        for series_idx in range(self.data_interface.labels.shape[0]):
            variables_orig = self.data_interface.variables[series_idx]
            fail_time_orig = self.data_interface.labels[series_idx, 0]
            sample_times_orig = self.data_interface.times[series_idx]

            for _ in range(self.num):
                time_scale = random.normal([1], mean=1, stddev=TIME_FACTOR / 2)[0]
                time_scale = np.clip(time_scale, 1-4*TIME_FACTOR, 1+4*TIME_FACTOR)

                #time_end_scaled = random.uniform([1], sample_times_orig[0]/time_scale + TIME_STEP*MIN_STEPS, sample_times_orig[-1]/time_scale - TIME_END_MIN)[0]
                rand = random.uniform([1], 0, 1)[0]
                #rand = rand**1.25

                max_time = sample_times_orig[-1]/time_scale - TIME_END_MIN
                min_time = sample_times_orig[0]/time_scale + TIME_STEP*MIN_STEPS
                time_end_scaled = max_time - rand*(max_time - min_time)

                variables_new, fail_time_new, sample_times_new = self.generate_one(time_scale, time_end_scaled, variables_orig, fail_time_orig, sample_times_orig, add_error=False)

                variables_new = self._pad_array(variables_new)
                sample_times_new = self._pad_array(sample_times_new)

                variables_all.append(variables_new)
                labels_all.append([fail_time_new])
                times_all.append(sample_times_new)

        self._variables = np.stack(variables_all)
        self._labels = np.stack(labels_all)
        self._times = np.stack(times_all)

    @classmethod
    def generate_one(cls, time_scale, time_end_scaled, variables_orig, fail_time_orig, sample_times_orig, add_error=False):
        sample_times_scaled = sample_times_orig/time_scale
        fail_time_scaled = fail_time_orig/time_scale

        t_out = np.arange(sample_times_scaled[0], time_end_scaled, TIME_STEP)
        var_out = np.zeros((t_out.shape[0], variables_orig.shape[1]))
        for i in range(variables_orig.shape[1]):
            var_out[:, i] = np.interp(t_out, sample_times_scaled, variables_orig[:, i])

        if add_error:
            error = random.normal(var_out.shape, mean=1, stddev=VARIABLE_ERROR / 2).numpy()
            var_out *= error

        fail_time_out = fail_time_scaled-t_out[-1]  # Referenced from most recent sample
        fail_time_out /= TIME_STEP  # Normalize wrt time step size
        # TODO: NOTE data processor does not normalize times like this. This should be standardized.

        return var_out, fail_time_out, t_out

    @classmethod
    def _pad_array(self, array):
        zero_idxs = np.where(array == 0)
        array[zero_idxs] = ZERO_ESCAPE_VAL

        num_valid = min(MAX_STEPS, array.shape[0])

        new_shape = list(array.shape)
        new_shape[0] = MAX_STEPS
        new_array = np.zeros(new_shape)
        new_array[0:num_valid] = array[-num_valid:]

        return new_array

    @property
    def variables(self) -> np.array:
        return self._variables

    @property
    def labels(self) -> np.array:
        return self._labels

    @property
    def times(self) -> np.array:
        return self._times
