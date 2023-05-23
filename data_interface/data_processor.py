import numpy as np
from collections import namedtuple

from data_interface.training_data_interface import TrainingDataInterface
from data_interface.raw_data_interface import RawDataInterface
from util.util import *

Quantity = namedtuple('Quantity', ['name', 'min', 'max', 'freq_ranges'])

QUANTITIES = {q.name:q for q in [
    Quantity(PSI_COL, 5, 40, [(20, 40), ]),
    Quantity(CURR_COL, -0.6, 0.6, [(20, 40), ]),
    Quantity(VOLT_COL, -200, 200, [(20, 40), ]),
    Quantity(ANG_VEL_COL, 500, 1000, [(20, 40), ]),
    Quantity(TEMP_DIFF_COL, 0, 100, []),
]}

FAIL_RATIO = 0.85
FAIL_METRIC = PSI_OUT_COL

PHASE_REF_QUANTITY = QUANTITIES[VOLT_COL]

class DataProcessor(TrainingDataInterface):

    def __init__(self, raw_data: RawDataInterface, start_time):
        self.raw_data = raw_data.data_samples
        self.start_time = start_time

        self._process_data()

    def _process_data(self):
        sample_variables = []
        sample_times = []
        max_fail_metric = None
        min_fail_metric = self.raw_data[0][FAIL_METRIC].iloc[0]

        for sample in self.raw_data:
            if sample[TIME].iloc[0] < self.start_time:
                continue

            if not max_fail_metric:
                max_fail_metric = sample[FAIL_METRIC].iloc[0]

            variables = self._process_sample(sample)
            sample_variables.append(variables)
            t_avg = (sample[TIME].iloc[0] + sample[TIME].iloc[-1])/2
            sample_times.append(t_avg)

            if sample[FAIL_METRIC].iloc[0] - min_fail_metric < FAIL_RATIO*(max_fail_metric - min_fail_metric):
                self.fail_time = t_avg
                print(f"Start time: {self.start_time}, Fail time: {self.fail_time}, Num samples: {len(sample_times)}")
                break

        self._times = np.array([sample_times])
        self._variables = np.array([np.row_stack(sample_variables)])
        self._labels = np.array([[self.fail_time]])


    def _process_sample(self, sample):
        sample = sample.copy()
        time = sample[TIME].values
        time -= time[0]

        ref_phase = None
        qty_f = []
        qty_a = []
        qty_p = []
        qty_m = []
        for q in QUANTITIES.values():
            raw_values = sample[q.name].values

            normalized_values = (raw_values - q.min)/(q.max - q.min)

            f_arr, a_arr, p_arr = fft(time, normalized_values)

            freq, amplitude, phase = self._process_quantity(q, f_arr, a_arr, p_arr)
            mean = np.mean(normalized_values)

            if q == PHASE_REF_QUANTITY:
                ref_phase = phase[0]

            qty_f.append(freq)
            qty_a.append(amplitude)
            qty_p.append(phase)
            qty_m.append(mean)

        sample_f = np.concatenate(qty_f)
        sample_a = np.concatenate(qty_a)
        sample_p = np.concatenate(qty_p)
        sample_m = np.array(qty_m)  # Only one per qty

        sample_p -= ref_phase
        sample_p = (sample_p + np.pi) % (2*np.pi) - np.pi
        sample_p /= np.pi
        sample_p = (sample_p+1)/2

        variables = np.concatenate([sample_f, sample_a, sample_p, sample_m])

        return variables

    def _process_quantity(self, qty: Quantity, freq_arr, amplitude_arr, phase_arr):
        freq = []
        amplitude = []
        phase = []

        for range in qty.freq_ranges:
            start = np.searchsorted(freq_arr, range[0], side='left')
            end = np.searchsorted(freq_arr, range[1], side='right')

            max_idx = np.argmax(amplitude_arr[start:end]) + start

            raw_freq = freq_arr[max_idx]
            normalized_freq = (raw_freq - range[0]) / (range[1] - range[0])

            freq.append(normalized_freq)
            amplitude.append(amplitude_arr[max_idx])
            phase.append(phase_arr[max_idx])

        return np.array(freq), np.array(amplitude), np.array(phase)

    @property
    def variables(self) -> np.array:
        return self._variables

    @property
    def labels(self) -> np.array:
        return self._labels

    @property
    def times(self) -> np.array:
        return self._times