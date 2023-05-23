from tensorflow import random
import os
from time import perf_counter

from data_interface.training_data_interface import TrainingDataInterface
from data_interface.raw_data_interface import DataContainter
from util.util import *
from compressor_model import compute_equilibrium, step, PSI_TO_PA, SAMPLE_TIME
from data_interface.data_processor import DataProcessor

NUM_WEAR_POINTS = 10
Y_INIT = [1.0983972138975649e-05, 350.7055202697288, 6.283185307179587, 185.4405744273075, 0.00011121524980722459, 429.2559995834747]

NUM_TRAINING = 30000
NOISE = 0.5

ABS_MIN_TIME = 1E-8

CACHE_FILE = "cache_sim_wear_data.npz"

class SimWearData(TrainingDataInterface):

    def __init__(self, num_samples, sample_time, in_leak_ratio_max, wear_rate_min, wear_rate_max, fail_time_min, fail_time_max):
        self.num_samples = num_samples
        self.sample_time = sample_time
        self.in_leak_ratio_max = in_leak_ratio_max
        self.wear_rate_min = wear_rate_min
        self.wear_rate_max = wear_rate_max
        self.fail_time_min = fail_time_min
        self.fail_time_max = fail_time_max

        if not self._load_training_data():
            print("Could not find wear sim cache, running new simulation...")
            start_time = perf_counter()
            self._compute_sim_data()
            self._generate_training_data()
            end_time = perf_counter()
            print(f"Done simulating. Took {end_time-start_time}s")
            self._save_training_data()

    def _compute_sim_data(self):
        self._leak_ratio_arr = np.linspace(0, self.in_leak_ratio_max, NUM_WEAR_POINTS)
        leak_data = []
        y = Y_INIT
        for leak_r in self._leak_ratio_arr:
            # TODO: Save solutions in cache, not processed result / variables + labels
            sol = compute_equilibrium(y_init=y, piston_in_leak_ratio=leak_r, piston_out_leak_ratio=0)[-1]

            piston_m, piston_t, theta, d_theta_dt, outlet_m, outlet_t = sol.y
            (d_piston_m_dt, d_piston_t_dt, d_theta_dt, d_dthetadt_dt, d_outlet_m_dt, d_outlet_t_dt), (
            piston_p, piston_v, piston_rho, s, I_2, T_m, outlet_p, outlet_rho) \
                = step(None, sol.y, piston_in_leak_ratio=leak_r,
                       piston_out_leak_ratio=0)

            y = np.array(sol.y)[:,-1]

            t_sample = np.arange(0, sol.t[-1], SAMPLE_TIME)
            p_sample = np.interp(t_sample, sol.t, piston_p)

            data = {
                TIME: t_sample,
                'PSI': p_sample / PSI_TO_PA,
                'CURR': 2.4*np.ones_like(p_sample),
                'VOLT': 0*np.ones_like(p_sample),
            }

            data = pd.DataFrame(data=data)

            processed = DataProcessor(DataContainter([data]), None)

            leak_data.append(processed.variables[0])

        self._leak_data_arr = np.array(leak_data)

    def _generate_training_data(self):
        wear_rate_arr = random.normal([NUM_TRAINING], mean=(self.wear_rate_max+self.wear_rate_min)/2, stddev=(self.wear_rate_max-self.wear_rate_min)/2/4)
        wear_rate_arr = np.clip(wear_rate_arr, self.wear_rate_min, self.wear_rate_max)
        fail_time_arr = random.normal([NUM_TRAINING], mean=(self.fail_time_max + self.fail_time_min) / 2, stddev=(self.fail_time_max - self.fail_time_min) / 2 / 4)
        fail_time_arr = np.clip(fail_time_arr, ABS_MIN_TIME, np.inf)

        # TODO: DOING IT LIKE THIS SETS t=0 BEFORE THE FIRST SAMPLE, BUT REALISTICALLY PREDICTIONS ARE MADE WITH t=0 AFTER LAST SAMPLE
        t_arr = np.linspace(0, self.sample_time*self.num_samples, self.num_samples)

        wear_init = self.in_leak_ratio_max / np.exp(wear_rate_arr*fail_time_arr)

        # TODO: GARBAGE...
        self._variables = np.zeros((NUM_TRAINING, self.num_samples, len(self._leak_data_arr[0])))
        fail_time_noisy_arr = random.normal([NUM_TRAINING], mean=fail_time_arr, stddev=NOISE)
        fail_time_noisy_arr = np.clip(fail_time_noisy_arr, 0, np.inf)
        self._labels = fail_time_noisy_arr
        for i in range(NUM_TRAINING):
            for j in range(len(self._leak_data_arr[0])):
                wear_arr = wear_init[i] * np.exp(wear_rate_arr[i] * t_arr)
                self._variables[i, :, j] = np.interp(wear_arr, self._leak_ratio_arr, self._leak_data_arr[:, j])


    def _save_training_data(self):
        print("Saving wear sim data as cache...")
        np.savez(CACHE_FILE, self._variables, self._labels)
        print("Done saving.")

    def _load_training_data(self):
        if os.path.isfile(CACHE_FILE):
            print("Found wear sim cache, loading...")
            load = np.load(CACHE_FILE)
            self._variables, self._labels = [load[file] for file in load.files]
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