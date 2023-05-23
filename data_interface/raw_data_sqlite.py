import os.path
from typing import List

import numpy as np

from data_interface.raw_data_interface import RawDataInterface
from util.util import *

import sqlite3

SAMPLE_SPLIT_LENGTH = 2  #s
DROP_TIME = 2  # s
MAX_COUNTER = 2**16-1
ENCODER_PPR = 120

# Common columns
SAMPLE_COL = "SECTION"
TIME_COL = "TIME"

# Slow data columns
ACCELERATION_COL = "ACCELERATE"
PRESS_OUT_RAW_COL = "AIN6"
TEMP_1_RAW_COL = "AIN8_EF_READ_A"
TEMP_2_RAW_COL = "AIN10_EF_READ_A"
TEMP_3_RAW_COL = "AIN12_EF_READ_A"

# Fast data columns
CURR_RAW_COL = "AIN0"
VOLT_RAW_COL = "AIN2"
PRESS_RAW_COL = "AIN4"
ANG_RAW_COL = "DIO6_EF_READ_A"
ANG_RAW_IDX_COL = "DIO3_EF_READ_A"

class RawDataSQLite(RawDataInterface):
    def __init__(self, file_name, test_num, start=None, end=None):
        self.file_path = os.path.join(RAW_DATA_PATH, file_name)
        self.test_num = test_num
        self.start = start
        self.end = end

        fast_table, slow_table = self._open_data()

        self._convert_values(fast_table, slow_table)

        large_samples = self._create_samples(fast_table, slow_table)

        split_samples = self.split_samples(large_samples)

        self._sample_data = split_samples

    @property
    def data_samples(self) -> List[pd.DataFrame]:
        return self._sample_data

    def _open_data(self):
        with sqlite3.connect(self.file_path) as conn:

            fast_table = pd.read_sql_query(f"SELECT * FROM FASTDATA{self.test_num}", conn, parse_dates=[TIME_COL])
            fast_table[TIME_COL] = (fast_table[TIME_COL] - fast_table[TIME_COL].iloc[0]).dt.total_seconds()
            slow_table = pd.read_sql_query(f"SELECT * FROM SLOWDATA{self.test_num}", conn, parse_dates=[TIME_COL])
            slow_table[TIME_COL] = (slow_table[TIME_COL] - slow_table[TIME_COL].iloc[0]).dt.total_seconds()
            
            return fast_table, slow_table


    def _convert_values(self, fast_table, slow_table):
        fast_table.rename(columns={TIME_COL: TIME, }, inplace=True)

        fast_table[PRESS_RAW_COL] = (fast_table[PRESS_RAW_COL] - 0.4) / 61
        fast_table[PSI_COL] = (fast_table[PRESS_RAW_COL] * 1000 + 9.24) / 17.2 / 5 * 100

        fast_table[CURR_COL] = fast_table[CURR_RAW_COL]

        fast_table[VOLT_COL] = fast_table[VOLT_RAW_COL] * 42 / 2


    def _create_samples(self, fast_table, slow_table):
        sample_nums = slow_table[SAMPLE_COL]
        sample_fast_idxs = list(np.searchsorted(fast_table[SAMPLE_COL], sample_nums, side='left'))
        sample_fast_idxs.append(fast_table.shape[0] - 1)

        sample_data = []

        for i in range(len(sample_fast_idxs) - 1):
            start = sample_fast_idxs[i]
            end = sample_fast_idxs[i + 1]

            sample = fast_table.iloc[start:end].copy()

            if sample.shape[0] < 2E3:
                continue

            sample[ACCELERATION_COL] = slow_table[ACCELERATION_COL].iloc[i] * np.ones(sample.shape[0])
            sample[PRESS_OUT_RAW_COL] = slow_table[PRESS_OUT_RAW_COL].iloc[i] * np.ones(sample.shape[0])
            sample[PRESS_OUT_RAW_COL] = (sample[PRESS_OUT_RAW_COL] - 0.4) / 61
            sample[PSI_OUT_COL] = (sample[PRESS_OUT_RAW_COL] * 1000 + 9.24) / 17.2 / 5 * 100
            sample[TEMP_1_RAW_COL] = slow_table[TEMP_1_RAW_COL].iloc[i] * np.ones(sample.shape[0])
            #sample[TEMP_2_RAW_COL] = slow_table[TEMP_2_RAW_COL].iloc[i] * np.ones(sample.shape[0])
            sample[TEMP_3_RAW_COL] = slow_table[TEMP_3_RAW_COL].iloc[i] * np.ones(sample.shape[0])
            sample[TEMP_DIFF_COL] = sample[TEMP_1_RAW_COL] - sample[TEMP_3_RAW_COL]

            start_time = sample[TIME].iloc[0] + DROP_TIME
            end_time = sample[TIME].iloc[-1]

            if start_time > end_time:
                continue

            keep_times(start_time, end_time, sample)

            if np.min(sample[PSI_COL]) < 0:
                continue

            sample[TIME] += start_time

            ang_vel = np.diff(sample[ANG_RAW_COL], append=sample[ANG_RAW_COL].iloc[-1])
            ang_vel = ang_vel % MAX_COUNTER  # Takes care of rollover
            ang_vel[-1] = ang_vel[-2]

            ang = np.cumsum(ang_vel)
            ang = moving_average(ang, 50)
            sample[ANG_RAW_COL] = ang

            ang_vel = np.diff(ang, append=ang[-1])
            ang_vel[-1] = ang_vel[-2]
            dt = np.diff(sample[TIME], append=sample[TIME].iloc[-1])
            dt[-1] = dt[-2]
            ang_vel = ang_vel / ENCODER_PPR * 2 * np.pi / dt
            avg = np.mean(ang_vel)
            ang_vel = moving_average(ang_vel, 50)
            ang_vel = ang_vel[50:-50]

            ang_vel_full = np.ones_like(ang) * avg
            ang_vel_full[50:ang_vel.shape[0]+50] = ang_vel
            sample[ANG_VEL_COL] = ang_vel_full

            sample_data.append(sample)

        return sample_data

    def split_samples(self, large_samples):
        if not SAMPLE_SPLIT_LENGTH:
            return large_samples

        split_samples = []

        for large_sample in large_samples:
            split_times = np.arange(large_sample[TIME].iloc[0], large_sample[TIME].iloc[-1], SAMPLE_SPLIT_LENGTH)

            for i in range(len(split_times) - 1):
                sample = large_sample.copy()

                start = split_times[i]
                stop = split_times[i+1]

                keep_times(start, stop, sample)

                sample[TIME] += start

                split_samples.append(sample)

        return split_samples