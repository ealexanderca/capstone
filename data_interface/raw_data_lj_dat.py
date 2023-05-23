from typing import List

from data_interface.raw_data_interface import RawDataInterface
from util.util import *

SAMPLE_DURATION = 1  # s

class RawDataLjDat(RawDataInterface):
    def __init__(self, folder_path, start=None, end=None):
        self.folder_path = folder_path
        self.start = start
        self.end = end

        self._open_data()

        self._convert_values()

        self._create_samples()

    @property
    def data_samples(self) -> List[pd.DataFrame]:
        return self._sample_data

    def _open_data(self):
        self._all_data = open_lj_dat(self.folder_path)

        if not self.start:
            self.start = self._all_data[TIME][0]

        if not self.end:
            self.end = self._all_data[TIME][-1]

        keep_times(self.start, self.end, self._all_data)

    def _convert_values(self):
        self._all_data[PSI_COL] = (self._all_data[V0] * 1000 + 9.24) / 17.2 / 5 * 100

        self._all_data[CURR_COL] = self._all_data[V1]

        self._all_data[VOLT_COL] = self._all_data[V2] * 42 / 2

    def _create_samples(self):
        time = self._all_data[TIME].values

        sample_ts = np.arange(time[0], time[-1], SAMPLE_DURATION)
        sample_is = time_to_idx(sample_ts, time)
        self._sample_data = []

        for i in range(len(sample_is) - 1):
            start = sample_is[i]
            end = sample_is[i + 1]
            self._sample_data.append(self._all_data.iloc[start:end])
