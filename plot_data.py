from data_interface.data_processor import DataProcessor
from data_interface.raw_data_sqlite import *
from util.util import *

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

DATA_FILE = r"Test_Mar18_Paint_CompressorB_100Duty_Paint_from11_49_CapB.db"
TEST_NUM = 1
START_TIME = 704
END_TIME = 1000

def get_freq(sample, data_col, main_freq, freq_tolerance, time_col=TIME):
    freq, amplitude, phase = fft(sample[time_col].values, sample[data_col].values)

    start = np.searchsorted(freq, main_freq - freq_tolerance, side='left')
    end = np.searchsorted(freq, main_freq + freq_tolerance, side='right')

    max_idx = np.argmax(amplitude[start:end]) + start

    return freq[max_idx], amplitude[max_idx], phase[max_idx]

print("Loading data...")
raw = RawDataSQLite(DATA_FILE, TEST_NUM)

data = pd.DataFrame()
good_samples = []

for sample in raw.data_samples:
    sample = sample.copy()

    good_samples.append(sample)
    data = pd.concat([data, sample], ignore_index=True, axis=0)

proc = DataProcessor(raw, START_TIME)

accel_start_idx = np.searchsorted(data[ACCELERATION_COL], 1, side='left')
accel_start_time = data[TIME].iloc[accel_start_idx]
fail_time = proc.labels[0][0]

plt.plot(data[TIME], data[PSI_COL], 'C0', label='PSI')
plt.plot(data[TIME], data[PSI_OUT_COL], 'C1', label='PSI_OUT')
plt.axvline(x=accel_start_time, color='r')
plt.axvline(x=fail_time, color='r')
plt.legend(loc="upper left")
plt.ylabel("PSI")
plt.xlabel("Time")
ax2 = plt.twinx()
ax2.plot(data[TIME], data[TEMP_1_RAW_COL] - 273.15, 'C2', label='TC1')
#ax2.plot(data[TIME], data[TEMP_2_RAW_COL] - 273.15, 'C2', label='TC2')
ax2.plot(data[TIME], data[TEMP_3_RAW_COL] - 273.15, 'C3', label='TC3')
ax2.plot(data[TIME], data[TEMP_DIFF_COL], 'C4', label='TEMP DIFF')
ax2.legend(loc="upper right")
ax2.set_ylabel("deg C")
ax2.set_xlabel("Time")

plt.show()

plt.plot(data[TIME], data[VOLT_COL], 'C0', label='Volt')
plt.legend(loc="upper left")
plt.ylabel("V")
plt.xlabel("Time")
ax2 = plt.twinx()
ax2.plot(data[TIME], data[CURR_COL], 'C1', label='Curr')
ax2.legend(loc="upper right")
ax2.set_ylabel("A")
ax2.set_xlabel("Time")

plt.show()

plt.plot(data[TIME], data[ANG_RAW_COL], 'C0', label='Angle')
plt.legend(loc="upper left")
plt.ylabel("Angle")
plt.xlabel("Time")
ax2 = plt.twinx()
ax2.plot(data[TIME], data[ANG_RAW_IDX_COL], 'C1', label='Idx')
ax2.legend(loc="upper right")
ax2.set_ylabel("Idx")
ax2.set_xlabel("Time")

plt.show()


plt.plot(data[TIME], data[ANG_VEL_COL], 'C0', label='Velocity')
plt.legend(loc="upper left")
plt.ylabel("Angular Velocity")
plt.xlabel("Time")
ax2 = plt.twinx()
ax2.plot(data[TIME], data[ANG_RAW_IDX_COL], 'C1', label='Idx')
ax2.legend(loc="upper right")
ax2.set_ylabel("Idx")
ax2.set_xlabel("Time")

plt.show()

print("Calculating FFT...")
t_avg = []
power_real = []
current = []
press_avg = []
press_amp = []
press_freq = []
vel_avg = []
vel_amp = []
vel_freq = []
temp_diff = []
press_out = []
for sample in good_samples:
    t_avg.append((sample[TIME].values[0]+sample[TIME].values[-1])/2)

    v_freq, v_amp, v_phase = get_freq(sample, VOLT_RAW_COL, 60, 10)
    i_freq, i_amp, i_phase = get_freq(sample, CURR_RAW_COL, 60, 10)
    pf_angle = i_phase - v_phase
    power_real.append(v_amp*i_amp*np.cos(pf_angle))
    current.append(i_amp)

    p_freq, p_amp, p_phase = get_freq(sample, 'PSI', 30, 10)
    press_avg.append(np.mean(sample[PSI_COL].values))
    press_amp.append(p_amp)
    press_freq.append(p_freq)

    freq, amp, phase = get_freq(sample, ANG_VEL_COL, 30, 10)
    vel_avg.append(np.mean(sample[ANG_VEL_COL].values))
    vel_amp.append(amp)
    vel_freq.append(freq)

    temp_diff.append(sample[TEMP_DIFF_COL].iloc[0])
    press_out.append(sample[PRESS_OUT_RAW_COL].iloc[0])

t_avg = np.array(t_avg)
power_real = np.array(power_real)
current = np.array(current)
press_avg = np.array(press_avg)
press_amp = np.array(press_amp)
press_freq = np.array(press_freq)
vel_avg = np.array(vel_avg)
vel_amp = np.array(vel_amp)
vel_freq = np.array(vel_freq)
temp_diff = np.array(temp_diff)
press_out = np.array(press_out)

plt.plot(t_avg, press_avg, 'b-', label='press_avg')
plt.plot(t_avg, press_avg - press_amp, 'b--', label='press_amp')
plt.plot(t_avg, press_avg + press_amp, 'b--', label='press_amp')
plt.legend(loc="upper left")
plt.ylabel("PSI")
plt.xlabel("Time")
ax2 = plt.twinx()
ax2.plot(t_avg, press_freq, 'C1', label='press_freq')
ax2.legend(loc="upper right")
ax2.set_ylabel("press_freq")
ax2.set_xlabel("Time")
plt.show()

plt.plot(t_avg, current, 'C0', label='Curr')
plt.legend(loc="upper left")
plt.ylabel("Curr")
plt.xlabel("Time")
ax2 = plt.twinx()
ax2.plot(t_avg, power_real, 'C1', label='Power')
ax2.legend(loc="upper right")
ax2.set_ylabel("Power")
ax2.set_xlabel("Time")
plt.show()

start_idx = time_to_idx(START_TIME, t_avg)
end_idx = time_to_idx(END_TIME, t_avg)
plt.title("Sensor Values vs. Time")
#plt.plot(t_avg, power_real/power_real[start_idx]-1, label='power_real')
plt.plot(t_avg, current/current[start_idx]*100-100, label='Current Amplitude')
plt.plot(t_avg, press_avg/press_avg[start_idx]*100-100, label='Piston Pressure Mean')
plt.plot(t_avg, press_amp/press_amp[start_idx]*100-100, label='Piston Pressure Amplitude')
#plt.plot(t_avg, press_freq/press_freq[start_idx]*100-100, label='press_freq')
plt.plot(t_avg, vel_avg/vel_avg[start_idx]*100-100, label='Velocity Mean')
plt.plot(t_avg, vel_amp/vel_amp[start_idx]*100-100, label='Velocity Amplitude')
plt.plot(t_avg, temp_diff/temp_diff[start_idx]*100-100, label='Temperature Difference')
plt.plot(t_avg, press_out/press_out[start_idx]*100-100, label='Outlet Pressure')
#plt.plot(t_avg, vel_freq/vel_freq[start_idx]*100-100, label='vel_freq')
plt.plot(t_avg[[0,-1]], [1,1], 'k--', label='Neutral')
plt.legend(loc="lower left")
plt.ylabel("% Change")
plt.xlabel("Time (s)")
plt.show()

plt.plot(t_avg, press_avg, 'b-', label='press_avg')
plt.plot(t_avg, press_avg - press_amp, 'b--', label='press_amp')
plt.plot(t_avg, press_avg + press_amp, 'b--', label='press_amp')
plt.legend(loc="upper left")
plt.ylabel("PSI")
plt.xlabel("Time")
plt.ylim(np.min((press_avg - press_amp)[start_idx:end_idx]), np.max((press_avg + press_amp)[start_idx:end_idx]))
ax2 = plt.twinx()
ax2.plot(t_avg, vel_avg, 'r-', label='vel_avg')
ax2.plot(t_avg, vel_avg - vel_amp, 'r--', label='vel_amp')
ax2.plot(t_avg, vel_avg + vel_amp, 'r--', label='vel_amp')
ax2.legend(loc="upper right")
ax2.set_ylabel("Vel")
ax2.set_xlabel("Time")
ax2.set_ylim(np.min((vel_avg - vel_amp)[start_idx:end_idx]), np.max((vel_avg + vel_amp)[start_idx:end_idx]))
plt.show()

MAX_FREQ = 5000
MIN_FREQ = 5
FREQ_RATIO = 1.01
num_freqs = int(np.log(MAX_FREQ/MIN_FREQ)/np.log(FREQ_RATIO))
freqs = MIN_FREQ*FREQ_RATIO**(np.arange(0, num_freqs))

def plot_fft(data_col, title):
    A = []
    for sample in good_samples:
        freq, amplitude, phase = fft(sample[TIME].values, sample[data_col].values)

        idxs = np.concatenate([[0], np.searchsorted(freq, freqs, side='right')])

        amplitude_cumsum = np.cumsum(amplitude)[idxs]

        amps = np.diff(amplitude_cumsum)
        A.append(amps)

    #f = np.arange(0, num_freqs)
    T, F = np.meshgrid(t_avg, freqs)
    A = np.stack(A).T

    min_amp = np.min(A[np.where(A > 0)])
    A[np.where(A <= 0)] = min_amp

    plt.pcolormesh(T, F, A, cmap='hot', norm=matplotlib.colors.LogNorm(vmin=np.max(A)*1E-3, vmax=np.max(A)))
    plt.yscale('log')
    plt.ylabel("Frequency")
    plt.xlabel("Time")
    plt.title(title)
    plt.show()

plot_fft('PSI', 'Pressure')
plot_fft(CURR_RAW_COL, 'Current')
plot_fft(VOLT_RAW_COL, 'Voltage')
plot_fft(ANG_VEL_COL, 'Angular Velocity')