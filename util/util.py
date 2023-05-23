from scipy.optimize import curve_fit
import pandas as pd
import os.path
import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import rfft, rfftfreq
from scipy.io import wavfile

TIME = 'Time'

# Processed Data
PSI_COL = 'PSI'
PSI_OUT_COL = 'PSI_OUT'
CURR_COL = 'CURR'
VOLT_COL = 'VOLT'
ANG_VEL_COL = 'ANG_VEL'
TEMP_DIFF_COL = 'TEMP_DIFF'

RAW_DATA_PATH = os.environ['RAW_DATA_PATH']

def open_lj_dat(folder):
    files = []
    idx = 0
    data = pd.DataFrame()
    while True:
        path = os.path.join(folder, f"data_{idx}.dat")
        if os.path.exists(path):
            files.append(path)
            idx += 1
            #with open(path) as f:
            #    for _ in range(6):
            #        f.readline()

            new = pd.read_csv(path, sep='\t', skiprows=5)

            data = pd.concat([data, new], ignore_index=True, axis=0)
        else:
            break

    if idx == 0:
        raise Exception("No data found in folder")
    return data

def moving_average(x, width):
    return np.convolve(x, np.ones(width), 'same') / np.convolve(np.ones(len(x)), np.ones(width), 'same')

def time_to_idx(t, times):
    return np.searchsorted(times, t, side='left')

def keep_times(start, end, data, time_col=TIME):
    data.reset_index(inplace=True, drop=True)
    start_idx = time_to_idx(start, data[time_col])
    end_idx = time_to_idx(end, data[time_col])
    initial_len = len(data)

    data.drop(range(0, start_idx), inplace=True)
    data.drop(range(end_idx, initial_len), inplace=True)
    data.reset_index(inplace=True)

    data[time_col] -= data[time_col][0]

def open_mono_wav(file):
    a = wavfile.read(file)

    # a[0] gives us sample rate
    data = np.array(a[1], dtype=float)

    # Why are they different?? There should only be one anyways!
    data = np.mean(data, axis=1)

    t = np.linspace(0, len(data)/a[0], len(data))

    return t, data

def fft(t, y, normalize=False):
    coefs = rfft(y)
    amplitude = 1 / len(y) * np.abs(coefs)
    phase = np.angle(coefs)

    if normalize:
        amplitude = amplitude / np.max(amplitude)

    freq = rfftfreq(len(t), d=(t[-1] - t[0])/len(t))

    return freq, amplitude, phase

def plot_all(plot, data, x_col_name=None, plot_cols=None, subplots=False):
    x_col = data.columns[0] if x_col_name is None else x_col_name
    x = data[x_col]

    if plot_cols is None:
        cols = list(data.columns)
        cols.remove(x_col)
    elif isinstance(plot_cols, str):
        cols = [plot_cols]
    else:
        cols = plot_cols

    if subplots:
        fig, axs = plot.subplots(len(cols))
    else:
        fig = plot

    for i, col in enumerate(cols):
        if subplots:
            ax = axs[i]
            ax.set_title(col)
            ax.set_xlabel(x_col)
            ax.set_ylabel(col)
        else:
            ax = plot
            ax.xlabel(x_col)
            ax.ylabel(col)

        ax.plot(x, data[col], label=col)

    return fig


def sin(x, amplitude, freq, phase, mean):
    return amplitude * np.sin(freq * x + phase) + mean


def fit_sin(t, y):
    a = np.abs(np.fft.fft(y))
    freq = np.fft.fftfreq(len(t), (t[1] - t[0]))
    f_guess = abs(freq[np.argmax(a[1:]) + 1]) * 2 * np.pi
    a_guess = np.std(y) * np.sqrt(2)
    m_guess = np.mean(y)
    init_point = (y[0] - m_guess) if np.abs(y[0] - m_guess) < a_guess else np.sign(y[0])*a_guess
    p_guess = np.arcsin(init_point / a_guess)


    fit1, cov1 = curve_fit(sin, t, y, p0=(a_guess, f_guess, p_guess, m_guess), bounds=((0.2*a_guess, 0.1*f_guess, -np.pi, -np.inf),(5*a_guess, 10*f_guess, np.pi, np.inf)))
    stderr1 = np.std(sin(t, *fit1) - y)

    fit2, cov2 = curve_fit(sin, t, y, p0=(a_guess, f_guess, (2*np.pi-p_guess) % (2*np.pi) - np.pi, m_guess), bounds=((0.2*a_guess, 0.1*f_guess, -np.pi, -np.inf),(5*a_guess, 10*f_guess, np.pi, np.inf)))
    stderr2 = np.std(sin(t, *fit2) - y)

    amplitude, freq, phase, mean = fit1 if stderr1 < stderr2 else fit2

    return amplitude, freq, phase, mean