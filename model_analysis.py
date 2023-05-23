import tensorflow as tf

from data_interface.data_processor import DataProcessor
from data_interface.raw_data_sqlite import *
from util.util import *
from train import weibull_pdf, weibull_cdf, weibull_mean, weibull_neg_log_likelihood_loss, VALIDATION_DATA_PATH, \
    MODEL_CHECKPOINT_PATH, weibull_mode, weibull_median, weibull_inv_cdf
from data_interface.training_data_fuzzer import TrainingDataFuzzer, TIME_STEP

tf.config.set_visible_devices([], 'GPU')  # Disable GPU

CONFIDENCE_INTERVAL = 0.90

DATA_FILE = r"Test_Mar18_Paint_CompressorB_100Duty_Paint_from11_49_CapB.db"
TEST_NUM = 1
START_TIME = 704

def get_freq(sample, data_col, main_freq, freq_tolerance, time_col=TIME):
    freq, amplitude, phase = fft(sample[time_col].values, sample[data_col].values)

    start = np.searchsorted(freq, main_freq - freq_tolerance, side='left')
    end = np.searchsorted(freq, main_freq + freq_tolerance, side='right')

    max_idx = np.argmax(amplitude[start:end]) + start

    return freq[max_idx], amplitude[max_idx], phase[max_idx]

def main():
    load = np.load(VALIDATION_DATA_PATH)
    x_real, y_real, source_idx = [load[file] for file in load.files]
    model = tf.keras.models.load_model(MODEL_CHECKPOINT_PATH, custom_objects={'weibull_neg_log_likelihood_loss': weibull_neg_log_likelihood_loss})

    time_real = y_real[:, 0]
    y_pred = model.predict(x_real)
    scale_pred = y_pred[:, 0]
    shape_pred = y_pred[:, 1]
    time_mean_pred = weibull_mean(scale_pred, shape_pred)

    plt.title("Real vs. Predicted MTBF Times")
    plt.xlabel("Real")
    plt.ylabel("Predicted MTBF")
    #plt.plot(time_real*TIME_STEP, time_mean_pred*TIME_STEP, "o")
    plt.scatter(time_real * TIME_STEP, time_mean_pred * TIME_STEP, color=[f'C{i}' for i in source_idx])
    plt.plot([np.min(time_real*TIME_STEP), np.max(time_real*TIME_STEP)], [np.min(time_real*TIME_STEP), np.max(time_real*TIME_STEP)], "r-")
    plt.show()

    cum_prob = weibull_cdf(scale_pred, shape_pred, time_real)
    cum_prob = np.sort(cum_prob)
    portion = (np.arange(len(cum_prob)) + 1)/len(cum_prob)

    plt.title("Q-Q Plot")
    plt.xlabel("% samples")
    plt.ylabel("Predicted CDF")
    plt.plot(portion*100, cum_prob*100)
    plt.plot([0, 100], [0, 100], "r-")
    plt.show()


    # Validation
    print("Loading data...")
    raw = RawDataSQLite(DATA_FILE, TEST_NUM)
    data_processor = DataProcessor(raw, START_TIME)
    print("Loaded raw data.")

    t_avg = []
    power_real = []
    current = []
    press_avg = []
    press_amp = []
    rot_freq = []
    temp = []
    for sample in raw.data_samples:
        sample = sample.copy()
        t_avg.append((sample[TIME].values[0] + sample[TIME].values[-1]) / 2)

        v_freq, v_amp, v_phase = get_freq(sample, VOLT_RAW_COL, 60, 10)
        i_freq, i_amp, i_phase = get_freq(sample, CURR_RAW_COL, 60, 10)
        pf_angle = i_phase - v_phase
        power_real.append(v_amp * i_amp * np.cos(pf_angle))
        current.append(i_amp)

        p_freq, p_amp, p_phase = get_freq(sample, 'PSI', 30, 10)
        press_avg.append(np.mean(sample[PSI_COL].values))
        press_amp.append(p_amp)
        rot_freq.append(p_freq)

        temp.append(sample[TEMP_DIFF_COL].iloc[0])

    t_avg = np.array(t_avg)
    power_real = np.array(power_real)
    current = np.array(current)
    press_avg = np.array(press_avg)
    press_amp = np.array(press_amp)
    rot_freq = np.array(rot_freq)
    temp = np.array(temp)

    predicted_mean = []
    predicted_mode = []
    predicted_median = []
    predicted_top_percentile = []
    predicted_bottom_percentile = []
    predicted_scale = []
    predicted_shape = []
    real_times = []
    current_times = []
    start_idx = np.searchsorted(t_avg, data_processor.times[0][0])
    for i, end_time in enumerate(data_processor.times[0][2:]):
        if end_time > data_processor.labels[0, 0]:
            break
        var_out, fail_time_out, t_out = TrainingDataFuzzer.generate_one(1, end_time, data_processor.variables[0], data_processor.labels[0, 0], data_processor.times[0], add_error=False)
        var_out = TrainingDataFuzzer._pad_array(var_out)
        y_pred = np.array(model(np.array([var_out]))[0])
        scale_pred = y_pred[0]
        shape_pred = y_pred[1]
        time_mean_pred = weibull_mean(scale_pred, shape_pred)
        time_mode_pred = weibull_mode(scale_pred, shape_pred)
        time_median_pred = weibull_median(scale_pred, shape_pred)

        percentile = (1-CONFIDENCE_INTERVAL)/2
        top_percentile_pred = weibull_inv_cdf(scale_pred, shape_pred, 1-percentile)
        bottom_percentile_pred = weibull_inv_cdf(scale_pred, shape_pred, percentile)

        predicted_mean.append(time_mean_pred * TIME_STEP)
        predicted_mode.append(time_mode_pred * TIME_STEP)
        predicted_median.append(time_median_pred * TIME_STEP)
        predicted_top_percentile.append(top_percentile_pred * TIME_STEP)
        predicted_bottom_percentile.append(bottom_percentile_pred * TIME_STEP)
        predicted_scale.append(scale_pred)
        predicted_shape.append(shape_pred)
        real_times.append(fail_time_out*TIME_STEP)
        current_times.append(end_time)

        idxs = np.where((t_avg <= end_time) & (t_avg >= t_avg[start_idx]))

        #plt.plot(t_avg, power_real / power_real[start_idx], label='power_real')
        plt.plot(t_avg[idxs]-end_time, current[idxs] / current[start_idx], label='Current')
        plt.plot(t_avg[idxs]-end_time, press_avg[idxs] / press_avg[start_idx], label='Pressure Mean')
        plt.plot(t_avg[idxs]-end_time, press_amp[idxs] / press_amp[start_idx], label='Pressure Amplitude')
        #plt.plot(t_avg[idxs]-end_time, rot_freq[idxs] / rot_freq[start_idx], label='rot_freq')
        plt.plot(t_avg[idxs] - end_time, temp[idxs] / temp[start_idx], label='Temperature')
        #plt.plot(t_avg[[0, -1]], [1, 1], 'k--', label='neutral')
        plt.legend(loc="upper left")
        plt.ylabel("Ratio")
        plt.xlabel("Time")
        plt.ylim(0.5, 1.5)
        ax2 = plt.twinx()
        ts = np.linspace(end_time, t_avg[-1]*3, 1000)
        ax2.plot(ts-end_time, weibull_pdf(scale_pred.astype(np.float32), shape_pred.astype(np.float64), np.clip((ts-end_time)/TIME_STEP, 0, np.inf)).numpy(), 'r-', label='PDF')
        ax2.axvline(x=fail_time_out*TIME_STEP,  color='k', linestyle='--', label='Real Fail Time')
        #ax2.legend(loc="upper right")
        ax2.set_ylabel("PDF")
        ax2.set_xlabel("Time")
        ax2.set_ylim(0, 0.40)
        ax2.legend(loc="upper right")
        plt.xlim(-(data_processor.times[0][-1]-t_avg[start_idx])*1.25, (data_processor.times[0][-1]-t_avg[start_idx])*1.25)
        #plt.show()
        plt.savefig(f'./cache/frame_{i}.png')
        plt.clf()

    plt.title("Real vs. Predicted MTBF Times")
    plt.plot(real_times, predicted_mean, label='Predicted Mean')
    plt.plot(real_times, predicted_mode, label='Predicted Mode')
    plt.plot(real_times, predicted_median, label='Predicted Median')
    plt.plot(real_times, predicted_top_percentile, 'b--', label=f'Predicted {CONFIDENCE_INTERVAL*100}% Confidence Interval')
    plt.plot(real_times, predicted_bottom_percentile, 'b--')
    plt.plot(real_times, real_times, 'r--', label='Ideal')
    plt.legend(loc="upper left")
    plt.ylabel("Predicted Time to Failure (s)")
    plt.xlabel("Real Time to Failure (s)")
    plt.legend()
    plt.show()

    plt.plot(real_times, predicted_shape, 'C0', label='predicted_shape')
    plt.ylabel("Shape")
    plt.xlabel("Real Time to Failure")
    plt.legend(loc="upper left")
    ax2 = plt.twinx()
    ax2.plot(real_times, predicted_scale, 'C1', label='predicted_scale')
    ax2.set_ylabel("Scale")
    #ax2.set_xlabel("Time")
    ax2.legend(loc="upper right")
    plt.legend()
    plt.show()

    real_times = np.array(real_times)
    predicted_mean = np.array(predicted_mean)
    abs_error = np.abs(real_times-predicted_mean)
    mean_abs_error = np.mean(abs_error)
    print(f"Mean Abs. Error = {mean_abs_error}")

    predicted_top_percentile = np.array(predicted_top_percentile)
    predicted_bottom_percentile = np.array(predicted_bottom_percentile)
    predicted_conf = np.abs(predicted_top_percentile - predicted_bottom_percentile)/2
    #predicted_conf = np.min([np.abs(predicted_top_percentile-predicted_mean), np.abs(predicted_bottom_percentile-predicted_mean)], axis=0)

    plt.title("Prediction Absolute Error")
    plt.plot(real_times, abs_error, 'C0', label='Prediction Absolute Error')
    plt.axhline(y=mean_abs_error,  color='r', linestyle='--', label=f'Prediction Mean Absolute Error ({mean_abs_error:.1f}s)')
    plt.plot(real_times, predicted_conf, 'C1', label=f'Predicted {CONFIDENCE_INTERVAL*100}% Confidence Interval')
    plt.ylabel("Absolute Error (s)")
    plt.xlabel("Time to Failure (s)")
    plt.legend()
    plt.legend()
    plt.show()

    plt.title("Error Distribution")
    plt.hist(abs_error, density=True)
    plt.ylabel("PDF")
    plt.xlabel("Absolute Error (s)")
    plt.show()


if __name__ == '__main__':
    main()
