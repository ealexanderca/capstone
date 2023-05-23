from util.util import *
import scipy.stats

STEP_SIZE = 100

def psi_stats(path, name, num_tape, color):
    data = open_lj_dat(path)
    keep_times(0.5, 60.5, data)

    data[PSI_COL] = (data[V0]*1000 + 9.24) / 17.2 / 5 * 100
    data[PSI_COL] = moving_average(data[PSI_COL], 5)

    psi_mean = np.mean(data[PSI_COL])
    psi = data[PSI_COL]

    start_idxs = []
    end_idxs = []
    max_idxs = []
    idx = 0
    while True:
        idx += STEP_SIZE
        if idx >= len(psi):
            break

        if len(start_idxs) <= len(end_idxs) and psi[idx] > psi_mean > psi[idx - STEP_SIZE]:
            start_idxs.append(idx)

        if len(start_idxs) > len(end_idxs) and psi[idx] < psi_mean < psi[idx - STEP_SIZE]:
            end_idxs.append(idx-STEP_SIZE)
            max_idx = np.argmax(psi[start_idxs[-1]:end_idxs[-1]]) + start_idxs[-1]
            max_idxs.append(max_idx)
            idx = max_idx


    max_psis = psi[max_idxs]

    """
    plt.plot(data[TIME], psi, 'b-', label=name)
    plt.plot(data[TIME][start_idxs], psi[start_idxs], 'go', label="Start")
    plt.plot(data[TIME][end_idxs], psi[end_idxs], 'ro', label="End")
    plt.plot(data[TIME][max_idxs], max_psis, 'yo', label="End")
    plt.show()
    """

    max_num = len(max_psis)
    max_mean = np.mean(max_psis)
    max_stdev = np.std(max_psis)
    max_sterr = max_stdev/np.sqrt(max_num)

    print(f"{name}: max_num = {max_num}")
    print(f"{name}: max_mean = {max_mean}")
    print(f"{name}: max_stdev = {max_stdev}")
    print(f"{name}: max_sterr = {max_sterr}")

    return max_num, max_mean, max_stdev, max_sterr, max_psis, name, num_tape, color

stats = [
    psi_stats(r"C:\Users\Andrei\Desktop\Capstone Test 2 (CFP)\No Tape Redo", "0 Tape", 0, "C1"),
    psi_stats(r"C:\Users\Andrei\Desktop\Capstone Test 2 (CFP)\1 Tape", "1 Tape", 1, "C2"),
    psi_stats(r"C:\Users\Andrei\Desktop\Capstone Test 2 (CFP)\2 Tape", "2 Tape", 2, "C3"),
    psi_stats(r"C:\Users\Andrei\Desktop\Capstone Test 2 (CFP)\4 Tape", "4 Tape", 4, "C4"),
    psi_stats(r"C:\Users\Andrei\Desktop\Capstone Test 2 (CFP)\6 Tape", "6 Tape", 6, "C5"),
]

for stat in stats:
    max_num, max_mean, max_stdev, max_sterr, max_psis, name, num_tape, color = stat
    plt.hist(max_psis, bins=50, color=color, label=f"{name}, {max_num} samples")

plt.title("Peak Pressure Distributions")
plt.xlabel("PSI")
plt.ylabel("num")
plt.legend(loc="upper left")
plt.show()

num_tapes = []
max_means = []
max_sterrs = []

for i in range(1, len(stats)):
    max_num_ref, max_mean_ref, max_stdev_ref, _, _, _, _, _ = stats[i-1]
    stat = stats[i]
    max_num, max_mean, max_stdev, max_sterr, max_psis, name, num_tape, color = stat
    num_tapes.append(num_tape)
    max_means.append(max_mean)
    max_sterrs.append(max_sterr)

    z_score = (max_mean_ref - max_mean)/np.sqrt(max_stdev_ref**2/max_num_ref + max_stdev**2/max_num)
    p_score = 1-scipy.stats.norm.cdf(z_score)

    print(f"{name}: z_score = {z_score}")
    print(f"{name}: p_score = {p_score}")

plt.errorbar(np.array(num_tapes)*0.4/6, max_means, yerr = max_sterrs, capsize=10, fmt ='o', label="Mean Peak Pressure, Standard Error")
plt.title("Mean Peak Pressure vs. Tape Thickness")
plt.xlabel("mm of tape")
plt.ylabel("Mean Peak PSI")
plt.legend(loc="upper right")
plt.show()