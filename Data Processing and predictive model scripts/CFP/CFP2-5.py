from util.util import *
r"""
def plot_psi(path, name, color):
    data = open_lj_dat(path)
    keep_times(0.5, 10.5, data)

    data[PSI_COL] = (data[V0]*1000 + 9.24) / 17.2 / 5 * 100
    max_curr_idx = np.argmax(moving_average(data[PSI_COL], 100))
    max_curr_time = data[TIME][max_curr_idx]

    data[TIME] -= max_curr_time

    plt.plot(data[TIME], moving_average(data[PSI_COL], 5), color, label=name)

plot_psi(r"c:\Users\Andrei\Desktop\Capstone Test 2 (CFP)\No Tape Redo", "0 Tape", "C1")
plot_psi(r"c:\Users\Andrei\Desktop\Capstone Test 2 (CFP)\1 Tape", "1 Tape", "C2")
plot_psi(r"c:\Users\Andrei\Desktop\Capstone Test 2 (CFP)\2 Tape", "2 Tape", "C3")
plot_psi(r"c:\Users\Andrei\Desktop\Capstone Test 2 (CFP)\4 Tape", "4 Tape", "C4")
plot_psi(r"c:\Users\Andrei\Desktop\Capstone Test 2 (CFP)\6 Tape", "6 Tape", "C5")

plt.title("PSI")
plt.xlim(0.01, 0.06)
plt.ylim(12.8, 17.8)
plt.legend()
plt.ylabel("PSI")
plt.xlabel("Time (s)")

plt.show()

def plot_curr(path, name, color):
    data = open_lj_dat(path)
    keep_times(0.5, 10.5, data)

    data[CURR_COL] = data[V1]
    max_curr_idx = np.argmax(moving_average(data[CURR_COL], 100))
    max_curr_time = data[TIME][max_curr_idx]

    data[TIME] -= max_curr_time

    plt.plot(data[TIME], moving_average(data[CURR_COL], 5), color, label=name)

plot_curr(r"c:\Users\Andrei\Desktop\Capstone Test 2 (CFP)\No Tape Redo", "0 Tape", "C1")
plot_curr(r"c:\Users\Andrei\Desktop\Capstone Test 2 (CFP)\1 Tape", "1 Tape", "C2")
plot_curr(r"c:\Users\Andrei\Desktop\Capstone Test 2 (CFP)\2 Tape", "2 Tape", "C3")
plot_curr(r"c:\Users\Andrei\Desktop\Capstone Test 2 (CFP)\4 Tape", "4 Tape", "C4")
plot_curr(r"c:\Users\Andrei\Desktop\Capstone Test 2 (CFP)\6 Tape", "6 Tape", "C5")

plt.title("Current Sensor Vout")
plt.xlim(0.005, 0.03)
plt.ylim(2.2,2.8)
plt.legend()
plt.ylabel("CURR")
plt.xlabel("Time (s)")

plt.show()
"""

def plot_volt(path, name, color):
    data = open_lj_dat(path)
    keep_times(0.5, 60.5, data)

    data[VOLT_COL] = data[V2]*42/2
    max_volt_idx = np.argmin(moving_average(data[VOLT_COL], 100))
    max_volt_time = data[TIME][max_volt_idx]

    data[TIME] -= max_volt_time

    plt.plot(data[TIME], moving_average(data[VOLT_COL], 5), color, label=name)

plot_volt(r"c:\Users\Andrei\Desktop\Capstone Test 2 (CFP)\No Tape Redo", "0 Tape", "C1")
plot_volt(r"c:\Users\Andrei\Desktop\Capstone Test 2 (CFP)\1 Tape", "1 Tape", "C2")
plot_volt(r"c:\Users\Andrei\Desktop\Capstone Test 2 (CFP)\2 Tape", "2 Tape", "C3")
plot_volt(r"c:\Users\Andrei\Desktop\Capstone Test 2 (CFP)\4 Tape", "4 Tape", "C4")
plot_volt(r"c:\Users\Andrei\Desktop\Capstone Test 2 (CFP)\6 Tape", "6 Tape", "C5")

plt.title("Voltage In")
plt.xlim(0.012, 0.036)
plt.ylim(-200,200)
plt.legend()
plt.ylabel("V")
plt.xlabel("Time (s)")

plt.show()

