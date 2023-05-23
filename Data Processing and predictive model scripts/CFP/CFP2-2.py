from util.util import *

data_a = open_lj_dat(r"C:\Users\Andrei\Desktop\Capstone Test 2 (CFP)\No Tape Redo")
data_b = open_lj_dat(r"C:\Users\Andrei\Desktop\Capstone Test 2 (CFP)\6 Tape")

keep_times(0.5, 60.5, data_a)
keep_times(0.5, 60.5, data_b)

data_a[PSI_COL] = (data_a[V0]*1000 + 9.24) / 17.2 / 5 * 100
data_b[PSI_COL] = (data_b[V0]*1000 + 9.24) / 17.2 / 5 * 100

print(f"No Tape: Min. PSI = {min(data_a[PSI_COL])}")
print(f"6 Tape: Min. PSI = {min(data_b[PSI_COL])}")

print(f"No Tape: Max. PSI = {max(data_a[PSI_COL])}")
print(f"6 Tape: Max. PSI = {max(data_b[PSI_COL])}")


mean_psi_a = np.mean(data_a[PSI_COL])
mean_psi_b = np.mean(data_b[PSI_COL])

print(f"No Tape: Mean PSI = {mean_psi_a}")
print(f"6 Tape: Mean PSI = {mean_psi_b}")

rms_psi_a = np.sqrt(np.mean((data_a[PSI_COL] - mean_psi_a)**2))
rms_psi_b = np.sqrt(np.mean((data_b[PSI_COL] - mean_psi_b)**2))

print(f"No Tape: RMS PSI = {rms_psi_a}")
print(f"6 Tape: RMS PSI = {rms_psi_b}")

max_psi_idx_a = np.argmax(moving_average(data_a[PSI_COL], 100))
max_psi_time_a = data_a[TIME][max_psi_idx_a]

max_psi_idx_b = np.argmax(moving_average(data_b[PSI_COL], 100))
max_psi_time_b = data_b[TIME][max_psi_idx_b]

plt.title("Pressure")
plt.xlim(max_psi_time_a,max_psi_time_a+0.1)
plt.ylim(12.8, 17.8)
plt.plot(data_a[TIME], data_a[PSI_COL], 'r', label='No Tape')
plt.legend(loc="upper left")
plt.ylabel("PSI")
plt.xlabel("Time")
ax2 = plt.twiny()
ax2.plot(data_b[TIME], data_b[PSI_COL], 'b', label='6 Tape')
ax2.set_xlim(max_psi_time_b,max_psi_time_b+0.1)
ax2.legend(loc="upper right")
ax2.set_ylabel("PSI")
ax2.set_xlabel("Time")

plt.show()

data_a[CURR_COL] = data_a[V1]
data_b[CURR_COL] = data_b[V1]

print(f"No Tape: Min. CURR = {min(data_a[CURR_COL])}")
print(f"6 Tape: Min. CURR = {min(data_b[CURR_COL])}")

print(f"No Tape: Max. CURR = {max(data_a[CURR_COL])}")
print(f"6 Tape: Max. CURR = {max(data_b[CURR_COL])}")

mean_curr_a = np.mean(data_a[CURR_COL])
mean_curr_b = np.mean(data_b[CURR_COL])

print(f"No Tape: Mean CURR = {mean_curr_a}")
print(f"6 Tape: Mean CURR = {mean_curr_b}")

rms_curr_a = np.sqrt(np.mean((data_a[CURR_COL] - mean_curr_a)**2))
rms_curr_b = np.sqrt(np.mean((data_b[CURR_COL] - mean_curr_b)**2))

print(f"No Tape: RMS CURR = {rms_curr_a}")
print(f"6 Tape: RMS CURR = {rms_curr_b}")

max_curr_idx_a = np.argmax(moving_average(data_a[CURR_COL], 100))
max_curr_time_a = data_a[TIME][max_curr_idx_a]

max_curr_idx_b = np.argmax(moving_average(data_b[CURR_COL], 100))
max_curr_time_b = data_b[TIME][max_curr_idx_b]

plt.title("Current Sensor Vout")
plt.xlim(max_curr_time_a,max_curr_time_a+0.1)
plt.ylim(2.2,2.8)
plt.plot(data_a[TIME], data_a[CURR_COL], 'r', label='No Tape')
plt.legend(loc="upper left")
plt.ylabel("CURR")
plt.xlabel("Time")
ax2 = plt.twiny()
ax2.plot(data_b[TIME], data_b[CURR_COL], 'b', label='6 Tape')
ax2.set_xlim(max_curr_time_b,max_curr_time_b+0.1)
ax2.legend(loc="upper right")
ax2.set_ylabel("CURR")
ax2.set_xlabel("Time")

plt.show()

data_a[VOLT_COL] = data_a[V2]*42/2
data_b[VOLT_COL] = data_b[V2]*42/2

print(f"No Tape: Min. VOLT = {min(data_a[VOLT_COL])}")
print(f"6 Tape: Min. VOLT = {min(data_b[VOLT_COL])}")

print(f"No Tape: Max. VOLT = {max(data_a[VOLT_COL])}")
print(f"6 Tape: Max. VOLT = {max(data_b[VOLT_COL])}")

mean_volt_a = np.mean(data_a[VOLT_COL])
mean_volt_b = np.mean(data_b[VOLT_COL])

print(f"No Tape: Mean VOLT = {mean_volt_a}")
print(f"6 Tape: Mean VOLT = {mean_volt_b}")

rms_volt_a = np.sqrt(np.mean((data_a[VOLT_COL] - mean_volt_a)**2))
rms_volt_b = np.sqrt(np.mean((data_b[VOLT_COL] - mean_volt_b)**2))

print(f"No Tape: RMS VOLT = {rms_volt_a}")
print(f"6 Tape: RMS VOLT = {rms_volt_b}")

max_volt_idx_a = np.argmax(moving_average(data_a[VOLT_COL], 100))
max_volt_time_a = data_a[TIME][max_volt_idx_a]

max_volt_idx_b = np.argmax(moving_average(data_b[VOLT_COL], 100))
max_volt_time_b = data_b[TIME][max_volt_idx_b]

plt.title("Current Sensor Vout")
plt.xlim(max_volt_time_a,max_volt_time_a+0.1)
plt.ylim(-200,200)
plt.plot(data_a[TIME], data_a[VOLT_COL], 'r', label='No Tape')
plt.legend(loc="upper left")
plt.ylabel("VOLT")
plt.xlabel("Time")
ax2 = plt.twiny()
ax2.plot(data_b[TIME], data_b[VOLT_COL], 'b', label='6 Tape')
ax2.set_xlim(max_volt_time_b,max_volt_time_b+0.1)
ax2.legend(loc="upper right")
ax2.set_ylabel("VOLT")
ax2.set_xlabel("Time")

plt.show()

def plot_psi(path, name, color):
    data = open_lj_dat(path)
    keep_times(0.5, 60.5, data)

    data[PSI_COL] = (data[V0]*1000 + 9.24) / 17.2 / 5 * 100
    max_curr_idx = np.argmax(moving_average(data[PSI_COL], 100))
    max_curr_time = data[TIME][max_curr_idx]

    data[TIME] -= max_curr_time

    plt.plot(data[TIME], moving_average(data[PSI_COL], 5), color, label=name)

plot_psi(r"C:\Users\Andrei\Desktop\Capstone Test 2 (CFP)\No Tape Redo", "0 Tape", "C1")
plot_psi(r"C:\Users\Andrei\Desktop\Capstone Test 2 (CFP)\1 Tape", "1 Tape", "C2")
plot_psi(r"C:\Users\Andrei\Desktop\Capstone Test 2 (CFP)\2 Tape", "2 Tape", "C3")
plot_psi(r"C:\Users\Andrei\Desktop\Capstone Test 2 (CFP)\4 Tape", "4 Tape", "C4")
plot_psi(r"C:\Users\Andrei\Desktop\Capstone Test 2 (CFP)\6 Tape", "6 Tape", "C5")

plt.xlim(0, 0.1)
plt.ylim(12.8, 17.8)
plt.legend()
plt.ylabel("PSI")
plt.xlabel("Time")

plt.show()

