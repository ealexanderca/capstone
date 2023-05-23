from util.util import *


def plot_audio(file, name, color):
    t, y = open_mono_wav(file)
    end_idx = time_to_idx(81, t)
    t = t[0:end_idx]
    y = y[0:end_idx]

    f, a, _ = fft(t, y)

    xvals = np.array(range(10000))/1000
    xvals = np.exp(xvals)
    xvals = xvals/xvals[-1]*f[-1]
    avals = np.interp(xvals, f, a)
    avals = moving_average(avals, 10)

    plt.fill_between(xvals, avals, 1E-8, color=color,  label=name)
    #plt.plot(xvals, avals, color, label=name, linewidth=1)

    return xvals, avals


a = plot_audio(r"C:\Users\Andrei\Desktop\Capstone Test 2 (CFP)\No Tape Redo\audio.wav", "0 Tape", "C1")
#plot_audio(r"C:\Users\Andrei\Desktop\Capstone Test 2 (CFP)\1 Tape\audio.wav", "1 Tape", "C2")
#plot_audio(r"C:\Users\Andrei\Desktop\Capstone Test 2 (CFP)\2 Tape\audio.wav", "2 Tape", "y")
#plot_audio(r"C:\Users\Andrei\Desktop\Capstone Test 2 (CFP)\4 Tape\audio.wav", "4 Tape", "y")
b = plot_audio(r"C:\Users\Andrei\Desktop\Capstone Test 2 (CFP)\6 Tape\audio.wav", "6 Tape", "C0")

plt.title('Audio FFT')
plt.xlim(10, 20E3)
plt.ylim(5E-8, 1E-3)
plt.xscale('log')
plt.yscale('log')
plt.ylabel("Amplitude")
plt.xlabel("Frequency (Hz)")
plt.legend()
plt.show()

plt.title('Audio FFT Difference')
plt.fill_between(a[0], moving_average(np.abs(a[1]-b[1]),50), 1E-20)
plt.xlim(10, 20E3)
plt.ylim(2E-8, 1E-4)
plt.xscale('log')
plt.yscale('log')
plt.ylabel("Amplitude")
plt.xlabel("Frequency (Hz)")
plt.legend()
plt.show()

plt.title('Audio FFT Ratio')
plt.fill_between(a[0], moving_average(np.abs(a[1]/b[1]),50), 1E-20)
plt.xlim(10, 20E3)
plt.ylim(0.35, 4)
plt.xscale('log')
plt.yscale('log')
plt.ylabel("Amplitude")
plt.xlabel("Frequency (Hz)")
plt.legend()
plt.show()