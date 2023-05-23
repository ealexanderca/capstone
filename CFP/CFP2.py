from util.util import *

data = open_lj_dat(r"C:\Users\Andrei\Desktop\Capstone Test 2 (CFP)\6 Tape")

data[PSI_COL] = (data[V0]*1000 + 9.24) / 17.2 / 5 * 100

plot_all(plt, data, plot_cols='PSI')
#plt.xlim(12, 12.1)
plt.ylim(12.8, 17.8)
plt.show()
#data['Amps'] = (data[V0]*1000 - 2.503) / 41.67 / 6

plot_all(plt, data, plot_cols=V1)
#plt.xlim(30,30.1)
plt.ylim(2.2,2.8)
plt.ylabel("Current sensor Vout")
plt.show()

data['Voltage'] = data[V2]*42/2

plot_all(plt, data, plot_cols='Voltage')
#plt.xlim(30,30.1)
plt.ylim(-200,200)
plt.show()