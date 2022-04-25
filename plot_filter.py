import csv
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
import numpy as np
from scipy.fft import ifft, fft, fftfreq, fftshift


header = []
data = []


filename = "bode_filter2_copy.csv"
with open(filename) as csvfile:
    csvreader = csv.reader(csvfile)

    header = next(csvreader)

    for datapoint in csvreader:

        values = [float(value) for value in datapoint]
        data.append(values)

print(header)
print(data[0])
#print(data[1])

freq = [p[0] for p in data]
# ch1 = [p[1] for p in data]
ch2 = [p[2] for p in data]


plt.plot(freq,ch2)
plt.xscale('log')
#plt.plot(time,ch1)
# plt.grid(color='black', linestyle='-', linewidth=1)
plt.xlabel('Frekvens [Hz]', fontsize=20)
plt.ylabel('Amplitude [dB]', fontsize=20)
plt.grid(b=None, which='major', axis='both')
plt.grid(b=None, which='minor', axis='both')
plt.tick_params(axis='both', which='major', labelsize=16)

plt.tight_layout()
#plt.xlim([-0.002,0.002])
# matplotlib.rc('xtick', labelsize=15) 
# matplotlib.rc('ytick', labelsize=10)

plt.show()
