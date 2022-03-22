import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
import csv

# TODO: Velge område for signal og noise

with open('test_for_22_2.csv', 'r') as file:
    reader = csv.reader(file)
    data_tr=[]
    for row in reader:
        data_tr.append(row)

# print(len(data_tr))
data_red=[float(row[0]) for row in data_tr]
data_green=[float(row[1]) for row in data_tr]
data_blue=[float(row[2]) for row in data_tr]

data_red = signal.detrend(data_red, axis=0)
data_green = signal.detrend(data_green, axis=0)
data_blue = signal.detrend(data_blue, axis=0)


# Filtrering
fs=40 # fs=fps
fc_high = 0.6  # Cut-off frequency of the filter (tilsvarer 3 slag per sekund * 60 sekunder = 180 slag per minutt)
w = fc_high / (fs / 2) # Normalize the frequency
b, a = signal.butter(6, w, 'high') # kan velge høyere eller lavere ordens, men må være forsiktig med hvor mye det tar av ønskede frekvenser
data_red = signal.filtfilt(b, a, data_red)
data_green = signal.filtfilt(b, a, data_green)
data_blue = signal.filtfilt(b, a, data_blue)

fc_high = 130/60  # Cut-off frequency of the filter (tilsvarer 3 slag per sekund * 60 sekunder = 180 slag per minutt)
w = fc_high / (fs / 2) # Normalize the frequency
b, a = signal.butter(6, w, 'low') # kan velge høyere eller lavere ordens, men må være forsiktig med hvor mye det tar av ønskede frekvenser
data_red = signal.filtfilt(b, a, data_red)
data_green = signal.filtfilt(b, a, data_green)
data_blue = signal.filtfilt(b, a, data_blue)



# print(data_red[len(data_red)-1])
# print(data_green[len(data_green)-1])
# print(data_blue[len(data_blue)-1])

freq=np.fft.fftfreq(n=len(data_red), d=1/40)
freq=np.fft.fftshift(freq,axes=0)
spectrum_red = np.fft.fft(data_red, axis=0)  # takes FFT of all channels fftshift på data og frekvensakser 
spectrum_green = np.fft.fft(data_green, axis=0)  # takes FFT of all channels fftshift på data og frekvensakser 
spectrum_blue = np.fft.fft(data_blue, axis=0)  # takes FFT of all channels fftshift på data og frekvensakser 
spectrum_red=np.fft.fftshift(spectrum_red, axes=0)
spectrum_green=np.fft.fftshift(spectrum_green, axes=0)
spectrum_blue=np.fft.fftshift(spectrum_blue, axes=0)


plt.plot(data_red, color="r") # 'ro'
plt.plot(data_green, color="g") # 'ro'
plt.plot(data_blue, color="b") # 'ro'
plt.grid(True, which='both')
plt.show()

plt.title("Power spectrum of signal")
plt.xlabel("Frequency [Hz]")
plt.ylabel("Power [dB]")
plt.plot(freq, 20*np.log10(np.abs(spectrum_red)), color="r") # get the power spectrum
plt.plot(freq, 20*np.log10(np.abs(spectrum_green)), color="g") # get the power spectrum
plt.plot(freq, 20*np.log10(np.abs(spectrum_blue)), color="b") # get the power spectrum
plt.show()

# # Upsampling
# func_1_up = interpolate.interp1d(time, func_1, kind='cubic')
# func_2_up = interpolate.interp1d(time, func_2, kind='cubic')
# func_3_up = interpolate.interp1d(time, func_3, kind='cubic')
# func_1_new = func_1_up(time_upsampled)
# func_2_new = func_2_up(time_upsampled)
# func_3_new = func_3_up(time_upsampled)
# func_1=func_1_new.tolist()
# func_2=func_2_new.tolist()
# func_3=func_3_new.tolist()


# # Filtrering
# fs=1/diff_t_upsample
# fc = 28000  # Cut-off frequency of the filter
# w = fc / (fs / 2) # Normalize the frequency
# b, a = signal.butter(15, w, 'low')
# func_1 = signal.filtfilt(b, a, func_1)
# func_2 = signal.filtfilt(b, a, func_2)
# func_3 = signal.filtfilt(b, a, func_3)

# # SNR
# SNR_lower_freq=Xf[Sf_max_index]-20
# SNR_upper_freq=Xf[Sf_max_index]+20
# print(Xf)
# SNR_arr=[]
# for i in range(len(Xf)):
#     if int(Xf[i])<int(SNR_lower_freq) and int(Xf[i])>int(SNR_upper_freq):
#         SNR_arr.append(Xf[i])
#         # print("hei")
# print(len(SNR_arr))
# print(SNR_arr)