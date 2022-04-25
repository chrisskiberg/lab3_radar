from cProfile import label
from re import I
import matplotlib.pyplot as plt
import scipy.signal as signal
import numpy as np
from scipy.fft import ifft, fft, fftfreq, fftshift

# TODO - Regne ut SNR, hvor stort avvik fra farten skal ha. Cirka  0.1 m/s

def raspi_import(path, channels=2):
    """
    Import data produced using adc_sampler.c.
    Returns sample period and ndarray with one column per channel.
    Sampled data for each channel, in dimensions NUM_SAMPLES x NUM_CHANNELS.
    """

    with open(path, 'r') as fid:
        sample_period = np.fromfile(fid, count=1, dtype=float)[0]
        data = np.fromfile(fid, dtype=np.uint16)
        data = data.reshape((-1, channels))
    return sample_period, data


# Import data from bin file
filename="bil_sno2_maal1"
sample_period, data = raspi_import(filename+'.bin')

VDD=3.3
data=data*VDD/(2**12)


files = {
"bil_sno1_maal1": [23000,80,230],
"bil_sno1_maal2": [20000,85,340],
"bil_sno1_maal3": [19000,85,220],

"bil_sno2_maal1": [22000,80,275],
"bil_sno2_maal2": [18000,50,235],
"bil_sno2_maal3": [15000,40,280],

"bil_sno3_maal1_2": [0,220,400], 
"bil_sno3_maal2": [16000,50,580], 
"bil_sno3_maal3": [15000,70,630], 

"bil_sno4_maal1": [22000,100,250],
"bil_sno4_maal2": [20000,90,290],
"bil_sno4_maal3": [20000,110,185],

"bil_sno5_maal1": [0,140,310],
"bil_sno5_maal2": [0,110,340],
"bil_sno5_maal3": [20000,105,270],

"bil_sno6_maal3": [20000,20,280],
"bil_sno6_maal4": [20000,30,250],
"bil_sno6_maal5": [16000,30,240],
}

# sno1_maal1_samples=num_of_samples # N=23000, fc_high=130, fc_low=180, Nth=6
# sno1_maal2_samples=num_of_samples # N=20000, fc_high=100, fc_low=150, Nth=6
# sno1_maal3_samples=num_of_samples # N=19000, fc_high=80, fc_low=125, Nth=6

# sno2_maal1_samples=num_of_samples # N=22000, fc_high=100, fc_low=155, Nth=6
# sno2_maal2_samples=num_of_samples # N=18000, fc_high=50, fc_low=130, Nth=6
# sno2_maal3_samples=num_of_samples # N=15000, fc_high=80, fc_low=125, Nth=6

# sno3_maal1_samples=num_of_samples # N=16000, fc_high=160, fc_low=330, Nth=6
# sno3_maal2_samples=num_of_samples # N=16000, fc_high=115, fc_low=390, Nth=6
# sno3_maal3_samples=num_of_samples # N=15000, fc_high=90, fc_low=420, Nth=6

# sno4_maal1_samples=num_of_samples # N=22000, fc_high=100, fc_low=275, Nth=6 
# sno4_maal2_samples=num_of_samples # N=20000, fc_high=75, fc_low=200, Nth=6 
# sno4_maal3_samples=num_of_samples # N=20000, fc_high=85, fc_low=200, Nth=6

# sno5_maal1_samples=num_of_samples # N=22000, fc_high=165, fc_low=400, Nth=6
# sno5_maal2_samples=num_of_samples # N=22000, fc_high=150, fc_low=365, Nth=6
# sno5_maal3_samples=num_of_samples # N=20000, fc_high=115, fc_low=260, Nth=6

# sno6_maal1_samples=num_of_samples
# sno6_maal2_samples=num_of_samples 
# sno6_maal3_samples=num_of_samples # N=20000, fc_high=NaN, fc_low=300, Nth=6
# sno6_maal4_samples=num_of_samples # N=20000, fc_high=30, fc_low=300,Nth=6
# sno6_maal5_samples=num_of_samples # N=16000, fc_high=30, fc_low=300,Nth=6


sample_period *= 1e-6  # change unit to micro seconds

# Generate time axis
num_of_samples = data.shape[0]  # returns shape of matrix
# t = np.linspace(start=files[filename][0], stop=num_of_samples, num=num_of_samples-files[filename][0])
t = []
for i in range(files[filename][0], num_of_samples):
# for i in range(0, num_of_samples):
    t.append(i*sample_period)
IF_I=data[files[filename][0]:,0]
IF_Q=data[files[filename][0]:,1]
# IF_I=data[0:,0]
# IF_Q=data[0:,1]


# Filtrering

# fs=1/diff_t_upsample
# fc_low = 28000  # Cut-off frequency of the filter
# fc_high = 28000  # Cut-off frequency of the filter
# w = fc / (fs / 2) # Normalize the frequency



# # fc_high = files[filename][1]    # Cut-off frequency of the filter (tilsvarer 3 slag per sekund * 60 sekunder = 180 slag per minutt)
# fc_high = 90    # Cut-off frequency of the filter (tilsvarer 3 slag per sekund * 60 sekunder = 180 slag per minutt)
# w = fc_high / (31250 / 2) # Normalize the frequency
# b, a = signal.butter(6, w, 'high') # kan velge høyere eller lavere ordens, men må være forsiktig med hvor mye det tar av ønskede frekvenser
# IF_I = signal.filtfilt(b, a, data[files[filename][0]:,0])
# IF_Q = signal.filtfilt(b, a, data[files[filename][0]:,1])

# fc_low = files[filename][2]  # Cut-off frequency of the filter (tilsvarer 3 slag per sekund * 60 sekunder = 180 slag per minutt)
# w = fc_low / (31250 / 2) # Normalize the frequency
# b, a = signal.butter(6, w, 'low') # kan velge høyere eller lavere ordens, men må være forsiktig med hvor mye det tar av ønskede frekvenser
# IF_I = signal.filtfilt(b, a, data[files[filename][0]:,0])
# IF_Q = signal.filtfilt(b, a, data[files[filename][0]:,1])

IF_I = signal.detrend(IF_I, axis=0)  # removes DC component for each channel
IF_Q = signal.detrend(IF_Q, axis=0)  # removes DC component for each channel



# Channel 1 er IF_Q og channel 2 er IF_I
# x_k= IF_I[k] + j*IF_Q[k]
x_k=[IF_I[i]+1j*IF_Q[i] for i in range(len(IF_I))]
Xf = fftfreq(num_of_samples-files[filename][0], sample_period)
# Xf = fftfreq(num_of_samples, sample_period)
Xf = fftshift(Xf)
Sf = fft(x_k)
Sf_max=np.max(np.abs(Sf))
Sf_max_index=(fftshift(np.abs(Sf)).tolist()).index(Sf_max)
print(Xf[Sf_max_index]) 
print("v_r= " + str(Xf[Sf_max_index]/160.978))
print(20*np.log10(np.abs(Sf_max)))  


# sf_fc_high = 100    # Cut-off frequency of the filter (tilsvarer 3 slag per sekund * 60 sekunder = 180 slag per minutt)
# w = sf_fc_high / (31250 / 2) # Normalize the frequency
# b, a = signal.butter(6, w, 'high') # kan velge høyere eller lavere ordens, men må være forsiktig med hvor mye det tar av ønskede frekvenser
# Sf = signal.filtfilt(b, a, Sf)
# # IF_Q = signal.filtfilt(b, a, data[files[filename][0]:,1])


# SNR
SNR_lower_freq=Xf[Sf_max_index]-5
SNR_upper_freq=Xf[Sf_max_index]+5
SNR_arr=[]
for i in range(len(Xf)):
    if int(Xf[i])<int(SNR_lower_freq) or int(Xf[i])>int(SNR_upper_freq):
        SNR_arr.append(i)
        # print("hei")
# print(len(SNR_arr))
# print(SNR_arr[len(SNR_arr)-1])

sum=0
for i in SNR_arr:
    sum+=np.abs(Sf[i])
SNR_avg=sum/len(SNR_arr)
print(SNR_avg)    
print("SNR=" + str(20*np.log10(np.abs(Sf_max/SNR_avg))))



# Generate frequency axis and take FFT
freq = np.fft.fftfreq(n=num_of_samples-files[filename][0], d=sample_period)
# freq = np.fft.fftfreq(n=num_of_samples, d=sample_period)
freq=np.fft.fftshift(freq,axes=0)
spectrum0= np.fft.fft(IF_I, axis=0)  # takes FFT of all channels fftshift på data og frekvensakser 
spectrum0=np.fft.fftshift(spectrum0, axes=0)
spectrum1= np.fft.fft(IF_Q, axis=0)  # takes FFT of all channels fftshift på data og frekvensakser 
spectrum1=np.fft.fftshift(spectrum1, axes=0)
IF_FFT_max=np.max(np.abs(spectrum1))
IF_FFT_max_db=20*np.log10(IF_FFT_max)


# Plot the results in two subplots
# NOTICE: This lazily plots the entire matrixes. All the channels will be put into the same plots.
# If you want a single channel, use data[:,n] to get channel n
plt.style.use('seaborn-paper')

# ax = plt.gca()  # or any other way to get an axis object
# ax.plot(t, IF_I, label=r'$IF_I$')
# ax.plot(t, IF_Q, label=r'$IF_Q$')

# # plt.plot(t, IF_I, label="IF_I") 
# # plt.plot(t, IF_Q, label="IF_Q") 
# # ax.xlabel('Tid [s]', fontsize=20)
# # ax.ylabel('Spenning [V]', fontsize=20)
# ax.set_xlabel('Tid [s]', fontsize=20)
# ax.set_ylabel('Spenning [V]', fontsize=20)
# ax.grid(b=None, which='major', axis='both')
# ax.grid(b=None, which='minor', axis='both')
# ax.tick_params(axis='both', which='major', labelsize=16)
# ax.legend(prop={'size': 16})
# ax.set_xticks(np.arange(0, 1.1, 0.1))
# # plt.tight_layout()
# # plt.ylim(-0.1, 1.80)
# # plt.ylim(1.1, 1.55)
# plt.show()


ax = plt.gca()  # or any other way to get an axis object
ax.plot(t, IF_I, label=r'$IF_I$')
ax.plot(t, IF_Q, label=r'$IF_Q$')
ax.set_xlabel('Tid [s]', fontsize=20)
ax.set_ylabel('Spenning [V]', fontsize=20)
ax.grid(b=None, which='major', axis='both')
ax.grid(b=None, which='minor', axis='both')
ax.tick_params(axis='both', which='major', labelsize=16)
ax.legend(prop={'size': 16})
# ax.set_xticks(np.arange(0, 1.1, 0.1))
plt.ylim((-0.25, 0.25)) # set the xlim to left, right

plt.show()

ax = plt.gca()  # or any other way to get an axis object
ax.plot(freq, 20*np.log10(np.abs(spectrum0)), label=r'$IF_I$', color="C0")
ax.plot(freq, 20*np.log10(np.abs(spectrum1)), label=r'$IF_Q$', color="C1")
ax.set_xlabel("Frekvens [Hz]", fontsize=20)
ax.set_ylabel("Effekt [dB]", fontsize=20)
ax.grid(b=None, which='major', axis='both')
ax.grid(b=None, which='minor', axis='both')
ax.tick_params(axis='both', which='major', labelsize=16)
ax.legend(prop={'size': 16})
plt.xlim((-1000, 1000)) # set the xlim to left, right
plt.ylim((-10, 60)) # set the xlim to left, right
# ax.set_yticks(np.arange(-10, 65, 5))
plt.axhline(y=IF_FFT_max_db, linestyle='--', color="black")


# plt.subplot(2, 1, 2)
# plt.xlabel("Frekvens [Hz]")
# plt.ylabel("Effekt [dB]")
# plt.plot(freq, 20*np.log10(np.abs(spectrum0))) # get the power spectrum
# plt.plot(freq, 20*np.log10(np.abs(spectrum1))) # get the power spectrum
# plt.xlim((-1000, 1000)) # set the xlim to left, right
# plt.ylim((-30, 60)) # set the xlim to left, right
plt.show()


yplot = fftshift(Sf)
# plt.plot(Xf, 20*np.log10(np.abs(yplot)))
# plt.plot(Xf, np.abs(yplot))
# plt.title("Power spectrum of signal")
# plt.xlabel("Frequency [Hz]")
# plt.ylabel("Power [dB]")
# plt.grid(True, which='both')
# plt.show()
ax = plt.gca()  # or any other way to get an axis object
ax.plot(Xf, 20*np.log10(np.abs(yplot)), label=r'$S(f)$', color="C2")
ax.set_xlabel("Frekvens [Hz]", fontsize=20)
ax.set_ylabel("Effekt [dB]", fontsize=20)
ax.grid(b=None, which='major', axis='both')
ax.grid(b=None, which='minor', axis='both')
ax.tick_params(axis='both', which='major', labelsize=16)
ax.legend(prop={'size': 16})
plt.xlim((-1000, 1000)) # set the xlim to left, right
plt.ylim((-10, 60)) # set the xlim to left, right
# ax.set_yticks(np.arange(-10, 65, 5))
plt.axhline(y=20*np.log10(np.abs(Sf_max)), linestyle='--', color="black")
plt.show()


# Butterworth
# # Sample rate and desired cutoff frequencies (in Hz).
# fs = 5000.0
# lowcut = 500.0
# highcut = 1250.0

# # Plot the frequency response for a few different orders.
# plt.figure(1)
# plt.clf()
# for order in [3, 6, 9]:
#     b, a = butter_bandpass(lowcut, highcut, fs, order=order)
#     w, h = freqz(b, a, worN=2000)
#     plt.plot((fs * 0.5 / np.pi) * w, abs(h), label="order = %d" % order)




# # Channel 1 er IF_Q og channel 2 er IF_I
# # x_k= IF_I[k] + j*IF_Q[k]
# x_k=[data[i,1]+1j*data[i,0] for i in range(len(data[0:,1]))]
# Xf = fftfreq(num_of_samples, sample_period)
# Xf = fftshift(Xf)
# Sf = fft(x_k)
# Sf_max=np.max(Sf)
# Sf_max_index=(fftshift(Sf).tolist()).index(Sf_max)
# print(Xf[Sf_max_index])

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

# sum=0
# for fq in SNR_arr:
#     sum+=Sf



# # Generate frequency axis and take FFT
# freq = np.fft.fftfreq(n=num_of_samples, d=sample_period)
# freq=np.fft.fftshift(freq,axes=0)
# spectrum = np.fft.fft(data, axis=0)  # takes FFT of all channels fftshift på data og frekvensakser 
# spectrum=np.fft.fftshift(spectrum, axes=0)