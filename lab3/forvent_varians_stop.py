import statistics
from numpy import sqrt

sno_1_stop=[0.7246, 0.7605, 0.7273]
sno_1_stop_mean= statistics.mean(sno_1_stop)

sno_1_stop_s_sum=0
for result_i in sno_1_stop:
    sno_1_stop_s_sum+=(result_i-sno_1_stop_mean)**2

sno_1_stop_varians=sno_1_stop_s_sum/(len(sno_1_stop)-1)
sno_1_stop_std=sqrt(sno_1_stop_varians)

print("Sno 1 - Stop - Forventningsverdi: ", sno_1_stop_mean)
print("Sno 1 - Stop - Varians: ", sno_1_stop_varians)
print("Sno 1 - Stop - Standardavvik: ", sno_1_stop_std)

sno_2_stop=[0.7117, 0.6897, 0.6494 ]
sno_2_stop_mean= statistics.mean(sno_2_stop)

sno_2_stop_s_sum=0
for result_i in sno_2_stop:
    sno_2_stop_s_sum+=(result_i-sno_2_stop_mean)**2

sno_2_stop_varians=sno_2_stop_s_sum/(len(sno_2_stop)-1)
sno_2_stop_std=sqrt(sno_2_stop_varians)

print("Sno 2 - Stop - Forventningsverdi: ", sno_2_stop_mean)
print("Sno 2 - Stop - Varians: ", sno_2_stop_varians)
print("Sno 2 - Stop - Standardavvik: ", sno_2_stop_std)

sno_3_stop=[1.5385,1.7094, 1.7857]
sno_3_stop_mean= statistics.mean(sno_3_stop)

sno_3_stop_s_sum=0
for result_i in sno_3_stop:
    sno_3_stop_s_sum+=(result_i-sno_3_stop_mean)**2

sno_3_stop_varians=sno_3_stop_s_sum/(len(sno_3_stop)-1)
sno_3_stop_std=sqrt(sno_3_stop_varians)

print("Sno 3 - Stop - Forventningsverdi: ", sno_3_stop_mean)
print("Sno 3 - Stop - Varians: ", sno_3_stop_varians)
print("Sno 3 - Stop - Standardavvik: ", sno_3_stop_std)

sno_4_stop=[0.8772, 0.8403 , 0.9009 ]
sno_4_stop_mean= statistics.mean(sno_4_stop)

sno_4_stop_s_sum=0
for result_i in sno_4_stop:
    sno_4_stop_s_sum+=(result_i-sno_4_stop_mean)**2

sno_4_stop_varians=sno_4_stop_s_sum/(len(sno_4_stop)-1)
sno_4_stop_std=sqrt(sno_4_stop_varians)

print("Sno 4 - Stop - Forventningsverdi: ", sno_4_stop_mean)
print("Sno 4 - Stop - Varians: ", sno_4_stop_varians)
print("Sno 4 - Stop - Standardavvik: ", sno_4_stop_std)

sno_5_stop=[0.3984, 0.3824 , 0.3992]
sno_5_stop_mean= statistics.mean(sno_5_stop)

sno_5_stop_s_sum=0
for result_i in sno_5_stop:
    sno_5_stop_s_sum+=(result_i-sno_5_stop_mean)**2

sno_5_stop_varians=sno_5_stop_s_sum/(len(sno_5_stop)-1)
sno_5_stop_std=sqrt(sno_5_stop_varians)

print("Sno 5 - Stop - Forventningsverdi: ", sno_5_stop_mean)
print("Sno 5 - Stop - Varians: ", sno_5_stop_varians)
print("Sno 5 - Stop - Standardavvik: ", sno_5_stop_std)

sno_6_stop=[0.5333, 0.5249 , 0.5405 ]
sno_6_stop_mean= statistics.mean(sno_6_stop)

sno_6_stop_s_sum=0
for result_i in sno_6_stop:
    sno_6_stop_s_sum+=(result_i-sno_6_stop_mean)**2

sno_6_stop_varians=sno_6_stop_s_sum/(len(sno_6_stop)-1)
sno_6_stop_std=sqrt(sno_6_stop_varians)

print("Sno 6 - Stop - Forventningsverdi: ", sno_6_stop_mean)
print("Sno 6 - Stop - Varians: ", sno_6_stop_varians)
print("Sno 6 - Stop - Standardavvik: ", sno_6_stop_std)




