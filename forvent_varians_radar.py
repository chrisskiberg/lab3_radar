import statistics
from numpy import sqrt

sno_1_radar=[0.9882773365982251, 0.8455261657562593, 0.8557386850427782]
sno_1_radar_mean= statistics.mean(sno_1_radar)

sno_1_radar_s_sum=0
for result_i in sno_1_radar:
    sno_1_radar_s_sum+=(result_i-sno_1_radar_mean)**2

sno_1_radar_varians=sno_1_radar_s_sum/(len(sno_1_radar)-1)
sno_1_radar_std=sqrt(sno_1_radar_varians)

print("Sno 1 - radar - Forventningsverdi: ", sno_1_radar_mean)
print("Sno 1 - radar - Varians: ", sno_1_radar_varians)
print("Sno 1 - radar - Standardavvik: ", sno_1_radar_std)

sno_2_radar=[0.7974902059808072, 0.6006914808703498, 0.6689877355434138 ]
sno_2_radar_mean= statistics.mean(sno_2_radar)

sno_2_radar_s_sum=0
for result_i in sno_2_radar:
    sno_2_radar_s_sum+=(result_i-sno_2_radar_mean)**2

sno_2_radar_varians=sno_2_radar_s_sum/(len(sno_2_radar)-1)
sno_2_radar_std=sqrt(sno_2_radar_varians)

print("Sno 2 - radar - Forventningsverdi: ", sno_2_radar_mean)
print("Sno 2 - radar - Varians: ", sno_2_radar_varians)
print("Sno 2 - radar - Standardavvik: ", sno_2_radar_std)

sno_3_radar=[1.484674924523848, 1.7057620540348755,1.8755549014342139]
sno_3_radar_mean= statistics.mean(sno_3_radar)

sno_3_radar_s_sum=0
for result_i in sno_3_radar:
    sno_3_radar_s_sum+=(result_i-sno_3_radar_mean)**2

sno_3_radar_varians=sno_3_radar_s_sum/(len(sno_3_radar)-1)
sno_3_radar_std=sqrt(sno_3_radar_varians)

print("Sno 3 - radar - Forventningsverdi: ", sno_3_radar_mean)
print("Sno 3 - radar - Varians: ", sno_3_radar_varians)
print("Sno 3 - radar - Standardavvik: ", sno_3_radar_std)

sno_4_radar=[-0.923409712188303, -0.7765036216128912 , -0.8800374378279433 ]
sno_4_radar_mean= statistics.mean(sno_4_radar)

sno_4_radar_s_sum=0
for result_i in sno_4_radar:
    sno_4_radar_s_sum+=(result_i-sno_4_radar_mean)**2

sno_4_radar_varians=sno_4_radar_s_sum/(len(sno_4_radar)-1)
sno_4_radar_std=sqrt(sno_4_radar_varians)

print("Sno 4 - radar - Forventningsverdi: ", sno_4_radar_mean)
print("Sno 4 - radar - Varians: ", sno_4_radar_varians)
print("Sno 4 - radar - Standardavvik: ", sno_4_radar_std)

sno_5_radar=[-0.34787362248257525, -0.3540856514554784 , -0.39135782529289714]
sno_5_radar_mean= statistics.mean(sno_5_radar)

sno_5_radar_s_sum=0
for result_i in sno_5_radar:
    sno_5_radar_s_sum+=(result_i-sno_5_radar_mean)**2

sno_5_radar_varians=sno_5_radar_s_sum/(len(sno_5_radar)-1)
sno_5_radar_std=sqrt(sno_5_radar_varians)

print("Sno 5 - radar - Forventningsverdi: ", sno_5_radar_mean)
print("Sno 5 - radar - Varians: ", sno_5_radar_varians)
print("Sno 5 - radar - Standardavvik: ", sno_5_radar_std)

sno_6_radar=[-0.4659021729677347, -0.43139090089605064 , -0.5091827026969777]
sno_6_radar_mean= statistics.mean(sno_6_radar)

sno_6_radar_s_sum=0
for result_i in sno_6_radar:
    sno_6_radar_s_sum+=(result_i-sno_6_radar_mean)**2

sno_6_radar_varians=sno_6_radar_s_sum/(len(sno_6_radar)-1)
sno_6_radar_std=sqrt(sno_6_radar_varians)

print("Sno 6 - radar - Forventningsverdi: ", sno_6_radar_mean)
print("Sno 6 - radar - Varians: ", sno_6_radar_varians)
print("Sno 6 - radar - Standardavvik: ", sno_6_radar_std)




