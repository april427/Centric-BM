# %%
from channel_functions import *
from motion import *
import numpy as np
import math
import random
import gymnasium as gym
from scipy.linalg import hadamard
import matplotlib.pyplot as plt

from beam_management_env_letter import BeamManagementEnv

env = gym.make('BeamManagementEnv-v1', N_f=1000)
num_episodes = 10    #TTI
episode_durations = []

# %%
# variable initialization
N_UEs = 2#env.N_UEs
TTI = 10
N = 16 # number of samples
user_positions = np.zeros((TTI+1, N_UEs, 2))
d = np.zeros(N_UEs)
AoD = np.zeros(N_UEs) # real value
aod = np.zeros(N_UEs) # estimated value
a_BS = np.zeros((N_UEs, env.number_of_antennas)) + 1j
mmWave_LoS_NLoS_channel = np.zeros((N_UEs, env.number_of_antennas)) + 1j
current_beam = np.empty([TTI+1, N_UEs,], dtype=object)
config = np.arange(N_UEs)
r = np.zeros(N_UEs)
edge_snr_db = 0
noise_power = 1 #Pt/N_UEs / (50*np.sqrt(2)) ** 2 * 10 **(-edge_snr_db / 10)
Pt = N_UEs * (50*np.sqrt(2)) ** 2 * 10 **(-edge_snr_db / 10)
y_user = np.zeros((N_UEs, N))
y_BS = np.zeros((env.number_of_antennas, N))
waveform = hadamard(N)
rate = np.zeros((N_UEs,TTI))
nitta = np.zeros((env.number_of_antennas)) # temperal variable for AoD estimation
beamwidth = 2*np.arcsin(1.2/env.number_of_antennas) # radians, consider 3 dB beamwidth
error = np.zeros((N_UEs, TTI)) # AoD estimation error
pathloss = np.zeros((N_UEs, TTI)) # pathloss

genie_beam = np.empty([TTI+1, N_UEs,], dtype=object)
genie_s_BS = np.zeros((env.number_of_antennas, N)).astype(complex)
genie_y_user = np.zeros((N_UEs, N))
genie_rate = np.zeros((N_UEs,TTI))
rx_snr_db = np.zeros((TTI, N_UEs))

 # Used for AoD estimation
A = np.zeros((100,env.number_of_antennas),dtype=complex)
for m in range(100):
    A[m,:] = np.array([np.exp(1j * np.pi * i * np.cos(m/100*np.pi)) for i in range(env.number_of_antennas)])


# beam initialization using beam sweeping
np.random.shuffle(config)
for i in range(N_UEs):
    AoD[i] = np.random.rand() * np.pi /N_UEs + np.pi / N_UEs * config[i] #radians
    r[i] = (2 * np.random.rand() - 1) * 50
    user_positions[0][i][0] = 50 + r[i] * np.cos(AoD[i])
    user_positions[0][i][1] = 50 + r[i] * np.sin(AoD[i])

    d[i] = np.absolute(r[i])

    a_BS[i] = compute_ULA_response(env.number_of_antennas, AoD[i])  # current array response

    mmWave_LoS_NLoS_channel[i] = compute_mmWave_LoS_NLoS_channel(env.number_of_antennas, a_BS[i], d[i])  # current LoS channel

    current_beam[0,i] = np.argmax(np.abs(np.sqrt(Pt) * mmWave_LoS_NLoS_channel[i] @ env.codebook))
    

# %%
for n in range(TTI):
    s_BS = np.zeros((env.number_of_antennas, N)).astype(complex)
    mono_LoS_channel = np.zeros((env.number_of_antennas, env.number_of_antennas), dtype=np.complex128)

    for u in range(N_UEs):
        # update user position
        user_positions[n+1,u] = update_position(user_positions[n,u], sample_velocity(), 5, config[u])

        # update distance
        d[u] = np.linalg.norm(env.base_station_position - user_positions[n+1,u])

        # update AoD real
        AoD[u] = compute_AoD_at_BS(env.base_station_position, user_positions[n+1,u])

        # update array response
        a_BS[u] = compute_ULA_response(env.number_of_antennas, AoD[u])

        # update LoS channel
        mmWave_LoS_NLoS_channel[u] = compute_mmWave_LoS_NLoS_channel(env.number_of_antennas, a_BS[u], d[u])
        pathloss[u,n] = np.linalg.norm(mmWave_LoS_NLoS_channel[u])

        mono_LoS_channel += np.sqrt(env.number_of_antennas) * env.reflection_coefficient[u] * np.outer(a_BS[u,:] , np.conj(a_BS[u,:]))

        # benchmark - genie-aided - upper bound
        genie_beam[n+1, u] = np.argmax(np.abs( env.codebook @ a_BS[u,:]))
        genie_s_BS += np.sqrt(Pt / N_UEs ) * np.outer((env.codebook[:,genie_beam[n+1,u]]), waveform[u, :])
        genie_y_user[u] = mmWave_LoS_NLoS_channel[u] @ genie_s_BS
        genie_rate[:,n] = np.mean(np.log2(1 +  np.abs(genie_y_user[u,:]) ** 2 / noise_power))

        # transmit signal
        if current_beam[n,u].size == 1:
            transmit_beam = env.codebook[:, current_beam[n,u]]
        else:
            transmit_beam = env.codebook[:, current_beam[n,u]].sum(axis=1)
        
        s_BS += np.sqrt(Pt / N_UEs / np.size(waveform[u,:])) * np.outer((transmit_beam), waveform[u, :])

        # user received signal noise-less
        y_user[u] = mmWave_LoS_NLoS_channel[u] @ s_BS 

    rx_snr_db[n,:] = np.mean(np.abs(genie_y_user)**2, axis=1) / noise_power

    # calculate communication rate of each user
    if True: #current_beam[n,u].size == 1:
        rate[:,n] = np.mean(np.log2(1 +  np.abs(y_user[u,:]) ** 2 / noise_power))
    else:
        rate[:,n] = 0
    # received echos at BS
    y_BS = mono_LoS_channel @ s_BS + np.random.normal(0, noise_power, size = (env.number_of_antennas, N))
    
    # Estimate AoDs from the echo
    # extract each user's signal
    for u in range(N_UEs):
        nitta = np.matmul(y_BS, waveform[u, :])
        # estimate AoD
        aod[u] = np.argmax(np.abs(A @ nitta))/100*np.pi
        # map the estimated aod to the closest beam in the codebook
        current_beam[n+1, u] = np.argmax(env.codebook @ compute_ULA_response(env.number_of_antennas, aod[u]))
        # estimation error
        error[u,n] = np.abs(aod[u] - AoD[u])

        if error[u,n] > beamwidth:
            # update beam - use multiple beams
            if np.max(current_beam[n+1, u])>=31:
                current_beam[n+1,u] = np.add(current_beam[n+1,u],[-1,0])
            elif np.min(current_beam[n+1,u])==0:
                current_beam[n+1,u] = np.add(current_beam[n+1,u],[0,1])
            else:
                current_beam[n+1,u] = np.add(current_beam[n+1,u],[-1,0,1])
            # else keep the current beam

    
# TODO - exhaustive beam sweeping 

average_rate = np.mean(rate, axis=1)
genie_average_rate = np.mean(genie_rate, axis=1)

SNR_dB = 10*np.log10(np.mean((Pt/N_UEs)* pathloss,axis=1) /noise_power)
rx_snr_db = 10*np.log10(np.mean(rx_snr_db, axis=0))

nmse = np.mean((error/np.pi)**2, axis = 1)

# %%
for i in range(N_UEs):
    ue_pos_x = user_positions[:,i,0]
    ue_pos_y = user_positions[:,i,1]
    du = np.diff(ue_pos_x)
    dv = np.diff(ue_pos_y)
    pos_x = ue_pos_x[:-1] + du/2
    pos_y = ue_pos_y[:-1] + dv/2
    norm = np.sqrt(du**2+dv**2) 

    plt.plot(ue_pos_x,ue_pos_y, marker="o")
    plt.quiver(pos_x, pos_y, du/norm, dv/norm, angles="xy", zorder=5, pivot="mid")
    plt.xlim(0,100)
    plt.ylim(0,100)
    # hold the plot
    
plt.show()

# %%
