# %%
from channel_functions import *
#from motion import *
import numpy as np
import gymnasium as gym
from scipy.linalg import hadamard
import matplotlib.pyplot as plt
from road import *

from beam_management_env_letter import BeamManagementEnv

env = gym.make('BeamManagementEnv-v1', N_f=1000)
episode_durations = []

# %%
# variable initialization
N_UEs = 1 #env.N_UEs
TTI = 100
episode =  1
N = 512 # number of samples in one TTI
user_positions = np.zeros((TTI+1, N_UEs, 2))
d = np.zeros(N_UEs)
AoD = np.zeros(N_UEs) # real value
aod = np.zeros(N_UEs) # estimated value
a_BS = np.zeros((N_UEs, env.number_of_antennas)) + 1j

## fixed map
cross_pos = np.array([[27, 80], [30, 70]])
road_matrix = road(cross_pos)
single_lane_width = 5

dt = 0.01 # time step 10 ms

y_BS = np.zeros((env.number_of_antennas, N)) + 1j
waveform = hadamard(N)
waveform[[0, -1],:] = waveform[[-1,0],:]

nitta = np.zeros((env.number_of_antennas)) + 1j # temperal variable for AoD estimation
beamwidth = 2*np.arcsin(1.2/env.number_of_antennas) # radians, consider 3 dB beamwidth

pathloss = np.zeros((N_UEs, TTI)) # pathloss
mmWave_LoS_NLoS_channel = np.zeros((N_UEs, env.number_of_antennas)) + 1j

genie_beam = np.empty([TTI, N_UEs,], dtype=object)

config = np.empty([N_UEs, 2], dtype = int)
edge_snr_db = np.array([-5,5,15,25])        
noise_power = 10**((-109+109) /10) # -109 dBm/Hz in mW


SNR_user = np.zeros((np.size(edge_snr_db),episode, N_UEs))
SNR_echo = np.zeros((np.size(edge_snr_db),episode, N_UEs))
SNR_echo_numerical = np.zeros((np.size(edge_snr_db),episode, N_UEs))
nmse = np.zeros((np.size(edge_snr_db), N_UEs))
nmse_theo = np.zeros((np.size(edge_snr_db), N_UEs))
nmse_alternative = np.zeros((np.size(edge_snr_db), N_UEs))

 # Used for AoD estimation
N_fft = 1024*128

# initialize user positions
user_positions[0,:,:] = np.array([[48,71]])
#user_positions[0,:,:], config, if_turned = user_pos_direction_init(road_matrix, cross_pos, N_UEs, single_lane_width)
d = np.linalg.norm(env.base_station_position - user_positions[0], axis = 1)
Pt_mW = 10 **(edge_snr_db / 10) * (4* np.pi * np.max(d))**2 / (env.lambda_ **2)

# For power allocation
xi = np.zeros(N_UEs)
threshold = [1e-5, 2**4 -1] # [SU, CU] threshold
ue_type = np.zeros(N_UEs)
if_dynamic = False # dynamic power allocation
Ratio = np.zeros((np.size(edge_snr_db), N_UEs))

# %%
for nPt in range(np.size(edge_snr_db)): 
    
    error = np.zeros((episode, N_UEs, TTI)) # AoD estimation error
    variance = np.zeros((episode, N_UEs, TTI)) # variance of AoD estimation error
    variance2 = np.zeros((episode, N_UEs, TTI)) # variance of AoD estimation error
    ratio = np.zeros((episode, N_UEs, TTI)) # ratio of z_BS to variance
    Pt = Pt_mW[nPt] / N_UEs *np.ones(N_UEs)

    for l in range(episode):

        current_beam = np.empty([TTI+1, N_UEs,], dtype=object)
        transmit_beam = np.zeros((N_UEs, env.number_of_antennas)) + 1j
        transmit_beam_multi = np.zeros((N_UEs, env.number_of_antennas)) + 1j
        
        for i in range(N_UEs):
            
            AoD[i] = compute_AoD_at_BS(env.base_station_position, user_positions[0][i])

            a_BS[i] = compute_ULA_response(env.number_of_antennas, AoD[i])  # current array response

            mmWave_LoS_NLoS_channel[i] = compute_mmWave_LoS_NLoS_channel(env.number_of_antennas, a_BS[i], d[i], env.lambda_)  # current LoS channel

            current_beam[0,i] = np.argmax(np.abs( env.codebook @ mmWave_LoS_NLoS_channel[i]))

        # Monte Carlo simulation
        for n in range(TTI):
            s_BS = np.zeros(( N_UEs, env.number_of_antennas, N), dtype = complex)
            mono_LoS_channel = np.zeros((env.number_of_antennas, env.number_of_antennas), dtype=complex)
            mmWave_LoS_NLoS_channel = np.zeros((N_UEs, env.number_of_antennas)) + 1j

            genie_s_BS = np.zeros((N_UEs, env.number_of_antennas, N), dtype = complex)

            for u in range(N_UEs):
                # update user position
                user_positions[n+1, u,:] = user_positions[n,u,:]

                # update distance
                d[u] = np.linalg.norm(env.base_station_position - user_positions[n+1,u])

                # update AoD real
                AoD[u] = compute_AoD_at_BS(env.base_station_position, user_positions[n+1,u])

                # update array response
                a_BS[u] = compute_ULA_response(env.number_of_antennas, AoD[u])

                # update LoS channel
                mmWave_LoS_NLoS_channel[u] = compute_mmWave_LoS_NLoS_channel(env.number_of_antennas, a_BS[u], d[u], env.lambda_)
                pathloss[u,n] = np.linalg.norm(mmWave_LoS_NLoS_channel[u])**2

                reflection_coefficient = env.lambda_ * 5 / (4* np.pi)**(1.5) / d[u]**2 

                mono_LoS_channel += compute_mono_static_channel(env.number_of_antennas, AoD[u], reflection_coefficient)

                # benchmark - genie-aided - upper bound
                genie_beam[n, u] = np.argmax(np.abs( np.sqrt(Pt[u])* env.codebook @ (mmWave_LoS_NLoS_channel[u])))

                genie_s_BS[u,:,:] = np.outer((env.codebook[genie_beam[n,u],:]), waveform[u, :])

                # transmit signal
                transmit_beam[u,] = env.codebook[genie_beam[n,u],:]
                    
                transmit_beam_multi[u,] = env.codebook[(genie_beam[n,u]+[-1,0,1]),:].sum(axis=0)
                
                s_BS[u,:,:] = np.outer(transmit_beam[u,], waveform[u, :])

            for u in range(N_UEs):
                s_BS[u,] = np.sqrt(Pt[u]/np.size(current_beam[n,u]) )* s_BS[u,]
            s_BS = np.sum(s_BS,axis=0)  

            Noise = ( np.random.normal(0, np.sqrt(noise_power/2), size = (env.number_of_antennas, N)) + 1j * np.random.normal(0, np.sqrt(noise_power/2), size = (env.number_of_antennas, N)))
            y_BS_noiseless = mono_LoS_channel @ s_BS
            # received echos at BS
            y_BS = y_BS_noiseless + Noise
            
            # Estimate AoDs from the echo
            aod = estimate_AoD_from_echo(y_BS, waveform, N_UEs, N, N_fft, env.number_of_antennas)
            # estimation error
            error[l,:,n] = np.abs(aod - AoD)
            
            for u in range(N_UEs):

                #beamforming output
                z_BS = np.sum(np.abs(transmit_beam[u,] @ y_BS)**2)

                # map the estimated aod to the closest beam in the codebook
                current_beam[n+1, u] = np.argmax(np.abs(env.codebook @ np.conj(compute_ULA_response(env.number_of_antennas, aod[u]))))
                
                # CLRB
                derivative2_a_BS = np.array([np.exp(-1j * np.pi * i * np.cos(AoD[u]))*(1j * np.pi * i * np.cos(AoD[u]) - (np.pi*i * np.sin(AoD[u]))**2 ) for i in range(env.number_of_antennas)])

                variance[l,u,n] = env.number_of_antennas/ (2*N) / (np.abs(np.sqrt(Pt[u])* reflection_coefficient * np.dot(a_BS[u], transmit_beam[u,] ) )**2) / ( np.linalg.norm(derivative2_a_BS)**2 + np.dot( derivative2_a_BS, a_BS[u]) + np.dot(a_BS[u], derivative2_a_BS))
                
                variance2[l,u,n] = env.number_of_antennas/ (2*N) / (np.abs(np.sqrt(Pt[u]/np.size(transmit_beam_multi[u,]))* reflection_coefficient * np.dot(a_BS[u], transmit_beam_multi[u,] ) )**2) / ( np.linalg.norm(derivative2_a_BS)**2 + np.dot( derivative2_a_BS, a_BS[u]) + np.dot(a_BS[u], derivative2_a_BS))

                ratio[l,u,n] = variance[l,u,n]/z_BS

                if error[l,u,n] > beamwidth:
                    #beam_index = np.mod( np.add(current_beam[n+1, u], [-1,0,1]), 31 )
                    beam_index = np.mod( np.add(current_beam[n, u], [-1,0,1]), 31 )
                    current_beam[n+1,u] = beam_index[beam_index>=0]


        SNR_user[nPt, l, :] = 10*np.log10(np.mean(Pt* pathloss,axis=1) /noise_power)
        SNR_echo[nPt, l, :] = 10*np.log10((Pt* env.number_of_antennas**2 * reflection_coefficient**2 ) /(noise_power*env.number_of_antennas))   
        SNR_echo_numerical[nPt, l, :] = 20*np.log10(np.mean( np.linalg.norm(y_BS_noiseless, axis=0) / np.linalg.norm(Noise, axis=0) ))
    
    nmse[nPt] = (np.mean(np.mean((error/np.pi*180), axis = 2), axis = 0)) # expected error in degree
    nmse_theo[nPt] = np.mean(np.mean(np.sqrt(variance), axis=2), axis = 0) # standard deviation in degree
    nmse_alternative[nPt] = np.sqrt(np.mean(np.mean(variance2, axis=2), axis = 0))

    Ratio[nPt] = np.mean(np.mean(ratio, axis=2), axis = 0)
# %%
plt.imshow(road_matrix.T)

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
plt.legend(['UE1','','UE2','', 'UE3', ''])    
plt.show()

# %%
for i in range(N_UEs):
    plt.semilogy(np.mean(SNR_echo[:,i], axis=1), nmse[:,i], '-ro')
    plt.semilogy(np.mean(SNR_echo[:,i], axis=1), nmse_theo[:,i], '-b>')
    plt.semilogy(np.mean(SNR_echo[:,i], axis=1), nmse_alternative[:,i], '-k*')
    plt.xlabel('Echo SNR_dB')
    plt.ylabel('NMSE')
    plt.legend(['Simulation', 'CRLB'])
    plt.show()

# %%
# plot beam pattern
beams = [9]
w = (1/N)*np.matmul(y_BS, waveform[u, :])#y_BS[:,0]#env.codebook[beams,:].sum(axis=0) # sum of 3 beams
w_padded = np.concatenate((w, np.zeros(N_fft - env.number_of_antennas)))
w_fft_dB = 10*np.log10(np.abs(np.fft.fftshift(np.fft.fft(w_padded)))**2) # magnitude of fft in dB
w_fft_dB -= np.max(w_fft_dB) # normalize to 0 dB at peak

# Map the FFT bins to angles in radians
theta_bins = np.arcsin(np.linspace(-1, 1, N_fft)) # in radians

# find max so we can add it to plot
top_three_indices = np.argsort(w_fft_dB)[-np.size(beams):]
theta_max = theta_bins[top_three_indices]
#theta_max = theta_bins[np.argmax(w_fft_dB)]

fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
ax.plot(theta_bins, w_fft_dB) # MAKE SURE TO USE RADIAN FOR POLAR
ax.plot([theta_max], [w_fft_dB[top_three_indices]],'ro')
ax.text(theta_max - 0.1, w_fft_dB[top_three_indices] - 4, np.round(theta_max * 180 / np.pi))
ax.set_theta_zero_location('N') # make 0 degrees point up
ax.set_theta_direction(-1) # increase clockwise
ax.set_rlabel_position(55)  # Move grid labels away from other labels
ax.set_thetamin(-90) # only show top half
ax.set_thetamax(90)
ax.set_ylim([-30, 1]) # because there's no noise, only go down 30 dB
plt.show()


