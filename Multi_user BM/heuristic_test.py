# %%
from channel_functions import *
#from motion import *
import numpy as np
import gymnasium as gym
from scipy.linalg import hadamard
import matplotlib.pyplot as plt
from road import *
from itertools import groupby

from beam_management_env_letter import BeamManagementEnv

env = gym.make('BeamManagementEnv-v1', N_f=1000)
episode_durations = []

# %%
# variable initialization
N_UEs = 2 #env.N_UEs
TTI = 100
episode =  100
N = 64 # number of samples

d = np.zeros(N_UEs)
AoD = np.zeros(N_UEs) # real value
aod = np.zeros(N_UEs) # estimated value
a_BS = np.zeros((N_UEs, env.number_of_antennas)) + 1j
mmWave_LoS_NLoS_channel = np.zeros((N_UEs, env.number_of_antennas)) + 1j

edge_snr_db = np.array([10])
noise_power = 1 #Pt/N_UEs / (50*np.sqrt(2)) ** 2 * 10 **(-edge_snr_db / 10)
Pt = 10 **(edge_snr_db / 10) * (4* np.pi * 50)**2 / (env.lambda_ **2)

# fixed map is imported from env_letter

y_user = np.zeros((N_UEs, N)) + 1j
y_BS = np.zeros((env.number_of_antennas, N)) + 1j
waveform = hadamard(N)

nitta = np.zeros((env.number_of_antennas)) + 1j # temperal variable for AoD estimation
beamwidth = 2*np.arcsin(1.2/env.number_of_antennas) # radians, consider 3 dB beamwidth
nmse = np.zeros((np.size(Pt), episode, N_UEs))
pathloss = np.zeros((N_UEs, TTI)) # pathloss
var_aod = np.zeros((np.size(Pt), episode, N_UEs))
 
genie_beam = np.empty([TTI, N_UEs,], dtype=object)

genie_y_user = np.zeros((N_UEs, N))+ 1j
rx_snr = np.zeros((TTI, N_UEs))
SNR_user = np.zeros((np.size(Pt),episode, N_UEs))
SNR_echo = np.zeros((np.size(Pt),episode, N_UEs))

# Used for AoD estimation
N_fft = 1024

# %%
# beam initialization using beam sweeping
for nPt in range(np.size(Pt)): 
    rate_all_episodes = []
    genie_rate_all_episodes = []
    sinr_user = []
    genie_sinr_user = []
    error = np.zeros((episode, N_UEs, TTI)) # AoD estimation error
    variance = np.zeros((episode, N_UEs, TTI)) # variance of AoD estimation error

    for l in range(episode):
        rate = np.zeros((N_UEs,TTI))
        genie_rate = np.zeros((N_UEs,TTI))
        sinr = np.zeros((N_UEs,TTI))
        genie_sinr = np.zeros((N_UEs,TTI))
        current_beam = np.empty([TTI+1, N_UEs,], dtype=object)
        user_positions = np.zeros((TTI+1, N_UEs, 2))

        config = np.empty([N_UEs, 2], dtype = int)
        
        # initialize user positions in every episode
        user_positions[0,:,:], config, if_turned = user_pos_direction_init(env.road_matrix, env.cross_pos, N_UEs, env.single_lane_width)
        
        for i in range(N_UEs):
            
            AoD[i] = compute_AoD_at_BS(env.base_station_position, user_positions[0][i])

            d[i] = np.linalg.norm(env.base_station_position - user_positions[0][i])

            a_BS[i] = compute_ULA_response(env.number_of_antennas, AoD[i])  # current array response

            mmWave_LoS_NLoS_channel[i] = compute_mmWave_LoS_NLoS_channel(env.number_of_antennas, a_BS[i], d[i], env.lambda_)  # current LoS channel

            current_beam[0,i] = np.argmax(np.abs(np.sqrt(Pt[nPt]) * env.codebook @ mmWave_LoS_NLoS_channel[i]))

        # Moving users
        for n in range(TTI):
            s_BS = np.zeros(( N_UEs, env.number_of_antennas, N), dtype = complex)
            mono_LoS_channel = np.zeros((env.number_of_antennas, env.number_of_antennas), dtype=complex)
            mmWave_LoS_NLoS_channel = np.zeros((N_UEs, env.number_of_antennas)) + 1j
            genie_s_BS = np.zeros((N_UEs, env.number_of_antennas, N), dtype = complex)
            genie_user_signal = np.zeros((N_UEs, N), dtype = complex)
            user_signal = np.zeros((N_UEs, N), dtype = complex)

            for u in range(N_UEs):
                # update user position
                user_positions[n+1, u,:], config[u,:], if_turned[u] = update_positions(user_positions[n,u,:], env.cross_pos, sample_velocity(), env.dt, env.single_lane_width, config[u,:], if_turned[u]) 

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
                genie_beam[n, u] = np.argmax(np.abs( np.sqrt(Pt[nPt]/N_UEs)* env.codebook @ (mmWave_LoS_NLoS_channel[u])))

                genie_s_BS[u,:,:] = np.sqrt( Pt[nPt] / N_UEs ) * np.outer((env.codebook[genie_beam[n,u],:]), waveform[u, :])

                # target signal for user
                genie_user_signal[u] = mmWave_LoS_NLoS_channel[u] @ genie_s_BS[u]

                # transmit signal
                if current_beam[n,u].size == 1:
                    transmit_beam = env.codebook[current_beam[n,u],:]
                else:
                    transmit_beam = env.codebook[current_beam[n,u],:].sum(axis=0)
                
                s_BS[u,:,:] = np.sqrt(Pt[nPt] / N_UEs / np.size(current_beam[n,u])) * np.outer((transmit_beam), waveform[u, :])

                user_signal[u] = mmWave_LoS_NLoS_channel[u] @ s_BS[u]

            # user received signal noise-less including interference
            y_user = mmWave_LoS_NLoS_channel @ np.sum(s_BS, axis=0)
            genie_y_user = mmWave_LoS_NLoS_channel @ np.sum(genie_s_BS, axis=0)

            # genie-aided rate
            genie_rate[:,n], genie_sinr[:,n] = compute_communication_rate(genie_y_user, genie_user_signal, noise_power, N_UEs, N) 

            # calculate communication rate of each user
            rate[:,n], sinr[:,n] = compute_communication_rate(y_user, user_signal, noise_power, N_UEs, N)
            # for u in range(N_UEs):
            #     if current_beam[n,u].size > 1:
            #         rate[u,n] = 0

            s_BS = np.sum(s_BS,axis=0)    
            
            # received echos at BS
            y_BS = mono_LoS_channel @ s_BS +  ( np.random.normal(0, np.sqrt(noise_power/2), size = (env.number_of_antennas, N)) + 1j * np.random.normal(0, np.sqrt(noise_power/2), size = (env.number_of_antennas, N)))
            
            # Estimate AoDs from the echo
            aod = estimate_AoD_from_echo(y_BS, waveform, N_UEs, N, N_fft, env.number_of_antennas)
            # estimation error
            error[l,:,n] = np.abs(aod - AoD)
            # extract each user's signal
            for u in range(N_UEs):
                
                # map the estimated aod to the closest beam in the codebook
                current_beam[n+1, u] = np.argmax(np.abs(env.codebook @ np.conj(compute_ULA_response(env.number_of_antennas, aod[u]))))

                # CLRB
                derivative_a_BS = np.array([np.exp(-1j * np.pi * i * np.cos(AoD[u]))*(1j * np.pi * i * np.sin(AoD[u])) for i in range(env.number_of_antennas)])
                variance[l,u,n] = 1/2/ N/ abs(np.sqrt(Pt[nPt]/N_UEs)* reflection_coefficient * env.number_of_antennas * np.dot(a_BS[u], transmit_beam ) )**2 / ( np.linalg.norm(derivative_a_BS)**2 - np.abs( np.dot( derivative_a_BS, a_BS[u]))**2 )

                if error[l,u,n] > beamwidth:
                    beam_index = np.insert(np.array(current_beam[n,u]), 0, [np.min(current_beam[n,u])-1, np.max(current_beam[n,u])+1])
                    beam_index = np.unique(np.mod( beam_index, 31 ))
                    #beam_index = np.mod( np.add(np.min(current_beam[n, u]), [-1,0,1]), 31 ) # use the previous AoD
                    #beam_index = np.mod( np.add(current_beam[n+1, u], [-1,0,1]), 31 ) # use the estimated AoD
                    current_beam[n+1,u] = beam_index[beam_index >= 0]
                # else keep the current beam
                else:   
                    current_beam[n+1,u] = current_beam[n,u]

        rate_all_episodes.append(rate)
        genie_rate_all_episodes.append(genie_rate)
        sinr_user.append(10*np.log10(sinr))
        genie_sinr_user.append(10*np.log10(genie_sinr))

        SNR_user[nPt, l, :] = 10*np.log10(np.mean((Pt[nPt]/N_UEs)* pathloss,axis=1) /noise_power)
        SNR_echo[nPt, l, :] = 10*np.log10(((Pt[nPt]/N_UEs)* env.number_of_antennas**2 * reflection_coefficient**2) /(noise_power*env.number_of_antennas))
    nmse[nPt, :, :] = np.mean((error)**2, axis = 2)
    var_aod[nPt, :,:] = np.mean(variance, axis = 2)


    rate_all_episodes = np.array(rate_all_episodes)
    genie_rate_all_episodes = np.array(genie_rate_all_episodes)
    for i in range (N_UEs):
        rate_cdf = np.sort(rate_all_episodes[:,i,:].flatten())
        yvals = np.arange(len(rate_cdf))/float(len(rate_cdf)-1)
        plt.plot(rate_cdf, yvals)

        genie_rate_cdf = np.sort(genie_rate_all_episodes[:,i,:].flatten())
        yvals = np.arange(len(genie_rate_cdf))/float(len(genie_rate_cdf)-1)
        plt.plot(genie_rate_cdf, yvals)

    plt.xlabel('Rate (bps/Hz)')
    plt.ylabel('CDF')
    plt.legend(['UE1-Empirical','UE1-Genie','UE2-Empirical','UE2-Genie'])

# %%
# plot beam pattern
N_fft = 1024
beams = [9]
w = y_BS[:,0]#env.codebook[beams,:].sum(axis=0) # sum of 3 beams
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

# %%
plt.imshow(env.road_matrix.T)

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

temp = np.ceil(SNR_echo.flatten())
temp_nmse = nmse.flatten()
temp_var = var_aod.flatten()/np.pi**2
temp_list = list(zip(temp, temp_nmse, temp_var))
temp_list.sort(key = lambda x: x[0])
snr_grouped = [np.mean([x[0] for x in g]) for k, g in groupby(temp_list, lambda x: x[0])]
nmse_grouped = [np.mean([x[1] for x in g]) for k, g in groupby(temp_list, lambda x: x[0])]
var_grouped = [np.mean([x[2] for x in g]) for k, g in groupby(temp_list, lambda x: x[0])]
plt.semilogy(snr_grouped, nmse_grouped)
#plt.semilogy(snr_grouped, var_grouped)
plt.xlabel('Echo SNR_dB')
plt.ylabel('NMSE')
plt.legend(['NMSE','Variance'])
plt.show()


# %%
