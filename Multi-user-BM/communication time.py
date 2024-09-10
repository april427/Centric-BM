# %%
from channel_functions import *
import numpy as np
import gymnasium as gym
from scipy.linalg import hadamard
import matplotlib.pyplot as plt
from road import *
from plot_functions import *

from beam_management_env_letter import BeamManagementEnv

env = gym.make('BeamManagementEnv-v1')
episode_durations = []

# %%
# variable initialization
N_UEs = env.unwrapped.N_UEs
TTI = env.unwrapped.num_steps 
episode =  100
N_samples = env.unwrapped.N_samples # number of samples
number_of_antennas = env.unwrapped.number_of_antennas
base_station_position = env.unwrapped.base_station_position
codebook = env.unwrapped.codebook

d = np.zeros(N_UEs)
AoD = np.zeros(N_UEs) # real value
aod = np.zeros(N_UEs) # estimated value
a_BS = np.zeros((N_UEs, number_of_antennas), dtype = complex) 
mmWave_LoS_NLoS_channel = np.zeros((N_UEs, number_of_antennas), dtype = complex) 

edge_snr_db = np.array([30])
noise_power = env.unwrapped.noise_power # -109 dBm/Hz in mW
Pt_mW = np.zeros(1) + 1j
Pt_mW[0] = env.unwrapped.Pt_mW # 20 dBm in mW, SNR of 1 user = 33 dB

# fixed map is imported from env_letter

y_user = np.zeros((N_UEs, N_samples), dtype = complex) 
y_BS = np.zeros((number_of_antennas, N_samples), dtype = complex) 
waveform = hadamard(N_samples)
waveform[[0, -1],:] = waveform[[-1,0],:]

nitta = np.zeros((number_of_antennas), dtype = complex) # temperal variable for AoD estimation
beamwidth = 2*np.arcsin(1.2/number_of_antennas) # radians, consider 3 dB beamwidth
nmse = np.zeros((np.size(Pt_mW), episode, N_UEs))
pathloss = np.zeros((N_UEs, TTI)) # pathloss
var_aod = np.zeros((np.size(Pt_mW), episode, N_UEs))
 
echo_bf = np.zeros((episode, TTI, N_UEs))
echo_bf_genie = np.zeros((episode, TTI, N_UEs))

genie_y_user = np.zeros((N_UEs, N_samples))+ 1j
rx_snr = np.zeros((TTI, N_UEs))
SNR_user = np.zeros((np.size(Pt_mW), episode, N_UEs))
SNR_echo = np.zeros((np.size(Pt_mW), episode, N_UEs))

# Used for AoD estimation
N_fft = 1024

# For power allocation
xi = np.zeros(N_UEs)
#threshold = [1e-5, 2**4 -1] # [SU, CU] threshold
threshold = [ 5/(2*beamwidth**2)/(number_of_antennas)**4 , 2**4-1]
ue_type = np.zeros(N_UEs, dtype = int)
if_dynamic = True # dynamic power allocation
genie_transmitted_rate = np.zeros(episode)
heu_transmitted_rate = np.zeros(episode)

# %%
# beam initialization using beam sweeping
for nPt in range(np.size(Pt_mW)): 
    rate_all_episodes = []
    genie_rate_all_episodes = []
    sinr_user = []
    genie_sinr_user = []
    user_count_sensing = np.zeros(episode)
    error = np.zeros((episode, N_UEs, TTI)) # AoD estimation error
    variance = np.zeros((episode, N_UEs, TTI)) # variance of AoD estimation error
    transmission_time = np.zeros(episode)
    reward = np.zeros((episode, TTI))
    reward_genie = np.zeros((episode, TTI))

    for l in range(episode):
        rate = np.zeros((N_UEs,TTI))
        genie_rate = np.zeros((N_UEs,TTI))
        sinr = np.zeros((N_UEs,TTI))
        genie_sinr = np.zeros((N_UEs,TTI))
        current_beam = np.empty([TTI+1, N_UEs,], dtype=object)
        genie_beam = np.empty([TTI, N_UEs,], dtype=object)
        user_positions = np.zeros((TTI+1, N_UEs, 2))

        config = np.empty([N_UEs, 2], dtype = int)
        datapool = np.random.binomial(n=TTI, p=0.6, size=N_UEs).astype(float)
        total_num_packets = np.sum(datapool)
        genie_transmitted = 0


        # initialize user positions in every episode
        #user_positions[0,:,:], config, if_turned = user_pos_direction_init(env.unwrapped.road_matrix, env.unwrapped.cross_pos, N_UEs, env.unwrapped.single_lane_width)
        dice = np.random.randint(0, 6)
        if dice == 0:
            user_positions[0,:,:] = np.array([[27., 65.],[79., 42.]])
            config = np.array([[4,2],[4,2]])
            if_turned = np.array([False, False])
        elif dice == 1:
            user_positions[0,:,:] = np.array([[31., 39.],[59., 72.]])
            config = np.array([[3,3],[1,1]])
            if_turned = np.array([False, True])
        elif dice == 2:
            user_positions[0,:,:] = np.array([[23., 79.],[61., 29.]])
            config = np.array([[3,3],[1,1]])
            if_turned = np.array([False, False])
        elif dice == 3:
            user_positions[0,:,:] = np.array([[14., 33.],[75., 34.]])
            config = np.array([[4,2],[4,2]])
            if_turned = np.array([False, False])
        elif dice == 4:
            user_positions[0,:,:] = np.array([[30., 21.],[75., 74.]])
            config = np.array([[2,2],[1,1]])
            if_turned = np.array([False, False])
        elif dice == 5:
            user_positions[0,:,:] = np.array([[35., 73.],[64., 33.]])
            config = np.array([[4,2],[4,2]])
            if_turned = np.array([False, False])
        
        for i in range(N_UEs):
            
            AoD[i] = compute_AoD_at_BS(base_station_position, user_positions[0][i])

            d[i] = np.linalg.norm(base_station_position - user_positions[0][i])

            a_BS[i] = compute_ULA_response(number_of_antennas, AoD[i])  # current array response

            mmWave_LoS_NLoS_channel[i] = compute_mmWave_LoS_NLoS_channel(number_of_antennas, a_BS[i], d[i], env.lambda_)  # current LoS channel

            current_beam[0,i] = np.argmax(np.abs( codebook @ mmWave_LoS_NLoS_channel[i]))

        # Moving users calculate the number of TTIs
        for n in range(TTI):
            s_BS = np.zeros(( N_UEs, number_of_antennas, N_samples), dtype = complex)
            mono_LoS_channel = np.zeros((number_of_antennas, number_of_antennas), dtype=complex)
            mmWave_LoS_NLoS_channel = np.zeros((N_UEs, number_of_antennas), dtype=complex) 
            genie_s_BS = np.zeros((N_UEs, number_of_antennas, N_samples), dtype = complex)
            genie_user_signal = np.zeros((N_UEs, N_samples), dtype = complex)
            user_signal = np.zeros((N_UEs, N_samples), dtype = complex)
            Pt = Pt_mW[nPt] / N_UEs * np.ones(N_UEs) 
            transmit_beam = np.zeros((N_UEs, number_of_antennas), dtype = complex)
            genie_transmit_beam = np.zeros((N_UEs, number_of_antennas), dtype = complex)

            for u in range(N_UEs):
                # update user position
                user_positions[n+1, u,:], config[u,:], if_turned[u] = update_positions(user_positions[n,u,:], env.cross_pos, sample_velocity(), env.unwrapped.dt, env.unwrapped.single_lane_width, config[u,:], if_turned[u]) 

                # update distance
                d[u] = np.linalg.norm(base_station_position - user_positions[n+1,u])

                # update AoD real
                AoD[u] = compute_AoD_at_BS(base_station_position, user_positions[n+1,u])

                # update array response
                a_BS[u] = compute_ULA_response(number_of_antennas, AoD[u])

                # update LoS channel
                mmWave_LoS_NLoS_channel[u] = compute_mmWave_LoS_NLoS_channel(number_of_antennas, a_BS[u], d[u], env.lambda_)
                pathloss[u,n] = np.linalg.norm(mmWave_LoS_NLoS_channel[u])**2

                reflection_coefficient = env.lambda_ * 5 / (4* np.pi)**(1.5) / d[u]**2  

                mono_LoS_channel += compute_mono_static_channel(number_of_antennas, AoD[u], reflection_coefficient)

                # benchmark - genie-aided - upper bound
                genie_beam[n, u] = np.argmax(np.abs(codebook @ (mmWave_LoS_NLoS_channel[u])))
                genie_transmit_beam[u,:] = codebook[genie_beam[n,u],:]

                genie_s_BS[u,:,:] = np.outer(codebook[genie_beam[n,u],:], waveform[u, :])

                # transmit signal
                if current_beam[n,u].size == 1:
                    transmit_beam[u,] = codebook[current_beam[n,u],:]
                    xi[u] = np.abs(transmit_beam[u,] @ mmWave_LoS_NLoS_channel[u])**2 # for power allocation
                    ue_type[u] = 1
                else:
                    transmit_beam[u,] = codebook[current_beam[n,u],:].sum(axis=0)
                    xi[u] = reflection_coefficient**2 * N_samples * np.abs(a_BS[u] @ transmit_beam[u,])**2 # for power allocation
                    ue_type[u] = 0
                    user_count_sensing[l] += 1
                
                s_BS[u,:,:] = np.outer(transmit_beam[u,], waveform[u, :])

            # Allocate power
            if if_dynamic:
                Pt = dynamic_power_allocation(d, xi, Pt, Pt_mW, ue_type, threshold, N_UEs, datapool)
            
            for u in range(N_UEs):
                genie_s_BS[u,] = np.sqrt(Pt[u]) * genie_s_BS[u,]

                # target signal for user
                genie_user_signal[u] = mmWave_LoS_NLoS_channel[u] @ genie_s_BS[u]

                s_BS[u,] = np.sqrt(Pt[u]/np.size(current_beam[n,u]) )* s_BS[u,]

                user_signal[u] = mmWave_LoS_NLoS_channel[u] @ s_BS[u]
            
            # user received signal noise-less including interference
            y_user = mmWave_LoS_NLoS_channel @ np.sum(s_BS,axis=0) 
            genie_y_user = mmWave_LoS_NLoS_channel @ np.sum(genie_s_BS, axis=0)

            # genie-aided rate
            genie_rate[:,n], genie_sinr[:,n], interference_db = compute_communication_rate(genie_y_user, genie_user_signal, noise_power, N_UEs, N_samples) 

            if genie_transmitted < total_num_packets:
                genie_transmitted += np.sum(((genie_rate[:,n] >= 4)  ))
                genie_ue_type = np.array([1,1])
            else:
                genie_ue_type = np.array([0,0])

            # calculate communication rate of each user
            rate[:,n], sinr[:,n], interference_db = compute_communication_rate(y_user, user_signal, noise_power, N_UEs, N_samples)
            
            datapool -= ((rate[:,n] >= 4) * ue_type )
            datapool[datapool < 0] = 0
            if np.sum(datapool) == 0:
                break   
            
            genie_y_BS = mono_LoS_channel @ np.sum(genie_s_BS,axis=0) + ( np.random.normal(0, np.sqrt(noise_power/2), size = (number_of_antennas, N_samples)) + 1j * np.random.normal(0, np.sqrt(noise_power/2), size = (number_of_antennas, N_samples)))

            # received echos at BS
            y_BS = mono_LoS_channel @ np.sum(s_BS,axis=0)  +  ( np.random.normal(0, np.sqrt(noise_power/2), size = (number_of_antennas, N_samples)) + 1j * np.random.normal(0, np.sqrt(noise_power/2), size = (number_of_antennas, N_samples)))
            
            # Estimate AoDs from the echo
            aod = estimate_AoD_from_echo(y_BS, waveform, N_UEs, N_samples, N_fft, number_of_antennas)
            # estimation error
            error[l,:,n] = np.abs(aod - AoD)

            echo_bf[l,n,:] = np.mean(np.abs(np.conj(transmit_beam) @ y_BS)**2, axis = 1)

            echo_bf_genie[l,n,:] = np.mean(np.abs(np.conj(genie_transmit_beam) @ genie_y_BS)**2, axis = 1)

            for u in range(N_UEs):

                # CLRB
                derivative_a_BS = np.array([np.exp(-1j * np.pi * i * np.cos(AoD[u]))*(1j * np.pi * i * np.sin(AoD[u])) for i in range(number_of_antennas)])

                variance[l,u,n] = 1/2/ N_samples/ abs(np.sqrt(Pt[u])* reflection_coefficient * number_of_antennas * np.dot(a_BS[u], transmit_beam[u,] ) )**2 / ( np.linalg.norm(derivative_a_BS)**2 - np.abs( np.dot( derivative_a_BS, a_BS[u]))**2 )


                # map the estimated aod to the closest beam in the codebook
                current_beam[n+1, u] = np.argmax(np.abs(codebook @ np.conj(compute_ULA_response(number_of_antennas, aod[u]))))
                if np.sqrt(variance[l,u,n]) > beamwidth: #error[l,u,n] > 2*beamwidth: #echo_bf[u]< beamwidth*1e4: 
                    # beam_index = np.insert(np.array(current_beam[n,u]), 0, [np.min(current_beam[n,u])-1, np.max(current_beam[n,u])+1])
                    beam_index = np.mod( np.add(np.min(current_beam[n, u]), [-1,0,1]), 31 ) # use the previous AoD
                    #beam_index = np.mod( np.add(current_beam[n+1, u], [-1,0,1]), 31 ) # use the estimated AoD
                    beam_index = np.unique(np.mod( beam_index, 31 ))
                    current_beam[n+1,u] = beam_index[beam_index >= 0]
                # else keep the current beam
                else:   
                    if current_beam[n,u].size == 1:
                        current_beam[n+1,u] = current_beam[n,u] #CU=>CU
                    else:
                        current_beam[n+1,u] =current_beam[n+1,u] #SU=>CU

            #reward += np.sum(0.75 * echo_bf[l,n] * ue_type + 0.25* echo_bf[l,n] * (1-ue_type))
            echo_bf_db = 10*np.log10(echo_bf[l,n,:]) 
            hyper_sinr_commu = hyperbolic_func(sinr[:,n], 0.03, 0.4, 12) #snr in dB
            hyper_sinr_sensing = hyperbolic_func(echo_bf_db,  0.03, 0.4, 20)
            reward[l,n]= 0.75*np.sum(hyper_sinr_commu * ue_type ) + 0.25*np.sum((1-ue_type) * hyper_sinr_sensing)

            echo_bf_genie_db = 10*np.log10(echo_bf_genie[l,n,:])
            reward_genie[l,n] = 0.75*np.sum(hyperbolic_func(genie_sinr[:,n],  0.03, 0.4, 12) * genie_ue_type ) + 0.25*np.sum((1-genie_ue_type) * hyperbolic_func(echo_bf_genie_db, 0.03, 0.4, 20))


        # calculate the percentage of transmitted packets    
        heu_transmitted_rate[l] = (total_num_packets - np.sum(datapool))/total_num_packets
        genie_transmitted_rate[l] = genie_transmitted/total_num_packets

        transmission_time[l] = n

        sinr_user.append(sinr)
        genie_sinr_user.append(genie_sinr)

        user_count_sensing[l] = user_count_sensing[l] / (n+1) 
        #SNR_user[nPt, l, :] = 10*np.log10(np.mean(Pt[u]* pathloss,axis=1) /noise_power)
        #SNR_echo[nPt, l, :] = 10*np.log10( np.mean((Pt[u]* number_of_antennas**2 * reflection_coefficient**2)) /(noise_power*number_of_antennas))

    nmse[nPt, :, :] = np.mean((error)**2, axis = 2)
    var_aod[nPt, :,:] = np.mean(variance, axis = 2)

#echo_bf_db = 10*np.log10(np.mean(echo_bf, axis = 0) )
# %%
# print(dice)
# print(genie_beam)

fig_sinr,ax_sinr = plt.subplots()
inx = 0
variable = np.zeros([episode,N_UEs*TTI])
for item in sinr_user:
    variable[inx,:] = item.flatten()
    inx+=1
variable = variable.flatten()
variable = variable[variable != -np.inf]
variable_cdf = np.sort(variable)
yvals = np.arange(len(variable))/float(len(variable)-1)
ax_sinr.plot(variable_cdf, yvals, '--')
ax_sinr.grid(True, which='both')
ax_sinr.xlabel('SINR')
ax_sinr.legend(['heuristic','genie'])
#fig_sinr.show()

# hist, bins = np.histogram(variable, bins=15)
# bin_centers = (bins[1:]+bins[:-1])*0.5
# plt.plot(bin_centers, hist)

# %%
figure_handle = plt.figure()
ax = figure_handle.add_subplot()
heu_transmitted_rate_cdf = np.sort(heu_transmitted_rate)
yvals = np.arange(len(heu_transmitted_rate))/float(len(heu_transmitted_rate)-1)
ax.plot(heu_transmitted_rate_cdf, yvals, '-')
#
genie_transmitted_rate_cdf = np.sort(genie_transmitted_rate)
yvals = np.arange(len(genie_transmitted_rate))/float(len(genie_transmitted_rate)-1)
ax.plot(genie_transmitted_rate_cdf, yvals, '-')
ax.legend(['Heuristic', 'Genie'])
ax.set_ylabel('CDF')
ax.set_xlabel('Packet throughput')

# # %%
# # # plot beam pattern
# # plot_beam_pattern(y_BS, number_of_antennas)
# plot_road_trraffic(env.unwrapped.road_matrix, N_UEs, user_positions)

# # plot_nmse_vs_snr(SNR_echo, nmse, var_aod)

# np.savetxt('results_bm/Thp_heuristic', heu_transmitted_rate)
# np.savetxt('results_bm/Thp_genie', genie_transmitted_rate)

# # %%
# moving_avg = np.convolve(reward_episode, np.ones(10)/10, mode='valid')
# plt.plot(moving_avg)
# plt.plot(reward_episode)
# plt.show()
# %%
