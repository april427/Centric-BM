# %%
from channel_functions import *
import numpy as np
import gymnasium as gym
from scipy.linalg import hadamard
import matplotlib.pyplot as plt
from road import *

from beam_management_env_letter import BeamManagementEnv

env = gym.make('BeamManagementEnv-v1')
episode_durations = []

# %%
# variable initialization
N_UEs = 1#env.unwrapped.N_UEs #env.N_UEs
TTI = env.unwrapped.num_steps  #env.TTI
episode =  1
N_samples = env.unwrapped.N_samples # number of samples in one TTI
number_of_antennas = env.unwrapped.number_of_antennas
base_station_position = env.unwrapped.base_station_position

d = np.zeros(N_UEs)
AoD = np.zeros(N_UEs) # real value
aod = np.zeros(N_UEs) # estimated value
a_BS = np.zeros((N_UEs, number_of_antennas),dtype = complex) 

## fixed map
cross_pos = np.array([[27, 80], [30, 70]])
road_matrix = road(cross_pos)
single_lane_width = 5

dt = env.unwrapped.dt # time step 10 ms

y_BS = np.zeros((number_of_antennas, N_samples), dtype = complex)
waveform = hadamard(N_samples)
waveform[[0, -1],:] = waveform[[-1,0],:]

nitta = np.zeros((number_of_antennas), dtype = complex) # temperal variable for AoD estimation
beamwidth = 2*np.arcsin(1.2/number_of_antennas) # radians, consider 3 dB beamwidth

pathloss = np.zeros((N_UEs, TTI)) # pathloss
mmWave_LoS_NLoS_channel = np.zeros((N_UEs, number_of_antennas), dtype = complex) # LoS channel

genie_beam = np.empty([TTI, N_UEs,], dtype=object)

config = np.empty([N_UEs, 2], dtype = int)       
noise_power = env.unwrapped.noise_power # -109 dBm/Hz in mW
Pt_mW = np.zeros(1) + 1j
Pt_mW[0] = env.unwrapped.Pt_mW # 20 dBm in mW, SNR of 1 user = 33 dB

SNR_user = np.zeros((np.size(Pt_mW),episode,N_UEs))
SNR_echo = np.zeros((np.size(Pt_mW),episode, TTI))
SNR_echo_numerical = np.zeros((np.size(Pt_mW),episode, TTI))
nmse = np.zeros((np.size(Pt_mW), TTI))
nmse_theo = np.zeros((np.size(Pt_mW), TTI))
nmse_alternative = np.zeros((np.size(Pt_mW), TTI))

 # Used for AoD estimation
N_fft = 1024
user_positions = np.zeros((TTI+1, N_UEs, 2))
reflection_coefficient = np.zeros(TTI)

# initialize user positions
#user_positions[0,:,:], config, if_turned = user_pos_direction_init(road_matrix, cross_pos, N_UEs, single_lane_width)
# user_positions[0,:,:] = env.unwrapped.user_positions
# config = env.unwrapped.config
# if_turned = env.unwrapped.if_turned
# d = np.linalg.norm(base_station_position - user_positions[0], axis = 1)

# For power allocation
xi = np.zeros((N_UEs, TTI))
threshold = [1e-5, 2**4 -1] # [SU, CU] threshold
ue_type = np.zeros(N_UEs)
if_dynamic = False # dynamic power allocation

echo_bf = np.zeros((episode, N_UEs, TTI))

# %%
for nPt in range(np.size(Pt_mW)):  
    
    error = np.zeros((episode, N_UEs, TTI)) # AoD estimation error
    variance = np.zeros((episode, N_UEs, TTI)) # variance of AoD estimation error
    variance2 = np.zeros((episode, N_UEs, TTI)) # variance of AoD estimation error
    ratio = np.zeros((episode, N_UEs, TTI)) # ratio of z_BS to variance
    temp = np.zeros((episode, N_UEs, TTI)) 
    Pt = Pt_mW[nPt] / N_UEs *np.ones(N_UEs)



    for l in range(episode):

        current_beam = np.empty([TTI+1, N_UEs,], dtype=object)
        transmit_beam = np.zeros((N_UEs, number_of_antennas), dtype = complex)
        transmit_beam_multi = np.zeros((N_UEs, number_of_antennas), dtype = complex)

        config = np.empty([N_UEs, 2], dtype = int)
        user_positions[0,:,:] = np.array([[31,39]])
        config = np.array([[3,3]])
        if_turned = np.array([False])
        for i in range(N_UEs):
            
            AoD[i] = compute_AoD_at_BS(base_station_position, user_positions[0][i])

            a_BS[i] = compute_ULA_response(number_of_antennas, AoD[i])  # current array response

            d[i] = np.linalg.norm(base_station_position - user_positions[0][i])

            mmWave_LoS_NLoS_channel[i] = compute_mmWave_LoS_NLoS_channel(number_of_antennas, a_BS[i], d[i], env.lambda_)  # current LoS channel

            current_beam[0,i] = np.argmax(np.abs( env.codebook @ mmWave_LoS_NLoS_channel[i]))

        # Monte Carlo simulation
        for n in range(TTI):
            s_BS = np.zeros(( N_UEs, number_of_antennas, N_samples), dtype = complex)
            mono_LoS_channel = np.zeros((number_of_antennas, number_of_antennas), dtype=complex)
            mmWave_LoS_NLoS_channel = np.zeros((N_UEs, number_of_antennas), dtype = complex)

            genie_s_BS = np.zeros((N_UEs, number_of_antennas, N_samples), dtype = complex)

            for u in range(N_UEs):
                # update user position
                #user_positions[n+1, u,:] = user_positions[n,u,:]
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

                reflection_coefficient[n] = env.lambda_ * 5 / (4* np.pi)**(1.5) / d[u]**2 

                mono_LoS_channel += compute_mono_static_channel(number_of_antennas, AoD[u], reflection_coefficient[n])

                # benchmark - genie-aided - upper bound
                genie_beam[n, u] = np.argmax(np.abs( np.sqrt(Pt[u])* env.codebook @ (mmWave_LoS_NLoS_channel[u])))

                genie_s_BS[u,:,:] = np.outer((env.codebook[genie_beam[n,u],:]), waveform[u, :])

                # transmit signal
                transmit_beam[u,] = env.codebook[genie_beam[n,u],:]
                    
                transmit_beam_multi[u,] = env.codebook[(genie_beam[n,u]+[-1,0,1]),:].sum(axis=0)
                
                s_BS[u,:,:] = np.outer(transmit_beam[u,], waveform[u, :])

                xi[u,n] = reflection_coefficient[n]**2 * N_samples * np.abs(a_BS[u] @ transmit_beam[u,])**2

            for u in range(N_UEs):
                s_BS[u,] = np.sqrt(Pt[u]/np.size(current_beam[n,u]) )* s_BS[u,]
            s_BS = np.sum(s_BS,axis=0)  

            Noise = ( np.random.normal(0, np.sqrt(noise_power/2), size = (number_of_antennas, N_samples)) + 1j * np.random.normal(0, np.sqrt(noise_power/2), size = (number_of_antennas, N_samples)))
            y_BS_noiseless = mono_LoS_channel @ s_BS
            # received echos at BS
            y_BS = y_BS_noiseless + Noise
            
            # Estimate AoDs from the echo
            aod = estimate_AoD_from_echo(y_BS, waveform, N_UEs, N_samples, N_fft, number_of_antennas)
            # estimation error
            error[l,:,n] = np.abs(aod - AoD)
            
            for u in range(N_UEs):

                #beamforming output
                echo_bf[l,u,n] = np.sum(np.abs(transmit_beam[u,] @ y_BS)**2)

                # map the estimated aod to the closest beam in the codebook
                current_beam[n+1, u] = np.argmax(np.abs(env.codebook @ np.conj(compute_ULA_response(number_of_antennas, aod[u]))))
                
                # CLRB
                derivative2_a_BS = np.array([np.exp(-1j * np.pi * i * np.cos(AoD[u]))*(1j * np.pi * i * np.cos(AoD[u]) - (np.pi*i * np.sin(AoD[u]))**2 ) for i in range(number_of_antennas)])

                variance[l,u,n] = number_of_antennas/ (2*N_samples) / (np.abs(np.sqrt(Pt[u])* reflection_coefficient[n] * np.dot(a_BS[u], transmit_beam[u,] ) )**2) / ( np.linalg.norm(derivative2_a_BS)**2 + np.dot( derivative2_a_BS, a_BS[u]) + np.dot(a_BS[u], derivative2_a_BS))
                
                variance2[l,u,n] = number_of_antennas/ (2*N_samples) / (np.abs(np.sqrt(Pt[u]/np.size(transmit_beam_multi[u,]))* reflection_coefficient[n] * np.dot(a_BS[u], transmit_beam_multi[u,] ) )**2) / ( np.linalg.norm(derivative2_a_BS)**2 + np.dot( derivative2_a_BS, a_BS[u]) + np.dot(a_BS[u], derivative2_a_BS))


                temp[l,u,n] = 6/((number_of_antennas **2 -1)* np.cos(AoD)**2 * np.pi**2 *echo_bf[l,u,n])
                ratio[l,u,n] = (temp[l,u,n])/variance[l,u,n]

                if error[l,u,n] > beamwidth:
                    #beam_index = np.mod( np.add(current_beam[n+1, u], [-1,0,1]), 31 )
                    beam_index = np.mod( np.add(current_beam[n, u], [-1,0,1]), 31 )
                    current_beam[n+1,u] = beam_index[beam_index>=0]


        SNR_user[nPt, l, :] = 10*np.log10(Pt* np.mean(pathloss,axis=1) /noise_power)
        SNR_echo[nPt, l,:] = 10*np.log10((Pt* number_of_antennas**2 * reflection_coefficient**2 ) /(noise_power*number_of_antennas))   
        SNR_echo_numerical[nPt, l, :] = 20*np.log10(np.mean( np.linalg.norm(y_BS_noiseless, axis=0) / np.linalg.norm(Noise, axis=0) ))
    
    nmse[nPt,:] = np.mean(error/np.pi*180, axis = 0) # expected error in degree
    nmse_theo[nPt,:] = np.mean(np.sqrt(variance), axis = 0) # standard deviation in degree
    nmse_alternative[nPt,:] = np.sqrt(np.mean(variance2, axis = 0))


# %%
fig, ax = plt.subplots()
ax.semilogy(np.mean(SNR_echo[0,], axis=0), nmse[0,], '-ro')
ax.semilogy(np.mean(SNR_echo[0,], axis=0), nmse_theo[0,], '-b>')
ax.semilogy(np.mean(SNR_echo[0,], axis=0), nmse_alternative[0,], '-k*')
ax.set_xlabel('Echo SNR_dB')
ax.set_ylabel('NMSE')
ax.legend(['Simulation', 'CRLB'])
fig.show()

fig2, ax2 = plt.subplots()
ax2.semilogy(np.mean(echo_bf[:,0,:]  , axis = 0), '-ro')

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
# plot beam pattern
beams = [9]
w = (1/N_samples)*np.matmul(y_BS, waveform[u, :])#y_BS[:,0]#env.codebook[beams,:].sum(axis=0) # sum of 3 beams
w_padded = np.concatenate((w, np.zeros(N_fft - number_of_antennas)))
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


