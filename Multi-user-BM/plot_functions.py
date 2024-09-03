# %%
import matplotlib.pyplot as plt
import numpy as np
from itertools import groupby

# %%
def plot_beam_pattern(y_BS, number_of_antennas):
    N_fft = 1024
    beams = [9]
    w = y_BS[:,0]#env.codebook[beams,:].sum(axis=0) # sum of 3 beams
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

def plot_road_trraffic(road_matrix, N_UEs, user_positions):
    plt.imshow(road_matrix.T)
    plt.plot(50,50, 'ro')

    for i in range(N_UEs):
        ue_pos_x = user_positions[:,i,0]
        ue_pos_y = user_positions[:,i,1]
        ind = np.where(ue_pos_x == 0) and np.where(ue_pos_y == 0)
        ue_pos_x = np.delete(ue_pos_x, ind)
        ue_pos_y = np.delete(ue_pos_y, ind)
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
    plt.legend(['BS'])    
    plt.show()

def plot_nmse_vs_snr(SNR_echo, nmse, var_aod):

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
import numpy as np
import matplotlib.pyplot as plt
def plot_column_from_csv(file_path, column_name):
    data = np.genfromtxt(file_path, delimiter=',', names=True)
    column_data = data[column_name]
    window_size = 100
    moving_avg = np.convolve(column_data, np.ones(window_size)/window_size, mode='valid')
    plt.plot(moving_avg)
    plt.show()
    plt.plot(column_data)
    plt.xlabel('Index')
    plt.ylabel(column_name)
    plt.show()

# plot_column_from_csv('PPO_logs/BeamManagementEnv-v1/PPO_BeamManagementEnv-v1_log_11.csv', 'reward')

# %%
import numpy as np
import matplotlib.pyplot as plt
def plot_cdf_from_csv(file_path, column_name1, column_name2):
    data = np.genfromtxt(file_path, delimiter=',', names=True)
    tot = data[column_name1]
    tp = data[column_name2]
    th = tp/tot
    th_cdf = np.sort(th)
    yvals = np.arange(len(th))/float(len(th)-1)
    plt.plot(th_cdf, yvals, '-')
    plt.ylabel('CDF')
    plt.show()

#plot_cdf_from_csv('/Users/xiyu/Documents/AAU/CENTRIC_codes/PPO_logs/BeamManagementEnv-v1/TEST_PPO_BeamManagementEnv-v1_log_10.csv', 'total_packet_num', 'transmitted_packets')
# %%
