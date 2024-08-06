# %%
import numpy as np
import math
#import tensorflow as tf

from sionna.channel.tr38901 import Antenna, AntennaArray, CDL, UMi, UMa, RMa
from sionna.channel import gen_single_sector_topology as gen_topology
from sionna.channel import subcarrier_frequencies, cir_to_ofdm_channel, cir_to_time_channel
from sionna.channel import ApplyOFDMChannel, ApplyTimeChannel, OFDMChannel


# %%
def dft_codebook(n, m):
    """
    Creates a DFT codebook for a ULA BS.

    Args:
        n (int): The number of antenna elements in the ULA.
        m (int): The number of beams in the codebook.
        angles_deg (list): A list of angles (in degrees) to form the beams.
        d (float): The antenna spacing (in wavelengths).

    Returns:
        codebook (numpy.ndarray): A numpy array of shape (n, m) representing the codebook.
    """
    dft_matrix = np.ones((m, m)) * complex(1, 0)
    for i in range(1, m+1):
        dft_matrix[i-1,:] = 1/np.sqrt(n) * np.array([np.exp(-1j * np.pi * j *(2 * i - 1 - m)/m) for j in range(n)])
    return dft_matrix

def compute_AoD_at_BS(BS_pos, UE_pos):
    """
        Computes AoD at ULA BS given positions.

        Args:
            BS_pos (numpy.ndarray): Float array with the coordinates of the BS.
            UE_pos [numpy.ndarray]: Float array with the coordinates of the UE.
        Returns:
            AoD (numpy.ndarray): Angle of departure in radians.
        """
    # Compute angle of departure at BS
    phi_radi = np.arctan2( UE_pos[1] - BS_pos[1], - BS_pos[0] + UE_pos[0]) 
    AoD = np.array([phi_radi])
    if AoD < 0:
        AoD = np.add(AoD,np.pi)
    return AoD

def compute_ULA_response(M, AoD):
    # Compute array response
    a_BS = 1/np.sqrt(M) * np.array([np.exp(-1j * np.pi * i * np.cos(AoD)) for i in range(M)])
    return a_BS

def compute_mmWave_LoS_NLoS_channel(M, a_BS, d, wavelength, blockage_coefficient=1):
    """
        Compute mmWave LoS channel. Dimension is (1,M)
        Consider Rayleigh block fading. * 1/np.sqrt(2)* complex(np.random.normal(0, 1), np.random.normal(0, 1))

        Args:  
            M (int): The number of antenna elements in the ULA.
            a_BS (numpy.ndarray): The array response at the BS.
            d (float): The distance between the BS and the UE.
            blockage_coefficient (float): The blockage coefficient.  
             
    """
    mmWave_LOS_channel = np.sqrt(M) * wavelength / (4 *np.pi * d) * blockage_coefficient  * np.conj(a_BS).T
    return mmWave_LOS_channel

def compute_realistic_LoS_channel(sce, carrier_frequency, direction, num_user):
    """
        TODO - Compute realistic LoS channel.

        Args:  
            sce (sionna.Scenario): The scenario.
            cf : carrier frequency (Hz)
            dir (sionna.Direction): The direction.
            num_user (int): The number of users.

    """
    
def compute_mono_static_channel(M, AoD, reflection_coefficient):
    """
        Compute monostatic echo channel from one UE. LoS. (M, M)
    """
    a_BS = compute_ULA_response(M, AoD)
    mono_LoS_channel = M * reflection_coefficient * np.outer(a_BS, np.conj(a_BS))
    return mono_LoS_channel

"""
def compute_power_given_beam(Pt, mmWave_LOS_channel, selected_precoder, SNR_dB=30):
    # Compute signal power
    signal_power = np.abs(np.sqrt(Pt) * mmWave_LOS_channel @ selected_precoder) ** 2

    # Compute noise power
    SNR_linear = 10 ** (SNR_dB / 10)
    noise_power = signal_power / SNR_linear

    # Generate complex Gaussian noise
    noise = np.sqrt(noise_power / 2) * complex(np.random.normal(0, 1), np.random.normal(0, 1))

    # Add noise to the output
    output = np.sqrt(Pt) * mmWave_LOS_channel @ selected_precoder 

    return np.abs(output)[0]
"""

def compute_communication_rate(noiseless_output, signal, noise_power, N_UEs, N):
    """
        Compute the communication rate, consider inter user interference.
        Args:
            noiseless_output (numpy.ndarray): The noiseless output of the channel.
            noise_power (float): The noise power.
            N_UEs (int): The number of UEs.
            N (int): The number of samples within one TTI.
    """
    # Generate complex Gaussian noise
    noise = (np.random.normal(0, np.sqrt(noise_power/2), size = (N_UEs, N)) + 1j* np.random.normal(0, np.sqrt(noise_power/2), size = (N_UEs, N)))
    
    interference = noiseless_output - signal
    # Compute the communication rate
    rate = np.log2(1 + np.abs(signal)**2 / ( np.abs(interference)**2 + np.abs(noise) ** 2 ))

    # Compute SINR
    sinr = 10*np.log10(np.abs(signal)**2 / ( np.abs(interference)**2 + np.abs(noise) ** 2 ))

    rate = np.mean(rate, axis=1)
    return rate, np.mean(sinr,axis=1)

def estimate_AoD_from_echo(y_BS, waveform, N_UEs, N, N_fft, number_of_antennas):
    """
        Estimate AoDs from the echo.
        Args:
            y_BS (numpy.ndarray): The received signal at the BS.
            waveform (numpy.ndarray): The waveform.
            N_UEs (int): The number of UEs.
            N (int): The number of samples within one TTI.
            N_fft (int): The number of samples in the FFT.
            env (Environment): The environment.
    """
    AoD_est = np.zeros(N_UEs)
    for u in range(N_UEs):
        nitta = np.matmul(y_BS, waveform[u, :])
        # estimate AoD
        w_padded = np.concatenate((nitta, np.zeros(N_fft - number_of_antennas)))
        w_fft_dB = 10*np.log10(np.abs(np.fft.fftshift(np.fft.fft(w_padded)))**2) # magnitude of fft in dB
        # Map the FFT bins to angles in radians
        theta_bins = np.arccos(np.linspace(-1, 1, N_fft)) # in radians
        AoD_est[u] = np.pi - theta_bins[np.argmax(w_fft_dB)]
    return AoD_est

def dynamic_power_allocation(d, xi, Pt, ue_type, threshold, N_UEs):
    """
        Compute dynamic power allocation.
        Args:
            d (float): The distance between the BS and the UE.
            xi (float): The blockage coefficient.
            Pt_mW (float): The total transmit power in mW.
        
    """
    Pt_mW = np.sum(Pt)
    counter = -1
    sorted_user = np.argsort(d)
    weak_ue_ind = sorted_user[counter]
    if ue_type[weak_ue_ind] == 1: # CU
        Pt[weak_ue_ind] = threshold[1] / xi[weak_ue_ind]
    else: # SU
        Pt[weak_ue_ind] = threshold[0] / xi[weak_ue_ind]
    for u in range(N_UEs):
        Pt[u] = Pt[weak_ue_ind] * xi[u]/xi[weak_ue_ind]
    while np.sum(Pt) - Pt_mW > 0 :
        Pt[sorted_user[counter]] = 0
        if counter == -N_UEs:
            Pt[sorted_user[counter]] = Pt_mW
            break
        counter -= 1
    
    return Pt

# %%
# #Constants
# fc = 28e9  # carrier frequency (Hz)
# c = 3e8  # speed of light (m/s)
# lambda_ = c / fc  # wavelength (m)
# M = 32  # number of antennas at the BS
# Pt = 10 #mW
# # BS and UE positions
# BS_pos = np.array([25, 25, 60])  # BS at origin with height 10 meters
# UE_pos = np.array([68, 93, 1.7])  # UE at (50, 50, 1.5) meters
# d = np.linalg.norm(BS_pos-UE_pos)

# AoD = compute_AoD_at_BS(BS_pos, UE_pos)
# a_BS = compute_ULA_response(M, AoD)
# mmWave_LOS_channel = compute_mmWave_LoS_NLoS_channel(M, a_BS, d)
# mono_LoS_channel = compute_mono_static_channel(M, AoD, 0.5)
# codebook = dft_codebook(M, M)

# y = np.sqrt(Pt)*mmWave_LOS_channel @ codebook
# print(np.argmax(np.abs(y), axis=1))
# print(compute_power_given_beam(Pt, mmWave_LOS_channel, codebook[:, np.argmax(np.abs(y), axis=1)]))
# print("--------------------------------------------------------------------------------------------")
# for i in range(len(codebook)):
#     print("Beam: " + str(compute_power_given_beam(Pt, mmWave_LOS_channel, codebook[:, i])))
#     print("Ratio: " + str(compute_power_given_beam(Pt, mmWave_LOS_channel, codebook[:, i])/compute_power_given_beam(Pt, mmWave_LOS_channel, codebook[:, np.argmax(np.abs(y), axis=1)]))) 

# %%
