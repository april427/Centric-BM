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
        for j in range(m):
            dft_matrix[j][i-1] = np.exp(-1j * math.pi * j * (2 * i - 1 - m)/m)
        dft_matrix[:, i-1] = 1/np.sqrt(n) * dft_matrix[:, i-1]

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
        Consider Rayleigh block fading.

        Args:  
            M (int): The number of antenna elements in the ULA.
            a_BS (numpy.ndarray): The array response at the BS.
            d (float): The distance between the BS and the UE.
            blockage_coefficient (float): The blockage coefficient.  
             
    """
    mmWave_LOS_channel = np.sqrt(M) * wavelength / (4 *np.pi * d) * blockage_coefficient * 1/np.sqrt(2)* complex(np.random.normal(0, 1), np.random.normal(0, 1)) * np.conj(a_BS).T
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
