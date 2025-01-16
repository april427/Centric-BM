# %%
import os
if os.getenv("CUDA_VISIBLE_DEVICES") is None:
    gpu_num = 0 # Use "" to use the CPU
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_num}"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Import Sionna
try:
    import sionna
except ImportError as e:
    # Install Sionna if package is not already installed
    import os
    os.system("pip install sionna")
    import sionna

# Configure the notebook to use only a single GPU and allocate only as much memory as needed
# For more details, see https://www.tensorflow.org/guide/gpu
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError as e:
        print(e)
# Avoid warnings from TensorFlow
tf.get_logger().setLevel('ERROR')

# Set random seed for reproducibility
sionna.config.seed = 42

########################################################################################
# %%
import matplotlib.pyplot as plt
import numpy as np
import time
import pickle

from sionna.mimo import StreamManagement

from sionna.ofdm import ResourceGrid, ResourceGridMapper, LSChannelEstimator, LMMSEEqualizer
from sionna.ofdm import OFDMModulator, OFDMDemodulator, ZFPrecoder, RemoveNulledSubcarriers

from sionna.channel.tr38901 import Antenna, AntennaArray, CDL, UMi, UMa, RMa
from sionna.channel import gen_single_sector_topology as gen_topology
from sionna.channel import subcarrier_frequencies, cir_to_ofdm_channel, cir_to_time_channel
from sionna.channel import ApplyOFDMChannel, ApplyTimeChannel, OFDMChannel

from sionna.fec.ldpc.encoding import LDPC5GEncoder
from sionna.fec.ldpc.decoding import LDPC5GDecoder

from sionna.mapping import Mapper, Demapper

from sionna.utils import BinarySource, ebnodb2no, sim_ber, QAMSource
from sionna.utils.metrics import compute_ber

# %%
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib.colorbar
import matplotlib.colors
import matplotlib.cm


########################################################################################
# %%
import gymnasium as gym
import sys
sys.path.append('/Users/xiyu/Documents/AAU/CENTRIC_codes/PPO_beam_management/Extension_BM/')
from exroad import *
from beam_management_env_letter import BeamManagementEnv
from channel_functions import *
from util_functions import *

env = gym.make('BeamManagementEnv-v1', reward_inx = 0, if_training=False)

########################################################################################
# %%
scenario = "rma"
carrier_frequency = 28e9
wavelength = 3e8 / carrier_frequency
direction = "downlink"
N_UTs = 2 # env.unwrapped.N_UTs 
batch_size = 1 # number of steps
number_of_antennas = 16
fft_size = 128
subcarrier_spacing = 30e3
TTI = 1 # channel coherence time


# Define the UT antenna array
ut_array = Antenna(polarization="single",
                   polarization_type="V",
                   antenna_pattern="omni",
                   carrier_frequency=carrier_frequency)

# Define the BS antenna array
bs_array = AntennaArray(num_rows=1,
                        num_cols=number_of_antennas,
                        polarization="single",
                        polarization_type="V",
                        antenna_pattern="38.901",    # something to be changed
                        carrier_frequency=carrier_frequency)

# Create channel model
channel_model = RMa(carrier_frequency=carrier_frequency,
                    ut_array=ut_array,
                    bs_array=bs_array,
                    direction=direction,
                    enable_pathloss=True,
                    enable_shadow_fading=False)

# Generate the topology
topo = gen_topology(batch_size, N_UTs, scenario)

bs_loc = tf.constant([50, 50, 0], dtype = tf.float32)
bs_loc = tf.expand_dims(tf.expand_dims(bs_loc,axis=0),axis=0)
ut_loc = np.zeros((batch_size, N_UTs, 3))

ut_loc[0,:,:], config, if_turned = user_pos_direction_init(env.unwrapped.road_matrix, env.unwrapped.cross_pos, N_UTs, env.unwrapped.single_lane_width) 
for s in range(1, batch_size):
    for i in range(N_UTs):
        ut_loc[s, i,:], config[i,:], if_turned[i] = update_positions(ut_loc[s-1,i,:], env.unwrapped.cross_pos, sample_velocity(), 0.1, env.unwrapped.single_lane_width, config[i,:], if_turned[i]) 

ut_velocities = np.zeros((batch_size, N_UTs, 3))
ut_velocities[:,:,0:2] = np.random.normal(loc = 20, scale=2, size = (batch_size, N_UTs, 2))
ut_velocities[:,0,0] = 0
ut_velocities[:,1,1] = 0
ut_velocities = tf.convert_to_tensor(ut_velocities, dtype = tf.float32)


########################################################################################
# %%
# Set the topology
channel_model.set_topology(ut_loc=tf.convert_to_tensor(ut_loc,  dtype = tf.float32), bs_loc=tf.convert_to_tensor(bs_loc,  dtype = tf.float32), ut_orientations=topo[2], bs_orientations=topo[3],  ut_velocities=ut_velocities, in_state=topo[5])

# Visualize the topology
channel_model.show_topology()

########################################################################################
# %%
num_streams_per_tx = N_UTs # Number of transmitted streams is equal to the number of UTs
rx_tx_association = np.ones([N_UTs, 1])

sm = StreamManagement(rx_tx_association, num_streams_per_tx,)
#  resource grid
rg = ResourceGrid(num_ofdm_symbols=14, 
                  fft_size=fft_size,
                  subcarrier_spacing=subcarrier_spacing,
                  num_tx = 1, # one BS
                  num_streams_per_tx=num_streams_per_tx, 
                  cyclic_prefix_length=20,
                  pilot_pattern="kronecker",
                  pilot_ofdm_symbol_indices=[2,11]
                 )
rg.show()

########################################################################################
# %%
num_bits_per_symbol = 2 # QPSK modulation
coderate = 0.5 # The code rate
n = int(rg.num_data_symbols*num_bits_per_symbol) # Number of coded bits
k = int(n*coderate) # Number of information bits

# The binary source will create batches of information bits
binary_source = BinarySource()
qam_source = QAMSource(num_bits_per_symbol)

# The encoder maps information bits to coded bits
encoder = LDPC5GEncoder(k, n)

# The mapper maps blocks of information bits to constellation symbols
mapper = Mapper("qam", num_bits_per_symbol)

# The resource grid mapper maps symbols onto an OFDM resource grid
rg_mapper = ResourceGridMapper(rg)
# This function removes nulled subcarriers from any tensor having the shape of a resource grid
remove_nulled_scs = RemoveNulledSubcarriers(rg)

ls_est = LSChannelEstimator(rg, interpolation_type="nn")

# The LMMSE equalizer will provide soft symbols together with noise variance estimates
lmmse_equ = LMMSEEqualizer(rg, sm)

# The demapper produces LLR for all coded bits
demapper = Demapper("app", "qam", num_bits_per_symbol)

# The decoder provides hard-decisions on the information bits
decoder = LDPC5GDecoder(encoder, hard_out=True)

# OFDM CHannel
ofdm_channel = OFDMChannel(channel_model, rg, add_awgn=False, normalize_channel=False, return_channel=True)

frequencies = subcarrier_frequencies(rg.fft_size, rg.subcarrier_spacing)

########################################################################################
# %%
ebno_db = 10
no = ebnodb2no(ebno_db, num_bits_per_symbol, coderate, rg)
b = binary_source([batch_size, 1, rg.num_streams_per_tx, encoder.k])
c = encoder(b)
x = mapper(c)
x_rg = rg_mapper(x) # batch_size, num_tx, num_streams_per_tx, num_ofdm_symbols, fft_size

########################################################################################
# %%
#        a : [batch size, num_rx, num_rx_ant, num_tx, num_tx_ant, #
#              num_paths, num_time_samples], tf.complex
            # Path coefficients
#       tau : [batch size, num_rx, num_tx, num_paths], tf.float
            # Path delays [s]
# a, tau = channel_model(num_time_samples=rg.num_ofdm_symbols, sampling_frequency=1/rg.ofdm_symbol_duration) #batch_size, num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths, num_ofdm_symbols
# h_freq = cir_to_ofdm_channel(frequencies, a, tau, normalize=True) # batch_size, num_rx, num_rx_ant, num_tx, num_tx_ant, num_ofdm_symbols, fft_size

# Calculate distances between BS and each UT for each batch
Rd = np.zeros((batch_size, N_UTs)) # distance between BS and UTs
aod = np.zeros((batch_size, N_UTs)) # AoD at the BS
reflection_coefficient = np.zeros((batch_size, N_UTs)) # BS-UT-BS path loss 
a_c = np.zeros((batch_size, N_UTs)) # BS-UT path loss
for s in range(batch_size):
    for i in range(N_UTs):
        Rd[s, i] = np.linalg.norm(ut_loc[s, i, :] - bs_loc.numpy()[0, 0, :])
        reflection_coefficient[s, i] = wavelength * 5 / (4* np.pi)**(1.5) / (Rd[s, i]**2) # path loss, RCS = 5 
        a_c[s,i] = (wavelength / (4 * np.pi * Rd[s,i]))**2
        aod[s,i] = compute_AoD_at_BS(bs_loc.numpy()[0, 0, :], ut_loc[s, i, :])

tau_s = Rd*2/3e8 # two-way distance for sensing, reflected signal at the BS
tau_c = tau_s/2

N = 1 # consider one path
H_UT = np.zeros([batch_size, N_UTs,1,1, number_of_antennas, rg.num_ofdm_symbols, fft_size], dtype=np.complex64) #[ batch size, num_users, num_subcarriers, num_BS_antennas ]
for s in range(batch_size):
    for i in range(N_UTs):
        H_f,_,_ = freqency_communication_channel(rg.num_ofdm_symbols, fft_size, N, number_of_antennas, a_c[s,i],tau_c[s,i],aod[s,i])
        H_UT[s, i,0,0, ] = H_f
H_UT = tf.convert_to_tensor(H_UT, dtype = tf.complex64)
        
H_BS = tf.zeros([batch_size, 1 , number_of_antennas, 1, number_of_antennas, rg.num_ofdm_symbols, fft_size])
H_BS, A_t = freqency_sensing_channel(batch_size, rg.num_ofdm_symbols, fft_size , number_of_antennas, N_UTs, subcarrier_spacing, rg.ofdm_symbol_duration, reflection_coefficient, ut_velocities.numpy(), wavelength, tau_s, aod)
H_BS = tf.convert_to_tensor(H_BS, dtype = tf.complex64)    

########################################################################################
# %%
codebook = dft_codebook(number_of_antennas, number_of_antennas)

########################################################################################
# %%
tx_sig_shape = list(x_rg.shape) #[batch_size, num_tx, num_tx_antennas, num_ofdm_symbols, fft_size]
tx_sig_shape[2] = number_of_antennas
tx_sig_fd = tf.zeros(tx_sig_shape, dtype=x_rg.dtype)

# the first beam is set with known AoDs
current_beam = np.empty([TTI, N_UTs], dtype = object) 

for u in range(N_UTs):
    current_beam[0, u] = np.argmax(np.abs(np.transpose(np.conj(A_t[:,0,u])) @ codebook))

channel_freq = ApplyOFDMChannel(add_awgn=False)

num_rangebin = int(fft_size)*3
num_dopplerbin = int(rg.num_ofdm_symbols)*3
num_anglebin = number_of_antennas

n1 = np.array(range(num_dopplerbin)) 
n2 = np.array(range(num_rangebin))
n3 = np.array(range(num_anglebin))

# %%
z_BS = np.zeros([batch_size, 1, N_UTs, num_dopplerbin, num_rangebin], dtype = np.complex64)
for nTTI in range(TTI):
    beam_index = current_beam[nTTI, :].astype('int')
    tx_sig_fd = tf.einsum('kj,nijom -> nikom', codebook[:,beam_index], x_rg)
    
    # Apply the OFDM channel
    y_UT = channel_freq([tx_sig_fd, H_UT])

    h_hat, err_var = ls_est ([y_UT, no])
    x_hat, no_eff = lmmse_equ([y_UT, h_hat, err_var, no])
    llr = demapper([x_hat, no_eff])
    b_hat = decoder(llr)
    ber = np.zeros(N_UTs)
    for uu in range(N_UTs): ber[uu] = compute_ber(b[:,:,uu,:], b_hat[:,:,uu,:])
    print("BER: {}".format(ber))

    y_BS = channel_freq([tx_sig_fd, H_BS])

    y_BS2 = np.einsum('uj, nijom->niuom ' , np.transpose(np.conj((codebook[:,beam_index]))), y_BS)
    y_BS2 = tf.multiply(y_BS2, tf.math.conj(x_rg[nTTI,]))

z_BS = np.fft.fft2(y_BS2, [num_dopplerbin, num_rangebin])
    #Y_BS = np.fft.fft(y_BS_r_d, num_anglebin, axis = 2)
  
#%%
# fig = plt.figure()
# ax = fig.add_subplot(projection = '3d')
# ax.set_xlabel('Angle bin')
# ax.set_zlabel('Range bin')
# ax.set_ylabel('Doppler bin')
# plotMatrix(ax, n3, n1,n2, np.abs(np.squeeze(Y_BS[0,0,:,:,:])), cmap="jet")


# %%
indices = np.zeros([N_UTs, 2])
R_estimate = np.zeros(N_UTs)
v_estiamte = np.zeros(N_UTs)
fig1 = plt.figure()
for u in range(N_UTs):
    ax1 = fig1.add_subplot(projection = '3d')
    _x = np.arange(num_rangebin)
    _y = np.arange(num_dopplerbin)
    _xx, _yy = np.meshgrid(_x, _y)
    x, y = _xx.ravel(), _yy.ravel()
    top = np.abs(z_BS[0,0,u,:,:]).ravel()
    colors = plt.cm.jet(top.flatten()/float(top.max()))
    ax1.set_xlabel('Range bin')
    ax1.set_ylabel('Doppler bin')
    ax1.bar3d(x, y, np.zeros_like(top), 1, 1, top, shade=True, color=colors)
    
    indices[u,:] = np.concatenate(np.unravel_index(np.argsort(top, axis=None)[-1:], z_BS[0,0,u,:,:].shape)).astype('int')

# fft shift
    indices[u,0] = indices[u,0] if indices[u,0]<np.ceil(num_dopplerbin/2) else indices[u,0]-num_dopplerbin
    indices[u,1] = indices[u,1] if indices[u,1]<np.ceil(num_rangebin/2) else indices[u,1]-num_rangebin

    R_estimate[u] = - 3e8*indices[u,1]/(2*subcarrier_spacing*num_rangebin)
    v_estiamte[u] = 3e8*indices[u,0]/(2*rg.ofdm_symbol_duration*carrier_frequency*num_dopplerbin) 
    print(''.join(['R estimate: ', str(R_estimate[u]), ' v estimate: ', str(v_estiamte[u]), '\nR_real: ', str(Rd[0,u]), ' v_real: ', str(ut_velocities[0,u,0])]))
# %% Plot user locations
for i in range(N_UTs):
    ue_pos_x = ut_loc[:,i,0]
    ue_pos_y = ut_loc[:,i,1]
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
