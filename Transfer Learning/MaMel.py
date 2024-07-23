import os
gpu_num = 0 # Use "" to use the CPU
os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_num}"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'




#######
##########'


# Colab does currently not support the latest version of ipython.
# Thus, the preview does not work in Colab. However, whenever possible we
# strongly recommend to use the scene preview mode.
try: # detect if the notebook runs in Colab
    import google.colab
    colab_compat = True # deactivate preview
except:
    colab_compat = False
resolution = [480,320] # increase for higher quality of renderings

# Allows to exit cell execution in Jupyter
class ExitCell(Exception):
    def _render_traceback_(self):
        pass

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

tf.random.set_seed(1) # Set global random seed for reproducibility




#matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
import time

# Import Sionna RT components
from sionna.rt import load_scene, Transmitter, Receiver, PlanarArray, Camera

# For link-level simulations
from sionna.channel import cir_to_ofdm_channel, subcarrier_frequencies, OFDMChannel, ApplyOFDMChannel, CIRDataset, GenerateOFDMChannel
from sionna.nr import PUSCHConfig, PUSCHTransmitter, PUSCHReceiver
from sionna.utils import compute_ber, ebnodb2no, PlotBER
from sionna.ofdm import KBestDetector, LinearDetector
from sionna.mimo import StreamManagement

import pickle
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model
from tensorflow.keras.layers import Layer, Conv2D, LayerNormalization
from tensorflow.nn import relu

from sionna.channel.tr38901 import Antenna, AntennaArray, CDL
from sionna.channel import OFDMChannel
from sionna.mimo import StreamManagement
from sionna.ofdm import ResourceGrid, ResourceGridMapper, LSChannelEstimator, LMMSEEqualizer, RemoveNulledSubcarriers, ResourceGridDemapper
from sionna.utils import BinarySource, ebnodb2no, insert_dims, flatten_last_dims, log10, expand_to_rank
from sionna.fec.ldpc.encoding import LDPC5GEncoder
from sionna.fec.ldpc.decoding import LDPC5GDecoder
from sionna.mapping import Mapper, Demapper
from sionna.utils.metrics import compute_ber
from sionna.utils import sim_ber

##########################################################


num_tx = 1#4 # Number of users
num_rx = 1 # Only one receiver considered
num_tx_ant = 1#4 # Each user has 4 antennas
num_rx_ant = 2 # The receiver is equipped with 2 antennas
num_time_steps = 14 # Total number of ofdm symbols per slot
#model_weights_path = "munich_10" # Location to save the neural receiver weights once training is done
##########################################################
####  get the meta training file
#filename = "/home/es.aau.dk/mw88bt/centric_tf/myDRx/rt_cir_simo_10k_simo(2Tx_ant).h5"
filename = "C:\\Users\\MW88BT\\OneDrive - Aalborg Universitet\\my_vscode_scripts\\rt_cir_simo_10k_simo(2Tx_ant).h5"
import h5py
with h5py.File(filename, "r") as f:
    a = f['a'][()]
    tau = f['tau'][()]
#    h_freq= f['h_freq'][()]

print("a_from_file", a.shape)
print("tau_from_file", tau.shape)

###########################################################################

##################################################
#print("hfreq_from_file", h_freq.shape)
validation_split = .5
shuffle_dataset = True
random_seed= 42

def dataset_split(x,y, split_ratio):
    dataset_size = len(x)
    indices = list(range(dataset_size))
    split = int(np.floor(split_ratio* dataset_size))
    if shuffle_dataset :
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    train_indices, val_indices = indices[split:], indices[:split]
    train_a= x[train_indices]
    train_tau= y[train_indices]
    test_a= x[val_indices]
    test_tau= y[val_indices]
    return train_a, train_tau, test_a, test_tau
###############################################################################################
##################################################################################################
############### Get the meta finetuning and test file 
#filename_paris = "/home/es.aau.dk/mw88bt/centric_tf/myDRx/rt_paris_test.h5"
filename_paris = "C:\\Users\\MW88BT\\OneDrive - Aalborg Universitet\\my_vscode_scripts\\rt_paris_test.h5"
import h5py
with h5py.File(filename_paris, "r") as f:
    a_paris = f['a'][()]
    tau_paris = f['tau'][()]
#    h_freq= f['h_freq'][()]
a_paris = a_paris[:,:,:,:,:1,:,:]
print("a_from_file_paris", a_paris.shape)
print("tau_from_file_paris", tau_paris.shape)

a_train_paris, tau_train_paris, a_test_paris, tau_test_paris= dataset_split(a_paris, tau_paris,0.2)

######################################################################################################
#################################################################################################
def dtset_meta_split(zz,tt,sup_size):
    point_split = .5
    meta_size = len(zz)
    indx = list(range(meta_size))
    meta_split = int(np.floor(point_split * meta_size))
    np.random.shuffle(indx)
    supp_indices = indx[meta_split:meta_split+sup_size]

    supp_a= zz[supp_indices]
    supp_tau= tt[supp_indices]
    
    return supp_a, supp_tau

################################################################################################
###############################################################################################
## dataset 1
#a_d1 = a[:1000,:,:,:,:,:,:]
#tau_d1 = tau[:1000,:,:,:]

a_d1, tau_d1 = dtset_meta_split(a,tau,1000)
supp_a1, supp_tau1, querry_a1, querr_tau1= dataset_split(a_d1, tau_d1,0.5)
###############################################################################################
## dataset 1
#a_d2 = a[1000:,:,:,:,:,:,:]
#tau_d2 = tau[1000:,:,:,:]
a_d2, tau_d2 = dtset_meta_split(a,tau,1000)

supp_a2, supp_tau2, querry_a2, querr_tau2= dataset_split(a_d2, tau_d2,0.5)
#############################################################################
## dataset 3
#a_d3 = a[3000:4000,:,:,:,:,:,:]
#tau_d3 = tau[3000:4000,:,:,:]
a_d3, tau_d3 = dtset_meta_split(a,tau,1000)
supp_a3, supp_tau3, querry_a3, querr_tau3= dataset_split(a_d3, tau_d3,0.5)

################################################################################################
subcarrier_spacing = 60e3
batch_size = 32
batch_size = batch_size
cyclic_prefix_length = 0# cyclic_prefix_length
pilot_ofdm_symbol_indices = [2, 11]# pilot_ofdm_symbol_indices
num_rx = 1


carrier_frequency=60e9
carrier_frequency_2=28e9
##########

####################
##################################################################################################
######################################
######################################
class CIRGenerator:
    """Creates a generator from a given dataset of channel impulse responses.

    The generator samples ``num_tx`` different transmitters from the given path
    coefficients `a` and path delays `tau` and stacks the CIRs into a single tensor.

    Note that the generator internally samples ``num_tx`` random transmitters
    from the dataset. For this, the inputs ``a`` and ``tau`` must be given for
    a single transmitter (i.e., ``num_tx`` =1) which will then be stacked
    internally.

    Parameters
    ----------
    a : [batch size, num_rx, num_rx_ant, 1, num_tx_ant, num_paths, num_time_steps], complex
        Path coefficients per transmitter.

    tau : [batch size, num_rx, 1, num_paths], float
        Path delays [s] per transmitter.

    num_tx : int
        Number of transmitters

    Output
    -------
    a : [batch size, num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths, num_time_steps], tf.complex
        Path coefficients

    tau : [batch size, num_rx, num_tx, num_paths], tf.float
        Path delays [s]
    """

    def __init__(self,
                 a,
                 tau,
                 num_tx):

        # Copy to tensorflow
        self._a = tf.constant(a, tf.complex64)
        self._tau = tf.constant(tau, tf.float32)
        self._dataset_size = self._a.shape[0]

        self._num_tx = num_tx

    def __call__(self):

        # Generator implements an infinite loop that yields new random samples
        while True:
            # Sample 4 random users and stack them together
            idx,_,_ = tf.random.uniform_candidate_sampler(
                            tf.expand_dims(tf.range(self._dataset_size, dtype=tf.int64), axis=0),
                            num_true=self._dataset_size,
                            num_sampled=self._num_tx,
                            unique=True,
                            range_max=self._dataset_size)

            a = tf.gather(self._a, idx)
            tau = tf.gather(self._tau, idx)

            # Transpose to remove batch dimension
            a = tf.transpose(a, (3,1,2,0,4,5,6))
            tau = tf.transpose(tau, (2,1,0,3))
            # And remove batch-dimension
            a = tf.squeeze(a, axis=0)
            tau = tf.squeeze(tau, axis=0)
            yield a, tau
###########################################
            ###################################

batch_size = 32 # Must be the same for the BER simulations as CIRDataset returns fixed batch_size
num_paths = None 


#################
######################
# Channel configuration
carrier_frequency = 3.5e9 # Hz
delay_spread = 100e-9 # s
speed = 10.0 # Speed for evaluation and training [m/s]
# SNR range for evaluation and training [dB]
ebno_db_min = -20.0
ebno_db_max = 20.0

############################################
## OFDM waveform configuration
subcarrier_spacing = 15e3 # Hz
fft_size = 32 # Number of subcarriers forming the resource grid, including the null-subcarrier and the guard bands
num_ofdm_symbols = 14 # Number of OFDM symbols forming the resource grid
dc_null = True # Null the DC subcarrier
num_guard_carriers = [5, 6] # Number of guard carriers on each side
pilot_pattern = "kronecker" # Pilot pattern
pilot_ofdm_symbol_indices = [2, 11] # Index of OFDM symbols carrying pilots
cyclic_prefix_length = 0 # Simulation in frequency domain. This is useless

############################################
## Modulation and coding configuration
num_bits_per_symbol = 2 # QPSK
coderate = 0.5 # Coderate for LDPC code
direction = 'uplink'
############################################
## Neural receiver configuration
num_conv_channels = 32 # Number of convolutional channels for the convolutional layers forming the neural receiver
#################

stream_manager = StreamManagement(np.array([[1]]), # Receiver-transmitter association matrix
                                  1)               # One stream per transmitter
#########################


##########################
############################
resource_grid = ResourceGrid(num_ofdm_symbols = num_ofdm_symbols,
                             fft_size = fft_size,
                             subcarrier_spacing = subcarrier_spacing,
                             num_tx = 1,
                             num_streams_per_tx = 1,
                             cyclic_prefix_length = cyclic_prefix_length,
                             dc_null = dc_null,
                             pilot_pattern = pilot_pattern,
                             pilot_ofdm_symbol_indices = pilot_ofdm_symbol_indices,
                             num_guard_carriers = num_guard_carriers) 
##########################
# Codeword length. It is calculated from the total number of databits carried by the resource grid, and the number of bits transmitted per resource element
n = int(resource_grid.num_data_symbols*num_bits_per_symbol)
# Number of information bits per codeword
k = int(n*coderate) 
##########################
## Transmitter
binary_source = BinarySource()
mapper = Mapper("qam", num_bits_per_symbol)
rg_mapper = ResourceGridMapper(resource_grid)


## Receiver

rg_demapper = ResourceGridDemapper(resource_grid, stream_manager) # Used to extract data-carrying resource elements

####
batch_size = 32
ebno_db = tf.fill([batch_size], 5.0)
no = ebnodb2no(ebno_db, num_bits_per_symbol, coderate)


## Transmitter
# Generate codewords
c = binary_source([batch_size, 1, 1, n])
print("c shape: ", c.shape)

# Map bits to QAM symbols
x = mapper(c)
print("x shape: ", x.shape)
# Map the QAM symbols to a resource grid
x_rg = rg_mapper(x)
print("x_rg shape: ", x_rg.shape)

######################################
## Channel

# A batch of new channel realizations is sampled and applied at every inference
no_ = expand_to_rank(no, tf.rank(x_rg))




class ResidualBlock(Layer):
    r"""
    This Keras layer implements a convolutional residual block made of two convolutional layers with ReLU activation, layer normalization, and a skip connection.
    The number of convolutional channels of the input must match the number of kernel of the convolutional layers ``num_conv_channel`` for the skip connection to work.
    
    Input
    ------
    : [batch size, num time samples, num subcarriers, num_conv_channel], tf.float
        Input of the layer
    
    Output
    -------
    : [batch size, num time samples, num subcarriers, num_conv_channel], tf.float
        Output of the layer
    """
                        
    def build(self, input_shape):
        
        # Layer normalization is done over the last three dimensions: time, frequency, conv 'channels'
        self._layer_norm_1 = LayerNormalization(axis=(-1, -2, -3))
        self._conv_1 = Conv2D(filters=num_conv_channels,
                              kernel_size=[3,3],
                              padding='same',
                              activation=None)
        # Layer normalization is done over the last three dimensions: time, frequency, conv 'channels'
        self._layer_norm_2 = LayerNormalization(axis=(-1, -2, -3))
        self._conv_2 = Conv2D(filters=num_conv_channels,
                              kernel_size=[3,3],
                              padding='same',
                              activation=None)
    
    def call(self, inputs):
        z = self._layer_norm_1(inputs)
        z = relu(z)
        z = self._conv_1(z)
        z = self._layer_norm_2(z)
        z = relu(z)
        z = self._conv_2(z) # [batch size, num time samples, num subcarriers, num_channels]
        # Skip connection
        z = z + inputs
        
        return z

class NeuralReceiver(Model):
    r"""
    Keras layer implementing a residual convolutional neural receiver.
    
    This neural receiver is fed with the post-DFT received samples, forming a resource grid of size num_of_symbols x fft_size, and computes LLRs on the transmitted coded bits.
    These LLRs can then be fed to an outer decoder to reconstruct the information bits.
    
    As the neural receiver is fed with the entire resource grid, including the guard bands and pilots, it also computes LLRs for these resource elements.
    They must be discarded to only keep the LLRs corresponding to the data-carrying resource elements.
    
    Input
    ------
    y : [batch size, num rx antenna, num ofdm symbols, num subcarriers], tf.complex
        Received post-DFT samples.
    
    no : [batch size], tf.float32
        Noise variance. At training, a different noise variance value is sampled for each batch example.
    
    Output
    -------
    : [batch size, num ofdm symbols, num subcarriers, num_bits_per_symbol]
        LLRs on the transmitted bits.
        LLRs computed for resource elements not carrying data (pilots, guard bands...) must be discarded.
    """
    
    def build(self, input_shape):
        
        # Input convolution
        self._input_conv = Conv2D(filters=num_conv_channels,
                                  kernel_size=[3,3],
                                  padding='same',
                                  activation=None)
        # Residual blocks
        self._res_block_1 = ResidualBlock()
        self._res_block_2 = ResidualBlock()
        self._res_block_3 = ResidualBlock()
        self._res_block_4 = ResidualBlock()
        # Output conv
        self._output_conv = Conv2D(filters=num_bits_per_symbol,
                                   kernel_size=[3,3],
                                   padding='same',
                                   activation=None)
        
    def call(self, inputs):
        y, no = inputs
        
        # Feeding the noise power in log10 scale helps with the performance
        no = log10(no)
        
        # Stacking the real and imaginary components of the different antennas along the 'channel' dimension
        y = tf.transpose(y, [0, 2, 3, 1]) # Putting antenna dimension last
        no = insert_dims(no, 3, 1)
        no = tf.tile(no, [1, y.shape[1], y.shape[2], 1])
        # z : [batch size, num ofdm symbols, num subcarriers, 2*num rx antenna + 1]
        z = tf.concat([tf.math.real(y),
                       tf.math.imag(y),
                       no], axis=-1)
        # Input conv
        z = self._input_conv(z)
        # Residual blocks
        z = self._res_block_1(z)
        z = self._res_block_2(z)
        z = self._res_block_3(z)
        z = self._res_block_4(z)
        # Output conv
        z = self._output_conv(z)
        
        return z
    
#####################################################
    ##########################################

## Receiver
#neural_receiver = NeuralReceiver()
#rg_demapper = ResourceGridDemapper(resource_grid, stream_manager) # Used to extract data-carrying resource elements
#y = tf.squeeze(y, axis=1)
############################################
#######################################
#llr = neural_receiver([y, no])
#print("llr shape: ", llr.shape)
# Reshape the input to fit what the resource grid demapper is expected
#llr = insert_dims(llr, 2, 1)
# Extract data-carrying resource elements. The other LLRs are discarded
#llr = rg_demapper(llr)
#llr = tf.reshape(llr, [batch_size, 1, 1, n])
#print("Post RG-demapper LLRs: ", llr.shape)
###################################################################
###################################################################
############################################################################################


cir_generator_sup_1= CIRGenerator(supp_a1,
                            supp_tau1,
                            num_tx)

cir_generator_sup_2= CIRGenerator(supp_a2,
                            supp_tau2,
                            num_tx)

cir_generator_sup_3= CIRGenerator(supp_a3,
                            supp_tau3,
                            num_tx)
# Initialises a channel model that can be directly used by OFDMChannel layer
channel_model_1 = CIRDataset(cir_generator_sup_1,
                        batch_size,
                        num_rx,
                        num_rx_ant,
                        num_tx,
                        num_tx_ant,
                        num_paths,
                        num_time_steps)

channel_model_2 = CIRDataset(cir_generator_sup_2,
                        batch_size,
                        num_rx,
                        num_rx_ant,
                        num_tx,
                        num_tx_ant,
                        num_paths,
                        num_time_steps)

channel_model_3 = CIRDataset(cir_generator_sup_3,
                        batch_size,
                        num_rx,
                        num_rx_ant,
                        num_tx,
                        num_tx_ant,
                        num_paths,
                        num_time_steps)

chaneel_model_supp = [channel_model_1 , channel_model_2 , channel_model_3 ]

##



##############################################################

cir_generator_querry_1= CIRGenerator(querry_a1,
                            querr_tau1,
                            num_tx)

cir_generator_querry_2= CIRGenerator(querry_a2,
                            querr_tau2,
                            num_tx)

cir_generator_querry_3= CIRGenerator(querry_a3,
                            querr_tau3,
                            num_tx)
# Initialises a channel model that can be directly used by OFDMChannel layer
channel_model_q1 = CIRDataset(cir_generator_querry_1,
                        batch_size,
                        num_rx,
                        num_rx_ant,
                        num_tx,
                        num_tx_ant,
                        num_paths,
                        num_time_steps)

channel_model_q2 = CIRDataset(cir_generator_querry_2,
                        batch_size,
                        num_rx,
                        num_rx_ant,
                        num_tx,
                        num_tx_ant,
                        num_paths,
                        num_time_steps)

channel_model_q3 = CIRDataset(cir_generator_querry_3,
                        batch_size,
                        num_rx,
                        num_rx_ant,
                        num_tx,
                        num_tx_ant,
                        num_paths,
                        num_time_steps)

channel_model_supp = [channel_model_1 , channel_model_2 , channel_model_3 ]
channel_model_querry = [channel_model_q1 , channel_model_q2 , channel_model_q3 ]
##############################################################################################
cir_generator = CIRGenerator(a_train_paris,
                            tau_train_paris,
                            num_tx)
# Initialises a channel model that can be directly used by OFDMChannel layer
channel_model = CIRDataset(cir_generator,
                        batch_size,
                        num_rx,
                        num_rx_ant,
                        num_tx,
                        num_tx_ant,
                        num_paths,
                        num_time_steps)
##
# Init CIR generator
cir_generator_test = CIRGenerator(a_test_paris,
                            tau_test_paris,
                            num_tx)
# Initialises a channel model that can be directly used by OFDMChannel layer
channel_model_test = CIRDataset(cir_generator_test,
                        batch_size,
                        num_rx,
                        num_rx_ant,
                        num_tx,
                        num_tx_ant,
                        num_paths,
                        num_time_steps)



#############################################################################################
###############################################################################################

sionna.config.xla_compat=False # not supported in CIRDataset
class E2ESystem(Model):
    r"""
    Keras model that implements the end-to-end systems.
    
    As the three considered end-to-end systems (perfect CSI baseline, LS estimation baseline, and neural receiver) share most of
    the link components (transmitter, channel model, outer code...), they are implemented using the same Keras model.

    When instantiating the Keras model, the parameter ``system`` is used to specify the system to setup,
    and the parameter ``training`` is used to specified if the system is instantiated to be trained or to be evaluated.
    The ``training`` parameter is only relevant when the neural 
    
    At each call of this model:
    * A batch of codewords is randomly sampled, modulated, and mapped to resource grids to form the channel inputs
    * A batch of channel realizations is randomly sampled and applied to the channel inputs
    * The receiver is executed on the post-DFT received samples to compute LLRs on the coded bits.
      Which receiver is executed (baseline with perfect CSI knowledge, baseline with LS estimation, or neural receiver) depends
      on the specified ``system`` parameter.
    * If not training, the outer decoder is applied to reconstruct the information bits
    * If training, the BMD rate is estimated over the batch from the LLRs and the transmitted bits
    
    Parameters
    -----------
    system : str
        Specify the receiver to use. Should be one of 'baseline-perfect-csi', 'baseline-ls-estimation' or 'neural-receiver'
    
    training : bool
        Set to `True` if the system is instantiated to be trained. Set to `False` otherwise. Defaults to `False`.
        If the system is instantiated to be trained, the outer encoder and decoder are not instantiated as they are not required for training.
        This significantly reduces the computational complexity of training.
        If training, the bit-metric decoding (BMD) rate is computed from the transmitted bits and the LLRs. The BMD rate is known to be
        an achievable information rate for BICM systems, and therefore training of the neural receiver aims at maximizing this rate.
    
    Input
    ------
    batch_size : int
        Batch size
    
    no : scalar or [batch_size], tf.float
        Noise variance.
        At training, a different noise variance should be sampled for each batch example.
    
    Output
    -------
    If ``training`` is set to `True`, then the output is a single scalar, which is an estimation of the BMD rate computed over the batch. It
    should be used as objective for training.
    If ``training`` is set to `False`, the transmitted information bits and their reconstruction on the receiver side are returned to
    compute the block/bit error rate. 
    """
    
    def __init__(self, system, training=False):
        super().__init__()
        self._system = system
        self._training = training
        
    
        ######################################
        ## Transmitter
        self._binary_source = BinarySource()
        # To reduce the computational complexity of training, the outer code is not used when training,
        # as it is not required
        if not training:
            self._encoder = LDPC5GEncoder(k, n)
        self._mapper = Mapper("qam", num_bits_per_symbol)
        self._rg_mapper = ResourceGridMapper(resource_grid)
        
        ######################################
        ## Channel
        # A 3GPP CDL channel model is used
        #a, tau = channel_model()


        self._channel = ApplyOFDMChannel(add_awgn=True)
        #self._channel = OFDMChannel(channel_model, resource_grid, normalize_channel=True, return_channel=True)
        #self._channel = OFDMChannel(channel_model, resource_grid, normalize_channel=True, return_channel=True)
        ######################################
        ## Receiver
        # Three options for the receiver depending on the value of `system`
        if "baseline" in system:
            if system == 'baseline-perfect-csi': # Perfect CSI
                self._removed_null_subc = RemoveNulledSubcarriers(resource_grid)
            elif system == 'baseline-ls-estimation': # LS estimation
                self._ls_est = LSChannelEstimator(resource_grid, interpolation_type="nn")
            # Components required by both baselines
            self._lmmse_equ = LMMSEEqualizer(resource_grid, stream_manager, )
            self._demapper = Demapper("app", "qam", num_bits_per_symbol)
        elif system == "neural-receiver": # Neural receiver
            self._neural_receiver = NeuralReceiver()
            self._rg_demapper = ResourceGridDemapper(resource_grid, stream_manager) # Used to extract data-carrying resource elements
        # To reduce the computational complexity of training, the outer code is not used when training,
        # as it is not required
        if not training:
            self._decoder = LDPC5GDecoder(self._encoder, hard_out=True)
    
   # @tf.function()
    def call(self, batch_size, ebno_db, h_freq):
        #ofdm_channel = sionna.channel.GenerateOFDMChannel(CIR_supp_scene_1, resource_grid, normalize_channel=True)
        # If `ebno_db` is a scalar, a tensor with shape [batch size] is created as it is what is expected by some layers
        if len(ebno_db.shape) == 0:
            ebno_db = tf.fill([batch_size], ebno_db)
        
        ######################################
        ## Transmitter
        no = ebnodb2no(ebno_db, num_bits_per_symbol, coderate)
        # Outer coding is only performed if not training
        if self._training:
            c = self._binary_source([batch_size, 1, 1, n])
            x = self._mapper(c)
            x_rg = self._rg_mapper(x)
            no_ = expand_to_rank(no, tf.rank(x_rg))
            #ofdm_channel = OFDMChannel(CIR, rg, normalize_channel=True)
            #h_freq = ofdm_channel()
            y = self._channel([x_rg,h_freq, no_])
            h = h_freq
        else:
            k = int(n * coderate)
            b = self._binary_source([batch_size, 1, 1, k])
            c = self._encoder(b)
            ## channel
            # Compute frequencies of subcarriers and center around carrier frequency
        # Modulation
            #ofdm_channel = OFDMChannel(CIR_test, rg, normalize_channel=True)
            #h_freq = ofdm_channel()
            x = self._mapper(c)
            x_rg = self._rg_mapper(x)
            #print("no", no.shape)
            ofdm_channel = sionna.channel.GenerateOFDMChannel(channel_model_test, resource_grid, normalize_channel=True)
            h_freq = ofdm_channel()
            no_ = expand_to_rank(no, tf.rank(x_rg))
            y = self._channel([x_rg,h_freq, no_])
            h = h_freq
            print("h shape", h.shape)
        #chan = GenerateOFDMChannel(channel_model, resource_grid, normalize_channel=True)
        ######################################
        ## Receiver       
        # Three options for the receiver depending on the value of ``system``
        if "baseline" in self._system:
            if self._system == 'baseline-perfect-csi':
                h_hat = self._removed_null_subc(h) # Extract non-null subcarriers
                err_var = 0.0 # No channel estimation error when perfect CSI knowledge is assumed
            elif self._system == 'baseline-ls-estimation':
                print("y shape", y.shape)       
                h_hat, err_var = self._ls_est([y, no]) # LS channel estimation with nearest-neighbor
                print("h_hat shape", h_hat.shape)
                print("err_var shape", err_var.shape)
                print("no shape", no.shape)
            x_hat, no_eff = self._lmmse_equ([y, h_hat, err_var, no]) # LMMSE equalization
            no_eff_= expand_to_rank(no_eff, tf.rank(x_hat))
            llr = self._demapper([x_hat, no_eff_]) # Demapping
        elif self._system == "neural-receiver":
            # The neural receover computes LLRs from the frequency domain received symbols and N0
            y = tf.squeeze(y, axis=1)
            print('y', y.shape)
            llr = self._neural_receiver([y, no])
            llr = insert_dims(llr, 2, 1) # Reshape the input to fit what the resource grid demapper is expected
            llr = self._rg_demapper(llr) # Extract data-carrying resource elements. The other LLrs are discarded
            llr = tf.reshape(llr, [batch_size, 1, 1, n]) # Reshape the LLRs to fit what the outer decoder is expected

        # Outer coding is not needed if the information rate is returned
        if self._training:
            # Compute and return BMD rate (in bit), which is known to be an achievable
            # information rate for BICM systems.
            # Training aims at maximizing the BMD rate
            bce = tf.nn.sigmoid_cross_entropy_with_logits(c, llr)
            bce = tf.reduce_mean(bce)
            rate = tf.constant(1.0, tf.float32) - bce/tf.math.log(2.)
            return rate
        else:
            # Outer decoding
            b_hat = self._decoder(llr)
            return b,b_hat # Ground truth and reconstructed information bits returned for BER/BLER computation
        


num_validation_iter =60#30
num_training_iterations = 31#230 #116
training_batch_size = 32


epoch = 100#100#200
optimizer = tf.keras.optimizers.Adam()
losses =[]
losses_val =[]
losses_val_2 =[]
model_meta = E2ESystem('neural-receiver', training=True)
model = E2ESystem('neural-receiver', training=True)


ebno_db_min = -10.0
ebno_db_max = 30.0
#ebno_dbs = np.arange(ebno_db_min, ebno_db_max, 2) # Step
##
#### training of target model
epoch = 100
optimizer_train = tf.keras.optimizers.legacy.Adam(learning_rate=0.001)
optimizer_test = tf.keras.optimizers.legacy.Adam(learning_rate=0.005)
for epks in range(epoch):
    for j in range(len(channel_model_supp)):
        tot_loss = 0.0
    ## training using the support set
        for i in range(num_training_iterations):
            ofdm_channel = sionna.channel.GenerateOFDMChannel(channel_model_supp[j], resource_grid, normalize_channel=True)
            h_freq = ofdm_channel()
    # Compute the frequency response of the channel at frequencies.
            ebno_db = tf.random.uniform(shape=[], minval=ebno_db_min, maxval=ebno_db_max)
            if j >0:
                model_meta(32, tf.constant(10.0, tf.float32), h_freq)
                model_meta.load_weights("C:\\Users\\MW88BT\\wgts_querry")
            # Forward pass
            with tf.GradientTape() as tape_train:
                rate_train = model_meta(training_batch_size, ebno_db,h_freq)
                # Tensorflow optimizers only know how to minimize loss function.
                # Therefore, a loss function is defined as the additive inverse of the BMD rate
                loss_train = -rate_train
            # Computing and applying gradients        
            weights_train = model_meta.trainable_weights
            grads_train = tape_train.gradient(loss_train, weights_train)
            optimizer_train.apply_gradients(zip(grads_train, weights_train))

        model_meta.save_weights('C:\\Users\\MW88BT\\wgts_support',save_format='h5')
        # run an instance of the model 
        model_meta = E2ESystem('neural-receiver',training=True)
        # Run one inference to build the layers and loading the weights from the target model
        model_meta(32, tf.constant(10.0, tf.float32), h_freq)
        model_meta.load_weights("C:\\Users\\MW88BT\\wgts_support")
        # retraining using the querry set
        for kk in range(num_training_iterations):
            ofdm_channel = sionna.channel.GenerateOFDMChannel(channel_model_querry[j], resource_grid, normalize_channel=True)
            h_freq = ofdm_channel()
    # Compute the frequency response of the channel at frequencies.
            ebno_db = tf.random.uniform(shape=[], minval=ebno_db_min, maxval=ebno_db_max)
            # Forward pass
            with tf.GradientTape() as tape_test:
                rate_test = model_meta(training_batch_size, ebno_db, h_freq)
                # Tensorflow optimizers only know how to minimize loss function.
                # Therefore, a loss function is defined as the additive inverse of the BMD rate
                loss_test = -rate_test
            # Computing and applying gradients

        
            weights_test = model_meta.trainable_weights
            grads_test = tape_test.gradient(loss_test, weights_test)
            optimizer_test.apply_gradients(zip(grads_test, weights_test))
            model_meta.save_weights('C:\\Users\\MW88BT\\wgts_querry',save_format='h5')

#model = E2ESystem('neural-receiver', training=True) 
#model(32, tf.constant(10.0, tf.float32), h_freq) 
#model.save_weights('C:\\Users\\MW88BT\\meta_train_test',save_format='h5')
##############################################################

sionna.config.xla_compat=False # not supported in CIRDataset
class E2ESystem2(Model):
    r"""
    Keras model that implements the end-to-end systems.
    
    As the three considered end-to-end systems (perfect CSI baseline, LS estimation baseline, and neural receiver) share most of
    the link components (transmitter, channel model, outer code...), they are implemented using the same Keras model.

    When instantiating the Keras model, the parameter ``system`` is used to specify the system to setup,
    and the parameter ``training`` is used to specified if the system is instantiated to be trained or to be evaluated.
    The ``training`` parameter is only relevant when the neural 
    
    At each call of this model:
    * A batch of codewords is randomly sampled, modulated, and mapped to resource grids to form the channel inputs
    * A batch of channel realizations is randomly sampled and applied to the channel inputs
    * The receiver is executed on the post-DFT received samples to compute LLRs on the coded bits.
      Which receiver is executed (baseline with perfect CSI knowledge, baseline with LS estimation, or neural receiver) depends
      on the specified ``system`` parameter.
    * If not training, the outer decoder is applied to reconstruct the information bits
    * If training, the BMD rate is estimated over the batch from the LLRs and the transmitted bits
    
    Parameters
    -----------
    system : str
        Specify the receiver to use. Should be one of 'baseline-perfect-csi', 'baseline-ls-estimation' or 'neural-receiver'
    
    training : bool
        Set to `True` if the system is instantiated to be trained. Set to `False` otherwise. Defaults to `False`.
        If the system is instantiated to be trained, the outer encoder and decoder are not instantiated as they are not required for training.
        This significantly reduces the computational complexity of training.
        If training, the bit-metric decoding (BMD) rate is computed from the transmitted bits and the LLRs. The BMD rate is known to be
        an achievable information rate for BICM systems, and therefore training of the neural receiver aims at maximizing this rate.
    
    Input
    ------
    batch_size : int
        Batch size
    
    no : scalar or [batch_size], tf.float
        Noise variance.
        At training, a different noise variance should be sampled for each batch example.
    
    Output
    -------
    If ``training`` is set to `True`, then the output is a single scalar, which is an estimation of the BMD rate computed over the batch. It
    should be used as objective for training.
    If ``training`` is set to `False`, the transmitted information bits and their reconstruction on the receiver side are returned to
    compute the block/bit error rate. 
    """
    
    def __init__(self, system, training=False):
        super().__init__()
        self._system = system
        self._training = training
        
    
        ######################################
        ## Transmitter
        self._binary_source = BinarySource()
        # To reduce the computational complexity of training, the outer code is not used when training,
        # as it is not required
        if not training:
            self._encoder = LDPC5GEncoder(k, n)
        self._mapper = Mapper("qam", num_bits_per_symbol)
        self._rg_mapper = ResourceGridMapper(resource_grid)
        
        ######################################
        ## Channel
        # A 3GPP CDL channel model is used
       
        self._channel = ApplyOFDMChannel(add_awgn=True)
        
        #self._channel = OFDMChannel(channel_model, resource_grid, normalize_channel=True, return_channel=True)
        ######################################
        ## Receiver
        # Three options for the receiver depending on the value of `system`
        if "baseline" in system:
            if system == 'baseline-perfect-csi': # Perfect CSI
                self._removed_null_subc = RemoveNulledSubcarriers(resource_grid)
            elif system == 'baseline-ls-estimation': # LS estimation
                self._ls_est = LSChannelEstimator(resource_grid, interpolation_type="nn")
            # Components required by both baselines
            self._lmmse_equ = LMMSEEqualizer(resource_grid, stream_manager, )
            self._demapper = Demapper("app", "qam", num_bits_per_symbol)
        elif system == "neural-receiver": # Neural receiver
            self._neural_receiver = NeuralReceiver()
            self._rg_demapper = ResourceGridDemapper(resource_grid, stream_manager) # Used to extract data-carrying resource elements
        # To reduce the computational complexity of training, the outer code is not used when training,
        # as it is not required
        if not training:
            self._decoder = LDPC5GDecoder(self._encoder, hard_out=True)
    
    #@tf.function(jit_compile=False)
    def call(self, batch_size, ebno_db):
        ofdm_channel = sionna.channel.GenerateOFDMChannel(channel_model, resource_grid, normalize_channel=True)
        # If `ebno_db` is a scalar, a tensor with shape [batch size] is created as it is what is expected by some layers
        if len(ebno_db.shape) == 0:
            ebno_db = tf.fill([batch_size], ebno_db)
        
        ######################################
        ## Transmitter
        no = ebnodb2no(ebno_db, num_bits_per_symbol, coderate)
        # Outer coding is only performed if not training
        if self._training:
            c = self._binary_source([batch_size, 1, 1, n])
            x = self._mapper(c)
            x_rg = self._rg_mapper(x)
            h_freq = ofdm_channel()
            no_ = expand_to_rank(no, tf.rank(x_rg))
            y = self._channel([x_rg,h_freq, no_])
            h = h_freq
        else:
            b = self._binary_source([batch_size, 1, 1, k])
            c = self._encoder(b)
        # Modulation
            x = self._mapper(c)
            x_rg = self._rg_mapper(x)
            no_ = expand_to_rank(no, tf.rank(x_rg))
            ofdm_channel_test = sionna.channel.GenerateOFDMChannel(channel_model_test, resource_grid, normalize_channel=True)
            h_freq_t = ofdm_channel_test()
            y = self._channel([x_rg,h_freq_t, no_])
            h = h_freq_t
        #chan = GenerateOFDMChannel(channel_model, resource_grid, normalize_channel=True)
        ######################################
        ## Receiver       
        # Three options for the receiver depending on the value of ``system``
        if "baseline" in self._system:
            if self._system == 'baseline-perfect-csi':
                h_hat = self._removed_null_subc(h) # Extract non-null subcarriers
                err_var = 0.0 # No channel estimation error when perfect CSI knowledge is assumed
            elif self._system == 'baseline-ls-estimation':
                h_hat, err_var = self._ls_est([y, no]) # LS channel estimation with nearest-neighbor
            x_hat, no_eff = self._lmmse_equ([y, h_hat, err_var, no]) # LMMSE equalization
            no_eff_= expand_to_rank(no_eff, tf.rank(x_hat))
            llr = self._demapper([x_hat, no_eff_]) # Demapping
        elif self._system == "neural-receiver":
            # The neural receover computes LLRs from the frequency domain received symbols and N0
            y = tf.squeeze(y, axis=1)
            llr = self._neural_receiver([y, no])
            llr = insert_dims(llr, 2, 1) # Reshape the input to fit what the resource grid demapper is expected
            llr = self._rg_demapper(llr) # Extract data-carrying resource elements. The other LLrs are discarded
            llr = tf.reshape(llr, [batch_size, 1, 1, n]) # Reshape the LLRs to fit what the outer decoder is expected

        # Outer coding is not needed if the information rate is returned
        if self._training:
            # Compute and return BMD rate (in bit), which is known to be an achievable
            # information rate for BICM systems.
            # Training aims at maximizing the BMD rate
            bce = tf.nn.sigmoid_cross_entropy_with_logits(c, llr)
            bce = tf.reduce_mean(bce)
            rate = tf.constant(1.0, tf.float32) - bce/tf.math.log(2.)
            return rate
        else:
            # Outer decoding
            b_hat = self._decoder(llr)
            return b,b_hat # Ground truth and reconstructed information bits returned for BER/BLER computation



################## meta adaptaion or fine tuning 
#### create an instance of the source model and use it as target model without additional layers

## an instance of of the model and loading saved wieghts
model_all = E2ESystem2('neural-receiver',training = True)
model_all(32, tf.constant(10.0, tf.float32))
#model_all.load_weights("C:\\Users\\MW88BT\\meta_train_test",by_name=True)
model_all.load_weights("C:\\Users\\MW88BT\\wgts_querry",by_name=True)

#### training of target model
num_validation_iter =60#30
num_training_iterations = 32#302 #116
training_batch_size = 32



optimizer = tf.keras.optimizers.Adam()
losses =[]
losses_val =[]
losses_val_2 =[]


ebno_dbs = np.arange(-3, 18, 2) # sim SNR range
#ebno_dbs = np.arange(ebno_db_min, ebno_db_max, 2) # Step
##
#### training of target model
epoch = 100
optimizer = tf.keras.optimizers.Adam()
for j in range(epoch):
    tot_loss = 0.0
## training of the target model
    for i in range(num_training_iterations):
        ofdm_channel = sionna.channel.GenerateOFDMChannel(channel_model, resource_grid, normalize_channel=True)
        h_freq = ofdm_channel()
# Compute the frequency response of the channel at frequencies.
        ebno_db = tf.random.uniform(shape=[], minval=ebno_db_min, maxval=ebno_db_max)
        # Forward pass
        with tf.GradientTape() as tape:
            rate = model_all(training_batch_size, ebno_db)
            # Tensorflow optimizers only know how to minimize loss function.
            # Therefore, a loss function is defined as the additive inverse of the BMD rate
            loss = -rate
        # Computing and applying gradients        
        weights = model_all.trainable_weights
        grads = tape.gradient(loss, weights)
        optimizer.apply_gradients(zip(grads, weights))

        # Periodically printing the progress
        if i % 100 == 0:
            print('Iteration {}/{}  Rate: {:.4f} bit'.format(i, num_training_iterations, rate.numpy()), end='\r')


        
## save the weights in a hd5 format
model_all.save_weights('meta_wghts_all',save_format='h5')


####
# run an instance of the model 
model_all_= E2ESystem2('neural-receiver')

# Run one inference to build the layers and loading the weights from the target model
model_all_(32, tf.constant(10.0, tf.float32))

model_all_.load_weights("C:\\Users\\MW88BT\\meta_wghts_all")



BLER_all={}

# Evaluations of the target model
_,bler = sim_ber(model_all_, ebno_dbs, batch_size=32, num_target_block_errors=100, max_mc_iter=100)
BLER_all['neural-receiver'] = bler.numpy()

with open('meta_mun_2_par.pkl', 'wb') as f:
    pickle.dump(BLER_all, f)
