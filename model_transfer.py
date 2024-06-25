import os
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

import numpy as np
import tensorflow as tf
# Avoid warnings from TensorFlow
tf.get_logger().setLevel('ERROR')

# IPython "magic function" for inline plots
#matplotlib inline
import matplotlib.pyplot as plt


# For saving complex Python data structures efficiently
import pickle

# For the implementation of the neural receiver
from tensorflow.keras import Model
from tensorflow.keras import Sequential, optimizers
from tensorflow.keras.layers import Dense, Layer



##
#matplotlib inline
#matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
import pickle

from tensorflow.keras import Model
from tensorflow.keras.layers import Layer, Conv2D, LayerNormalization
from tensorflow.nn import relu
from tensorflow.keras import Model

from sionna.channel.tr38901 import Antenna, AntennaArray, CDL, UMa, UMi, RMa
from sionna.channel import OFDMChannel, ApplyOFDMChannel, ApplyTimeChannel, subcarrier_frequencies, cir_to_ofdm_channel, cir_to_time_channel
from sionna.channel import gen_single_sector_topology as gen_topology
from sionna.mimo import StreamManagement
from sionna.ofdm import ResourceGrid, ResourceGridMapper, LSChannelEstimator, LMMSEEqualizer, RemoveNulledSubcarriers, ResourceGridDemapper
from sionna.utils import BinarySource, ebnodb2no, insert_dims, flatten_last_dims, log10, expand_to_rank
from sionna.fec.ldpc.encoding import LDPC5GEncoder
from sionna.fec.ldpc.decoding import LDPC5GDecoder
from sionna.mapping import Mapper, Demapper
from sionna.utils.metrics import compute_ber
from sionna.utils import sim_ber
from NeuralReceiver import NeuralReceiver
from ResidualBlock import ResidualBlock


def model_transfer(scenario, target_model, weight_filepath):
############################################
## Channel configuration
    carrier_frequency = 3.5e9 # Hz
    delay_spread = 100e-9 # s
    speed = 10.0 # Speed for evaluation and training [m/s]
    # SNR range for evaluation and training [dB]
    ebno_db_min = -5.0
    ebno_db_max = 10.0

    ############################################
    ## OFDM waveform configuration
    subcarrier_spacing = 30e3 # Hz
    fft_size = 128 # Number of subcarriers forming the resource grid, including the null-subcarrier and the guard bands
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
    num_conv_channels = 128 # Number of convolutional channels for the convolutional layers forming the neural receiver
    ############################################
    ## Evaluation configuration
    #results_filename = "neural_receiver_results" # Location to save the results

    ##

    #if target_model =='UMi':
    #    scenario = 'umi'
    #elif target_model == 'UMa':
    #    scenario = 'uma'
    #else :
    #    scenario = 'rma'
    scenario = scenario

    num_ut = 1

    #####
    # The number of transmitted streams is equal to the number of UT antennas
    num_streams_per_tx = 1

    rx_tx_association = np.zeros([1, num_ut])
    rx_tx_association[0, :] = 1


    stream_manager = StreamManagement(rx_tx_association, num_streams_per_tx)
    #####


    #stream_manager = StreamManagement(np.array([[1]]), # Receiver-transmitter association matrix
                #                     1)               # One stream per transmitter


    ######

    resource_grid = ResourceGrid(num_ofdm_symbols = num_ofdm_symbols,
                                fft_size = fft_size,
                                subcarrier_spacing = subcarrier_spacing,
                                num_tx = num_ut,
                                num_streams_per_tx = 1,
                                cyclic_prefix_length = cyclic_prefix_length,
                                dc_null = dc_null,
                                pilot_pattern = pilot_pattern,
                                pilot_ofdm_symbol_indices = pilot_ofdm_symbol_indices,
                                num_guard_carriers = num_guard_carriers) 


    ###

    # Codeword length. It is calculated from the total number of databits carried by the resource grid, and the number of bits transmitted per resource element
    n = int(resource_grid.num_data_symbols*num_bits_per_symbol)
    # Number of information bits per codeword
    k = int(n*coderate)  
    ################################

    ###############################

    batch_size = 128
    ut_antenna = Antenna(polarization="single",
                        polarization_type="V",
                        antenna_pattern="38.901",
                        carrier_frequency=carrier_frequency)

    bs_array = AntennaArray(num_rows=1,
                            num_cols=1,
                            polarization="dual",
                            polarization_type="VH",
                            antenna_pattern="38.901",
                            carrier_frequency=carrier_frequency)


    channel_model = target_model(carrier_frequency=carrier_frequency,
                        o2i_model="low",
                        ut_array=ut_antenna,
                        bs_array=bs_array,
                        direction=direction,
                        enable_pathloss=False,
                        enable_shadow_fading=False)

    # Generate the topology
    topology = gen_topology(batch_size, num_ut, scenario)

    # Set the topology
    channel_model.set_topology(*topology)

    ##########################################################################################

    ##############################################################################################

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
            ##### An UMi
            self.channel_model = target_model(carrier_frequency=carrier_frequency,
                        o2i_model="low",
                        ut_array=ut_antenna,
                        bs_array=bs_array,
                        direction=direction,
                        enable_pathloss=False,
                        enable_shadow_fading=False, always_generate_lsp=True)

        
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
        
        @tf.function
        def call(self, batch_size, ebno_db):
            
            # If `ebno_db` is a scalar, a tensor with shape [batch size] is created as it is what is expected by some layers
            if len(ebno_db.shape) == 0:
                ebno_db = tf.fill([batch_size], ebno_db)
            
            ######################################
            ## Transmitter
            no = ebnodb2no(ebno_db, num_bits_per_symbol, coderate)
            # Outer coding is only performed if not training
            if self._training:
                c = self._binary_source([batch_size, num_ut, resource_grid.num_streams_per_tx, n])
            else:
                b = self._binary_source([batch_size, num_ut, resource_grid.num_streams_per_tx, k])
                c = self._encoder(b)
            # Modulation
            x = self._mapper(c)
            x_rg = self._rg_mapper(x)
            
            ######################################
            # Generate the topology
            self.topology = gen_topology(batch_size, num_ut, scenario)

            # Set the topology
            self.channel_model.set_topology(*self.topology)
            
            self._ofdm_channel = OFDMChannel(self.channel_model, resource_grid, add_awgn=True, normalize_channel=True, return_channel=True)
            
            
            ## Channel
            # A batch of new channel realizations is sampled and applied at every inference
            no_ = expand_to_rank(no, tf.rank(x_rg))
            y , h  = self._ofdm_channel([x_rg, no_])
            
            ######################################
            ## Receiver       
            # Three options for the receiver depending on the value of ``system``
            if "baseline" in self._system:
                if self._system == 'baseline-perfect-csi':
                    h_hat  = self._removed_null_subc(h ) # Extract non-null subcarriers
                    err_var = 0.0 # No channel estimation error when perfect CSI knowledge is assumed
                elif self._system == 'baseline-ls-estimation':
                    h_hat , err_var = self._ls_est([y , no]) # LS channel estimation with nearest-neighbor
                x_hat , no_eff  = self._lmmse_equ([y , h_hat , err_var, no]) # LMMSE equalization
                no_eff = expand_to_rank(no_eff , tf.rank(x_hat ))
                llr  = self._demapper([x_hat , no_eff ]) # Demapping
                print(y.shape), print(x_hat.shape), print(x_hat.shape)
            elif self._system == "neural-receiver":
                # The neural receover computes LLRs from the frequency domain received symbols and N0
                y  = tf.squeeze(y , axis=1)
                llr  = self._neural_receiver([y , no])
                llr  = insert_dims(llr , 2, 1) # Reshape the input to fit what the resource grid demapper is expected
                llr  = self._rg_demapper(llr ) # Extract data-carrying resource elements. The other LLrs are discarded
                llr  = tf.reshape(llr , [batch_size, num_ut, resource_grid.num_streams_per_tx, n]) # Reshape the LLRs to fit what the outer decoder is expected
            # Outer coding is not needed if the information rate is returned
            if self._training:
                # Compute and return BMD rate (in bit), which is known to be an achievable
                # information rate for BICM systems.
                # Training aims at maximizing the BMD rate
                bce  = tf.nn.sigmoid_cross_entropy_with_logits(c, llr )
                bce  = tf.reduce_mean(bce )
                rate  = tf.constant(1.0, tf.float32) - bce /tf.math.log(2.)
                return rate 
            
            else:
                # Outer decoding
                b_hat  = self._decoder(llr )
                return b,b_hat  # Ground truth and reconstructed information bits returned for BER/BLER computation
            
    #####################################################################################################################################################################

    # parameters for evaluation
    ebno_db_min = -5.0
    ebno_db_max = 10.0
    ebno_dbs = np.arange(ebno_db_min, # Min SNR for evaluation
                        ebno_db_max, # Max SNR for evaluation
                        0.5) # Step

    # an instance of source model to build layers        

    ## an instance of of the model and loading saved wieghts
    model = E2ESystem('neural-receiver',training = True)
    model(1, tf.constant(10.0, tf.float32))

    model.load_weights(weight_filepath) # wieights of base line or source model CDL C




    BLER_={}

    # Evaluations of the target model
    _,bler = sim_ber(model, ebno_dbs, batch_size=128, num_target_block_errors=100, max_mc_iter=10000)
    BLER_['neural-receiver'] = bler.numpy()


    return BLER_



#x_eval_y(target_model = UMi, weight_filepath = "C:\\Users\\MW88BT\\OneDrive - Aalborg Universitet\\my_vscode_scripts\\Comprehns_Evaluation\\weights_BL")