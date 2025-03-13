import sys
sys.path.append('/Users/xiyu/Documents/AAU/CENTRIC_codes/PPO_beam_management/Extension_BM/')
from channel_functions import *
import gymnasium as gym
from gymnasium.envs import register
from gymnasium.spaces import Box, Dict, MultiDiscrete
from exroad import *
import tensorflow as tf



# from sionna.ofdm import ResourceGrid
# from sionna.channel import ApplyOFDMChannel
from util_functions import *
import time
# import matplotlib.pyplot as plt

class BeamManagementEnv(gym.Env):

    def __init__(self,  args):
        self.if_training = args.if_training
        self.if_debugging = args.if_debugging

        self.N_UTs = args.N_UTs  # number of UEs 
        self.number_of_antennas = args.Nt  # number of antennas at the ULA BS

        self.num_steps = 100 # number of steps per episode

        ##################################################
        #          Define action and state space
        ##################################################
        self.action_space = Box(low =np.array([0]*self.number_of_antennas), high = np.array([1]*self.number_of_antennas),shape = (self.number_of_antennas, ))
        #self.action_space = MultiDiscrete(np.array([self.N_UTs+1]*self.number_of_antennas))
        #
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(1*self.number_of_antennas + 1*self.N_UTs,)) #

        self.carrier_frequency = 28e9  # carrier frequency (Hz)
        self.wavelength = 3e8/ self.carrier_frequency  # wavelength (m)

        # OFDM parameters
        self.batch_size = 1 # batch size
        self.fft_size = 144
        self.num_ofdm_symbols = 14
        self.subcarrier_spacing = 60e3
        self.cyclic_prefix_length = 20
        self.ofdm_symbol_duration = (1. + self.cyclic_prefix_length/self.fft_size) / self.subcarrier_spacing
        num_streams_per_tx = self.N_UTs # Number of transmitted streams is equal to the number of UTs
        rx_tx_association = np.ones([self.N_UTs, 1])
        #  resource grid
 #       self.rg = ResourceGrid(num_ofdm_symbols=14, 
                            #    fft_size=self.fft_size,
                            #    subcarrier_spacing=subcarrier_spacing,
                            #    num_tx = 1, # one BS
                            #    num_streams_per_tx=num_streams_per_tx, 
                            #    cyclic_prefix_length=20,
                            #    pilot_ofdm_symbol_indices=[2,11],
                            #    pilot_pattern="kronecker"
                            #    )

        self.SE_threshold = 4  # rate threshold for each UE
        self.SE = np.zeros((self.N_UTs,))  # spectral efficiency of each UE

        self.Pt_mW = (10**(args.Ptdbm/10))*(10**(109/10))  # 15dbm in linear scale
        self.noise_power = 10**(-109/10)*10**(109/10)  # -109 dbm in linear scale, 10**(109/10) is normalization

        self.beamwidth = 2*np.arcsin(1.2/self.number_of_antennas) 
        
        self.current_beam = np.empty([self.N_UTs,], dtype = object)  # current beam direction
        # self.action_beam = np.zeros((self.number_of_antennas))  # action beam direction
        # self.action_power = np.zeros((self.N_UTs))  # action power allocation

        self.datapool = np.zeros((self.N_UTs, ), dtype=np.float32)  # number of packets of each UE in the buffer
        self.total_traffic_episode = np.sum(self.datapool) # total downlink traffic per episode

        self.bs_loc = np.expand_dims(np.expand_dims(np.array([50, 20, 0], dtype=np.float32), axis=0), axis=0)   # BS position

        self.ut_loc = np.zeros((1, self.N_UTs, 3), dtype=np.float64)  # UE positions
        self.config = np.zeros((1, self.N_UTs,2 ), dtype=int)
        self.if_turned = np.zeros((1, self.N_UTs), dtype=bool)

        self.dt = args.dt  # time step 10 ms

        # self.Rd = np.zeros((1,self.N_UTs))
        # self.AoD = np.zeros((self.N_UTs))
        # self.aod = np.zeros((self.N_UTs))
        # self.a_BS = np.zeros((self.N_UTs, self.number_of_antennas), dtype=complex) 
        
        self.reflection_coefficient = np.zeros((1,self.N_UTs))
        
        self.nmse_R = np.zeros((self.N_UTs)) # normalized MSE of delay estimation
        self.nmse_v = np.zeros((self.N_UTs))

        # define the road traffic scenario
        self.cross_pos = np.array([[27, 80], [30, 70]])
        self.road_matrix = road(self.cross_pos)
        self.single_lane_width = 5

        # self.ut_loc[0,:,:], self.config, self.if_turned = user_pos_direction_init(self.road_matrix, self.cross_pos, self.N_UTs, self.single_lane_width)


    def reset(self, seed=32):
        super().reset(seed=seed)

        self.cstep = 0
        self.current_beam = np.empty([self.N_UTs,], dtype=object)

        self.Rd = np.zeros((1,self.N_UTs))
        self.AoD = np.zeros((self.N_UTs))
        
        # channel_freq = ApplyOFDMChannel(add_awgn=True)

        N = 1 # consider one path
        H_UT = np.zeros([self.batch_size, self.N_UTs,1,1, self.number_of_antennas, self.num_ofdm_symbols, self.fft_size], dtype=np.complex64) #[ batch size, num_rx, num_rx_ant, num_tx, num_tx_ant, num_ofdm_symbols, fft_size ]

        transmit_beam = np.zeros((self.number_of_antennas, self.N_UTs), dtype=np.complex64)

        codebook = dft_codebook(self.number_of_antennas, self.number_of_antennas)  # set of beamforming directions

        ########################################################################
        #                set limited options for initial positions              #
        ########################################################################
        initial_pos_training = np.array([[[36., 33.],[24., 62.],[72., 26.],[77., 68.],[17., 29.]],
            [[17., 28.],[22., 66.],[83., 30.],[79., 74.],[20., 27.]],  
            [[36., 25.],[26., 75.],[89., 30.],[77., 64.],[31., 29.]] ,
            [[18., 25.],[25., 77.],[71., 32.],[78., 76.],[32., 31.]] ,
            [[36., 25.],[26., 64.],[86., 27.],[75., 61.],[23., 31.]] ,
            [[31., 25.],[24., 78.],[87., 30.],[76., 62.],[27., 26.]] ,
            [[30., 29.],[23., 74.],[73., 25.],[78., 79.],[30., 32.]] ,
            [[28., 32.],[25., 63.],[77., 25.],[77., 62.],[18., 34.]] ,
            [[31., 27.],[26., 67.],[74., 26.],[75., 69.],[31., 26.]] ,
            [[30., 34.],[22., 72.],[88., 28.],[75., 68.],[20., 34.]],
            [[25., 35.],[46., 67.],[ 8., 71.],[76., 17.],[95., 30.]],
            [[ 9., 26.],[84., 44.],[46., 74.],[80., 73.],[31,68]]] )

        initial_config = np.array([[[1, 1],[2, 2],[0, 0],[4, 2],[0, 0]] ,
                                [[0, 0],[4, 2],[4, 2],[4, 2],[0, 0]] ,
                                [[4, 2],[4, 2],[1, 1],[2, 2],[4, 2]] ,
                                [[4, 2],[3, 3],[0, 0],[3, 3],[4, 2]] ,
                                [[4, 2],[2, 2],[1, 1],[4, 2],[4, 2]] ,
                                [[4, 2],[3, 3],[1, 1],[2, 2],[4, 2]] ,
                                [[4, 2],[4, 2],[4, 2],[3, 3],[4, 2]],
                                [[4, 2],[2, 2],[4, 2],[2, 2],[0, 0]],
                                [[4, 2],[4, 2],[0, 0],[4, 2],[4, 2]] ,
                                [[4, 2],[4, 2],[1, 1],[4, 2],[0, 0]], 
                                [[4, 2],[1, 1],[0, 0], [2, 2], [1, 1]],
                                [[0, 0],[3,3],[1, 1],[4, 2], [4,2]]] )

        initial_if_turned = np.array([False, False, False, False, False])

        # if self.if_training:
        #     dice = np.random.randint(0, 11)
        #     self.ut_loc[0,:,0:2] = initial_pos_training[dice, 0:self.N_UTs,:]
        #     self.config[0,:] = initial_config[dice, 0:self.N_UTs,:]
        #     self.if_turned[0,:] = initial_if_turned[0:self.N_UTs]
        # else:

        self.ut_loc[0,:,:], self.config[0,:], self.if_turned[0,:] = user_pos_direction_init(self.road_matrix, self.cross_pos, self.N_UTs, self.single_lane_width) 

        # set UT velocities and trajectories
        self.ut_velocities = np.zeros((1, self.N_UTs, 3))
        self.ut_velocities[0,:,0:2] = np.transpose(np.random.multivariate_normal( 6*np.arange(1,self.N_UTs+1,1), np.diag(4*np.ones([self.N_UTs])), size = (2))) 
        
        self.datapool = np.random.binomial(n=40, p=0.6, size=self.N_UTs) if self.if_training else np.random.binomial(n=self.num_steps, p=0.6, size=self.N_UTs)
        self.total_traffic_episode = np.sum(self.datapool) # total downlink traffic per episode
        
        for i in range(self.N_UTs):

            self.Rd[0,i] = np.linalg.norm(self.bs_loc[0,0,:] - self.ut_loc[0,i,:])  # euclidean distance between the BS and the UEs

            self.AoD[i] = compute_AoD_at_BS(self.bs_loc[0, 0, :], self.ut_loc[0,i,:])  # current AoD

            self.reflection_coefficient[0,i] = self.wavelength * 10/ self.Rd[0,i]**2 / (4*np.pi)**1.5 # reflection coefficient, 5 is the RCS of vehicle

            amp_c = (self.wavelength / (4 * np.pi * self.Rd[0,i]))**2

            H_f,_,_ = freqency_communication_channel(self.num_ofdm_symbols, self.fft_size, N, self.number_of_antennas, amp_c,self.Rd[0,i]/3e8, self.AoD[i])
            
            H_UT[0, i,0,0, ] = H_f

        
        H_BS, A_t = freqency_sensing_channel(self.batch_size, 
                                             self.num_ofdm_symbols, 
                                             self.fft_size , 
                                             self.number_of_antennas, 
                                             self.N_UTs, 
                                             self.subcarrier_spacing, 
                                             self.ofdm_symbol_duration, 
                                             self.reflection_coefficient, 
                                             np.array([self.ut_velocities[0,]]), 
                                             self.wavelength, 
                                             self.Rd*2/3e8, 
                                             self.AoD)

        #self.x_rg, self.encoder, self.b = gen_tx_signal(self.rg, self.N_UTs, self.batch_size)
        self.x_rg = np.zeros([self.batch_size, 1, self.N_UTs, self.num_ofdm_symbols, self.fft_size], dtype=np.complex64)
        self.x_rg[0,0,:] = np.exp(1j * np.deg2rad(np.random.choice([45, 135, 225, 315], size = (self.N_UTs, self.num_ofdm_symbols,self.fft_size ))))

        self.action_power = np.ones(self.N_UTs) * (self.Pt_mW/self.N_UTs)

        tx_sig_shape = list(self.x_rg.shape) #[batch_size, num_tx, num_tx_antennas, num_ofdm_symbols, fft_size]
        tx_sig_shape[2] = self.number_of_antennas
        tx_sig_fd = np.zeros(tx_sig_shape, dtype=self.x_rg.dtype)

        # the first beam is set with known AoDs
        y_UT_eff = np.zeros([self.batch_size, self.N_UTs, 1, self.num_ofdm_symbols, self.fft_size], dtype = np.complex64)

        self.action_beam = np.zeros((self.number_of_antennas))
        for u in range(self.N_UTs):
            self.current_beam[u,] = np.argmax(np.abs(np.transpose(np.conj(A_t[:,0,u])) @ codebook))
        unique_beams, counts = np.unique(self.current_beam, return_counts=True)
        duplicates = unique_beams[counts > 1]

        p = -1
        while len(duplicates)>0:
            p -= 1
            for dup in duplicates:
                indices = np.where(self.current_beam == dup)[0]
                for idx in indices[1:]:
                    self.current_beam[idx,] = np.argsort(np.abs(np.transpose(np.conj(A_t[:,0,idx])) @ codebook))[p]
            unique_beams, counts = np.unique(self.current_beam, return_counts=True)
            duplicates = unique_beams[counts > 1]

        beam_index = self.current_beam.astype('int')
        transmit_beam = np.matmul(codebook[:,beam_index] , np.diag(np.sqrt(self.action_power))).astype(np.complex64)
        
        tx_sig_fd = tf.einsum('kj,nijom -> nikom', transmit_beam, self.x_rg)

        for i in range(self.N_UTs):
            self.action_beam[self.current_beam[i]] = i+1
            temp1 = tf.einsum('kj,nijom -> nikom', tf.expand_dims(transmit_beam[:,i],1), tf.expand_dims(self.x_rg[:,:,i,:,:],2))
            
            # dedicated received signal at the UE
            temp = channel_freq(temp1, H_UT,0)
            y_UT_eff[:,i,:,:,:] = temp[:,i,:,:,:] # received signal targeted for the UE

        # Apply the OFDM channel
        y_UT = channel_freq(tx_sig_fd, H_UT, 0)
        # ber = decode_rx_signal(y_UT, self.rg, self.N_UTs, self.encoder, self.b, self.noise_power) 

        # spectral efficiency
        sig_pow = tf.squeeze(tf.square(tf.abs(y_UT_eff)))

        interference = y_UT - y_UT_eff
        inter_pow = tf.squeeze(tf.square(tf.abs(interference)))

        sinr_pow = sig_pow/tf.add(inter_pow ,self.noise_power)
        # sinr_db = 10*np.log10(sinr_pow)
        self.SE = tf.reduce_sum(tf.reduce_sum(tf.math.log(1+sinr_pow), axis=-1), axis = -1)/(self.num_ofdm_symbols*self.fft_size)
    
        y_BS = channel_freq(tx_sig_fd, H_BS, self.noise_power)

        # beamformed output of the sensing signal
        y_BS2 = tf.einsum('uj, nijom->niuom ' , tf.linalg.adjoint((transmit_beam)), y_BS)
        y_BS2 = tf.multiply(y_BS2, tf.math.conj(self.x_rg))
        # echo_bf_db = 10 * tf.math.log(tf.squeeze(tf.reduce_mean(tf.square(tf.abs(y_BS2)), axis=[-2,-1]))) / tf.math.log(10.0)

        echo_bf_db = np.zeros([ self.number_of_antennas])
        for u in range(self.N_UTs):
            # Assuming self.action_beam and codebook are already TensorFlow tensors
            action_beam_mask = np.equal(self.action_beam, u + 1)
            
            AA = action_beam_mask.astype('complex64') * codebook

            # Compute BB using TensorFlow operations
            # BB = tf.reduce_mean(tf.square(tf.abs(tf.einsum('ij, jnp->inp', tf.linalg.adjoint(AA), tf.squeeze(y_BS)))), axis=[-2, -1]).numpy()
            # echo_bf_db[action_beam_mask] = BB[action_beam_mask]
        echo_bf_db = tf.reduce_mean(tf.math.log1p(tf.square(tf.abs(tf.einsum('ij, jnp->inp', tf.linalg.adjoint(codebook), tf.squeeze(y_BS)))))/np.log(2), axis=[-2, -1]).numpy() 
        self.memory = np.array([echo_bf_db], dtype = np.float32)

        num_rangebin = int(self.fft_size)*3
        num_dopplerbin = int(self.num_ofdm_symbols)*3
        # num_anglebin = self.number_of_antennas

        z_BS = np.fft.fft2(y_BS2, [num_dopplerbin, num_rangebin])
        top = np.abs(z_BS[0,0,u,:,:]).ravel()
        indices = np.zeros([self.N_UTs, 2])
        for u in range(self.N_UTs):
            indices[u,:] = np.concatenate(np.unravel_index(np.argsort(top, axis=None)[-1:], z_BS[0,0,u,:,:].shape)).astype('int')
            # fft shift
            indices[u,0] = indices[u,0] if indices[u,0]<np.ceil(num_dopplerbin/2) else indices[u,0] - num_dopplerbin
            indices[u,1] = indices[u,1] if indices[u,1]<np.ceil(num_rangebin/2) else indices[u,1] - num_rangebin

        self.R_estimate = - 3e8*indices[:,1]/(2*self.subcarrier_spacing * num_rangebin)
        self.v_estiamte = 3e8 * indices[:,0]/(2 * self.ofdm_symbol_duration * self.carrier_frequency * num_dopplerbin)
        
        self.nmse_R = np.abs(self.R_estimate - self.Rd[0,] ) / (3e8/2/self.fft_size/self.subcarrier_spacing)
        self.nmse_v = np.abs(self.v_estiamte - self.ut_velocities[0,:,0]) / (3e8/2/self.num_ofdm_symbols/self.carrier_frequency/self.ofdm_symbol_duration)
        
        return self._get_state(), {}

########################################################################
#                             step function              
########################################################################

    def step(self, action): 
        '''
        Execute function
        '''
  
        reward = 0
        terminated = False 
        truncated = False
        self.cstep += 1

        codebook = dft_codebook(self.number_of_antennas, self.number_of_antennas)  # set of beamforming directions

        ################################################################
        #  update velocities and positions and generate channels
        ################################################################
        
        H_UT = np.zeros([self.batch_size, self.N_UTs,1,1, self.number_of_antennas, self.num_ofdm_symbols, self.fft_size], dtype=np.complex64) #[ batch size, num_users, num_subcarriers, num_BS_antennas ]
        # tx_sig_fd_ut = np.zeros([self.batch_size, self.N_UTs, self.number_of_antennas, self.num_ofdm_symbols, self.fft_size], dtype=np.complex64)
        y_UT_eff = np.zeros([self.batch_size, self.N_UTs, 1, self.num_ofdm_symbols, self.fft_size], dtype = np.complex64)
        
        # channel_freq = ApplyOFDMChannel(add_awgn=True)
        
        self.ut_velocities[0,:,0:2] = np.transpose(np.random.multivariate_normal( 6*np.arange(1,self.N_UTs+1,1), np.diag(4*np.ones([self.N_UTs])), size = (2)))
        for s in range(self.batch_size):
            positions = [update_positions(
                            self.ut_loc[s,i,:], 
                            self.cross_pos,
                            np.array([self.ut_velocities[s,i,0]]), 
                            self.dt, 
                            self.single_lane_width,
                            self.config[s,i], 
                            self.if_turned[s,i]
                            ) for i in range(self.N_UTs)]

            self.ut_loc[s,:,:] = np.array([p[0] for p in positions])
            self.config[s,:] = np.array([p[1] for p in positions])
            self.if_turned[s,:] = np.array([p[2] for p in positions])
        self.Rd[0,:] = np.linalg.norm(self.bs_loc[0,0,:] - self.ut_loc[0,:,:], axis=1)
        self.AoD = np.array([compute_AoD_at_BS(self.bs_loc[0, 0, :], 
                                          self.ut_loc[0,i,:]) for i in range(self.N_UTs)])
        self.reflection_coefficient[0,:] = (self.wavelength * 10 / 
                                      (self.Rd[0,:]**2) / ((4*np.pi)**1.5))
        for i in range(self.N_UTs):

            amp_c = (self.wavelength / (4 * np.pi * self.Rd[0,i]))**2

            H_UT[0, i,0,0, ],_,_ = freqency_communication_channel(self.num_ofdm_symbols, self.fft_size, 1, self.number_of_antennas, amp_c, self.Rd[0,i]/3e8, self.AoD[i])
        
        H_UT = tf.convert_to_tensor(H_UT, dtype = tf.complex64) 
  
        H_BS, A_t = freqency_sensing_channel(self.batch_size, self.num_ofdm_symbols, self.fft_size , self.number_of_antennas, self.N_UTs, self.subcarrier_spacing, self.ofdm_symbol_duration, self.reflection_coefficient, np.array([self.ut_velocities[0,]]), self.wavelength, self.Rd*2/3e8, self.AoD)
        H_BS = tf.convert_to_tensor(H_BS, dtype = tf.complex64)

        #############################################################
        #                         action mask              
        #############################################################
        if isinstance(action, str):
            if action == 'random':  # Random beam selection
                self.current_beam = np.random.choice(self.number_of_antennas, size = self.N_UTs, replace = False)
                self.action_beam = np.zeros((self.number_of_antennas))
                self.action_beam[self.current_beam] = np.arange(1, self.N_UTs+1)
                self.action_power = [1]*self.N_UTs

            elif (action == 'greedy'): # Greedy beam selection
                self.action_beam = np.zeros((self.number_of_antennas))
                for u in range(self.N_UTs):
                    self.current_beam[u,] = np.argmax(np.abs(np.transpose(np.conj(A_t[:,0,u])) @ codebook))    
                # Check for repetitive values in self.current_beam
                unique_beams, counts = np.unique(self.current_beam, return_counts=True)
                duplicates = unique_beams[counts > 1]

                p = -1
                while len(duplicates)>0:
                    p -= 1
                    for dup in duplicates:
                        indices = np.where(self.current_beam == dup)[0]
                        for idx in indices[1:]:
                            self.current_beam[idx,] = np.argsort(np.abs(np.transpose(np.conj(A_t[:,0,idx])) @ codebook))[p]
                    unique_beams, counts = np.unique(self.current_beam, return_counts=True)
                    duplicates = unique_beams[counts > 1]

                self.action_power = np.array(self.datapool!=0, dtype=np.int32)
            Pt = self.action_power/np.sum(self.action_power) * self.Pt_mW
            for u in range(self.N_UTs):
                self.action_beam[self.current_beam[u]] = u+1
            # print(self.current_beam)
                    ########### RL action ################
        else:        
            self.action_beam = np.floor(action[0:self.number_of_antennas]*(1+self.N_UTs)).astype('int')
            self.action_beam[self.action_beam == (1+self.N_UTs)] = self.N_UTs # set the illegal action to 0
            for u in range(self.N_UTs):
                self.current_beam[u,] = np.squeeze(np.argwhere(self.action_beam == u+1))
            self.action_power = np.ones(self.N_UTs)
            Pt = np.ones(self.N_UTs) * (self.datapool !=0) * (self.Pt_mW/len(self.datapool!=0))
        
        # Water-filling power allocation -- with known channel
        transmit_beam = np.zeros((self.number_of_antennas, self.N_UTs), dtype=np.complex64)
        beam_genie = np.argmax(np.abs(np.transpose(np.conj(A_t[:,0,:])) @ codebook), axis=1)
        transmit_beam = codebook[:,beam_genie]  
        transmit_beam = tf.convert_to_tensor(transmit_beam, dtype=tf.complex64) 
        AA = tf.einsum('kjom, ji -> kiom', tf.squeeze(H_UT,axis=[0,2,3]), (transmit_beam))
        xi = tf.linalg.tensor_diag_part(tf.reduce_mean(tf.square(tf.abs(AA)), axis = [-2,-1]))
        Pt = dynamic_power_allocation(self.Rd, xi, Pt, self.Pt_mW, np.ones([self.N_UTs,], dtype=np.int32), [0,2], self.N_UTs,self.datapool )
            

        #############################################################
        #               generate transmitted signal
        #############################################################
        
        # set the transmitted signal according to the action
        transmit_beam = np.zeros((self.number_of_antennas, self.N_UTs), dtype=np.complex64)
        for i in range(self.N_UTs):
            
            beam_index = self.current_beam[i].astype('int')
            if beam_index.size == 0:
                continue
            elif beam_index.size == 1:
                transmit_beam[:, i] = codebook[:, beam_index] * np.sqrt(Pt[i])
            else:
                transmit_beam[:, i] = np.sum( (codebook[:, beam_index]*(np.sqrt(Pt[i]/beam_index.size)) ), axis = 1) 
            
        transmit_beam = tf.convert_to_tensor(transmit_beam, dtype = tf.complex64) 

        # transmitted signal at the BS
        tx_sig_fd = tf.einsum('kj,nijom -> nikom', transmit_beam, self.x_rg)

        for i in range(self.N_UTs):
            temp1 = tf.einsum('kj,nijom -> nikom', tf.expand_dims(transmit_beam[:,i],1), tf.expand_dims(self.x_rg[:,:,i,:,:],2))
            
            # dedicated received signal at the UE
            temp = channel_freq( temp1, H_UT, 0)
            y_UT_eff[:,i,:,:,:] = temp[:,i,:,:,:] # received signal targeted for the UE
        
        #############################################################
        #               Received signal
        #############################################################

        # Apply the OFDM channel
        y_UT = channel_freq(tx_sig_fd, H_UT, 0)
        y_BS = channel_freq(tx_sig_fd, H_BS, self.noise_power)


        # spectral efficiency
        #y_UT_pow = tf.squeeze(tf.square(tf.abs(y_UT)))
        sig_pow = tf.squeeze(tf.square(tf.abs(y_UT_eff)))
        interference  = y_UT - y_UT_eff
        inter_pow = tf.squeeze(tf.square(tf.abs(interference)))
        
        sinr_pow = sig_pow/tf.add(inter_pow ,self.noise_power)
        # sinr_db= 10*np.log10(sinr_pow)
        # sinr_db[np.isinf(sinr_db)] = 0
        self.SE = tf.reduce_sum(tf.math.log(1+sinr_pow), axis=[-2,-1])/(self.num_ofdm_symbols*self.fft_size)

        # BER of receivers
        # ber = decode_rx_signal(y_UT, self.rg, self.N_UTs, self.encoder, self.b, self.noise_power) 
        
        # y_BS2 = tf.einsum('uj, nijom->niuom ' , tf.linalg.adjoint((transmit_beam)), y_BS)
        # y_BS2 = tf.multiply(y_BS2, tf.math.conj(self.x_rg))

        echo_bf_db = np.zeros([ self.number_of_antennas])
        num_rangebin = int(self.fft_size)*3
        num_dopplerbin = int(self.num_ofdm_symbols)*3
        num_subplots = self.N_UTs
        import matplotlib.pyplot as plt    
        for u in range(self.N_UTs):
            # Assuming self.action_beam and codebook are already TensorFlow tensors
            action_beam_mask = np.equal(self.action_beam, u + 1)
            
            AA = action_beam_mask.astype('complex64') * codebook
            coe = 1/np.sqrt(np.sum(action_beam_mask)) if np.sum(action_beam_mask) > 0 else 1
            # Compute BB using TensorFlow operations
            # BB = tf.reduce_mean(tf.square(tf.abs(tf.einsum('ij, jnp->inp', tf.linalg.adjoint(AA), tf.squeeze(y_BS)))), axis=[-2, -1]).numpy()
            # echo_bf_db[action_beam_mask] = BB[action_beam_mask]
            CC = tf.squeeze(tf.einsum('ij, jnp->inp', tf.linalg.adjoint(AA), tf.squeeze(y_BS)))
            CC = CC[action_beam_mask!=0]
            user_data = tf.broadcast_to(tf.squeeze(self.x_rg[:,:,u,]), (CC.shape))
            CC = tf.where(user_data != 0, tf.math.divide(coe*CC, user_data), CC)
            DD = tf.reduce_sum(CC, axis = 0)
            
            DD_spectrum = np.fft.fftshift(np.fft.fft2(DD, [num_dopplerbin, num_rangebin], axes=(-2, -1)))
            max_idx = np.array(np.unravel_index(np.argmax(np.abs(DD_spectrum)), DD_spectrum.shape))
            max_idx = np.array((max_idx[0] - num_dopplerbin // 2, max_idx[1] - num_rangebin // 2))
            self.R_estimate[u] = - 3e8*max_idx[1]/(2*self.subcarrier_spacing*num_rangebin)
            self.v_estiamte[u] = 3e8*max_idx[0]/(2*self.ofdm_symbol_duration*self.carrier_frequency*num_dopplerbin)

            if self.if_debugging:
                plt.subplot(num_subplots, 1, u+1)
                plt.imshow(np.square(np.abs(DD_spectrum)), aspect='auto', cmap='viridis')
                plt.colorbar()
                plt.title(f'Magnitude Spectrum {i+1}')
                plt.xlabel('Range Bin')
                plt.ylabel('Doppler Bin')
        echo_bf_db = tf.reduce_mean(tf.math.log1p(tf.square(tf.abs(tf.einsum('ij, jnp->inp', tf.linalg.adjoint(codebook), tf.squeeze(y_BS)))))/np.log(2), axis=[-2, -1]).numpy() 
        # plt.show()
        #self.memory = np.roll(self.memory, -2)
        self.memory = echo_bf_db
        # self.memory[self.memory < 0] = 0


        # normalized MSE of delay and Doppler
        self.nmse_R = np.abs(self.R_estimate - self.Rd[0,] ) / (3e8/2/self.fft_size/self.subcarrier_spacing) 
        self.nmse_v = np.abs(self.v_estiamte - self.ut_velocities[0,:,0]) / (3e8/2/self.num_ofdm_symbols/self.carrier_frequency/self.ofdm_symbol_duration) 

        # print(self.current_beam)
        #update buffer
        buffer_update = tf.cast(self.SE - 1.3 > 0, dtype=tf.int32)
        self.datapool = np.maximum(self.datapool - buffer_update, 0)
        terminated = bool(np.sum(self.datapool) == 0)

        ###################################################################
        #                       Reward Function
        ###################################################################
        #reward =   np.sum( self.SE - self.SE_threshold*np.ones(self.N_UTs) > 0 ) - np.sum(self.nmse_R) - np.sum(self.nmse_v)
        #reward =  float(np.sum( (1 + np.exp((self.total_traffic_episode - np.sum(self.datapool))/self.total_traffc_episode) ) /(1+np.exp(1.5-self.SE)) - 2) )
        if terminated:
            reward = 50
        else:
            reward = float(np.sum(  hyperbolic_func(self.SE, A1=0.8, A2=0.3, c=1.1) + (self.nmse_R<0.2) + (self.nmse_v <0.2) ) - 1* (np.sum(self.action_beam!=0) > (3*self.N_UTs))  )
            #(1 + np.exp((self.total_traffic_episode - np.sum(self.datapool))/self.total_traffic_episode) )
        
        if self.if_debugging:
            print(Pt, self.SE, self.current_beam)
            plt.figure()
            plt.plot(self.memory)
            plt.xlabel("beam index")
            print(beam_genie)
            print(reward, self.datapool)
            
            A = tf.einsum('ij, jknp -> iknp', tf.linalg.adjoint(codebook), tf.squeeze(H_BS))
            B = tf.square(tf.abs(tf.einsum('iknp, kq -> iqnp', A, codebook)))
            C = tf.reduce_mean(tf.math.log1p(tf.square(tf.abs(B))), axis=[-2,-1])
            D = tf.linalg.diag_part(C)
            import matplotlib.pyplot as plt
            plt.figure()
            plt.plot(D)
            plt.xlabel("beam index")


        if np.any(self.Rd[0,] > 100):
            truncated = True
        
        if np.isnan(reward):
            raise ValueError("NaN detected in reward")
        return self._get_state(), reward, terminated, truncated, {'buffer': self.datapool.copy(),
        'total_traffic': self.total_traffic_episode, 'nmse_R': self.nmse_R, 'nmse_v': self.nmse_v}


    def _get_state(self):
        return (np.concatenate((self.memory.flatten(), 
                                #self.action_beam, 
                                # self.R_estimate, self.v_estiamte, 
                                self.datapool
                                ), dtype = np.float32))


register(id='BeamManagementEnv-v3', entry_point=BeamManagementEnv)
