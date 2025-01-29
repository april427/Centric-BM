import sys
sys.path.append('/Users/xiyu/Documents/AAU/CENTRIC_codes/PPO_beam_management/Extension_BM/')
from channel_functions import *
import gymnasium as gym
from gymnasium.envs import register
from gymnasium.spaces import Box, Dict, MultiDiscrete
import random
from scipy.linalg import hadamard
from exroad import *
import tensorflow as tf
from itertools import product

from sionna.mimo import StreamManagement
from sionna.ofdm import ResourceGrid
from sionna.channel import ApplyOFDMChannel
from util_functions import *


class BeamManagementEnv(gym.Env):
    def generate_actions(self, num_users):
        """ Generate a set of actions that sum to 1 for the given number of users. """
        # Use a fixed granularity for simplicity (e.g., fractions of 0.1)
        granularity = 4  # 6 discrete steps for each user
        action_space = []

        for combination in product(range(granularity + 1), repeat=num_users):
            total = sum(combination)
            if total == granularity:
                # Convert counts to fractions
                action_space.append([x / granularity for x in combination])
        
        return np.array(action_space)

    def __init__(self,  if_training = 1):
        self.if_training = if_training

        self.N_UTs = 2
        self.number_of_antennas = 32  # number of antennas at the ULA BS

        self.num_steps = 100 # number of steps per episode
        self.config = np.zeros((self.N_UTs,))

        ##################################################
        #          Define action and state space
        ##################################################
        self.action_space = MultiDiscrete([self.N_UTs +1 ]*self.number_of_antennas)  #Allocate each beam to a user, 0 denotes a non-allocated beam
        #self.action_space = Box(low =0, high = 1, shape = (2*self.number_of_antennas, ))
        self.observation_space = Box(low=0, high=np.inf, shape=(self.N_UTs*3 ,)) 

        self.carrier_frequency = 28e9  # carrier frequency (Hz)
        self.wavelength = 3e8/ self.carrier_frequency  # wavelength (m)

        # OFDM parameters
        scenario = "rma"
        direction = "downlink"
        self.batch_size = 1 # batch size
        self.fft_size = 128
        subcarrier_spacing = 30e3
        num_streams_per_tx = self.N_UTs # Number of transmitted streams is equal to the number of UTs
        rx_tx_association = np.ones([self.N_UTs, 1])
        sm = StreamManagement(rx_tx_association, num_streams_per_tx,)
        #  resource grid
        self.rg = ResourceGrid(num_ofdm_symbols=14, 
                  fft_size=self.fft_size,
                  subcarrier_spacing=subcarrier_spacing,
                  num_tx = 1, # one BS
                  num_streams_per_tx=num_streams_per_tx, 
                  cyclic_prefix_length=20,
                  pilot_ofdm_symbol_indices=[2,11],
                  pilot_pattern="kronecker"
                 )

        self.SE_threshold = 8  # rate threshold for each UE
        self.SE = np.zeros((self.N_UTs,))  # spectral efficiency of each UE

        self.Pt_mW = (10**(15/10))*(10**(109/10))  # 15dbm in linear scale
        self.noise_power = 10**(-109/10)*10**(109/10)  # -109 dbm in linear scale, 10**(109/10) is normalization

        self.codebook = dft_codebook(self.number_of_antennas, self.number_of_antennas)  # set of beamforming directions

        self.beamwidth = 2*np.arcsin(1.2/self.number_of_antennas) 
        
        self.current_beam = np.empty([self.N_UTs,], dtype = object)  # current beam direction

        self.datapool = np.zeros(
            (self.N_UTs, ), dtype=np.float32)  # number of packets of each UE in the buffer
        self.total_traffic_episode = np.sum(self.datapool) # total downlink traffic per episode
        self.previous_beam = np.empty([self.N_UTs,], dtype = object)  # current beam direction
        self.previous_power = np.zeros((5, self.N_UTs))
        self.counter_sensing_slots = np.zeros((self.N_UTs), dtype=int) #Counter for sensing slots

        self.bs_loc = tf.expand_dims(tf.expand_dims(tf.constant([50, 50, 0], dtype = tf.float32),axis=0),axis=0)  # BS position

        self.ut_loc = np.zeros((self.batch_size, self.N_UTs, 3))
        self.config = np.zeros((self.batch_size, self.N_UTs,2 ), dtype=int)
        self.if_turned = np.zeros((self.batch_size, self.N_UTs), dtype=bool)

        self.dt = 0.01  # time step 10 ms

        self.Rd = np.zeros((1,self.N_UTs))
        self.AoD = np.zeros((self.N_UTs))
        self.aod = np.zeros((self.N_UTs))
        self.a_BS = np.zeros((self.N_UTs, self.number_of_antennas), dtype=complex) 
        
        self.reflection_coefficient = np.zeros((1,self.N_UTs))
        
        self.nmse_R = np.zeros((self.N_UTs)) # normalized MSE of delay estimation
        self.nmse_v = np.zeros((self.N_UTs))

        self.crlb = np.zeros((self.N_UTs)) # CRLB estimation error

        # define the road traffic scenario
        self.cross_pos = np.array([[27, 80], [30, 70]])
        self.road_matrix = road(self.cross_pos)
        self.single_lane_width = 5

        # self.ut_loc[0,:,:], self.config, self.if_turned = user_pos_direction_init(self.road_matrix, self.cross_pos, self.N_UTs, self.single_lane_width)
        
        self.num_rangebin = int(self.fft_size)*3
        self.num_dopplerbin = int(self.rg.num_ofdm_symbols)*3
        self.num_anglebin = self.number_of_antennas


    def reset(self, seed=None):
        super().reset(seed=seed)

        self.cstep = 0
        self.done = False
        self.current_beam = np.empty([self.N_UTs,], dtype=object)

        self.counter_sensing_slots = np.zeros((self.N_UTs), dtype=int)  # counter for sensing slots   
        self.previous_beam = np.empty([self.N_UTs,], dtype=object)  # previsou beam direction

        self.Rd = np.zeros((1,self.N_UTs))
        self.AoD = np.zeros((self.N_UTs))
        
        channel_freq = ApplyOFDMChannel(add_awgn=False)

        N = 1 # consider one path
        H_UT = np.zeros([self.batch_size, self.N_UTs,1,1, self.number_of_antennas, self.rg.num_ofdm_symbols, self.fft_size], dtype=np.complex64) #[ batch size, num_users, num_subcarriers, num_BS_antennas ]


        ############ set limited options for initial positions ###########
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

        if self.if_training:
            dice = np.random.randint(0, 11)
            self.ut_loc[0,:,0:2] = initial_pos_training[dice, 0:self.N_UTs,:]
            self.config[0,:] = initial_config[dice, 0:self.N_UTs,:]
            self.if_turned[0,:] = initial_if_turned[0:self.N_UTs]
        else:

            self.ut_loc[0,:,:], self.config[0,:], self.if_turned[0,:] = user_pos_direction_init(self.road_matrix, self.cross_pos, self.N_UTs, self.single_lane_width) 

        # set UT velocities and trajectories
        self.ut_velocities = np.zeros((self.batch_size, self.N_UTs, 3))
        self.ut_velocities[:,:,0:2] = np.random.normal(loc = 20, scale=2, size = (self.batch_size, self.N_UTs, 2)) 
        
        self.datapool = np.random.binomial(n=self.num_steps, p=0.6, size=self.N_UTs)
        self.total_traffic_episode = np.sum(self.datapool) # total downlink traffic per episode

        Pt = np.ones(self.N_UTs) * (self.Pt_mW/self.N_UTs)

        
        for i in range(self.N_UTs):

            self.Rd[0,i] = np.linalg.norm(self.bs_loc[0,0,:] - self.ut_loc[0,i,:])  # euclidean distance between the BS and the UEs

            self.AoD[i] = compute_AoD_at_BS(self.bs_loc.numpy()[0, 0, :], self.ut_loc[0,i,:])  # current AoD

            self.reflection_coefficient[0,i] = self.wavelength * 5/ self.Rd[0,i]**2 / (4*np.pi)**1.5 # reflection coefficient, 5 is the RCS of a vehicle

            amp_c = (self.wavelength / (4 * np.pi * self.Rd[0,i]))**2

            H_f,_,_ = freqency_communication_channel(self.rg.num_ofdm_symbols, self.fft_size, N, self.number_of_antennas, amp_c,self.Rd[0,i]/3e8, self.AoD[i])
            
            H_UT[0, i,0,0, ] = H_f

        H_UT = tf.convert_to_tensor(H_UT, dtype = tf.complex64)
        
        H_BS = tf.zeros([self.batch_size, 1 , self.number_of_antennas, 1, self.number_of_antennas, self.rg.num_ofdm_symbols, self.fft_size])
        H_BS, A_t = freqency_sensing_channel(self.batch_size, self.rg.num_ofdm_symbols, self.fft_size , self.number_of_antennas, self.N_UTs, self.rg.subcarrier_spacing, self.rg.ofdm_symbol_duration, self.reflection_coefficient, np.array([self.ut_velocities[0,]]), self.wavelength, self.Rd*2/3e8, self.AoD)
        H_BS = tf.convert_to_tensor(H_BS, dtype = tf.complex64) 

        x_rg, encoder, b = gen_tx_signal(self.rg, self.N_UTs, self.batch_size)
        tx_sig_shape = list(x_rg.shape) #[batch_size, num_tx, num_tx_antennas, num_ofdm_symbols, fft_size]
        tx_sig_shape[2] = self.number_of_antennas
        tx_sig_fd = tf.zeros(tx_sig_shape, dtype=x_rg.dtype)

        # the first beam is set with known AoDs

        for u in range(self.N_UTs):
            self.current_beam[u,] = np.argmax(np.abs(np.transpose(np.conj(A_t[:,0,u])) @ self.codebook))

        beam_index = self.current_beam.astype('int')
        transmit_beam = self.codebook[:,beam_index] * np.diag(np.sqrt(Pt))
        tx_sig_fd = tf.einsum('kj,nijom -> nikom', transmit_beam, x_rg)

        # Apply the OFDM channel
        y_UT = channel_freq([tx_sig_fd, H_UT])
        ber = decode_rx_signal(y_UT, self.rg, self.N_UTs, encoder, b, 10) 
    
        y_BS = channel_freq([tx_sig_fd, H_BS])

        # beamformed output of the sensing signal
        y_BS2 = np.einsum('uj, nijom->niuom ' , np.transpose(np.conj((self.codebook[:,beam_index]))), y_BS)
        y_BS2 = tf.multiply(y_BS2, tf.math.conj(x_rg[0,]))

        z_BS = np.fft.fft2(y_BS2, [self.num_dopplerbin, self.num_rangebin])
        top = np.abs(z_BS[0,0,u,:,:]).ravel()
        indices = np.zeros([self.N_UTs, 2])
        for u in range(self.N_UTs):
            indices[u,:] = np.concatenate(np.unravel_index(np.argsort(top, axis=None)[-1:], z_BS[0,0,u,:,:].shape)).astype('int')
            # fft shift
            indices[u,0] = indices[u,0] if indices[u,0]<np.ceil(self.num_dopplerbin/2) else indices[u,0] - self.num_dopplerbin
            indices[u,1] = indices[u,1] if indices[u,1]<np.ceil(self.num_rangebin/2) else indices[u,1] - self.num_rangebin

        self.R_estimate = - 3e8*indices[:,1]/(2*self.rg.subcarrier_spacing * self.num_rangebin)
        self.v_estiamte = 3e8 * indices[:,0]/(2 * self.rg.ofdm_symbol_duration * self.carrier_frequency * self.num_dopplerbin)
        
        return self._get_state(), {}



    def step(self, action): 
        '''
        Execute function
        '''
        reward = 0
        done = False 
        self.cstep += 1
        
        H_UT = np.zeros([self.batch_size, self.N_UTs,1,1, self.number_of_antennas, self.rg.num_ofdm_symbols, self.fft_size], dtype=np.complex64) #[ batch size, num_users, num_subcarriers, num_BS_antennas ]
        H_BS = tf.zeros([self.batch_size, 1 , self.number_of_antennas, 1, self.number_of_antennas, self.rg.num_ofdm_symbols, self.fft_size])
        
        channel_freq = ApplyOFDMChannel(add_awgn=False)

        self.previous_beam = self.current_beam.copy()
        self.current_beam = np.empty([self.N_UTs,], dtype=object)
        x_rg, encoder, b = gen_tx_signal(self.rg, self.N_UTs, self.batch_size)

        #update velocities and positions
        self.ut_velocities[:,:,0:2] = np.random.normal(loc = 20, scale=2, size = (self.batch_size, self.N_UTs, 2))
        for s in range(1, self.batch_size):
            for i in range(self.N_UTs):
                self.ut_loc[s, i,:], self.config[s,i], self.if_turned[s,i] = update_positions(self.ut_loc[s,i,:], self.cross_pos, np.array([self.ut_velocities[s,i,0]]), 0.1, self.single_lane_width, self.config[s,i], self.if_turned[s,i])

        for i in range(self.N_UTs):

            self.Rd[0,i] = np.linalg.norm(self.bs_loc[0,0,:] - self.ut_loc[0,i,])  

            self.AoD[i] = compute_AoD_at_BS(self.bs_loc.numpy()[0, 0, :], self.ut_loc[0,i,])  # current AoD

            self.reflection_coefficient[0,i] = self.wavelength * 5/ (self.Rd[0,i]**2) / ((4*np.pi)**1.5) # reflection coefficient, 5 is the RCS of a vehicle

            amp_c = (self.wavelength / (4 * np.pi * self.Rd[0,i]))**2

            H_UT[0, i,0,0, ],_,_ = freqency_communication_channel(self.rg.num_ofdm_symbols, self.fft_size, 1, self.number_of_antennas, amp_c, self.Rd[0,i]/3e8, self.AoD[i])

        H_UT = tf.convert_to_tensor(H_UT, dtype = tf.complex64)    
        H_BS, A_t = freqency_sensing_channel(self.batch_size, self.rg.num_ofdm_symbols, self.fft_size , self.number_of_antennas, self.N_UTs, self.rg.subcarrier_spacing, self.rg.ofdm_symbol_duration, self.reflection_coefficient, np.array([self.ut_velocities[0,]]), self.wavelength, self.Rd*2/3e8, self.AoD)
        H_BS = tf.convert_to_tensor(H_BS, dtype = tf.complex64)
        ##
        tx_sig_fd_ut = np.zeros([self.batch_size, self.N_UTs, self.number_of_antennas, self.rg.num_ofdm_symbols, self.fft_size], dtype=np.complex64)
        y_UT_eff = np.zeros([self.batch_size, self.N_UTs, 1, self.rg.num_ofdm_symbols, self.rg.fft_size], dtype = np.complex64)

        # set the transmitted signal according to the action
        transmit_beam = np.zeros((self.number_of_antennas, self.N_UTs), dtype=np.complex64)
        for i in range(self.N_UTs):
            self.current_beam[i,] = np.squeeze(np.argwhere(action == i))
            beam_index = self.current_beam[i].astype('int')
            transmit_beam[:, i] = np.sum(self.codebook[:, beam_index], axis = 1) if beam_index.size > 1 else self.codebook[:, beam_index]
            transmit_beam[:, i] = transmit_beam[:, i] / np.linalg.norm(transmit_beam[:, i])     # normalize the beamforming vector
            
            tx_sig_fd_ut[:,i,:,:,:] = tf.einsum('kj,nijom -> nikom', tf.expand_dims(transmit_beam[:,i],1), tf.expand_dims(x_rg[:,:,i,:,:],2))
            # dedicated received signal at the UE
            temp = channel_freq([ tf.expand_dims(tx_sig_fd_ut[:,i,:,:,:],1), H_UT])
            y_UT_eff[:,i,:,:,:] = temp[:,i,:,:,:] # received signal targeted for the UE
        

        # transmitted signal at the BS
        tx_sig_fd = tf.einsum('kj,nijom -> nikom', transmit_beam, x_rg)
            
        # Apply the OFDM channel
        y_UT = channel_freq([tx_sig_fd, H_UT])
        y_BS = channel_freq([tx_sig_fd, H_BS])

        # spectral efficiency
        y_UT_pow = np.squeeze(np.square(np.abs(y_UT)))
        sig_pow = np.squeeze(np.square(np.abs(y_UT_eff)))

        inter_pow = y_UT_pow - sig_pow
        inter_pow[inter_pow<0] = 0
        sinr_pow = (sig_pow/(inter_pow + self.noise_power))
        self.SE = np.sum(np.sum(np.log2(1+sinr_pow), axis=-1), axis = -1)/(self.rg.num_ofdm_symbols*self.rg.fft_size)
        
        # beamformed output of the sensing signal
        y_BS2 = np.einsum('uj, nijom->niuom ' , np.transpose(((transmit_beam))), y_BS)
        y_BS2 = tf.multiply(y_BS2, tf.math.conj(x_rg))
        
        z_BS = np.fft.fft2(y_BS2, [self.num_dopplerbin, self.num_rangebin])

        # Doppler and range estimation
        indices = np.zeros([self.N_UTs, 2])
        R_estimate = np.zeros(self.N_UTs)
        v_estiamte = np.zeros(self.N_UTs)
        for u in range(self.N_UTs):
            top = np.abs(z_BS[0,0,u,:,:]).ravel()
            indices[u,:] = np.concatenate(np.unravel_index(np.argsort(top, axis=None)[-1:], z_BS[0,0,u,:,:].shape)).astype('int')
            # fft shift
            indices[u,0] = indices[u,0] if indices[u,0]<np.ceil(self.num_dopplerbin/2) else indices[u,0]-self.num_dopplerbin
            indices[u,1] = indices[u,1] if indices[u,1]<np.ceil(self.num_rangebin/2) else indices[u,1]-self.num_rangebin

            R_estimate[u] = - 3e8*indices[u,1]/(2*self.rg.subcarrier_spacing*self.num_rangebin)
            v_estiamte[u] = 3e8*indices[u,0]/(2*self.rg.ofdm_symbol_duration*self.carrier_frequency*self.num_dopplerbin) 

        # normalized MSE of delay and Doppler
        self.nmse_R = np.abs(R_estimate - self.Rd[0,] ) / (3e8/2/self.rg.fft_size/self.rg.subcarrier_spacing)
        self.nmse_v = np.abs(v_estiamte - self.ut_velocities[0,:,0]) / (3e8/2/self.rg.num_ofdm_symbols/self.carrier_frequency/self.rg.ofdm_symbol_duration)

        #update buffer
        self.datapool = self.datapool - ((self.SE - self.SE_threshold*np.ones(self.N_UTs)) > 0)
        self.datapool[self.datapool < 0] = 0

        # reward
        reward = (1 + np.exp((self.total_traffic_episode - np.sum(self.datapool))/self.total_traffic_episode) ) * np.sum( self.SE - self.SE_threshold*np.ones(self.N_UTs) > 0 ) - np.sum(self.nmse_R) - np.sum(self.nmse_v)
###########################################################



        # #reward for communication
        # if self.reward_inx == 0:
        #     reward = (1 + np.exp((self.total_traffic_episode - np.sum(self.datapool))/self.total_traffic_episode) )*np.sum( self.R - self.SE_threshold*np.ones(self.N_UTs) > 0 )
        # elif self.reward_inx == 1:
        #     reward = (1 + np.exp((self.total_traffic_episode - np.sum(self.datapool))/self.total_traffic_episode) ) * np.sum( self.R - self.SE_threshold*np.ones(self.N_UTs) > 0 ) + 0.1*np.sum( (1-self.ue_type) * 10*np.log10(self.beamwidth*2*np.pi/180/self.est_error ) )
        # elif self.reward_inx == 2:
        #     hyper_sinr_commu = hyperbolic_func(sinr, 0.8, 0.2, 10) #snr in dB
        #     hyper_sinr_sensing = hyperbolic_func(echo_bf_db,  0.03, 0.4, 1)
        #     reward = np.sum(hyper_sinr_commu * self.ue_type ) + 0.25 * np.sum((1-self.ue_type) * hyper_sinr_sensing)
        # elif self.reward_inx == 3:
        #     # reward using hyperbolic tagent function, no penalization
        #     hyper_sinr_commu = hyperbolic_func(sinr, 0.8, 0, 10) #snr in dB
        #     hyper_sinr_sensing = hyperbolic_func(echo_bf_db,  0.03, 0, 1)
        #     reward = np.sum(hyper_sinr_commu * self.ue_type ) + 0.25 * np.sum((1-self.ue_type) * hyper_sinr_sensing)
        # elif self.reward_inx == 4:
        #     # reward uses beamforming output
        #     reward = np.sum(0.75 * (sinr -12) * self.ue_type + 0.25* (echo_bf_db - 20) * (1-self.ue_type))
        # elif self.reward_inx == 5:
        #             # reward considers transmitted packets and echo_bf
        #     hyper_sinr_sensing = hyperbolic_func(echo_bf_db,  0.03, 0.4, 1)
        #     reward = 3*np.sum(self.R - self.R_threshold*np.ones(self.N_UTs) > 0)
        #     reward += np.sum((1-self.ue_type) * hyper_sinr_sensing)
  

        if np.sum(self.datapool) == 0:
            done = True
        truncated = False
        return self._get_state(), reward, done, truncated, {
                                                 }


    def _get_state(self):
        return (np.concatenate(( self.SE, self.nmse_R, self.nmse_v), dtype = np.float32))
    
    

register(id='BeamManagementEnv-v1', entry_point=BeamManagementEnv)
