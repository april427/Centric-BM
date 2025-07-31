import sys, os
# sys.path.append('/Users/xiyu/Documents/AAU/CENTRIC_codes/PPO_beam_management/Extension_BM/')
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from channel_functions import *
import gymnasium as gym
from gymnasium.envs import register
from gymnasium.spaces import Box, Dict, MultiDiscrete
from exroad import *
# from types import SimpleNamespace
# import tensorflow as tf

from scipy.optimize import linear_sum_assignment
from util_functions import *
import time
# import matplotlib.pyplot as plt

class BeamManagementEnv(gym.Env):

    def __init__(self,  args):
        
        self.if_training = args.if_training
        self.if_debugging = args.if_debugging
        self.if_rician = args.if_rician

        self.N_UTs = args.N_UTs  # number of UEs 
        self.number_of_antennas = args.Nt  # number of antennas at the ULA BS
        self.reuse_beams = args.reuse_beams  # reuse beams for sweeping

        self.num_steps = 40 # number of steps per episode

        ##################################################
        #          Define action and state space
        ##################################################
        #self.action_space = Box(low =np.array([0]*self.number_of_antennas), high = np.array([1]*self.number_of_antennas),shape = (self.number_of_antennas, ))
        self.action_space = MultiDiscrete(np.ones(self.N_UTs) *4)
        #
        self.observation_space = Box(low=0, high=np.inf, shape=(2*self.number_of_antennas + 2*self.N_UTs,)) #

        self.carrier_frequency = 28e9  # carrier frequency (Hz)
        self.wavelength = 3e8/ self.carrier_frequency  # wavelength (m)

        # OFDM parameters
        self.batch_size = 1 # batch size
        self.fft_size = 144
        self.num_ofdm_symbols = 14
        self.subcarrier_spacing = 60e3
        self.cyclic_prefix_length = 20
        self.ofdm_symbol_duration = (1. + self.cyclic_prefix_length/self.fft_size) / self.subcarrier_spacing
     
        self.SE_threshold = 3  # rate threshold for each UE
        self.SE = np.zeros((self.N_UTs,))  # spectral efficiency of each UE

        self.Pt_mW = (10**(args.Ptdbm/10))*(10**(109/10))  # 15dbm in linear scale
        self.noise_power = 10**(-109/10)*10**(109/10)  # -109 dbm in linear scale, 10**(109/10) is normalization

        self.beamwidth = 2*np.arcsin(1.2/self.number_of_antennas) 
        self.crlb_a = np.zeros((self.N_UTs,))  # CRLB for AoA estimation
        
        self.current_beam = np.empty([self.N_UTs,], dtype = object)  # current beam direction

        self.max_latency = [6, 8]
        self.buffer_size = [4, 5]  # buffers size of each UE 
        if self.N_UTs == 3:
            self.max_latency = [6, 8, 10]
            self.buffer_size = [4, 5, 6]
        self.buffers = np.zeros((self.N_UTs, np.max(self.buffer_size)), dtype=np.int32)  # number of packets of each UE in the buffers
        self.total_traffic_episode = 0 # total downlink traffic per episode
        self.waiting_times = np.zeros((self.N_UTs, np.max(self.buffer_size)), dtype=np.int32)
        self.buffer_occupancy = np.zeros((self.N_UTs, ), dtype=np.int32)

        self.bs_loc = np.expand_dims(np.expand_dims(np.array([20, 50, 0], dtype=np.float32), axis=0), axis=0)   # BS position

        self.ut_loc = np.zeros((1, self.N_UTs, 3), dtype=np.float64)  # UE positions
        self.config = np.zeros((1, self.N_UTs,2 ), dtype=int)
        self.if_turned = np.zeros((1, self.N_UTs), dtype=bool)

        self.dt = args.dt  # time step 10 ms
        
        self.reflection_coefficient = np.zeros((1,self.N_UTs))
        
        self.nmse_R = np.zeros((self.N_UTs)) # normalized MSE of delay estimation
        self.nmse_v = np.zeros((self.N_UTs))

        # define the road traffic scenario
        self.cross_pos = np.array([[27, 80], [30, 70]])
        self.road_matrix = road(self.cross_pos)
        self.single_lane_width = 5

        self.arrival_probability = [max(0, 0.9 - 0.2 * i) for i in range(self.N_UTs)] # arrival probability of each UE

        # self.ut_loc[0,:,:], self.config, self.if_turned = user_pos_direction_init(self.road_matrix, self.cross_pos, self.N_UTs, self.single_lane_width)


    def reset(self, seed=None, options = None):
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
        # initial_pos_training = np.array([[[36., 33.],[24., 62.],[72., 26.],[77., 68.],[17., 29.]],
        #     [[17., 28.],[22., 66.],[83., 30.],[79., 74.],[20., 27.]],  
        #     [[36., 25.],[26., 75.],[89., 30.],[77., 64.],[31., 29.]] ,
        #     [[18., 25.],[25., 77.],[71., 32.],[78., 76.],[32., 31.]] ,
        #     [[36., 25.],[26., 64.],[86., 27.],[75., 61.],[23., 31.]] ,
        #     [[31., 25.],[24., 78.],[87., 30.],[76., 62.],[27., 26.]] ,
        #     [[30., 29.],[23., 74.],[73., 25.],[78., 79.],[30., 32.]] ,
        #     [[28., 32.],[25., 63.],[77., 25.],[77., 62.],[18., 34.]] ,
        #     [[31., 27.],[26., 67.],[74., 26.],[75., 69.],[31., 26.]] ,
        #     [[30., 34.],[22., 72.],[88., 28.],[75., 68.],[20., 34.]],
        #     [[25., 35.],[46., 67.],[ 8., 71.],[76., 17.],[95., 30.]],
        #     [[ 9., 26.],[84., 44.],[46., 74.],[80., 73.],[31,68]]] )

        initial_pos_training = np.array([[[36., 33.],[24., 62.],[17., 29.],[72., 26.],[77., 68.]],
            [[17., 28.],[22., 66.],[20., 27.],[83., 30.],[79., 74.]],  
            [[36., 25.],[26., 75.],[31., 29.],[89., 30.],[77., 64.]] ,
            [[18., 25.],[25., 77.],[32., 31.],[71., 32.],[78., 76.]] ,
            [[36., 25.],[26., 64.],[23., 31.],[86., 27.],[75., 61.]] ,
            [[31., 25.],[24., 78.],[27., 26.],[87., 30.],[76., 62.]]] )
        

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
            dice = np.random.randint(0, 6)
            self.ut_loc[0,:,0:2] = initial_pos_training[dice, 0:self.N_UTs,:]
            self.config[0,:] = initial_config[dice, 0:self.N_UTs,:]
            self.if_turned[0,:] = initial_if_turned[0:self.N_UTs]
        else:
            dice = np.random.randint(0, 3)
            self.ut_loc[0,:,0:2] = initial_pos_training[dice, 0:self.N_UTs,:]
            self.config[0,:] = initial_config[dice, 0:self.N_UTs,:]
            self.if_turned[0,:] = initial_if_turned[0:self.N_UTs]
            # self.ut_loc[0,:,:], self.config[0,:], self.if_turned[0,:] = user_pos_direction_init(self.road_matrix, self.cross_pos, self.N_UTs, self.single_lane_width) 
        # plt.imshow(self.road_matrix.T)
        # for i in range(self.N_UTs):
        #     plt.scatter(self.ut_loc[0,i,0], self.ut_loc[0,i, 1], label='UE'+str(i+1))
        # plt.scatter(self.bs_loc[0,0,0], self.bs_loc[0,0,1], label='BS')

        # set UT velocities and trajectories
        self.ut_velocities = np.zeros((1, self.N_UTs, 3))
        self.ut_velocities[0,:,0:2] = np.transpose(np.random.multivariate_normal( 6*np.arange(1,self.N_UTs+1,1), np.diag(4*np.ones([self.N_UTs])), size = (2))) 
        
        
        for i in range(self.N_UTs):

            self.Rd[0,i] = np.linalg.norm(self.bs_loc[0,0,:] - self.ut_loc[0,i,:])  # euclidean distance between the BS and the UEs

            self.AoD[i] = compute_AoD_at_BS(self.bs_loc[0, 0, :], self.ut_loc[0,i,:])  # current AoD

            self.reflection_coefficient[0,i] = self.wavelength * 10/ self.Rd[0,i]**2 / (4*np.pi)**1.5 # reflection coefficient, 5 is the RCS of vehicle

            amp_c = (self.wavelength / (4 * np.pi * self.Rd[0,i]))**2

            H_UT[0, i,0,0, ],_,_ = freqency_communication_channel(self.num_ofdm_symbols, self.fft_size, N, self.number_of_antennas, amp_c,self.Rd[0,i]/3e8, self.AoD[i], self.if_rician)
        print(self.if_rician)
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
                                             self.AoD,
                                             self.if_rician)
        
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
        
        tx_sig_fd = np.einsum('kj,nijom -> nikom', transmit_beam, self.x_rg)

        for i in range(self.N_UTs):
            self.action_beam[self.current_beam[i]] = i+1
            temp1 = np.einsum('kj,nijom -> nikom', np.expand_dims(transmit_beam[:,i], 1), np.expand_dims(self.x_rg[:,:,i,:,:], 2))
            
            # dedicated received signal at the UE
            temp = channel_freq(temp1, H_UT,0)
            y_UT_eff[:,i,:,:,:] = temp[:,i,:,:,:] # received signal targeted for the UE

        # Apply the OFDM channel
        y_UT = channel_freq(tx_sig_fd, H_UT, 0)
        # ber = decode_rx_signal(y_UT, self.rg, self.N_UTs, self.encoder, self.b, self.noise_power) 

        # Calculate spectral efficiency 
        sig_pow = np.squeeze(np.square(np.abs(y_UT_eff)))
        interference = y_UT - y_UT_eff
        inter_pow = np.squeeze(np.square(np.abs(interference)))

        sinr_pow = sig_pow / (inter_pow + self.noise_power)
        self.SE = np.sum(np.sum(np.log(1 + sinr_pow), axis=-1), axis=-1) / (self.num_ofdm_symbols * self.fft_size)
    
        y_BS = channel_freq(tx_sig_fd, H_BS, self.noise_power)

        self.music_est = music_alg(y_BS, self.number_of_antennas, self.N_UTs)
        # Order self.music_est to minimize the difference with self.AoD
        # Create a cost matrix where the cost is the absolute difference between AoD and music_est
        cost_matrix = np.abs(self.AoD[:, np.newaxis] - self.music_est[np.newaxis, :])
        # Use the Hungarian algorithm to find the optimal assignment
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        # Reorder self.music_est based on the optimal assignment
        self.music_est = self.music_est[col_ind]

        # beamformed output of the sensing signal
        y_BS2 = np.einsum('uj, nijom->niuom ' , np.conj(transmit_beam.T), y_BS)
        y_BS2 = y_BS2* np.conj(self.x_rg)

        echo_bf_db = np.zeros([self.N_UTs, self.number_of_antennas])
        for u in range(self.N_UTs):
            # Assuming self.action_beam and codebook are already TensorFlow tensors
            action_beam_mask = np.equal(self.action_beam, u + 1)
            
            AA = action_beam_mask.astype('complex64') * codebook

            # Compute BB using TensorFlow operations
            # BB = tf.reduce_mean(tf.square(tf.abs(tf.einsum('ij, jnp->inp', tf.linalg.adjoint(AA), tf.squeeze(y_BS)))), axis=[-2, -1]).numpy()
            # echo_bf_db[action_beam_mask] = BB[action_beam_mask]
        echo_bf_db = np.mean(10*np.log10(np.square(np.abs(np.einsum('ij, jnp->inp', np.conj(codebook.T), y_BS.squeeze())))), axis=(-2, -1)) 
        self.memory = echo_bf_db

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
        self.v_estimate = 3e8 * indices[:,0]/(2 * self.ofdm_symbol_duration * self.carrier_frequency * num_dopplerbin)
        self.old_R_estimate = self.R_estimate
        self.old_v_estimate = self.v_estimate
        
        self.nmse_R = np.abs(self.R_estimate - self.Rd[0,] ) / (3e8/2/self.fft_size/self.subcarrier_spacing)
        self.nmse_v = np.abs(self.v_estimate - self.ut_velocities[0,:,0]) / (3e8/2/self.num_ofdm_symbols/self.carrier_frequency/self.ofdm_symbol_duration)

        mbuser = np.argmax(self.buffer_size)
        mb = np.max(self.buffer_size)
        self.buffers = np.zeros((self.N_UTs, mb), dtype=np.int32)  # number of packets of each UE in the buffers
        for i in range(self.N_UTs):
            self.buffers[i, self.buffer_size[i]:] = -1

        self.waiting_times = np.zeros((self.N_UTs, np.max(self.buffer_size)), dtype=np.int32)
        self.buffer_occupancy = np.zeros((self.N_UTs, ), dtype=np.int32)

        # packets arrival at the first time slot
        self.buffers[:,0] = np.random.binomial(size = self.N_UTs, n=1, p = self.arrival_probability) # packets arrival
        # For each user with a new arrival, add packet to first empty buffer slot
        self.buffer_occupancy = np.sum(self.buffers>0, axis=1)
        

        self.total_traffic_episode = np.sum(self.buffers[self.buffers>0]) # total downlink traffic per episode
        self.successful_transmission = 0  # number of successful transmission
        
        return self._get_state(), {}


######################################################################
#                             step function              
######################################################################

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
        ch_alpha = np.zeros([self.N_UTs], dtype=np.float32)  # channel gain for each UE
        ch_steer = np.zeros([self.number_of_antennas, self.N_UTs], dtype=np.complex64) # channel steering vector for each UE
        
        y_UT_eff = np.zeros([self.batch_size, self.N_UTs, 1, self.num_ofdm_symbols, self.fft_size], dtype = np.complex64)
        
        # channel_freq = ApplyOFDMChannel(add_awgn=True)
        
        self.ut_velocities[0,:,0:2] = np.transpose(np.random.multivariate_normal(6*np.arange(1,self.N_UTs+1,1), np.diag(4*np.ones([self.N_UTs])), size = (2)))
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
            # np.random.choice([True, False], size=self.N_UTs)
        self.Rd[0,:] = np.linalg.norm(self.bs_loc[0,0,:] - self.ut_loc[0,:,:], axis=1)
        self.AoD = np.array([compute_AoD_at_BS(self.bs_loc[0, 0, :], 
                                          self.ut_loc[0,i,:]) for i in range(self.N_UTs)])
        self.reflection_coefficient[0,:] = (self.wavelength * 10 / 
                                      (self.Rd[0,:]**2) / ((4*np.pi)**1.5))
        for i in range(self.N_UTs):

            amp_c = (self.wavelength / (4 * np.pi * self.Rd[0,i]))**2

            H_UT[0, i,0,0, ], ch_steer[:,i] , ch_alpha[i] = freqency_communication_channel(self.num_ofdm_symbols, self.fft_size, 1, self.number_of_antennas, amp_c, self.Rd[0,i]/3e8, self.AoD[i],self.if_rician)
        
    
        H_BS, A_t = freqency_sensing_channel(self.batch_size, self.num_ofdm_symbols, self.fft_size , self.number_of_antennas, self.N_UTs, self.subcarrier_spacing, self.ofdm_symbol_duration, self.reflection_coefficient, np.array([self.ut_velocities[0,]]), self.wavelength, self.Rd*2/3e8, self.AoD, self.if_rician)

        beam_reuse_checker = np.zeros((self.N_UTs, self.number_of_antennas), dtype = bool)  # check if the beam is reused
        #############################################################
        #                         action mask              
        #############################################################
        if isinstance(action, str):
            if action == 'random':  # Random beam selection
                self.current_beam = np.random.choice(self.number_of_antennas, size = self.N_UTs, replace = False)
                self.action_beam = np.zeros((self.number_of_antennas))
                self.action_beam[self.current_beam] = np.arange(1, self.N_UTs+1)
                self.action_power = [1]*self.N_UTs
            
            elif action == 'other_work':
                scale = 1/(2*np.sqrt(2**self.SE_threshold))  # normalized power allocation
                delta_scale = 0.3
                k1 = np.sqrt((2**self.SE_threshold - 1)*self.noise_power) / ch_alpha[0]
                k2 = np.sqrt(self.noise_power*(2**self.SE_threshold -1)) / ch_alpha[1]
                mat_C = ch_steer.T 
                mat_E = (mat_C @ np.conj(mat_C).T)**(-1)
                while np.abs(delta_scale) > 0.1:
                    scale = (np.sqrt(scale) + delta_scale)**2
                    vec_b = np.array([np.sqrt((2**self.SE_threshold - 1)*self.noise_power/ scale / ch_alpha[0]), np.sqrt(self.noise_power*(2**self.SE_threshold -1) / ch_alpha[1]/ (np.abs(1-scale)*(2**self.SE_threshold)) ) ])

                    delta_b = np.array([-k1/scale, k2*(2**self.SE_threshold)*np.sqrt(scale)* ((1- scale*(2**self.SE_threshold)**(-1.5)))])

                    delta_b_2 = np.array([2*k1/scale**(1.5), ((k2*(2**self.SE_threshold)*(1-scale)* 2**self.SE_threshold )**(1.5) - 2**3.6 * scale * 1.5*k2 *np.abs(1-scale) * (2**self.SE_threshold))/(np.abs(1-scale)* (2**self.SE_threshold))**(1.5) ])

                    delta_scale = - (delta_b @ np.conj(mat_E) @ delta_b.T)/(delta_b @np.conj(mat_E) @ delta_b.T + delta_b_2@np.conj(mat_E)@vec_b.T  ) 
                    
                    delta_scale = np.real(delta_scale)
                    if np.isnan(scale) or np.isnan(delta_scale):
                        delta_scale = 0.1
                        scale = 1/(2*np.sqrt(2**3))
                scale = (np.sqrt(scale) + delta_scale)**2
                vec_u = np.conj(mat_C).T @ mat_E @ vec_b.T
                
                vec_u_norm = vec_u / np.linalg.norm(vec_u)


            elif (action == 'genie' or 'estimated'): # Greedy beam selection
                self.action_beam = np.zeros((self.number_of_antennas))
                if action == 'genie':
                    for u in range(self.N_UTs):
                        self.current_beam[u,] = np.argmax(np.abs(np.transpose(np.conj(A_t[:,0,u])) @ codebook))
                else:
                    if np.size(self.music_est) ==   1:
                        self.music_est = np.tile(self.music_est, self.N_UTs)
                    A_t_est = (1/np.sqrt(self.number_of_antennas)) * np.exp(1j* -np.pi * np.outer( np.sin(self.music_est), np.arange(self.number_of_antennas))).T  
                    for u in range(self.N_UTs):
                        self.current_beam[u,] = np.argmax(np.abs(np.transpose(np.conj(A_t_est[:,u])) @ codebook))  
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

                self.action_power = np.ones(self.N_UTs)
            Pt = self.action_power/np.sum(self.action_power) * self.Pt_mW
            for u in range(self.N_UTs):
                self.action_beam[self.current_beam[u]] = u+1
            
                ############### RL action ################
        else:        
            # self.action_beam = np.floor(action[0:self.number_of_antennas]*(1+self.N_UTs)).astype('int')
            # self.action_beam[self.action_beam == (1+self.N_UTs)] = self.N_UTs # set the illegal action to 0
            # for u in range(self.N_UTs):
            #     self.current_beam[u,] = np.squeeze(np.argwhere(self.action_beam == u+1))
            self.action_beam = np.zeros((self.number_of_antennas))
            self.action_power = np.ones(self.N_UTs)
            Pt = np.ones(self.N_UTs) * (self.Pt_mW/self.N_UTs)
            
            for i in range(self.N_UTs):
                if action[i] == 0: # beam stays the same
                    self.current_beam[i] = self.current_beam[i]
                    beam_reuse_checker[i, self.current_beam[i]] = True

                elif action[i] == 1: # beam is selected from multiple
                    if self.current_beam[i].size == 1:
                        self.current_beam[i,] = self.current_beam[i]
                    else:
                        index = np.argmax(self.memory[self.current_beam[i]])
                        self.current_beam[i,] = self.current_beam[i][index]
                    beam_reuse_checker[i, self.current_beam[i]] = True
                
                elif action[i] == 2: # beam is turned left
                    temp = np.sort(self.current_beam[i]) if self.current_beam[i].size > 1 else [self.current_beam[i]]
                    if self.reuse_beams==2:  #multi beams sweeping
                        beamspan = [-2, -1, 0]
                    elif self.reuse_beams==1:  # single beam sweeping
                        beamspan = -1   
                    if temp[0] == 0 and temp[-1] == self.number_of_antennas - 1:
                        if temp[1] == 1:
                            self.current_beam[i,] = np.mod(np.max(temp)+beamspan, self.number_of_antennas)
                        else:
                            mask = (temp != 0) & (temp != self.number_of_antennas - 1)
                            self.current_beam[i,] = np.mod(np.min(temp[mask])+beamspan, self.number_of_antennas)
                    else:
                        self.current_beam[i,] = np.mod(np.min(self.current_beam[i])+beamspan, self.number_of_antennas)
                    
                    beam_reuse_checker[i, self.current_beam[i]] = True  # mark the beam as reused
                    # deal with reused beams
                    if np.any(self.action_beam[self.current_beam[i,]] != 0):  # Check if the beam is already used
                        count = np.sum(self.action_beam[self.current_beam[i,]] != 0)
                        # Find the index of the first unused beam that is closest to the current beam
                        used_beams = []
                        for j in range(i+1):
                            if isinstance(self.current_beam[j], np.ndarray):
                                used_beams.extend(self.current_beam[j].flatten())
                            else:
                                used_beams.append(self.current_beam[j])
                        used_beams = np.unique(used_beams)
                        unused_beam = np.setdiff1d(np.arange(self.number_of_antennas), used_beams)
                        if self.reuse_beams == 2:
                            self.current_beam[i][np.where(self.action_beam[self.current_beam[i,]] != 0)[0]] = unused_beam[0:count]
                        elif self.reuse_beams == 1:
                            self.current_beam[i] = unused_beam[0:count]

                elif action[i] == 3: # beam is turned right
                    temp = np.sort(self.current_beam[i]) if self.current_beam[i].size > 1 else [self.current_beam[i]] 
                    if self.reuse_beams == 2:
                        beamspan = [0, 1, 2]
                    elif self.reuse_beams == 1:
                        beamspan = 1
                    if temp[0] == 0 and temp[-1] == self.number_of_antennas - 1:
                        if temp[-2] == self.number_of_antennas - 2:
                            self.current_beam[i,] = np.mod(np.min(temp)+beamspan, self.number_of_antennas)
                        else:
                            mask = (temp != 0) & (temp != self.number_of_antennas - 1)
                            self.current_beam[i,] = np.mod(np.max(temp[mask])+beamspan, self.number_of_antennas)
                    else:
                        self.current_beam[i,] = np.mod(np.max(self.current_beam[i])+beamspan, self.number_of_antennas)
                    
                    beam_reuse_checker[i, self.current_beam[i]] = True
                    # deal with reused beams
                    if np.any(self.action_beam[self.current_beam[i,]] != 0):  # Check if the beam is already used
                        count = np.sum(self.action_beam[self.current_beam[i,]] != 0)
                        # Find the index of the first unused beam that is closest to the current beam
                        used_beams = []
                        for j in range(i+1):
                            if isinstance(self.current_beam[j], np.ndarray):
                                used_beams.extend(self.current_beam[j].flatten())
                            else:
                                used_beams.append(self.current_beam[j])
                        used_beams = np.unique(used_beams)
                        unused_beam = np.setdiff1d(np.arange(self.number_of_antennas), used_beams)
                        if self.reuse_beams == 2:
                            self.current_beam[i][np.where(self.action_beam[self.current_beam[i,]] != 0)[0]] = unused_beam[0:count]
                        elif self.reuse_beams == 1:
                            self.current_beam[i] = unused_beam[0:count]

                elif action[i] == 4: # beam is expanded to the adjacent two beams
                    self.current_beam[i,] = np.mod(self.current_beam[i]+[-1, 0, 1], self.number_of_antennas) # for TDMA

                elif action[i] == 5:
                    a_t = (1/np.sqrt(self.number_of_antennas)) * np.exp(1j* -np.pi * np.outer( np.sin(self.music_est[i]), np.arange(self.number_of_antennas))).T 
                    index = np.argmax(np.abs(np.transpose(np.conj(a_t)) @ codebook), axis=1).squeeze()
                    if np.isnan(self.crlb_a[i]):
                        num_beams = 1
                    else:
                        num_beams = math.ceil(np.sqrt(self.crlb_a[i]) / self.beamwidth/2)
                    if num_beams > 3 :
                        num_beams = 3
                    temp = range(index - num_beams, index + num_beams)
                    self.current_beam[i,] = np.unique(np.mod(temp, self.number_of_antennas) )
                    if np.any(self.action_beam[self.current_beam[i,]] != 0):  # Check if the beam is already used
                        count = np.sum(self.action_beam[self.current_beam[i,]] != 0)
                        # Find the index of the first unused beam that is closest to the current beam
                        # First find the beams already used by users 0 to i-1
                        used_beams = []
                        for j in range(i+1):
                            if isinstance(self.current_beam[j], np.ndarray):
                                used_beams.extend(self.current_beam[j].flatten())
                            else:
                                used_beams.append(self.current_beam[j])
                        used_beams = np.unique(used_beams)
                        unused_beam = np.setdiff1d(np.arange(self.number_of_antennas), used_beams)
                        if self.reuse_beams == 2:
                            self.current_beam[i][np.where(self.action_beam[self.current_beam[i,]] != 0)[0]] = unused_beam[0:count]
                        elif self.reuse_beams == 1:
                            self.current_beam[i] = unused_beam[0:count]
                
                self.action_beam[self.current_beam[i]] = i+1
 
        
        # Water-filling power allocation -- with known channel
        transmit_beam = np.zeros((self.number_of_antennas, self.N_UTs), dtype=np.complex64)
        beam_genie = np.argmax(np.abs(np.transpose(np.conj(A_t[:,0,:])) @ codebook), axis=1)
        transmit_beam = codebook[:,beam_genie]  
        
        AA = np.einsum('kjom, ji -> kiom', H_UT.squeeze(), transmit_beam)
        xi = np.diag(np.mean(np.square(np.abs(AA)), axis = (-2,-1)))
        Pt = dynamic_power_allocation(self.R_estimate, xi, Pt, self.Pt_mW, np.ones([self.N_UTs,], dtype=np.int32), [0,3], self.N_UTs,[1,1] )
        if any(Pt == 0):
            Pt = np.ones(self.N_UTs) * (self.Pt_mW/self.N_UTs)    

        #############################################################
        #               generate transmitted signal
        #############################################################
        
        # set the transmitted signal according to the action
        transmit_beam = np.zeros((self.number_of_antennas, self.N_UTs), dtype=np.complex64)
        for i in range(self.N_UTs):
            if isinstance(action, str) and action == 'other_work':
                if scale < 0:
                    Pt = np.array([scale, 1-scale]) * (self.Pt_mW)
                else:
                    Pt = np.ones(self.N_UTs) * (self.Pt_mW/self.N_UTs)
                transmit_beam[:, i] = vec_u_norm * np.sqrt(Pt[i])
            else:
                beam_index = self.current_beam[i].astype('int')
                if beam_index.size == 0:
                    continue
                elif beam_index.size == 1 and isinstance(beam_index, np.int64):
                    transmit_beam[:, i] = codebook[:, beam_index] * np.sqrt(Pt[i])
                else:
                    transmit_beam[:, i] = np.sum( (codebook[:, beam_index]*(np.sqrt(Pt[i]/beam_index.size)) ), axis = 1) 
        
        # transmitted signal at the BS
        tx_sig_fd = np.einsum('kj,nijom -> nikom', transmit_beam, self.x_rg)

        for i in range(self.N_UTs):
            temp1 = np.einsum('kj,nijom -> nikom', np.expand_dims(transmit_beam[:,i],1), np.expand_dims(self.x_rg[:,:,i,:,:],2))
            
            # dedicated received signal at the UE
            temp = channel_freq( temp1, H_UT, 0)
            y_UT_eff[:,i,:,:,:] = temp[:,i,:,:,:] # received signal targeted for the UE
        

        # calculate the received sensing signal
        # su_idx = np.where(self.buffer_occupancy == 0)[0]
        # temp_sensing = np.einsum('uj, nijom->niuom', transmit_beam[:,su_idx], self.x_rg[:,:,su_idx,:,:])
        # y_UT_sensing = channel_freq(temp_sensing, H_UT, 0)

        #############################################################
        #               Received signal
        #############################################################

        # Apply the OFDM channel
        y_UT = channel_freq(tx_sig_fd, H_UT, 0)
        y_BS = channel_freq(tx_sig_fd, H_BS, self.noise_power)
        
        previous_est = self.music_est
        if np.any(np.isnan(y_BS)):
            print('vec_u_norm is NaN, setting to random')
        self.music_est = music_alg(y_BS, self.number_of_antennas, self.N_UTs)
        # Create a cost matrix where the cost is the absolute difference between AoD and music_est
        cost_matrix = np.abs(previous_est[:, np.newaxis] - self.music_est[np.newaxis, :])
        # Use the Hungarian algorithm to find the optimal assignment
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        # Reorder self.music_est based on the optimal assignment
        self.music_est = self.music_est[col_ind]

        # spectral efficiency
        #y_UT_pow = tf.squeeze(tf.square(tf.abs(y_UT)))
        sig_pow = np.squeeze(np.square(np.abs(y_UT_eff)))
        interference  = y_UT - y_UT_eff 
        inter_pow = np.squeeze(np.square(np.abs(interference)))
        
        sinr_pow = sig_pow/(inter_pow + self.noise_power)
        # sinr_db= 10*np.log10(sinr_pow)
        # sinr_db[np.isinf(sinr_db)] = 0
        self.SE = np.sum(np.sum(np.log2(1 + sinr_pow), axis=-1), axis=-1)/(self.num_ofdm_symbols*self.fft_size)

        if isinstance(action, str) and action == 'other_work':
            ue_inx = np.argmax(Pt)
            self.SE[ue_inx] = np.log2(1 + (1-scale)* self.Pt_mW* ch_alpha[ue_inx] * np.abs(np.inner(vec_u_norm, ch_steer[:,ue_inx]))**2  / (self.noise_power + scale*ch_alpha[ue_inx]*self.Pt_mW*(np.abs(np.inner(vec_u_norm, ch_steer[:,ue_inx]))**2)))
            ue_inx = np.argmin(Pt)
            self.SE[ue_inx] = np.log2(1 + scale* self.Pt_mW* ch_alpha[ue_inx] * np.abs(np.inner(vec_u_norm, ch_steer[:,ue_inx]))**2  / self.noise_power )

        #############################################################
        #                      Echo SINR
        #############################################################        
        echo_sinr = np.zeros(self.N_UTs)
        for i in range(self.N_UTs):
            h_bs_vec = channel_freq(tx_sig_fd, H_BS, 0 ).squeeze()
            noise_free_sig = np.einsum('j, jpq -> pq', transmit_beam[:,i].conj(), h_bs_vec)
            eff_sig = self.reflection_coefficient[0,i] * np.abs(np.vdot(transmit_beam[:,i],  A_t[:,0,i]))**2
            interference = noise_free_sig - eff_sig
            echo_sinr[i] = np.sum((np.abs( eff_sig)**2 / Pt[i]) / (np.abs(interference)**2 / Pt[i] + self.noise_power), axis = (-2, -1) )
            # noise_free_sig = np.einsum('ijk, i -> jk', transmit_beam[:,i].conj().T * H_BS.squeeze(),  transmit_beam[:,i])
            # eff_sig = transmit_beam[:,i].conj().T * A_t[:,0,i] * A_t[:,0,i].conj().T * transmit_beam[:,i]
            # interference = (noise_free_sig - eff_sig)
            # echo_sinr[i] = np.abs(eff_sig)**2/Pt[i] / (np.abs(interference)**2/Pt[i] + self.noise_power)
        crlb_R = 3* (3e8**2)/(8* echo_sinr * (np.pi* self.subcarrier_spacing)**2 * (self.fft_size**2 -1)) # CRLB for range estimation
        crlb_v = 3 * self.wavelength **2 /(8 * echo_sinr * (np.pi *  self.ofdm_symbol_duration)**2 * (self.num_ofdm_symbols**2 - 1) ) # CRLB for velocity estimation
        self.crlb_a = 6 * self.wavelength**2 / ( echo_sinr * (np.pi*self.wavelength* np.cos(self.AoD))**2 * (self.number_of_antennas**2 -1) ) # CRLB for AoA estimation
        
        self.crlb_a_est = 6 * self.wavelength**2 / ( echo_sinr * (np.pi*self.wavelength* np.cos(self.music_est))**2 * (self.number_of_antennas**2 -1) )

        self.old_R_estimate = self.R_estimate
        self.old_v_estimate = self.v_estimate

        echo_bf_db = np.zeros([ self.number_of_antennas])
        num_rangebin = int(self.fft_size)*3
        num_dopplerbin = int(self.num_ofdm_symbols)*3
        num_subplots = self.N_UTs
        import matplotlib.pyplot as plt    
        for u in range(self.N_UTs):
            # Assuming self.action_beam and codebook are already TensorFlow tensors
            self.action_beam[self.current_beam[u]] = u+1
            action_beam_mask = np.equal(self.action_beam, u + 1)
            
            AA = action_beam_mask.astype(np.complex64) * codebook
            coe = 1/np.sqrt(np.sum(action_beam_mask)) if np.sum(action_beam_mask) > 0 else 1

            CC = np.einsum('ij,jnp->inp', np.conj(AA.T), y_BS.squeeze())
            CC = CC[action_beam_mask != 0]
            user_data = np.broadcast_to(np.squeeze(self.x_rg[:,:,u,]), CC.shape)
            CC = np.where(user_data != 0, np.divide(coe * CC, user_data), CC)
            # DD = np.sum(CC, axis=0)
            # DD_spectrum = np.fft.fftshift(np.fft.fft2(DD, [num_dopplerbin, num_rangebin], axes=(-2, -1)))
            # max_idx = np.array(np.unravel_index(np.argmax(np.abs(DD_spectrum)), DD_spectrum.shape))
            # max_idx = np.array((max_idx[0] - num_dopplerbin // 2, max_idx[1] - num_rangebin // 2))

            DD_spectrum = np.fft.fftshift(np.fft.fft2(CC, [num_dopplerbin, num_rangebin], axes=(-2, -1)))
            if DD_spectrum.size == 0:
                print('Warning: DD_spectrum is empty for user', u)
            max_idx_temp = np.array(np.unravel_index(np.argmax(np.abs(DD_spectrum)), DD_spectrum.shape))
            max_idx = np.array((max_idx_temp[1] - num_dopplerbin // 2, max_idx_temp[2] - num_rangebin // 2))

            self.R_estimate[u] = - 3e8*max_idx[1]/(2*self.subcarrier_spacing*num_rangebin)
            self.v_estimate[u] = 3e8*max_idx[0]/(2*self.ofdm_symbol_duration*self.carrier_frequency*num_dopplerbin)

            # if self.if_debugging:
            #     plt.subplot(num_subplots, 1, u+1)
            #     plt.imshow(np.square(np.abs(DD_spectrum)), aspect='auto', cmap='viridis')
            #     plt.colorbar()
                
            #     plt.xticks(np.arange(0, num_rangebin, num_rangebin//6), np.arange(-num_rangebin//2, num_rangebin//2, num_rangebin//6)*3e8//(2*self.subcarrier_spacing*num_rangebin))
            #     plt.yticks(np.arange(0, num_dopplerbin, num_dopplerbin//6), np.arange(-num_dopplerbin//2, num_dopplerbin//2, num_dopplerbin//6)*3e8//(2*self.ofdm_symbol_duration*self.carrier_frequency*num_dopplerbin))
            #     plt.xlabel('Range')
            #     plt.ylabel('Doppler')
        echo_bf_db = np.mean(10*np.log10(np.square(np.abs(np.einsum('ij, jnp->inp', np.conj(codebook.T), y_BS.squeeze())))), axis=(-2, -1)) 

        self.memory = echo_bf_db

        # normalized MSE of delay and Doppler
        self.nmse_R = np.abs(self.R_estimate - self.Rd[0,] ) / (3e8/2/self.fft_size/self.subcarrier_spacing) 
        self.nmse_v = np.abs(self.v_estimate - self.ut_velocities[0,:,0]) / (3e8/2/self.num_ofdm_symbols/self.carrier_frequency/self.ofdm_symbol_duration)

        self.angle_error = np.abs(self.music_est - self.AoD)/(np.pi/self.number_of_antennas) # radius 

        ###################################################################
        #                       update buffers
        ###################################################################
        self.latency = -1*np.ones((self.N_UTs, ), dtype=np.int32)
        drop_overload = 0
        drop_latency = 0
        reward_transmission = np.zeros((self.N_UTs, ), dtype=np.float32)
        success_packet = np.multiply((self.SE>=3),(self.buffer_occupancy!=0)) 

        # reward_estimation1 = (np.sum(self.nmse_R[self.buffer_occupancy!=0] < 0.3) + np.sum(self.nmse_v[self.buffer_occupancy!=0] < 0.3)).astype('float32')
        # reward_estimation2 = (np.sum(self.nmse_R[self.buffer_occupancy==0] < 0.15) + np.sum(self.nmse_v[self.buffer_occupancy==0] < 0.2)).astype('float32')

        reward_estimation1 = (np.sum(np.log( self.nmse_R[self.buffer_occupancy!=0])/np.log(1/3)) + np.sum(np.log(self.nmse_v[self.buffer_occupancy!=0])/np.log(1/3))).astype('float32')
        reward_estimation2 = (np.sum( np.log(self.nmse_R[self.buffer_occupancy==0])/np.log(1/3)) + np.sum( np.log(self.nmse_v[self.buffer_occupancy==0])/np.log(1/3))).astype('float32')

        for user_id in np.where(success_packet != 0)[0]:
            pak_inx = np.argmax(self.waiting_times[user_id,])
            self.latency[user_id] = self.waiting_times[user_id, pak_inx]
            reward_transmission[user_id] = self.max_latency[user_id] - self.latency[user_id]
            self.buffers[user_id, pak_inx] = 0 # transmit this packet
            self.waiting_times[user_id, pak_inx] = 0
            self.buffer_occupancy[user_id] -= 1

        for u in range(self.N_UTs):
            for i in range(self.buffer_size[u]):
                if self.buffers[u, i] != 0:
                    self.waiting_times[u,i] += 1

                if self.waiting_times[u,i] >= self.max_latency[u]:
                    self.waiting_times[u,i] = 0
                    self.buffers[u,i] = 0
                    self.buffer_occupancy[u] -= 1
                    drop_latency += 1

        ###################################################################
        #                       Reward Function
        ###################################################################
        #reward_transmission = np.sum(hyperbolic_func(self.SE[success_packet],A1=4.9,A2=0.2, c=1.0).astype('float32'))
        # reward_transmission = - np.sum(self.SE[self.buffer_occupancy!=0]<1.3).astype('float32')

        reward_drop = - (drop_overload + drop_latency)
        #reward_estimation = - (np.sum(self.nmse_R>0.5) + np.sum(self.nmse_v >0.2)).astype('float32')

        reward_jump = - np.sum(np.abs(self.old_R_estimate - self.R_estimate)/ (3e8/2/self.fft_size/self.subcarrier_spacing)  > 2) - np.sum(np.abs(self.old_v_estimate - self.v_estimate)/ (3e8/2/self.num_ofdm_symbols/self.carrier_frequency/self.ofdm_symbol_duration) > 2)
        
        reward = float( np.sum(reward_transmission) - drop_latency + reward_estimation1 + reward_estimation2 + 0.5* reward_jump )# + np.sum(self.SE) ) #

        # packets arrival
        arrivals = np.random.binomial(size = self.N_UTs, n=1, p = self.arrival_probability) 
        for user_id in range(self.N_UTs):
            if arrivals[user_id] == 1 and self.buffer_occupancy[user_id] < self.buffer_size[user_id]:
                # Find first empty slot
                for i in range(self.buffer_size[user_id]):
                    if self.buffers[user_id, i] == 0:
                        self.buffers[user_id, i] = 1
                        self.waiting_times[user_id, i] = 0
                        self.buffer_occupancy[user_id] += 1
                        break
            elif arrivals[user_id] == 1 and self.buffer_occupancy[user_id] > self.buffer_size[user_id]:
                drop_overload += 1
        reward -= drop_overload  # penalize for overload

        # self.buffers = np.maximum(self.buffers - success_packet - drop_latency - drop_overload + arrivals, 0)
        self.successful_transmission += np.sum(success_packet)

        terminated = bool(self.cstep == self.num_steps) 
        if not terminated:
            self.total_traffic_episode += np.sum(arrivals)

        
        if self.if_debugging:
            print('Action =', action)
            print('Pt =', Pt, self.SE, self.current_beam)
            plt.plot(echo_bf_db)
            
            # plt.xlabel("beam index")
            # plt.ylabel("Echo power")
            print('GenieBeam =', beam_genie)
            print('reward =', reward, 'Buffer =', self.buffers)
            print('WaitingTime = ', self.waiting_times)
            print(self.buffer_occupancy, np.max(self.waiting_times, axis=1), reward_transmission)
            
            # A = np.einsum('ij, jknp -> iknp', np.conj(codebook.T), H_BS.squeeze())
            # B = np.square(np.abs(np.einsum('iknp, kq -> iqnp', A, codebook)))
            # C = np.mean(np.log1p(np.square(np.abs(B))), axis=(-2, -1))
            # D = np.diagonal(C)
            # import matplotlib.pyplot as plt
            # plt.plot(D)
            # plt.xlabel("beam index")
            # plt.ylabel("Pathloss")

        if np.any(self.Rd[0,] > 100):
            truncated = True
        
        info = {'buffers': self.buffers.copy(),
                'total_traffic': self.total_traffic_episode, 
                'transmitted_packets': self.successful_transmission, 
                'latency': self.latency, 
                'nmse_R': self.nmse_R, 
                'nmse_v': self.nmse_v,
                'angle_error': np.sqrt(self.crlb_a_est)/ (np.pi/self.number_of_antennas),
                'SE': self.SE,
                'allocated_beams': self.action_beam,
                'genie_beams': beam_genie,
                }

        if np.isnan(reward):
            raise ValueError("NaN detected in reward")
        return self._get_state(), reward, terminated, truncated, info


    def _get_state(self):
        return (np.concatenate(((self.memory-np.min(self.memory)).flatten(), 
                                self.action_beam, 
                                #self.AoD, self.ut_velocities[0,: , 0], 
                                self.buffer_occupancy,
                                np.max(self.waiting_times, axis=1)
                                ), dtype = np.float32))


register(id='BeamManagementEnv-v4', entry_point=BeamManagementEnv)
