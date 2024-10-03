import sys
sys.path.append('/Users/xiyu/Documents/AAU/CENTRIC_codes/PPO_beam_management/Multi-user-BM/')
from channel_functions import *
import gymnasium as gym
from gymnasium.envs import register
import random
from scipy.linalg import hadamard
from road import *


random.seed(1)
np.random.seed(1)

def bernoulli(p):
    return 1 if random.random() <= p else 0
    # return 1


class BeamManagementEnv(gym.Env):
    def __init__(self):
        self.N_UEs = 2
        self.N_samples = 128 # number of sample per TTI
        self.number_of_antennas = 32  # number of antennas at the ULA BS
        self.th = np.zeros((1000,))
        self.num_steps = 100 # number of steps per episode
        self.config = np.zeros((self.N_UEs,))
        self.action_space = gym.spaces.MultiDiscrete(np.ones(self.N_UEs) * 5) # gym.spaces.Box(low=0, high = 1, shape=(self.N_UEs,)) #gym.spaces.Discrete(2**self.N_UEs) #
        self.observation_space = gym.spaces.Box(low=0, high=np.inf, shape=(self.N_UEs*3 + self.N_UEs *2 + 1,)) #gym.spaces.Box(low=0, high=np.inf, shape=(self.N_UEs *4 + 1,))
        self.waveform = hadamard(self.N_samples)
        self.waveform[[0,-1],:] = self.waveform[[-1,0],:]

        self.fc = 28e9  # carrier frequency (Hz)
        self.c = 3e8  # speed of light (m/s)
        self.lambda_ = self.c / self.fc  # wavelength (m)
        self.Pt_mW = (10**(15/10))*(10**(109/10))  # 15dbm in linear scale
        self.noise_power = 10**(-109/10)*10**(109/10)  # -109 dbm in linear scale, 10**(109/10) is normalization

        self.d_ULA = np.arange(self.number_of_antennas) * self.lambda_ / 2  # ULA with half-wavelength spacing
        self.codebook = dft_codebook(self.number_of_antennas, self.number_of_antennas)  # set of beamforming directions

        self.beamwidth = 2*np.arcsin(1.2/self.number_of_antennas)
        
        self.current_beam = np.empty([self.N_UEs,], dtype = object)  # current beam direction
        self.current_traffic = np.zeros(
            (self.N_UEs, ))  # current downlink 
        self.datapool = np.zeros(
            (self.N_UEs, ), dtype=np.float32)  # number of packets of each UE in the buffer
        self.total_traffic_episode = np.sum(self.datapool) # total downlink traffic per episode
        self.previous_beam = np.empty([self.N_UEs,], dtype = object)  # current beam direction
        self.previous_power = np.zeros((5, self.N_UEs))
        self.counter_sensing_slots = np.zeros((self.N_UEs), dtype=int) #Counter for sensing slots
        self.failure_type = np.zeros(3)
        self.done = False  # flag indicating whether the episode is over
        self.base_station_position = np.array([50, 50])  # BS position
        self.scenario_dimensions = np.array([0, 0, 100, 100])  # [x_min, y_min, x_max, y_max]
        self.user_positions = np.zeros((self.N_UEs, 2))
        self.dt = 0.01  # time step 10 ms

        self.d = np.zeros((self.N_UEs,))
        self.AoD = np.zeros((self.N_UEs,))
        self.aod = np.zeros((self.N_UEs,))
        self.a_BS = np.zeros((self.N_UEs, self.number_of_antennas), dtype=complex) 
        self.mmWave_LoS_NLoS_channel = np.zeros((self.N_UEs, self.number_of_antennas), dtype=complex) 
        mono_LoS_channel = np.zeros((self.number_of_antennas, self.number_of_antennas), dtype = complex) 
        self.reflection_coefficient = np.zeros((self.N_UEs,))
        self.R = np.zeros((self.N_UEs,)) # user communication rate
        self.ue_type = np.zeros((self.N_UEs,), dtype = int) # user type (SU or PU)
        
        self.echo_bf = np.zeros((self.N_UEs, self.N_samples)) # beamforming output of echo signal at the BS
        self.dice = 0 # decide the starting position

        self.crlb = np.zeros((self.N_UEs)) # CRLB estimation error

        # define the road traffic scenario
        self.cross_pos = np.array([[27, 80], [30, 70]])
        self.road_matrix = road(self.cross_pos)
        self.single_lane_width = 5
        self.user_positions = np.array([[24.,61.],[95.,30.]])
        self.config = np.array([[2,2],[1,1]])
        self.if_turned = np.array([False, False])
        #self.user_positions, self.config, self.if_turned = user_pos_direction_init(self.road_matrix, self.cross_pos, self.N_UEs, self.single_lane_width)

        self.diff_sensing_weight = np.zeros((3,self.N_UEs))
        self.est_error = np.zeros((self.N_UEs,), dtype=np.float32)


        for i in range(self.N_UEs):
            self.d[i] = np.linalg.norm(self.base_station_position - self.user_positions[
                i])  # euclidean distance between the BS and the UEs

            self.AoD[i] = compute_AoD_at_BS(self.base_station_position, self.user_positions[i])  # current AoD

            self.a_BS[i] = compute_ULA_response(self.number_of_antennas, self.AoD[i])  # current array response

            self.reflection_coefficient[i] = self.lambda_ * 5/ self.d[i]**2 / (4*np.pi)**1.5 # reflection coefficient, 5 is the RCS of a vehicle

            self.mmWave_LoS_NLoS_channel[i] = compute_mmWave_LoS_NLoS_channel(self.number_of_antennas, self.a_BS[i], self.d[i], self.lambda_)  # current LoS channel

            mono_LoS_channel += compute_mono_static_channel(self.number_of_antennas, self.AoD[i], self.reflection_coefficient[i])
            self.current_beam[i] = np.argmax(np.abs(self.codebook @ self.mmWave_LoS_NLoS_channel[i]))
        
        self.power_threshold = 5/(2*self.beamwidth**2)/(self.number_of_antennas)**4 # beamwidth*1e9 
        self.R_threshold = 2**4 -1
        self.rate_threshold = 4
    

    def reset(self, seed=None):
        super().reset(seed=seed)
        # self.th = np.zeros((1000,))
        #self.th_episode = np.zeros((1000,))
        self.current_slot = 0
        self.done = False
        self.current_beam = np.empty([self.N_UEs,], dtype=object)
        self.current_traffic = np.zeros(
            (self.N_UEs, 1))  # current downlink traffic 
        
        self.lost_traffic_episode = np.zeros((self.N_UEs, 1))  # traffic lost per episode
        self.counter_sensing_slots = np.zeros((self.N_UEs), dtype=int)  # counter for sensing slots   
        self.previous_beam = np.empty([self.N_UEs,], dtype=object)  # previsou beam direction
        self.previous_traffic = np.zeros(
            (5, self.N_UEs, 1))  # current downlink and uplink traffic 

        self.d = np.zeros((self.N_UEs,))
        self.AoD = np.zeros((self.N_UEs,))
        #self.a_BS = np.zeros((self.N_UEs, self.number_of_antennas), dtype=complex) 
        self.mmWave_LoS_NLoS_channel = np.zeros((self.N_UEs, self.number_of_antennas), dtype = complex) 
        mono_LoS_channel = np.zeros((self.number_of_antennas, self.number_of_antennas), dtype = complex) 
        self.reflection_coefficient = np.zeros((self.N_UEs,))

        ############ set limited options for initial positions ###########
        self.dice = np.random.randint(0, 5)
        if self.dice == 0:
            self.user_positions = np.array([[27., 65.],[79., 42.]])
            self.config = np.array([[4,2],[4,2]])
            self.if_turned = np.array([False, False])
        elif self.dice == 1:
            self.user_positions = np.array([[31., 39.],[59., 72.]])
            self.config = np.array([[3,3],[1,1]])
            self.if_turned = np.array([False, False])
        elif self.dice == 2:
            self.user_positions = np.array([[23., 79.],[61., 29.]])
            self.config = np.array([[3,3],[1,1]])
            self.if_turned = np.array([False, False])
        elif self.dice == 3:
            self.user_positions = np.array([[14., 33.],[75., 34.]])
            self.config = np.array([[4,2],[4,2]])
            self.if_turned = np.array([False, False])
        elif self.dice == 4:
            self.user_positions = np.array([[30., 21.],[75., 74.]])
            self.config = np.array([[2,2],[1,1]])
            self.if_turned = np.array([False, False])
        elif self.dice == 5:
            self.user_positions = np.array([[35., 73.],[64., 33.]])
            self.config = np.array([[4,2],[4,2]])
            self.if_turned = np.array([False, False])
        elif self.dice == -1: # this case is for testing
            self.user_positions, self.config, self.if_turned = user_pos_direction_init(self.road_matrix, self.cross_pos, self.N_UEs, self.single_lane_width)
        
        self.datapool = np.random.binomial(n=self.num_steps, p=0.6, size=self.N_UEs)
        self.total_traffic_episode = np.sum(self.datapool) # total downlink traffic per episode

        y_BS = np.zeros((self.number_of_antennas, self.N_samples), dtype = complex) 
        Pt = np.ones(self.N_UEs) * (self.Pt_mW/self.N_UEs)
        xi = np.zeros((self.N_UEs,))
        s_BS = np.zeros((self.N_UEs, self.number_of_antennas, self.N_samples), dtype=complex) 
        transmit_beam = np.zeros((self.N_UEs, self.number_of_antennas), dtype=complex)

        self.crlb = np.zeros((self.N_UEs))

        for i in range(self.N_UEs):

            self.d[i] = np.linalg.norm(self.base_station_position - self.user_positions[
                i])  # euclidean distance between the BS and the UEs

            self.AoD[i] = compute_AoD_at_BS(self.base_station_position, self.user_positions[i])  # current AoD

            self.a_BS[i] = compute_ULA_response(self.number_of_antennas, self.AoD[i])  # current array response

            self.reflection_coefficient[i] = self.lambda_ * 5/ self.d[i]**2 / (4*np.pi)**1.5 # reflection coefficient, 5 is the RCS of a vehicle

            self.mmWave_LoS_NLoS_channel[i] = compute_mmWave_LoS_NLoS_channel(self.number_of_antennas, self.a_BS[i], self.d[i], self.lambda_)  # current LoS channel

            mono_LoS_channel += compute_mono_static_channel(self.number_of_antennas, self.AoD[i], self.reflection_coefficient[i])
            self.current_beam[i] = np.argmax(np.abs(self.codebook @ self.mmWave_LoS_NLoS_channel[i]))
        
            transmit_beam[i,] = self.codebook[self.current_beam[i],:]
            xi[i] = np.abs(transmit_beam[i,] @ self.mmWave_LoS_NLoS_channel[i])**2 # for power allocation
            self.ue_type[i] = 1
        
        Pt = dynamic_power_allocation(self.d, xi, Pt, self.Pt_mW, self.ue_type, [self.power_threshold, self.R_threshold], self.N_UEs, self.datapool)
        
        for i in range(self.N_UEs):
            s_BS[i,] = np.sqrt(Pt[i]/np.size(self.current_beam[i]) )* np.outer(transmit_beam[i,], self.waveform[i,])
            
            derivative_a_BS = np.array([np.exp(-1j * np.pi * ll * np.cos(self.AoD[i]))*(1j * np.pi * ll * np.sin(self.AoD[i])) for ll in range(self.number_of_antennas)])

            self.crlb[i] = 0.5/ self.N_samples/ abs(np.sqrt(Pt[i])* self.reflection_coefficient[i] * self.number_of_antennas * np.dot(self.a_BS[i], transmit_beam[i,] ) )**2 / ( np.linalg.norm(derivative_a_BS)**2 - np.abs( np.dot( derivative_a_BS, self.a_BS[i]))**2 )
 


        y_BS = mono_LoS_channel @ np.sum(s_BS,axis=0) +  ( np.random.normal(0, np.sqrt(self.noise_power/2), size = (self.number_of_antennas, self.N_samples)) + 1j * np.random.normal(0, np.sqrt(self.noise_power/2), size = (self.number_of_antennas, self.N_samples)))

        self.aod = self.AoD
        self.est_error = np.abs(self.aod - self.AoD)

        self.echo_bf = np.abs(np.conj(transmit_beam) @ y_BS)**2

        # change to db
        echo_bf_db = 10*np.log10(np.mean(self.echo_bf,axis =1)/10 )
        self.memory = np.array([echo_bf_db, echo_bf_db, echo_bf_db], dtype = np.float32)     
        self.diff_sensing_weight = 1/(1+np.exp(-(self.memory[0] - self.memory[1]))) 

        return self._get_state(), {}



    def step(self, action): 
        '''
        Execute function
        '''
        reward = 0
        done = False
        xi = np.zeros((self.N_UEs,))
        transmit_beam = np.zeros((self.N_UEs, self.number_of_antennas), dtype=complex) 
        s_BS = np.zeros((self.N_UEs, self.number_of_antennas, self.N_samples), dtype=complex) 
        
        mono_LoS_channel = np.zeros((self.number_of_antennas, self.number_of_antennas), dtype = complex) 
        self.mmWave_LoS_NLoS_channel = np.zeros((self.N_UEs, self.number_of_antennas), dtype = complex)

        a_BS_est = np.zeros((self.N_UEs, self.number_of_antennas), dtype=complex)
        
        Pt = np.ones(self.N_UEs) * (self.Pt_mW/self.N_UEs)

        # action = base(2, action, self.N_UEs) # bruteforce discrete action space

        self.previous_beam = self.current_beam.copy()
        self.current_beam = np.empty([self.N_UEs,], dtype=object)
        

        for i in range(self.N_UEs):
            
            # update position
            self.user_positions[i,:], self.config[i,:], self.if_turned[i] = update_positions(self.user_positions[i,:], self.cross_pos, sample_velocity(), self.dt, self.single_lane_width, self.config[i,:], self.if_turned[i])

            self.d[i] = np.linalg.norm(self.base_station_position - self.user_positions[i])  

            self.AoD[i] = compute_AoD_at_BS(self.base_station_position, self.user_positions[i])  # current AoD

            self.reflection_coefficient[i] = self.lambda_ * 5/ self.d[i]**2 / (4*np.pi)**1.5 # reflection coefficient, 5 is the RCS of a vehicle

            self.a_BS[i] = compute_ULA_response(self.number_of_antennas, self.AoD[i]) 

            self.mmWave_LoS_NLoS_channel[i] = compute_mmWave_LoS_NLoS_channel(self.number_of_antennas, self.a_BS[i], self.d[i], self.lambda_)

            mono_LoS_channel += compute_mono_static_channel(self.number_of_antennas, self.AoD[i], self.reflection_coefficient[i])
            
            # set the transmitted signal according to the action
            if action[i] == 0: 
                self.ue_type[i] = 1
                if self.previous_beam[i].size == 1: # CU in the previous step
                    self.current_beam[i,] = self.previous_beam[i]
                else: # SU in the previous step
                    index = np.argmax(np.abs(self.codebook @ np.conj(compute_ULA_response(self.number_of_antennas, self.aod[i]))))
                    self.current_beam[i,] = index
            elif action[i] == 1:
                self.ue_type[i] = 0
                if self.previous_beam[i].size > 1:
                    temp = np.sort(self.previous_beam[i])
                    if temp[0] == 0 and temp[-1] == self.number_of_antennas - 1:
                        mask = (temp != 0) & (temp != 1)
                        self.current_beam[i,] = np.mod(np.min(temp[mask])+[-1, 0, 1], self.number_of_antennas)
                    else:
                        self.current_beam[i,] = np.mod(np.min(self.previous_beam[i])+[-1, 0, 1], self.number_of_antennas)
                else:
                    self.current_beam[i,] = np.mod(np.min(self.previous_beam[i])+[-1, 0, 1], self.number_of_antennas)

                self.counter_sensing_slots[i] += 1
            elif action[i] == 2:
                self.ue_type[i] = 0
                self.current_beam[i,] = np.mod(np.min(self.previous_beam[i])+[-2, -1, 0], self.number_of_antennas)
                self.counter_sensing_slots[i] += 1
            elif action[i] == 3:
                self.ue_type[i] = 0
                self.current_beam[i,] = np.mod(np.max(self.previous_beam[i])+[-1, 0, 1], self.number_of_antennas)
                self.counter_sensing_slots[i] += 1
            elif action[i] == 4:
                self.ue_type[i] = 0
                self.current_beam[i,] = np.mod(np.max(self.previous_beam[i])+[0, 1, 2], self.number_of_antennas)
                self.counter_sensing_slots[i] += 1
            else:
                raise ValueError('Invalid action')
            
            if self.current_beam[i].size == 1:
                transmit_beam[i,] = self.codebook[self.current_beam[i],:]
                xi[i] = np.abs(transmit_beam[i,] @ self.mmWave_LoS_NLoS_channel[i])**2
            else:
                transmit_beam[i,] = np.sqrt(1/3)*self.codebook[self.current_beam[i],:].sum(axis=0)
                xi[i] = self.reflection_coefficient[i]**2 *self.N_samples * np.abs(self.a_BS[i] @ (transmit_beam[i,]))**2
           
        # allocation power
        Pt = dynamic_power_allocation(self.d, xi, Pt, self.Pt_mW, self.ue_type, [self.power_threshold, self.R_threshold], self.N_UEs, self.datapool)

        user_signal = np.zeros((self.N_UEs, self.N_samples), dtype=complex) 
        for i in range(self.N_UEs):
            s_BS[i,] = np.sqrt(Pt[i]) * np.outer(transmit_beam[i,], self.waveform[i])

            # user targeted signal
            user_signal[i] = self.mmWave_LoS_NLoS_channel[i,] @ s_BS[i,] 

            # Calculate CRLB estimates
            a_BS_est[i,] = compute_ULA_response(self.number_of_antennas, self.AoD[i]) 
            derivative_a_BS = np.array([np.exp(-1j * np.pi * ll * np.cos(self.AoD[i]))*(1j * np.pi * ll * np.sin(self.AoD[i])) for ll in range(self.number_of_antennas)])

            self.crlb[i] = 0.5/ self.N_samples/ abs(np.sqrt(Pt[i])* self.reflection_coefficient[i] * self.number_of_antennas * np.dot(a_BS_est[i], transmit_beam[i,] ) )**2 / ( np.linalg.norm(derivative_a_BS)**2 - 2*np.real( np.dot( derivative_a_BS, a_BS_est[i])) ) #6/(np.pi**2 * abs(np.sqrt(Pt[i])* self.reflection_coefficient[i] * self.number_of_antennas )**2)/(self.number_of_antennas**2 - 1) / np.sin(self.AoD[i])**2 #
 
        # compute the user received signal, noiseless
        y_user = self.mmWave_LoS_NLoS_channel @ np.sum(s_BS,axis=0)

        # compute the echo signal at the BS
        y_BS = mono_LoS_channel @ np.sum(s_BS,axis=0) + ( np.random.normal(0, np.sqrt(self.noise_power/2), size = (self.number_of_antennas, self.N_samples)) + 1j * np.random.normal(0, np.sqrt(self.noise_power/2), size = (self.number_of_antennas, self.N_samples)))

        self.aod = estimate_AoD_from_echo(y_BS, self.waveform, self.N_UEs, self.N_samples, 1024, self.number_of_antennas)
        self.est_error = np.abs(self.aod - self.AoD)

        # compute the beamforming output at the BS
        # beamforming output
        self.echo_bf = np.abs(np.conj(transmit_beam) @ y_BS)**2
        
        echo_bf_db = 10*np.log10(np.mean(self.echo_bf, axis =1)) 
        self.memory[0] = self.memory[1]
        self.memory[1] = self.memory[2] 
        self.memory[2] = echo_bf_db # current echo bf
        self.diff_sensing_weight = 1/(1+np.exp(-(self.memory[0] - self.memory[1]))) 
        
        # compute the rate at the user
        self.R, sinr, interference_db = compute_communication_rate(y_user, user_signal, self.noise_power, self.N_UEs, self.N_samples)
        # SU user does not have traffic
        self.R = self.R * self.ue_type

        #update buffer
        self.datapool = self.datapool - ((self.R - self.rate_threshold*np.ones(self.N_UEs)) > 0)
        self.datapool[self.datapool < 0] = 0

        #reward for communication
        reward = (1 + np.exp((self.total_traffic_episode - np.sum(self.datapool))/self.total_traffic_episode) )*np.sum( self.R - self.rate_threshold*np.ones(self.N_UEs) > 0 )

        #SU
        #reward -= np.sum(normalized_error * (1-self.ue_type) )

        #reward += 0.5*np.sum((0.08 - normalized_error) *(1-self.ue_type))

        # reward uses beamforming output
        #reward = np.sum(0.75 * (sinr -12) * self.ue_type + 0.25* (echo_bf_db - 20) * (1-self.ue_type))

        # reward uses sinr and estimation error according to the objective function
        #reward = np.sum((sinr-10) * self.ue_type ) - 0.1*np.sum(interference_db-30)

        # reward consider the sensing importance
        #reward = np.sum((sinr-11) * self.ue_type) + np.sum(self.diff_sensing_weight * (1-self.ue_type) * (sinr-3) )

        # reward using hyperbolic tagent function, no penalization
        # hyper_sinr_commu = hyperbolic_func(sinr, 0.8, 0.2, 10) #snr in dB
        #hyper_sinr_sensing = hyperbolic_func(echo_bf_db,  0.03, 0.4, 16)
        # reward = np.sum(hyper_sinr_commu * self.ue_type ) #+ 0 * np.sum((1-self.ue_type) * hyper_sinr_sensing)

        # reward considers transmitted packets and echo_bf
        #reward = 3*np.sum(self.R - self.R_threshold*np.ones(self.N_UEs) > 0)
        # reward += np.sum((1-self.ue_type) * hyper_sinr_sensing)

        #reward = np.sum((1 + np.exp((self.total_traffic_episode - np.sum(self.datapool))/self.total_traffic_episode) ) * (sinr -12) * self.ue_type + 0.25 * (1-self.ue_type) * 10*np.log10(0.025*3.14/self.est_error ) ) 


        if np.sum(self.datapool) == 0:
            done = True
        truncated = False
        return self._get_state(), reward, done, truncated, {'ue_position': self.user_positions.copy(),
                                                 'current_beam': self.current_beam.copy(),
                                                 'buffer': self.datapool.copy(),
                                                 'total_traffic': self.total_traffic_episode,
                                                 'num_sensing_slots': self.counter_sensing_slots.copy(),
                                                 'sinr': sinr.copy(),
                                                 'est_error': self.est_error.copy(),
                                                 'crlb': self.crlb.copy(),
                                                 }


    def _get_state(self):
        #return (np.concatenate((self.echo_bf, self.diff_sensing_weight,self.datapool,[self.total_traffic_episode],self.ue_type)))
        #return (np.concatenate((self.echo_bf, self.diff_sensing_weight)))
        return (np.concatenate((self.memory.flatten(),self.datapool,[self.total_traffic_episode], self.ue_type), dtype = np.float32))

register(id='BeamManagementEnv-v1', entry_point=BeamManagementEnv)
