from channel_functions import *
import gymnasium as gym
from gymnasium.envs import register
from gymnasium import spaces
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
        self.N_UEs = 3
        self.N_samples = 128 # number of sample per TTI
        self.number_of_antennas = 32  # number of antennas at the ULA BS
        self.th = np.zeros((1000,))
        self.th_episode = np.zeros((1000,))
        self.config = np.zeros((self.N_UEs,))
        self.action_space = spaces.MultiDiscrete(np.ones(self.N_UEs) * 5)#spaces.Discrete(5**self.N_UEs) ##gym.spaces.Box(low=0, high = 1, shape=(self.N_UEs,)) #
        self.observation_space = gym.spaces.Box(low=0, high=np.inf, shape=(self.N_UEs * self.N_samples,))
        self.waveform = hadamard(self.N_samples)

        self.fc = 28e9  # carrier frequency (Hz)
        self.c = 3e8  # speed of light (m/s)
        self.lambda_ = self.c / self.fc  # wavelength (m)
        self.Pt_mW = (10**(15/10))*(10**(109/10))  # 15dbm in linear scale
        self.noise_power = 10**(-109/10)*10**(109/10)  # -109 dbm in linear scale, 10**(109/10) is normalization

        self.d_ULA = np.arange(self.number_of_antennas) * self.lambda_ / 2  # ULA with half-wavelength spacing
        self.codebook = dft_codebook(self.number_of_antennas, self.number_of_antennas)  # set of beamforming directions
        self.current_slot = 0  # current slot/channel use
        self.current_beam = np.empty([self.N_UEs,], dtype = object)  # current beam direction
        self.current_traffic = np.zeros(
            (self.N_UEs, ))  # current downlink 
        self.total_traffic = np.zeros(
            (self.N_UEs, ))  # total downlink traffic
        self.datapool = np.zeros(
            (self.N_UEs, ))  # number of packets of each UE in the buffer
        self.previous_beam = np.empty([self.N_UEs,], dtype = object)  # current beam direction
        self.previous_power = np.zeros((5, self.N_UEs))
        self.counter_slots_blocked = np.zeros((self.N_UEs, )).astype(int) #Counter for slots blockages
        self.previous_counter = 0
        self.failure_type = np.zeros(3)
        self.done = False  # flag indicating whether the episode is over
        self.base_station_position = np.array([50, 50])  # [x_bs, y_bs, z_bs]
        self.scenario_dimensions = np.array([0, 0, 100, 100])  # [x_min, y_min, x_max, y_max]
        self.user_positions = np.zeros((self.N_UEs, 2))
        self.dt = 0.01  # time step 10 ms

        self.d = np.zeros((self.N_UEs,))
        self.AoD = np.zeros((self.N_UEs,))
        self.a_BS = np.zeros((self.N_UEs, self.number_of_antennas)) + 1j
        self.mmWave_LoS_NLoS_channel = np.zeros((self.N_UEs, self.number_of_antennas)) + 1j
        self.mono_LoS_channel = np.zeros((self.number_of_antennas, self.number_of_antennas)) + 1j
        self.reflection_coefficient = np.zeros((self.N_UEs,))
        self.R = np.zeros((self.N_UEs,)) # user communication rate
        self.ue_type = np.zeros((self.N_UEs,)) # user type (SU or PU)
        
        self.echo_bf = np.zeros((self.N_UEs, self.N_samples)) # beamforming output of echo signal at the BS
        
        # define the road traffic scenario
        self.cross_pos = np.array([[27, 80], [30, 70]])
        self.road_matrix = road(self.cross_pos)
        self.single_lane_width = 5
        self.user_positions, self.config, self.if_turned = user_pos_direction_init(self.road_matrix, self.cross_pos, self.N_UEs, self.single_lane_width)

        for i in range(self.N_UEs):
            self.d[i] = np.linalg.norm(self.base_station_position - self.user_positions[
                i])  # euclidean distance between the BS and the UEs

            self.AoD[i] = compute_AoD_at_BS(self.base_station_position, self.user_positions[i])  # current AoD

            self.a_BS[i] = compute_ULA_response(self.number_of_antennas, self.AoD[i])  # current array response

            self.reflection_coefficient[i] = self.lambda_ * 5/ self.d[i]**2 / (4*np.pi)**1.5 # reflection coefficient, 5 is the RCS of a vehicle

            self.mmWave_LoS_NLoS_channel[i] = compute_mmWave_LoS_NLoS_channel(self.number_of_antennas, self.a_BS[i], self.d[i], self.lambda_)  # current LoS channel

            self.mono_LoS_channel += compute_mono_static_channel(self.number_of_antennas, self.AoD[i], self.reflection_coefficient[i])
            self.current_beam[i] = np.argmax(np.abs(self.codebook @ self.mmWave_LoS_NLoS_channel[i]))
        
        self.power_threshold = 2*np.arcsin(1.2/self.number_of_antennas)*1e9 # beamwidth*1e9 
        self.R_threshold = 4
    

    def reset(self):
        # self.th = np.zeros((1000,))
        self.th_episode = np.zeros((1000,))
        self.current_slot = 0
        self.done = False
        self.current_beam = np.empty([self.N_UEs,], dtype=object)
        self.current_traffic = np.zeros(
            (self.N_UEs, 1))  # current downlink traffic 
        self.total_traffic_episode = np.zeros((self.N_UEs, 1))  # total downlink traffic per episode
        self.lost_traffic_episode = np.zeros((self.N_UEs, 1))  # traffic lost per episode
        self.failure_counter = np.zeros(
            (self.N_UEs, 1))  # Consecutive failures (packet lost due to SU type or bad beam)
        self.previous_beam = np.empty([self.N_UEs,], dtype=object)  # previsou beam direction
        self.previous_traffic = np.zeros(
            (5, self.N_UEs, 1))  # current downlink and uplink traffic 
        self.previous_power = np.zeros((5, self.N_UEs))
        self.previous_counter = 0

        self.d = np.zeros((self.N_UEs,))
        self.AoD = np.zeros((self.N_UEs,))
        self.a_BS = np.zeros((self.N_UEs, self.number_of_antennas)) + 1j
        self.mmWave_LoS_NLoS_channel = np.zeros((self.N_UEs, self.number_of_antennas)) + 1j
        self.mono_LoS_channel = np.zeros((self.number_of_antennas, self.number_of_antennas)) + 1j
        self.reflection_coefficient = np.zeros((self.N_UEs,))

        self.user_positions, self.config, self.if_turned = user_pos_direction_init(self.road_matrix, self.cross_pos, self.N_UEs, self.single_lane_width)
        self.datapool = np.ones((self.N_UEs,))*10

        Pt = np.ones(self.N_UEs) * (self.Pt_mW/self.N_UEs)
        xi = np.zeros((self.N_UEs,))
        s_BS = np.zeros((self.N_UEs, self.number_of_antennas, self.N_samples)) + 1j
        transmit_beam = np.zeros((self.N_UEs, self.number_of_antennas)) + 1j

        for i in range(self.N_UEs):

            self.d[i] = np.linalg.norm(self.base_station_position - self.user_positions[
                i])  # euclidean distance between the BS and the UEs

            self.AoD[i] = compute_AoD_at_BS(self.base_station_position, self.user_positions[i])  # current AoD

            self.a_BS[i] = compute_ULA_response(self.number_of_antennas, self.AoD[i])  # current array response

            self.reflection_coefficient[i] = self.lambda_ * 5/ self.d[i]**2 / (4*np.pi)**1.5 # reflection coefficient, 5 is the RCS of a vehicle

            self.mmWave_LoS_NLoS_channel[i] = compute_mmWave_LoS_NLoS_channel(self.number_of_antennas, self.a_BS[i], self.d[i], self.lambda_)  # current LoS channel

            self.mono_LoS_channel += compute_mono_static_channel(self.number_of_antennas, self.AoD[i], self.reflection_coefficient[i])
            self.current_beam[i] = np.argmax(np.abs(self.codebook @ self.mmWave_LoS_NLoS_channel[i]))
        
            transmit_beam[i,] = self.codebook[self.current_beam[i],:]
            xi[i] = np.abs(transmit_beam[i,] @ self.mmWave_LoS_NLoS_channel[i])**2 # for power allocation
            self.ue_type[i] = 1
        
        Pt = dynamic_power_allocation(self.d, xi, Pt, self.Pt_mW, self.ue_type, [self.power_threshold, self.R_threshold], self.N_UEs, self.datapool)
        
        for i in range(self.N_UEs):
            s_BS[i,] = np.sqrt(Pt[i]/np.size(self.current_beam[i]) )* np.outer(transmit_beam[i,], self.waveform[i])

        y_BS = self.mono_LoS_channel @ np.sum(s_BS,axis=0) +  ( np.random.normal(0, np.sqrt(self.noise_power/2), size = (self.number_of_antennas, self.N_samples)) + 1j * np.random.normal(0, np.sqrt(self.noise_power/2), size = (self.number_of_antennas, self.N_samples)))

        self.echo_bf = np.abs(transmit_beam @ y_BS)**2

        return self._get_state()



    def step(self, action): 
        '''
        Execute function
        '''
        reward = 0
        done = False
        xi = np.zeros((self.N_UEs,))
        transmit_beam = np.zeros((self.N_UEs, self.number_of_antennas)) + 1j
        s_BS = np.zeros((self.N_UEs, self.number_of_antennas, self.N_samples)) + 1j
        
        self.mono_LoS_channel = np.zeros((self.number_of_antennas, self.number_of_antennas)) +1j
        self.mmWave_LoS_NLoS_channel = np.zeros((self.N_UEs, self.number_of_antennas)) + 1j
        
        Pt = np.ones(self.N_UEs) * (self.Pt_mW/self.N_UEs)

        packet = np.zeros((self.N_UEs,)).astype(float)
        was_dropped = np.zeros((self.N_UEs, 2))
        drop = 0

        #action = (abs(action)*5).astype(int)
        action = base(5, action, self.N_UEs) #125

        self.previous_beam = self.current_beam.copy()
        self.current_beam = np.empty([self.N_UEs,], dtype=object)

        for i in range(self.N_UEs):
            
            # update position
            self.user_positions[i,:], self.config[i,:], self.if_turned[i] = update_positions(self.user_positions[i,:], self.cross_pos, sample_velocity(), self.dt, self.single_lane_width, self.config[i,:], self.if_turned[i])

            self.d[i] = np.linalg.norm(self.base_station_position - self.user_positions[i])  

            self.AoD[i] = compute_AoD_at_BS(self.base_station_position, self.user_positions[i])  # current AoD

            self.a_BS[i] = compute_ULA_response(self.number_of_antennas, self.AoD[i]) 

            self.mmWave_LoS_NLoS_channel[i] = compute_mmWave_LoS_NLoS_channel(self.number_of_antennas, self.a_BS[i], self.d[i], self.lambda_)

            self.mono_LoS_channel += compute_mono_static_channel(self.number_of_antennas, self.AoD[i], self.reflection_coefficient[i])
            
            # set the transmitted signal according to the action
            if action[i] == 0: 
                self.ue_type[i] = 1
                self.current_beam[i,] = self.previous_beam[i]
            elif action[i] == 1:
                self.ue_type[i] = 0
                self.current_beam[i,] = np.mod(np.min(self.previous_beam[i])+[-2, -1, 0], 31)
            elif action[i] == 2:
                self.ue_type[i] = 0
                self.current_beam[i,] = np.mod(np.min(self.previous_beam[i])+[-1, 0, 1], 31)
            elif action[i] == 3:
                self.ue_type[i] = 0
                self.current_beam[i,] = np.mod(np.max(self.previous_beam[i])+[-1, 0, 1], 31)
            elif action[i] == 4:
                self.ue_type[i] = 0
                self.current_beam[i,] = np.mod(np.max(self.previous_beam[i])+[0, 1, 2], 31)
            else:
                raise ValueError('Invalid action')
            
            if self.current_beam[i].size == 1:
                transmit_beam[i,] = self.codebook[self.current_beam[i],:]
                xi[i] = np.abs(transmit_beam[i,] @ self.mmWave_LoS_NLoS_channel[i])**2
            else:
                transmit_beam[i,] = self.codebook[self.current_beam[i],:].sum(axis=0)
                xi[i] = self.reflection_coefficient[i]**2 *self.N_samples * np.abs(self.a_BS[i] @ transmit_beam[i,])**2

        # allocation power
        #Pt = dynamic_power_allocation(self.d, xi, Pt, self.Pt_mW, self.ue_type, [self.power_threshold, self.R_threshold], self.N_UEs, self.datapool)

        user_signal = np.zeros((self.N_UEs, self.N_samples)) + 1j
        for i in range(self.N_UEs):
            s_BS[i,] = np.sqrt(Pt[i]/self.current_beam[i].size) * np.outer(transmit_beam[i,], self.waveform[i])

            # user targeted signal
            user_signal[i] = self.mmWave_LoS_NLoS_channel[i] @ s_BS[i,:,:] 

        # compute the user received signal, noiseless
        y_user = self.mmWave_LoS_NLoS_channel @ np.sum(s_BS,axis=0)


        # compute the echo signal at the BS
        y_BS = self.mono_LoS_channel @ np.sum(s_BS,axis=0) + ( np.random.normal(0, np.sqrt(self.noise_power/2), size = (self.number_of_antennas, self.N_samples)) + 1j * np.random.normal(0, np.sqrt(self.noise_power/2), size = (self.number_of_antennas, self.N_samples)))

        # compute the beamforming output at the BS
        for i in range(self.N_UEs):
            # beamforming output
            self.echo_bf[i,] = np.abs(transmit_beam[i,:] @ y_BS)**2
        
        # compute the rate at the user
        self.R = compute_communication_rate(y_user, user_signal, self.noise_power, self.N_UEs, self.N_samples)
        # SU user does not have traffic
        self.R = self.R * self.ue_type

        #update buffer
        self.datapool = self.datapool - (self.R - self.R_threshold*np.ones(self.N_UEs) > 0)
        self.datapool[self.datapool < 0] = 0

        #reward for communication
        reward = np.sum(self.R - self.R_threshold*np.ones(self.N_UEs) > 0)

        #SU
        #reward += np.sum(np.sum(self.echo_bf, axis = 1) * (1-self.ue_type) > self.power_threshold)

        if np.sum(self.datapool) == 0:
            done = True

        return self._get_state(), reward, done, {'ue_position': self.user_positions.copy(),
                                                 'bs_position': self.base_station_position.copy(),
                                                 'current_beam': self.current_beam.copy(),
                                                 'buffer': self.datapool.copy()}


    def _get_state(self):
        return (self.echo_bf.flatten())

register(id='BeamManagementEnv-v1', entry_point=BeamManagementEnv)
