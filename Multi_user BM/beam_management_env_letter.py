from channel_functions import *
import gymnasium as gym
from gymnasium.envs import register
from gymnasium import spaces
from motion import *
import random
from road import *

random.seed(1)
np.random.seed(1)

def bernoulli(p):
    return 1 if random.random() <= p else 0
    # return 1


class BeamManagementEnv(gym.Env):
    def __init__(self, N_f):
        self.N_f = N_f  # total number of channel uses
        self.N_UEs = 3
        self.r_h = []
        self.r_a = []
        self.r_p = []
        self.N_samples = 64 # number of sample per TTI
        self.number_of_antennas = 32  # number of antennas at the ULA BS
        self.th = np.zeros((1000,))
        self.th_episode = np.zeros((1000,))
        self.config = np.zeros((self.N_UEs,))
        self.action_space = spaces.Discrete(self.N_UEs * 5)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.number_of_antennas, self.N_samples))

        self.fc = 28e9  # carrier frequency (Hz)
        self.c = 3e8  # speed of light (m/s)
        self.lambda_ = self.c / self.fc  # wavelength (m)
        self.Pt = 15  # in linear scale
        self.d_ULA = np.arange(self.number_of_antennas) * self.lambda_ / 2  # ULA with half-wavelength spacing
        self.codebook = dft_codebook(self.number_of_antennas, self.number_of_antennas)  # set of beamforming directions
        self.current_slot = 0  # current slot/channel use
        self.current_beam = np.empty([self.N_UEs,], dtype = object)  # current beam direction
        self.current_traffic = np.zeros(
            (self.N_UEs, ))  # current downlink 
        self.total_traffic = np.zeros(
            (self.N_UEs, ))  # total downlink traffic
        self.failure_counter = np.zeros(
            (self.N_UEs, ))  # Consecutive failures (packet lost due to SU type or bad beam)
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
            self.current_beam[i] = np.argmax(np.abs(np.sqrt(self.Pt) * self.codebook @ self.mmWave_LoS_NLoS_channel[i]))
        
        self.power_threshold = 1
        self.seed = 1

    def reset(self):
        # initiate an episode
        self.r_h = []
        self.r_a = []
        self.r_p = []
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

        self.user_positions[0,:,:], self.config, self.if_turned = user_pos_direction_init(self.road_matrix, self.cross_pos, self.N_UEs, self.single_lane_width)

        for i in range(self.N_UEs):

            self.d[i] = np.linalg.norm(self.base_station_position - self.user_positions[
                i])  # euclidean distance between the BS and the UEs

            self.AoD[i] = compute_AoD_at_BS(self.base_station_position, self.user_positions[i])  # current AoD

            self.a_BS[i] = compute_ULA_response(self.number_of_antennas, self.AoD[i])  # current array response

            self.reflection_coefficient[i] = self.lambda_ * 5/ self.d[i]**2 / (4*np.pi)**1.5 # reflection coefficient, 5 is the RCS of a vehicle

            self.mmWave_LoS_NLoS_channel[i] = compute_mmWave_LoS_NLoS_channel(self.number_of_antennas, self.a_BS[i], self.d[i], self.lambda_)  # current LoS channel

            self.mono_LoS_channel += compute_mono_static_channel(self.number_of_antennas, self.AoD[i], self.reflection_coefficient[i])
            self.current_beam[i] = np.argmax(np.abs(np.sqrt(self.Pt) * self.codebook @ self.mmWave_LoS_NLoS_channel[i]))
        
        # TODO - the received echo signal at the BS    

        return self._get_state()

    def step(self, action): 
        '''
        Execute function
        '''
        print("action: " + str(action))
        reward = 0
        done = False
        
        self.mono_LoS_channel = np.zeros((self.number_of_antennas, self.number_of_antennas)) +1j
        self.mmWave_LoS_NLoS_channel = np.zeros((self.N_UEs, self.number_of_antennas)) + 1j

        packet = np.zeros((self.N_UEs,)).astype(float)
        was_dropped = np.zeros((self.N_UEs, 2))
        drop = 0
        for i in range(self.N_UEs):
            for j in range(2):
                self.previous_traffic[i][j] = self.current_traffic[i][j].copy() / self.dl_buffer_size

            self.previous_beam[i] = self.current_beam[i].copy()
            
            # update position
            self.user_positions[i,:], self.config[i,:], self.if_turned[i] = update_positions(self.user_positions[i,:], self.cross_pos, sample_velocity(), self.dt, self.single_lane_width, self.config[i,:], self.if_turned[i])

            self.d[i] = np.linalg.norm(self.base_station_position - self.user_positions[i])  

            self.AoD[i] = compute_AoD_at_BS(self.base_station_position, self.user_positions[i])  # current AoD

            self.a_BS[i] = compute_ULA_response(self.number_of_antennas, self.AoD[i]) 

            self.mmWave_LoS_NLoS_channel[i] = compute_mmWave_LoS_NLoS_channel(self.number_of_antennas, self.a_BS[i], self.d[i], self.lambda_)

            self.mono_LoS_channel += compute_mono_static_channel(self.number_of_antennas, self.AoD[i], self.reflection_coefficient[i])
            
            # set the transmitted signal according to the action
            if action == 0: 
                self.user_type[i] = 0
                self.current_beam[i] = self.previous_beam[i]
            elif action == 1:
                self.user_type[i] = 1
                self.current_beam[i] = np.min(self.previous_beam[i])+[-2, -1, 0]
            elif action == 2:
                self.user_type[i] = 1
                self.current_beam[i] = np.min(self.previous_beam[i])+[-1, 0, 1]
            elif action == 3:
                self.user_type[i] = 1
                self.current_beam[i] = np.max(self.previous_beam[i])+[-1, 0, 1]
            elif action == 4:
                self.user_type[i] = 1
                self.current_beam[i] = np.max(self.previous_beam[i])+[0, 1, 2]
            
            if self.current_beam[i].size == 1:
                transmit_beam = self.codebook[self.current_beam[i]]
            else:
                transmit_beam = self.codebook[self.current_beam[i]].sum(axis=0)
            
            

        return self._get_state(), reward, done, {'ue_position': self.user_positions.copy(),
                                                 'bs_position': self.base_station_position.copy(),
                                                 'lost_traffic': self.lost_traffic.copy(),
                                                 'lost_traffic_episode': self.lost_traffic_episode.copy(),
                                                 'total_traffic': self.total_traffic.copy(),
                                                 'total_traffic_episode': self.total_traffic_episode.copy(),
                                                 'failure_type': self.failure_type.copy(),
                                                 'tx_packets': self.th.copy(),
                                                 'tx_packets_episode': self.th_episode.copy()}


    def _get_state(self):
        return (self.current_beam[0] / (self.number_of_antennas - 1),
                self.current_beam[1] / (self.number_of_antennas - 1),
                self.current_beam[2] / (self.number_of_antennas - 1), self.current_traffic[0][0] / self.dl_buffer_size,
                self.current_traffic[0][1] / self.dl_buffer_size, self.current_traffic[1][0] / self.dl_buffer_size,
                self.current_traffic[1][1] / self.dl_buffer_size,
                self.current_traffic[2][0] / self.dl_buffer_size,
                self.current_traffic[2][1] / self.dl_buffer_size, self.R[0], self.R[1], self.R[2])

register(id='BeamManagementEnv-v1', entry_point=BeamManagementEnv)
