import numpy as np
import matplotlib.pyplot
from stable_baselines3.common.callbacks import BaseCallback
import os
from stable_baselines3.common.results_plotter import load_results, ts2xy
from scipy import signal

def cuboid_data(center, size=(1,1,1)):
    # suppose axis direction: x: to left; y: to inside; z: to upper
    # get the (left, outside, bottom) point
    o = [a - b / 2 for a, b in zip(center, size)]
    # get the length, width, and height
    l, w, h = size
    x = np.array([[o[0], o[0] + l, o[0] + l, o[0], o[0]],  # x coordinate of points in bottom surface
         [o[0], o[0] + l, o[0] + l, o[0], o[0]],  # x coordinate of points in upper surface
         [o[0], o[0] + l, o[0] + l, o[0], o[0]],  # x coordinate of points in outside surface
         [o[0], o[0] + l, o[0] + l, o[0], o[0]]])  # x coordinate of points in inside surface
    y = np.array([[o[1], o[1], o[1] + w, o[1] + w, o[1]],  # y coordinate of points in bottom surface
         [o[1], o[1], o[1] + w, o[1] + w, o[1]],  # y coordinate of points in upper surface
         [o[1], o[1], o[1], o[1], o[1]],          # y coordinate of points in outside surface
         [o[1] + w, o[1] + w, o[1] + w, o[1] + w, o[1] + w]])    # y coordinate of points in inside surface
    z = np.array([[o[2], o[2], o[2], o[2], o[2]],                        # z coordinate of points in bottom surface
         [o[2] + h, o[2] + h, o[2] + h, o[2] + h, o[2] + h],    # z coordinate of points in upper surface
         [o[2], o[2], o[2] + h, o[2] + h, o[2]],                # z coordinate of points in outside surface
         [o[2], o[2], o[2] + h, o[2] + h, o[2]]]  )              # z coordinate of points in inside surface
    return x, y, z

def plotCubeAt(pos=(0,0,0), c="b", alpha=0.1, ax=None):
    # Plotting N cube elements at position pos
    if ax !=None:
        X, Y, Z = cuboid_data( (pos[0],pos[1],pos[2]) )
        ax.plot_surface(X, Y, Z, color=c, rstride=1, cstride=1, alpha=0.1)

def plotMatrix(ax, x, y, z, data, cmap="jet", cax=None, alpha=0.1):
    # plot a Matrix 
    norm = matplotlib.colors.Normalize(vmin=data.min(), vmax=data.max())
    colors = lambda i,j,k : matplotlib.cm.ScalarMappable(norm=norm,cmap = cmap).to_rgba(data[i,j,k]) 
    for i, xi in enumerate(x):
            for j, yi in enumerate(y):
                for k, zi, in enumerate(z):
                    plotCubeAt(pos=(xi, yi, zi), c=colors(i,j,k), alpha=alpha,  ax=ax)



    if cax !=None:
        cbar = matplotlib.colorbar.ColorbarBase(cax, cmap=cmap,
                                norm=norm,
                                orientation='vertical')  
        cbar.set_ticks(np.unique(data))
        # set the colorbar transparent as well
        cbar.solids.set(alpha=alpha)  

def expand_to_rank(x, target_rank, axis=-1):
    num_dims = np.maximum(target_rank - np.ndim(x), 0)
    shape = np.shape(x)
    new_shape = np.concatenate([shape[:axis],
                           np.ones([num_dims], np.int32),
                           shape[axis:]], 0)
    return np.reshape(x, new_shape)

def channel_freq(x, h_freq, no):
    target_rank = np.ndim(h_freq)

    x_new = expand_to_rank(x, target_rank, axis=1)
    y = np.sum(np.sum(h_freq*x_new, axis = 4), axis = 3)
    noise = np.sqrt(no/2) * (np.random.randn(*y.shape) + 1j*np.random.randn(*y.shape))
    y += noise
    return y

def music_alg(y, Nt, N_UTs):
    y = np.squeeze(y)
    N_ofdm,  N_sc = y.shape[-2], y.shape[-1]
    R = np.zeros([Nt, Nt], dtype = np.complex64)
    for i in range(N_ofdm):
        for j in range(N_sc):
            R += np.outer(y[ :, i, j], np.conj(y[ :, i, j]))
    R = R/(N_ofdm*N_sc)
    eigenvalues, eigenvectors = np.linalg.eig(R)
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:,idx]

    noise_eigenvectors = eigenvectors[:, N_UTs:]
    
    angles = np.linspace(0, 2*np.pi, 180)
    pspectrum = np.zeros(len(angles))

    for i, angle in enumerate(angles): # angles in radians
        a = np.exp(-1j * np.pi * np.arange(Nt) * np.sin(angle))
        a = a.reshape(-1, 1)  # Column vector
        
        # MUSIC spectrum formula: 1 / (a^H * En * En^H * a)
        # where En is the noise subspace eigenvector matrix
        denominator = np.dot(a.conj().T, noise_eigenvectors)
        denominator = np.dot(denominator, noise_eigenvectors.conj().T)
        denominator = np.dot(denominator, a)
        pspectrum[i] = 1 / abs(denominator.squeeze())
    
    pspectrum = pspectrum / max(pspectrum)
    peaks, _ = signal.find_peaks(pspectrum, height=0.001)
    estimated_angles = angles[peaks][:N_UTs]
    return estimated_angles

# from sionna.mimo import StreamManagement
# from sionna.ofdm import ResourceGridMapper, LSChannelEstimator, LMMSEEqualizer, RemoveNulledSubcarriers
# from sionna.fec.ldpc.encoding import LDPC5GEncoder
# from sionna.fec.ldpc.decoding import LDPC5GDecoder

# # from sionna.mapping import Mapper, Demapper
# from sionna.utils import BinarySource, ebnodb2no, QAMSource
# from sionna.utils.metrics import compute_ber
# def gen_tx_signal(rg, N_UTs, batch_size):
#     # Generate the transmitted signal in the frequency domain
#     # rg: ResourceGrid object
#     # N_UTs: Number of UTs
#     # return: tx_signal_fd: 3D array of the transmitted signal in the frequency domain
    
#     num_bits_per_symbol = 2 # QPSK modulation
#     coderate = 0.5 # The code rate
#     n = int(rg.num_data_symbols*num_bits_per_symbol) # Number of coded bits
#     k = int(n*coderate) # Number of information bits

#     # The binary source will create batches of information bits
#     binary_source = BinarySource()
#     qam_source = QAMSource(num_bits_per_symbol)

#     # The encoder maps information bits to coded bits
#     encoder = LDPC5GEncoder(k, n)

#     # The mapper maps blocks of information bits to constellation symbols
#     mapper = Mapper("qam", num_bits_per_symbol)

#     # The resource grid mapper maps symbols onto an OFDM resource grid
#     rg_mapper = ResourceGridMapper(rg)
    
#     b = binary_source([batch_size, 1, rg.num_streams_per_tx, encoder.k])
#     c = encoder(b)
#     x = mapper(c)
#     x_rg = rg_mapper(x) # batch_size, num_tx, num_streams_per_tx, num_ofdm_symbols, fft_size

#     # ebno_db=10
#     # noise_power = ebnodb2no(ebno_db, num_bits_per_symbol, coderate, rg)

#     return x_rg, encoder, b

# def decode_rx_signal(y_UT, rg, N_UTs, encoder, b, ebno_db):

#     num_streams_per_tx = N_UTs # Number of transmitted streams is equal to the number of UTs
#     rx_tx_association = np.ones([N_UTs, 1])

#     sm = StreamManagement(rx_tx_association, num_streams_per_tx,)

#     num_bits_per_symbol = 2 # QPSK modulation
#     coderate = 0.5 # The code rate
#     # This function removes nulled subcarriers from any tensor having the shape of a resource grid
#     remove_nulled_scs = RemoveNulledSubcarriers(rg)

#     ls_est = LSChannelEstimator(rg, interpolation_type="nn")

#     # The LMMSE equalizer will provide soft symbols together with noise variance estimates
#     lmmse_equ = LMMSEEqualizer(rg, sm)

#     # The demapper produces LLR for all coded bits
#     demapper = Demapper("app", "qam", num_bits_per_symbol)

#     # The decoder provides hard-decisions on the information bits
#     decoder = LDPC5GDecoder(encoder, hard_out=True)

#     no = ebnodb2no(ebno_db, num_bits_per_symbol, coderate, rg)

#     h_hat, err_var = ls_est ([y_UT, no])
#     x_hat, no_eff = lmmse_equ([y_UT, h_hat, err_var, no])
#     llr = demapper([x_hat, no_eff])
#     b_hat = decoder(llr)
#     ber = np.zeros(N_UTs)
#     for uu in range(N_UTs): 
#         ber[uu] = compute_ber(b[:,:,uu,:], b_hat[:,:,uu,:])

#     return ber

class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq:
    :param log_dir: Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
    """
    def __init__(self, check_freq: int, log_dir: str, verbose: int = 1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, "best_model")
        self.best_mean_reward = -np.inf

    # def _init_callback(self) -> None:
    #     # Create folder if needed
    #     if self.save_path is not None:
    #         os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:

          # Retrieve training reward
          x, y = ts2xy(load_results(self.log_dir), "timesteps")
          if len(x) > 0:
              # Mean training reward over the last 100 episodes
              mean_reward = np.mean(y[-100:])
              if self.verbose >= 1:
                print(f"Num timesteps: {self.num_timesteps}")
                print(f"Best mean reward: {self.best_mean_reward:.2f} - Last mean reward per episode: {mean_reward:.2f}")

              # New best model, you could save the agent here
              if mean_reward > self.best_mean_reward:
                  self.best_mean_reward = mean_reward
                  # Example for saving best model
                  if self.verbose >= 1:
                    print(f"Saving new best model to {self.save_path}")
                  self.model.save(self.save_path)

        return True
    
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gymnasium import spaces
# Define a custom feature extractor (neural network architecture)
class CustomNetwork(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Space, features_dim: int = 128):
        super(CustomNetwork, self).__init__(observation_space, features_dim)
        # Define the neural network layers
        self.net = nn.Sequential(
            nn.Linear(observation_space.shape[0], 128),  # Input layer: 40 -> 128
            nn.ReLU(),                                  # Activation function
            nn.Linear(128, 128),                        # Hidden layer: 128 -> 128
            nn.ReLU(),                                   # Activation function
        )
        # Output dimension for the feature extractor
        self.features_dim = features_dim

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.net(observations)

# Define the policy network (actor and critic)
class CustomPolicy(nn.Module):
    def __init__(self, observation_space: spaces.Space, action_space: spaces.Space):
        super(CustomPolicy, self).__init__()
        # Feature extractor (shared between actor and critic)
        self.features_extractor = CustomNetwork(observation_space)
        # Actor network (outputs mean and log standard deviation)
        self.actor_mean = nn.Linear(128, action_space.shape[0])  # 128 -> 16 (mean)
        self.actor_logstd = nn.Parameter(torch.zeros(action_space.shape[0]))  # Log std (16)
        # Critic network (outputs value function)
        self.critic = nn.Linear(128, 1)  # 128 -> 1 (value)

    def forward(self, observations: torch.Tensor) -> tuple:
        # Extract features
        features = self.features_extractor(observations)
        # Actor: compute mean and log standard deviation
        mean = self.actor_mean(features)
        log_std = self.actor_logstd
        # Critic: compute value
        value = self.critic(features)
        return mean, log_std, value