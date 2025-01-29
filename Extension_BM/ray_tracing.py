import os # Configure which GPU 
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
# gpus = tf.config.list_physical_devices('GPU')
# if gpus:
#     try:
#         tf.config.experimental.set_memory_growth(gpus[0], True)
#     except RuntimeError as e:
#         print(e) 

# Avoid warnings from TensorFlow
tf.get_logger().setLevel('ERROR')

resolution = [480,320] # increase for higher quality of renderings

# Define magic cell command to skip a cell if needed
from IPython.core.magic import register_cell_magic
from IPython import get_ipython

@register_cell_magic
def skip_if(line, cell):
    if eval(line):
        return
    get_ipython().run_cell(cell)

# Set random seed for reproducibility
sionna.config.seed = 42
# %%
from sionna.rt import load_scene, PlanarArray, Transmitter, Receiver, Camera
from sionna.rt.utils import r_hat

no_preview=True
resolution = [480,320]
# %%
scene = load_scene(sionna.rt.scene.simple_street_canyon_with_cars)
scene.add(Camera("cam", position=[50,0,130], look_at=[10,0,0]))
if no_preview:
    scene.render(camera="cam", num_samples=512);

#%%
from sionna.channel.utils import subcarrier_frequencies, cir_to_ofdm_channel

import gymnasium as gym
import numpy as np
import sys
sys.path.append('/Users/xiyu/Documents/AAU/CENTRIC_codes/PPO_beam_management/Extension_BM/')
from beam_management_env_letter import BeamManagementEnv
env = gym.make('BeamManagementEnv-v1', reward_inx = 0, if_training=False)

# %%
# Parameters for ray tracing
max_depth = 2
diffraction = True
number_of_antennas = env.unwrapped.number_of_antennas
N_UTs = env.unwrapped.N_UTs
rg = env.unwrapped.rg

ut_velocities = np.zeros((1, N_UTs, 3))
ut_velocities[:,:,0:2] = np.random.normal(loc = 20, scale=2, size = (1, N_UTs, 2))
ut_velocities = tf.convert_to_tensor(ut_velocities, dtype = tf.float32)
# TX and RX have directional antennas
scene.tx_array = PlanarArray(1,number_of_antennas,0.5,0.5,"iso","VH")

scene.rx_array = PlanarArray(1,1,0.5,0.5,"iso","VH")


# scene.add(Transmitter("tx", position=[31.49, -8.65, 0.75 ], orientation=[np.pi,0,0], color=(0,1,0)))

# rx_ind = [2,7]
# for i in range(N_UTs):
#     rx = Receiver(name=f"rx{i}",
#                     position=scene.get(f"car_{rx_ind[i]}").position, 
#                     orientation=[np.pi,0,0])
#     scene.add(rx)
#     scene.get(f"car_{rx_ind[i]}").velocity = ut_velocities[0,i,:]

# only these coordinats have paths between nodes
scene.add(Receiver("rx0", position=[22.7, 5.6, 0.75], orientation=[np.pi,0,0]))
scene.add(Receiver("rx1", position=[-27.8,-4.9, 0.75]))
scene.add(Transmitter("tx", position=[31.49, -8.65, 4 ], orientation=[np.pi,0,-np.pi/4]))

# scene.remove('tx')
# scene.remove('rx0')
# scene.remove('rx1')

# %%
# Compute paths and apply Doppler shift for time evolution
paths = scene.compute_paths(max_depth=max_depth, diffraction=diffraction)
paths.normalize_delays = False
paths.apply_doppler(sampling_frequency=1/rg.ofdm_symbol_duration,
                    num_time_steps=rg.num_ofdm_symbols,
                    tx_velocities=[0,0,0],
                    rx_velocities=ut_velocities[0,:,:])
# %%
# Compute the corresponding channel frequency responses
frequencies = subcarrier_frequencies(rg.fft_size, rg.subcarrier_spacing)
h_dop = tf.squeeze(cir_to_ofdm_channel(frequencies, *paths.cir()))

# Visualize the scene and propagation paths
# %%
#scene.render(paths=paths, camera="cam", num_samples=512);

cm = scene.coverage_map(max_depth=3,           # Maximum number of ray scene interactions
                        num_samples=int(10e5), # If you increase: less noise, but more memory required
                        cm_cell_size=(2.5, 2.5),   # Resolution of the coverage map
                        cm_center=[0, 0, 0],   # Center of the coverage map
                        cm_size=[200, 200],    # Total size of the coverage map
                        cm_orientation=[0, 0, 0]) #
scene.preview(coverage_map=cm,
                     cm_metric="sinr",
                     camera="cam",
                     cm_vmin=-10,)
######################################################
# %%
scene = load_scene(sionna.rt.scene.simple_street_canyon_with_cars)
scene.add(Camera("cam", position=[50,0,130], look_at=[10,0,0]))

# Parameters for ray tracing
max_depth = 2
diffraction = True

# TX and RX have directional antennas
scene.tx_array = PlanarArray(1,16,0.5,0.5,"tr38901","VH")
scene.rx_array = PlanarArray(1,1,0.5,0.5,"tr38901","V")

# TX and RX are installed at the front of two different cars.
# The directive antennas ensure that paths reaching an antenna from the back are close to zero.
# scene.add(Transmitter("tx", position=[22.7, 5.6, 0.75], orientation=[np.pi,0,0]))
# scene.add(Receiver("rx", position=[-27.8,-4.9, 0.75]))
scene.add(Receiver("rx0", position=[22.7, 5.6, 0.75], orientation=[np.pi,0,0]))
scene.add(Receiver("rx1", position=[-27.8,-4.9, 0.75]))
scene.add(Transmitter("tx", position=[31.49, -8.65, 0.75 ], orientation=[np.pi,0,0]))


rg = env.unwrapped.rg

frequencies = subcarrier_frequencies(rg.fft_size, rg.subcarrier_spacing)

# Define a velocity vector and the corresponding displacement over the duration
# of one OFDM symbol
velocity_vec = np.array([10,0,0])
displacement_vec = velocity_vec*rg.ofdm_symbol_duration

# Assign velocity vector to cars driving in -x direction
for j in range(1,6):
    scene.get(f"car_{j}").velocity = -velocity_vec

# Assign velocity vector to cars driving in x direction
for j in range(6,9):
    scene.get(f"car_{j}").velocity = velocity_vec

# Compute paths and apply Doppler shift for time evolution
# %%
paths = scene.compute_paths(max_depth=max_depth, diffraction=diffraction)
paths.normalize_delays = False
paths.apply_doppler(sampling_frequency=1/rg.ofdm_symbol_duration,
                    num_time_steps=rg.num_ofdm_symbols,
                    tx_velocities=-velocity_vec,
                    rx_velocities=velocity_vec)

# Compute the corresponding channel frequency responses
h_dop = tf.squeeze(cir_to_ofdm_channel(frequencies, *paths.cir()))

# Visualize the scene and propagation paths
if no_preview:
    scene.render(paths=paths, camera="cam", num_samples=512);
# %%
