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
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError as e:
        print(e) 

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


# %%
# Parameters for ray tracing
max_depth = 2
diffraction = True

# TX and RX have directional antennas
scene.tx_array = PlanarArray(1,number_of_antennas,0.5,0.5,"tr38901","V")

scene.rx_array = PlanarArray(1,1,0.5,0.5,"tr38901","V")
# TX are installed on building_5
pos_bd5 = scene.get("building_5").position
scene.add(Transmitter("tx", position=[pos_bd5[0], pos_bd5[0]+15, 0.75 ], orientation=[np.pi,0,0]), color="red")

rx_ind = [2,4,7]
for i in range(N_UTs):
    rx = Receiver(name=f"rx{i}",
                    position=scene.get(f"car_{rx_ind[i]}").position, 
                    orientation=[0,0,0])
    scene.add(rx)
    scene.get(f"car_{rx_ind[i]}").velocity = ut_velocities[0,i,:]

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
h_dop = tf.squeeze(cir_to_ofdm_channel(frequencies, *paths.cir()))

# Visualize the scene and propagation paths

scene.render(paths=paths, camera="cam", num_samples=512)
# %%
