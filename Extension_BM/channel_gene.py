# %%
import DeepMIMOv3
import numpy as np

parameters = DeepMIMOv3.default_params()
parameters['scenario'] = 'dyn_doppler_DD1_28'
parameters['dataset_folder'] = r'../../../../Downloads/scenarios'
#dataset = DeepMIMOv3.generate_data(parameters)
# %%

parameters['dataset_folder'] = r'/Users/xiyu/Downloads/scenarios'

parameters['dynamic_scenario_scenes'] = np.arange(10)
parameters['num_paths'] = 5
parameters['active_BS'] = 4

parameters['ue_antenna']['shape'] = np.array([1,1])
parameters['ue_antenna']['radiation_pattern'] = 'isotropic'


parameters['bs_antenna']['shape'] = np.array([16, 1])
parameters['bs_antenna']['spacing'] = 0.5
parameters['bs_antenna']['radiation_pattern'] = 'isotropic'

parameters['enable_doppler'] = True

parameters['activate_OFDM'] = 1

parameters['OFDM']['bandwidth'] = 0.05  # 50 MHz
parameters['OFDM']['subcarriers'] = 256

dataset = DeepMIMOv3.generate_data(parameters)
# %%
