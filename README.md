# Updated 2024-07-04
Files in folder Multi_user BM are for the Centric project after 2024 March.
The codes are for sensing-assisted communication considering a multi-user V2X scenario.

motion.py - models vehicle's movement

channel_functions.py - includes channel modeling functions, calculating best beam, calculating AoD and the associated steering vector

beam_management_env_letter.py - gymnasium environment includes reinforcement learning environment, state, action

hueristic_test.py - implements hueristic beam management according to the AoD estimates from echos as well as genie-aided rate 

# V1
This documents the various parts of the transfer learning task. 
The documentation is subdivided into parts:

tl_config.py - is the main configuration file where all other files can be called. Parameters can be specified for each of the files here and where necessary parameters can also be updated within the files themself

FineTuning_Site_Specific.py - This function performs finetuning and feature extraction. It accepts as input the source and target domain file paths and returns the Save BLER for Finetuning, Finetuning+ and Feature extraction. 

The codes are written with site specific channel models in mind but can be easily adapted to 3GPP cahannel models.

MaMel.py - Implements the MAML algorithm and returns the BLER of the neural receiver evaluated in the target domain.

model_transfer.py - Implements a a function that trains the neural receiver on a source channel and simply evaluates on the target channel. It is a sort of benchmark for the case with zero target doamin data.

reptile.py - Implements the REPTILE meta learning algorithm which is first order MAML.

without_tl.py - A benchmark algorithm that implements the classical DNN. It can be configured for varios ratios of |Dt|/|Ds|.

plot_fig.py - This is the file for plotting the BLERs returned by the functions above. 
