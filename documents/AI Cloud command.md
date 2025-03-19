### AICloud command
```
scp -r ppo_bm fq35hc@es.aau.dk@ai-fe02.srv.aau.dk:~
ssh -l fq35hc@es.aau.dk ai-fe02.srv.aau.dk
srun hostname
srun --pty --gres=gpu:1 singularity shell --nv pytorch-24.07
```
After running, copy file from AIcloud to local
```
#copy Tensorboard event#

scp -r fq35hc@es.aau.dk@ai-fe02.srv.aau.dk:~/runs/PPO_BeamManagementEnv-v1_run_? .


#copy preTrained model#

scp fq35hc@es.aau.dk@ai-fe02.srv.aau.dk:~/PPO_preTrained/BeamManagementEnv-v1/PPO_BeamManagementEnv-v1_0_?.pth ./PPO_bem_management/PPO_preTrained/BeamManagementEnv-v1/

scp -r fq35hc@es.aau.dk@ai-fe02.srv.aau.dk:~/PPO_preTrained/BeamManagementEnv-v1/ ./PPO_preTrained


# copy data in csv 
scp -r fq35hc@es.aau.dk@ai-fe02.srv.aau.dk:~/PPO_logs/BeamManagementEnv-v1/PPO_BeamManagementEnv-v1_log_?.csv ./PPO_bem_management/PPO-logs/BeamManagementEnv-v1/
```

scp -r fq35hc@es.aau.dk@ai-fe02.srv.aau.dk:~/PPO_logs/BeamManagementEnv-v1/ ./PPO_logs

(. means current directly in my local machine)


---
scp -r Extension_BM fq35hc@es.aau.dk@ai-fe02.srv.aau.dk:~

# install packages to virtualenv
srun singularity exec --bind ~/my-virtual-env:/my-virtual-env pytorch_24.07-py3.sif /bin/bash -c "source ~/my-virtual-env/bin/activate && pip install stable-baselines3 ...."

srun --time 08:00:00 singularity exec --bind ~/my-virtual-env:/my-virtual-env pytorch_24.07-py3.sif /bin/bash -c "source ~/my-virtual-env/bin/activate && python Extension_BM/baseline_train.py --N_UTs ? --cdir echopow --timesteps 300000"

# tensorflow 
srun --time 08:00:00 singularity exec --bind ~/tf-my-venv:/tf-my-venv tensorflow_25.01-tf2-py3.sif /bin/bash -c "source ~/tf-my-venv/bin/activate && python Extension_BM/baseline_train.py --N_UTs 2 --cdir buffer --timesteps 9000000"