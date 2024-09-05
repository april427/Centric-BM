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

scp fq35hc@es.aau.dk@ai-fe02.srv.aau.dk:~/PPO_preTrained/BeamManagementEnv-v1/PPO_BeamManagementEnv-v1_0_?.pth .

# copy data in csv -- Not necessary
scp -r fq35hc@es.aau.dk@ai-fe02.srv.aau.dk:~/PPO_logs/BeamManagementEnv-v1/PPO_BeamManagementEnv-v1_log_?.csv .
```
(. means current directly in my local machine)