### AICloud command
```
scp -r ppo_bm fq35hc@es.aau.dk@ai-fe02.srv.aau.dk:~
ssh -l fq35hc@es.aau.dk ai-fe02.srv.aau.dk
srun hostname
srun --pty --gres=gpu:1 singularity shell --nv pytorch-24.07
```
After running, copy file from AIcloud to local
```
scp -r fq35hc@es.aau.dk@ai-fe02.srv.aau.dk:~/runs .
scp -r fq35hc@es.aau.dk@ai-fe02.srv.aau.dk:~/PPO_preTrained .
scp -r fq35hc@es.aau.dk@ai-fe02.srv.aau.dk:~/PPO_logs .
```
(. means current directly in my local machine)