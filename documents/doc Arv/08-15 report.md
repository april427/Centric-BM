
---
A BS optimizes the resource management in a multi-user communication network.
The BS decides the user type, beams, and power level for all the users.

### Buffer
We consider the buffer by assuming that packets for a user are generated before a frame which contains $N$ TTIs. Let $[B_{1,0}, \cdots, B_{U,0}]$ be the vector of packet number in the buffer of each user in the beginning of the frame. 

---
### Objective function

**Changed the objective function**
- We change the objective function to be
$$
\begin{alignat}{2}
&\arg \max_{\lambda, \boldsymbol{f}, P} && \sum_{n=1}^{N} \sum_{u=1}^U \left(\lambda_{u,n}\mathbf{1}_{(B_{u,n} > 0)} R_{u,n} + w(1-\lambda_{u,n}) S_{u,n}  \right) \nonumber \\
&\mathrm{subject~to} ~~ &&\sum_{u=1}^U P_{u,n} \leq P \nonumber \\
& &&\lambda_{u,n} \in\{0,1\} ~~ \text{user's state: 1=CU, 0= SU}\nonumber\\
\end{alignat}
$$
where $w$ is the weight for sensing and will affect the policy. We set $w=1$ to begin with. The indicator function $\mathbf{1}_{(B_{u,n} > 0)}$ implies that if the buffer is empty, transmitting is not needed anymore.
The sensing part is also an indicator function where
$$
S_{u,n} = \boldsymbol{1}_{(\text{var}(\hat{\varphi}_{u,n}) \le th)} .
$$
The communication part $R_{u,n}$, indicating if a packet went through, is defined as
$$
\begin{align}
& R_{u,n} = \mathbf{1}_{(\log (1 + \gamma_{u,n})> c) \cap (\text{var}(\hat{\varphi}_u) \leq th) }
 \\ 
= & \begin{cases} 1, (\log (1 + \gamma_{u,n})> c) \cap (\text{var}(\hat{\varphi}_{u,n-1}) \leq th)\\
0, (\text{var}(\hat{\varphi}_{u,,n-1}) > th) \cup \big((\log (1 + \gamma_{u,n})<c)\cap(\text{var}(\hat{\varphi}_{u,,n-1}) \le th )\big)
\end{cases}
\end{align}
$$

> Here in the communication part, estimation precision is regarding to the previous TTI, based on which the BS decides user types.  

---
### Power allocation

To simplify the machine learning model, we consider that BS decides the user type and user beams. After the action is taken, the BS calculates the power allocation.

The indicator functions in the sensing and communication parts:
- the SNR $\xi_c = \frac{|\boldsymbol{h}_u ^H \boldsymbol{f}_{u,n} |^2}{\sigma_{\omega}^2} \ge \iota_c$, 
- the signal strength of the echo signal $\xi_s = \frac{ \beta^2_{q,n} \big|\boldsymbol{a}^H (\varphi_q) \boldsymbol{f}_{q,n}\big|^2 \Delta T }{\sigma_{\omega}^2} \ge \iota_s$, where $\iota_s, \iota_c$ are the threshold of sensing and communication respectively.
After the action during TTI $n$ is determined, the objective function with respect to power allocation can be written as
$$
\arg \max_{P_u} \sum \boldsymbol{1}_{P_u \frac{\xi_u}{\iota_{u}} \ge 1} 
$$
This converts the problem to be maximizing the number of users who's signal strength is larger than the threshold. 
It can be solved by finding the weakest user and satisfying its requirement. The other users' power are allocated according to their channel conditions' ratio. 
Then, in order to satisfy the total transmit power, some weakest users are disposed. 

---
### Non-RL results
- *Heuristic method*: the BS estimates AoD from received echos. If the estimation error > beamwidth, the user is a sensing user. The BS assigns the beams according to $\hat{\theta}$ plus its two adjacent ones. 
- *Genie method:* the BS knows CSI $\boldsymbol{h}_u$ and selects the beam that maximizes the received power for each user.
**Simulation environment**
A frame contains 50 TTIs.
During each frame, the BS has 10 packets sending to each user.  
$R_u > 4$ bits/s/Hz, a packet is transmitted successfully.
$\text{var}^{1/2}(\hat{\varphi}) > beamwidth$, the user is identified as sensing user.

The CDF of transmission time for all packets and the CDF of the number of dropped packet.
The genie-aided method does not have dropped packets.

![[image-5.png|400]]
The x-axis is required transmitted time or number of dropped packets.

### PPO-based method

**Action**

The actions include 
$$
\begin{equation}
   a_n =   \{[\lambda_1, \cdots, \lambda_U], [\mathcal{C}_{1}, \cdots, \mathcal{C}_{U}]\} \in \mathcal{A} .
\end{equation}
$$
$\lambda_u$ --- User type, 1 denotes communication user, 0 denotes sensing user
$\mathcal{C}_{u}$ --- beam indices that are allocated for user $u$.
         The beamforming weight is $\boldsymbol{f}_u = \sum_{i\in\mathcal{C}_u} \mathbf{F}_{i,:}$, $\mathbf{F}$ is the codebook

##### Shrink the action space
For user $u$:

Case 1: $n-1$ TTI, $\lambda_{u,n-1} = 1$, and the beam index is $\mathcal{C}_{u, n-1}$ (the indices of the selected beam) 
   - $n$ TTI, $\lambda_{u,n} = 1$, the beam $\mathcal{C}_{u, n} = \mathcal{C}_{u, n-1}$
   - $n$ TTI, $\lambda_{u,n} = 0$, the beam can be    
      - $\mathcal{C}_{u, n} = \mathcal{C}_{u, n-1} + [-2,-1,0]$
      - $\mathcal{C}_{u, n} = \mathcal{C}_{u, n-1} + [-1,0,1]$
      - $\mathcal{C}_{u, n} = \mathcal{C}_{u, n-1} + [0, 1, 2]$

Case 2:  $n-1$ TTI, $\lambda_{u,n-1} = 0$, and the beam is $\mathcal{C}_{u, n-1}$ (the indices of the selected beams)
   - $n$ TTI, $\lambda_{u,n} = 1$, the beam $\mathcal{C}_{u, n}$ is the beam from the estimation result
   - $n$ TTI, $\lambda_{u,n} = 0$, the beam $\mathcal{C}_{u, n}$ needs to cover more area
      - $\mathcal{C}_{u, n} = min(\mathcal{C}_{u, n-1}) + [-2,-1,0]$
      - $\mathcal{C}_{u, n} = min(\mathcal{C}_{u, n-1}) + [-1,0, 1]$
      - $\mathcal{C}_{u, n} = max(\mathcal{C}_{u, n-1}) + [-1,0,1]$
      - $\mathcal{C}_{u, n} = max(\mathcal{C}_{u, n-1}) + [0, 1, 2]$

When a user is sensing user, the BS allocates some adjacent beams of its previous TTI's to enlarge its sweeping area. 
Action space dimension: $5^U$

---

### State
Instead of $\boldsymbol{Y}$, the state can be the beamforming output of the echos associated with user $u$-th beamformer $\boldsymbol{b}_u = |\boldsymbol{f}_u^H \boldsymbol{Y}|^2$. 

- The BS takes $\|\boldsymbol{b}_{u,n}\|^2$ as the state. 
```
self.observation_space = gym.spaces.Box(low=0, high=np.inf, 
                shape=(self.number_of_antennas, self.N_samples))
```

Regarding to reward, the BS compares needs $\|\boldsymbol{b}_{u,n}\|^2$ to different thresholds for SU and CU respectively. 

### Reward:
The number of successfully transmitted packets.







