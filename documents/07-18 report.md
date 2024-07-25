## Last week comments
- The objective function: the buffer penalty is repetitive with the communication part

- Consider the benchmark and debug the code that I already have. 
Consider what result should be presented.

- The ML model: cardinality of the action space and calculation of the reward.

---
### Code
- Two Cars
- Speed $\sim \mathcal{N}(30, 2)$
- TTI 10 ms second
- Expected received SINR at the user: 15 dB
- *Expected received SNR of the echos*: 0 dB -- estimation suffers from a bad performance
- Transmit power: distributed to each user equally.
$P_u = P_{tot}/U$
 If a user has multiple beams, $P_u$ is further distributed equally to each beam. 
 - LoS environment.

*Empirical method*: the BS estimates AoD from received echos. If the estimation error > beamwidth, the user is a sensing user. The BS assigns the beams according to $\hat{\theta}$ plus its two adjacent ones. 
*Genie method*: the BS knows CSI $\boldsymbol{h}_u$ and selects the beam that maximizes the received power for each user.

---

### Action space 
The action space is hybrid. 
$$
\begin{equation}
   a_n =   \{[\lambda_1, \cdots, \lambda_U], [\boldsymbol{f}_{1}, \cdots, \boldsymbol{f}_{U}], [P_1, \cdots, P_U]\} \in \mathcal{A} .
\end{equation}
$$
The actions are dependent.  

#### Consider discrete space
For beam alignment, it is difficult to determine the action space of $\boldsymbol{f}_1, \cdots, \boldsymbol{f}_U$, as each user can have one or multiple beams depending on $\lambda$. 

So, we consider to assign user to beams. 
Denote $\delta_{1}, \cdots, \delta_{N_t}$ for the $N_T$ beams' indicator. 
Let $\delta_{n_t,u} = 1$ denote that $u$-th user is assigned to $n_t$-th beam. 
$\sum_{u=1}^{U} \delta_{n_t,u} \in \{0, 1\}$, 0 means the beam is not assigned. A beam is only used for one user.
The action space is $N_t^{(U+1)}$. 
$\lambda$ is determined implicitly.


For the power, we consider the total power is divided to $M$ portion. 
The total number of beams in use is $N_t' = \sum_{u=1}^{U}\sum_{n_t=1}^{N_T} \delta_{n_t,u}$.
Distributing power action has $\binom{M-1}{N_t'-1}$, where $M$ is a tunable parameter.


#### Consider continuous space

Some work also makes the discrete space into continuous.
- Make the action space all continuous by using a softmax layer.
- $\hat{\delta}_{n_t} \in [0, 1]$, and the assigned beam is $\lceil\hat{\delta}*U\rceil$-th user.
- Normalize the transmit power $\hat{P}_{n_t}\in[0,1]$ 


### Comments
- [ ] Reduce the action space
   - [x] Relax the constraint that a beam can only be used by one user
   - [ ] Search in a set of beams by considering the largest area user can go calculated from its location and speed
   - [ ] The power can be calculated separately after beam assignment  
- [x] Debug non-ML implementation code



The objective function is
$$
\begin{alignat}{2}
&\arg \max_{\lambda, \boldsymbol{f}, P} && \sum_{n=1}^{N} \sum_{u=1}^U \left(\lambda_{u,n}\mathbf{1}_{(B_{u,n} > 0)} R_{u,n} + (1-\lambda_{u,n}) (1- \text{var}^{no}(\hat{\varphi}_{q,n})) - \frac{B_{u,n }- R_{u,n}}{B_{u,0}} \right) \\
&\mathrm{subject~to} ~~ &&\sum_{u=1}^U P_{u,n} \leq P \nonumber \\
& &&\lambda_{u,n} \in\{0,1\} ~~ \text{user's state: 1=CU, 0= SU}\nonumber\\
& && \text{var}^{no}(\hat{\varphi}_{q,n}) ~~ \text{normalized MSE,}  \nonumber \\
& &&  \text{var} (\hat{\varphi}_{q,n}) \propto \frac{\sigma_{\omega}^2}{\kappa^2 P_{q,n} \beta^2_{q,n} \big|\boldsymbol{a}^H (\varphi_q) \boldsymbol{f}_{q,n}\big|^2 \Delta T } \nonumber \\
& && B_{u,n} = B_{u,n-1} - R_{u, n-1}
\end{alignat}
$$
$R_{u,n}$, indicating if a packet went through, is defined as
$$
\begin{align}
& R_{u,n} = \mathbf{1}_{(\log (1 + \gamma_{u,n})> c) \cap (\text{var}(\hat{\varphi}_u) \leq th) }
 \\ 
= & \begin{cases} 1, (\log (1 + \gamma_{u,n})> c) \cap (\text{var}(\hat{\varphi}_u) \leq th)\\
0, (\text{var}(\hat{\varphi}_u) > th) \cup (\log (1 + \gamma_{u,n})<c)\cap(\text{var}(\hat{\varphi}_u))
\end{cases}
\end{align}
$$

> **Note**
> Adding buffer penalty prioritize data transmission when the buffer is not empty. 

- What value should be set for the outage probability？ 
- Optimize the power allocation when beams and types of users are set.  


### Decrease action space shape
- Search in a subspace considering the beam selection from previous TTI and car moving speed. 
- Multiple users can use the same beam.

<!-- **__Association:__** PPO algorithm's objective function set a clip reward function to avoid a big change in the policy. -->

For the actor network, allocating resouces to different users can be done by spliting to sub-nets. 



For user $u$:

- $n-1$ TTI, $\lambda_{u,n-1} = 1$, and the beam is $\boldsymbol{f}_{u,n-1}$ (the indix of the selected beam) 
   - $n$ TTI, $\lambda_{u,n} = 1$, the beam $\boldsymbol{f}_{u,n} = \boldsymbol{f}_{u,n-1}$
   - $n$ TTI, $\lambda_{u,n} = 0$, the beam can be    
      - $\boldsymbol{f}_{u,n} = \boldsymbol{f}_{u,n-1} + [-2,-1,0]$
      - $\boldsymbol{f}_{u,n} = \boldsymbol{f}_{u,n-1} + [-1,0,1]$
      - $\boldsymbol{f}_{u,n} = \boldsymbol{f}_{u,n-1} + [0, 1, 2]$
- $n-1$ TTI, $\lambda_{u,n-1} = 0$, and the beam is $\boldsymbol{f}_{u,n-1}$ (the indices of the selected beams)
   - $n$ TTI, $\lambda_{u,n} = 1$, the beam $\boldsymbol{f}_{u,n}$ is the beam from the estimation result
   - $n$ TTI, $\lambda_{u,n} = 0$, the beam $\boldsymbol{f}_{u,n}$ needs to cover more area
      - $\boldsymbol{f}_{u,n} = min(\boldsymbol{f}_{u,n-1}) + [-2,-1,0]$
      - $\boldsymbol{f}_{u,n} = min(\boldsymbol{f}_{u,n-1}) + [-1,0, 1]$
      - $\boldsymbol{f}_{u,n} = max(\boldsymbol{f}_{u,n-1}) + [-1,0,1]$
      - $\boldsymbol{f}_{u,n} = max(\boldsymbol{f}_{u,n-1}) + [0, 1, 2]$


---
### RL model

- **Input**: Env: $\boldsymbol{Y}$, $\boldsymbol{B}= [B_1, \cdots, B_U]$, perhaps previous beam allocation $[\boldsymbol{f}_1, \cdots, \boldsymbol{f}_U]$, $\mathbb{A}(\bar{\boldsymbol{\omega}}^a)$

- **Output** Two neural networks, *actor network* $\mathbb{A}(\bar{\boldsymbol{\omega}}^a)$ and *critic network* $\mathbb{C}(\bar{\boldsymbol{\omega}}^c)$ which are parameterized by $\bar{\boldsymbol{\omega}}^a$ and $\bar{\boldsymbol{\omega}}^c$, respectively.

**State**: is the number of packets in the buffer, the estimation precision which will determine the BS resources allocation.  
$$
\begin{equation}
    s_n  = \{[B_{1}, \cdots, B_{U}], [\mathbf{Y}]\} \in \mathcal{S}.
\end{equation}
$$

**Action**: The intellectual BS needs to extract features to tell 
- on which user to perform sensing
- which data to transmit based on the data buffer states
- which direction, i.e., the beam to use that is most useful


the agent action is decide user type, beams and power which is
$$
\begin{equation}
   a_n =   \{[\lambda_1, \cdots, \lambda_U], [\boldsymbol{f}_{1}, \cdots, \boldsymbol{f}_{U}], [P_1, \cdots, P_U]\} \in \mathcal{A} .
\end{equation}
$$
**Reward** is calculated according to the objective function. 
Advantage: how good the state is for the agent to be in  
Reward: how good the action is when the agent is at a state