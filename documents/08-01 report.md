### July 25 Comments
- [ ] Calculate power allocation. When users' type and beams are set, is it a convex problem to calculate the power alllocation?
- [ ] Check the estimation performance. See whether $\boldsymbol{b}_u = |\boldsymbol{f}_u^H \boldsymbol{Y}|^2$ can implicitly reflect the sensing performance.



---

### Objective function

The previous version:
$$
\begin{alignat}{2}
&\arg \max_{\lambda, \boldsymbol{f}, P} && \sum_{n=1}^{N} \sum_{u=1}^U \left(\lambda_{u,n}\mathbf{1}_{(B_{u,n} > 0)} R_{u,n} + (1-\lambda_{u,n}) (1- \text{var}^{no}(\hat{\varphi}_{q,n})) 
%- \frac{B_{u,n }- R_{u,n}}{B_{u,0}} 
\right) \\
&\mathrm{subject~to} ~~ &&\sum_{u=1}^U P_{u,n} \leq P \nonumber \\
& &&\lambda_{u,n} \in\{0,1\} ~~ \text{user's state: 1=CU, 0= SU}\nonumber\\
& && \text{var}^{no}(\hat{\varphi}_{q,n}) ~~ \text{normalized MSE,}  \nonumber \\
& &&  \text{var} (\hat{\varphi}_{q,n}) \propto \frac{\sigma_{\omega}^2}{P_{q,n} \beta^2_{q,n} \big|\boldsymbol{a}^H (\varphi_q) \boldsymbol{f}_{q,n}\big|^2 \Delta T } \nonumber \\
%& && B_{u,n} = B_{u,n-1} - R_{u, n-1}
\end{alignat}
$$
$R_{u,n}$, indicating if a packet went through, is defined as
$$
\begin{align}
& R_{u,n} = \mathbf{1}_{(\log (1 + \gamma_{u,n})> c) \cap (\text{var}(\hat{\varphi}_u) \leq th) }
 \\ 
= & \begin{cases} 1, (\log (1 + \gamma_{u,n})> c) \cap (\text{var}(\hat{\varphi}_u) \leq th)\\
0, (\text{var}(\hat{\varphi}_u) \le th) \cup (\log (1 + \gamma_{u,n})<c)\cap(\text{var}(\hat{\varphi}_u) > th )
\end{cases}
\end{align}
$$

> **__Change the objective function__**
The sensing part, the precision of estimating AOD, is directly added to the indicator function. As the precision is a linear function with respect to the signal strength, the power for the sensing part will be all allocated for the user with the best channel condition. This might cause a problem for the power allocation. Therefore, we change the objective function to be
$$
\arg \max_{\lambda, \boldsymbol{f}, P} \sum_{n=1}^{N} \sum_{u=1}^U \left(\lambda_{u,n}\mathbf{1}_{(B_{u,n} > 0)} R_{u,n} + w(1-\lambda_{u,n}) S_{u,n}  \right)
$$
where $w$ is the weight for sensing and will affect the policy. We set $w=1$ to begin with.
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

<!--

$$
\begin{align}
& R_{u,n} = \mathbf{1}_{(\log (1 + \gamma_{u,n})\ge c)  }
 \\ 
= & \begin{cases} 1, (\log (1 + \gamma_{u,n})\ge c) \\
0, (\log (1 + \gamma_{u,n})<c)
\end{cases}
\end{align}
$$
-->


<!--
> **Note**
> Adding buffer penalty prioritize data transmission when the buffer is not empty. 

- What value should be set for the outage probability？ 
- Optimize the power allocation when beams and types of users are set.  


**Reward**
The BS should prioritize sending out a packet from the data buffer.

Sensing: The sensing reward needs to be scaled to be comparable with the communication reward.

$if~\text{var}^no(\hat{\varphi}_u) < th $, 
$~~~~~~~$reward = 1. 
-->

---
#### RL model

- **Input**: Env: $\boldsymbol{Y}$, $\boldsymbol{B}= [B_1, \cdots, B_U]$, previous beam allocation $[\mathcal{C}_{1,n-1}, \cdots, \mathcal{C}_{U,n-1}]$, $\mathbb{A}(\bar{\boldsymbol{\omega}}^a)$

- **Output** Two neural networks, *actor network* $\mathbb{A}(\bar{\boldsymbol{\omega}}^a)$ and *critic network* $\mathbb{C}(\bar{\boldsymbol{\omega}}^c)$ which are parameterized by $\bar{\boldsymbol{\omega}}^a$ and $\bar{\boldsymbol{\omega}}^c$, respectively.

**State**: is the number of packets in the buffer, the estimation precision which will determine the BS resources allocation.  
$$
\begin{equation}
    s_n  = \{[B_{1}, \cdots, B_{U}], [\mathcal{C}_{1,n-1}, \cdots, \mathcal{C}_{U,n-1}],[\mathbf{Y}]\} \in \mathcal{S}.
\end{equation}
$$


<!--
**Reward** is calculated according to the objective function. 
Advantage: how good the state is for the agent to be in  
Reward: how good the action is when the agent is at a state
-->

**Action**
The BS needs to extract features to tell 
- on which user to perform sensing
- which direction, i.e., the beam and the power to use that is most useful

The actions include 
$$
\begin{equation}
   a_n =   \{[\lambda_1, \cdots, \lambda_U], [\mathcal{C}_{1}, \cdots, \mathcal{C}_{U}]\} \in \mathcal{A} .
\end{equation}
$$
$\lambda_u$ --- User type, 1 denotes communication user, 0 denotes sensing user
$\mathcal{C}_{u}$ --- beam indices that are allocated for user $u$.
         The beamforming weight is $\boldsymbol{f}_u = \sum_{i\in\mathcal{C}_u} \mathbf{F}_{i,:}$, $\mathbf{F}$ is the codebook

During one TTI $n$, when $\lambda_1, \cdots, \lambda_U$, $\mathcal{C}_{1}, \cdots, \mathcal{C}_{U}$ are set, the objective function regarding to power allocation is
$$
\begin{align}
\arg \max_{P} &\sum_{\forall u, \lambda_u=1} \log(1+\gamma_{u}) + \sum_{\forall u, \lambda_u = 0} (1-\text{var}^{no}(\hat{\varphi}_u)) \\
s.t. &\sum_{P_u} = P
\end{align}
$$

- But in the original objective function, we only need the rate to be larger than a threshold. How should we formulate it? 

>We can simplify the objective function to be
$$
\begin{align}
\arg \max_{P} &\sum_{\forall u, \lambda_u=1} \frac{P_{u,n}|\boldsymbol{h}_u ^H \boldsymbol{f}_{u,n} |^2}{\sigma_{\omega}^2} + \sum_{\forall u, \lambda_u = 0} \frac{P_{q,n} \beta^2_{q,n} \big|\boldsymbol{a}^H (\varphi_q) \boldsymbol{f}_{q,n}\big|^2 \Delta T }{\sigma_{\omega}^2}  \\
s.t. &\sum_u {P_u} = P
\end{align}
$$
We simplify the problem to be for the communication part, the SNR $\xi_c = \frac{P_{u,n}|\boldsymbol{h}_u ^H \boldsymbol{f}_{u,n} |^2}{\sigma_{\omega}^2} \ge \iota_c$, and for the sensing part, the signal strength of the echo signal $\xi_s = \frac{P_{q,n} \beta^2_{q,n} \big|\boldsymbol{a}^H (\varphi_q) \boldsymbol{f}_{q,n}\big|^2 \Delta T }{\sigma_{\omega}^2} \ge \iota_s$, where $\iota_s, \iota_c$ are the threshold of sensing and communication respectively.
After the action during TTI $n$ is determined, the objective function with respect to power allocation can be written as
$$
\arg \max_{P_u} \sum \boldsymbol{1}_{P_u \frac{\xi_u}{\iota_{u}} \ge 1} 
$$
This converts the problem to be maximizing the number of users who's signal strength is larger than the threshold. It can be solved by finding the weakest user and satisfying its requirement. The other users' power are allocated according to their channel conditions' ratio. Then, in order to satisfy the total transmit power, some weakest users are disposed. 

---

### Hypothesis 
Instead of $\boldsymbol{Y}$, the state can be the beamforming output of the echos associated with user $u$-th beamformer $\boldsymbol{b}_u = |\boldsymbol{f}_u^H \boldsymbol{Y}|^2$. It will be used for calculating the reward, seen from the objective function. And it decreases the $\mathrm{state\_ dim}$ to be $U*N$.

The SINR is
$$
\begin{align}
\gamma_{u,n} =& \frac{P_{u.n}|\boldsymbol{h}_u ^H \boldsymbol{f}_{u,n} {s}_{u,n}|^2}{\sum_{\substack{i=1, i \neq u}}^{U} |\sqrt{P_{i,n}} \boldsymbol{h}_u ^H \boldsymbol{f}_{i,n} \dot{s}_{i,n}|^2 + \sigma_{\omega}^2} \nonumber \\
= & \frac{P_{u,n}|\boldsymbol{h}_u ^H \boldsymbol{f}_{u,n} |^2}{\sum_{\substack{i=1, i \neq u}}^{U} P_{i,n} |\boldsymbol{h}_u ^H \boldsymbol{f}_{i,n} |^2 + \sigma_{\omega}^2}
\end{align}
$$
Due to the fact that multi-beam is considered, the interference term cannot be ignored. That is, a multi-beam may cover the other user. If we assume that every two users are separated enough, the interference term can be omitted.

As there is no feedback from the users, how can the agent evaluate the resource allocation. In the case of LoS scenario, the channel vector is $\boldsymbol{h}_u = \sum_{\ell=1}^{L} g_u^{\ell} \boldsymbol{a}(\varphi_u^{\ell})$, while $g_u^1 \gg g_u^{\ell}, \ell>1$, and the echo's channel is $\boldsymbol{G} = \sum_{u=1}^{U}
\beta_{u}\boldsymbol{a}(\varphi_u^1) \boldsymbol{a}^H(\varphi_u^1)$. The BS knows the beamformer used for each user $\boldsymbol{f}_{u,n}$. Then, leveraging each user's beamformer, the beamforming output is 
$$
\begin{align}
\boldsymbol{b}_{u,n} &= \boldsymbol{f}_{u,n}^H \boldsymbol{Y} = \boldsymbol{f}_{u,n}^H  \sum_{u=1}^{U}
\beta_{u}\boldsymbol{a}(\varphi_u^1) \boldsymbol{a}^H(\varphi_u^1) \sum_{u=1}^U \sqrt{P_{u,n}}\boldsymbol{f}_{u,n}s_{u,n} \\ 
&\approx \beta_u s_{u,n} \sqrt{P_{u,n}}|\boldsymbol{f}_{u,n}^H \boldsymbol{a}({\varphi_u})|^2  + \boldsymbol{I}_{inter-user} + \boldsymbol{\omega}
\end{align}
$$

$$
|\boldsymbol{b}_{u,n}|^2 = \beta_u^2P_{u,n} |\boldsymbol{f}_{u,n}^H \boldsymbol{a}({\varphi_u})|^4  
$$

