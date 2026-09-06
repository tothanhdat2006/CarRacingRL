# Foundation for Deep Q-Learning

Deep Q-Learning is Q-learning combine with Deep Learning: aims to learn the $Q(s, a)$ table under compression, $\hat{Q}(s, a; w)$ where $w$ is the compression parameters. This requires understanding about:
- Deep Learning: supervised between real Q(s, a) and compressed $\hat{Q}(s, a; w)$
- [Q-learning](#q-learning), which is off-policy learning
- Model-free generalized policy iteration. Which requires:
  - [$\epsilon$-greedy policy](#model-free-control)
  - [Monte Carlo or Temporal Difference](#model-free-policy-evaluation)
  - Policy evaluation:
    - [Bellman equation](#markov-reward-process-mrp)
    - [Markov Decision Process](#markov-decision-process-mdp)

## Markov
This part is a foundation and should be accepted as-is.
Important: Assume that we know the real transition model $P$ of the environments.

### Markov chain
A series of $X_0, X_1, \dots, X_n$ is called a Markov chain if it follows Markov property. Markov chain consists of state space $S$ where $X_i \in S,  \forall i \in [1, n]$, and transition matrix $P \in \mathcal{R}^{n \times n}$.

### Markov Reward Process (MRP)
Markov chain combine with reward function is MRP

Some notations:
- Return $G_t$ from time $t$ to horizon $H$ (end time) is $G_t = r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + \gamma^{H-1} r_{t+H-1}$
  - $\gamma$ is discount factor, which is a constant that controls the reward gain in the future (often $\gamma < 1$)
  - This will control the estimate will focus on reward or future reward more
  - Also to avoid infinite returns in *cyclic* Markov processes 
- State Value function $V(s) = E[G_t | s_t = s]$ (Expected return given current state)

To obtain $V(s)$, we use **Bellman Equations**: $V(s) = R(s) + \gamma \sum_{s' \in S} P(s' | s) V(s')$

Algorithm (O(|S|^2)):
- Initialize V
- Loop until convergence: assign based on Bellman Equations

### Markov Decision Process (MDP)
- MDP = MRP + actions ($a \in A$)
- MRP = MDP + policy ($\pi$)

Main components are model, **policy** and value function:
- Model are approximate of true environment dynamics including transition model and reward model
- Policy can be deterministic $\pi(s) = a$ or stochastic
- Value function $V^{\pi}(s)$ is the estimate reward at state $s$ under policy $\pi$

Some notations:
- Reward of a state under policy can be updated using the weighted reward based on probability of the action to achive that state: $R^{\pi}(s) = \sum_{a \in A} \pi(s | a) R(s, a)$
- Transition model of a state under policy can be updated using weighted probability of the action to achieve that state given a previous state: $P^{\pi}(s' | s) = \sum_{a \in A} \pi(s | a) P'(s' | s, a)$
- **State value function** is now: $$V^{\pi}(s) = \sum_{a \in A} \pi(s | a) \times [R(s, a) + \gamma \sum_{s' \in S} P'(s' | s, a) V^{\pi}(s')]$$
- **State-action value function**: $$Q^{\pi}(s, a) = R(s, a) + \gamma \sum_{s' \in S} P'(s' | s, a) V^{\pi}(s')$$
$$ => V^{\pi}(s) = \sum_{a \in A} \pi(s | a) Q^{\pi}(s, a)$$

## Policy + Value Iteration
Consists two steps: evaluate and improve policy

### Evaluation
Algorithm:
- Initialize V
- Loop until convergence: using State value function

An optimal policy:
- $\pi^*(s) = \argmax_s V^{\pi}(s)$


### Improvement
Bellman Optimality Equation:
- $V^*(s) = ...$
- $Q^*(s, a) = ...$
- $\pi^*(s) = \argmax_a Q^*(s, a)$

BOE is non-linear, therefore no closed-form solution. However, it can be calculated using iterative methods like **Q-learning**

### Overall
Bellman Backup Operator: ...

Convergence: at most $|A|^|S|$. Why converge? Contraction property of Bellman operator

## Model-free Policy Evaluation
The first assumption we made about knowing transition model is not feasible in real world. Therefore, we need to approximate it using algorithms such as:
- Monte Carlo: Estimate using sample mean. Some variants: First-visit MC, Every-visit MC
  - Pro: Don't need dynamic of environment or Markov property
  - Con: Episodic, Higher variance
- Temporal Difference (TD): Replace G_{i, t} with one-step average, consists of TD target and TD error. Some variants: TD(0), TD($\lambda$)
  - Pro: Learn online, applicable to task with no terminal state
  - Con: Biased
- Certainty-equivalence (least well-known than previous two methods)

## Model-free Control
Generalized Policy Iteration algorithm:
- Evaluation: estimate $Q^{\pi}(s, a)$
- Improvement: set new $\pi'$ greedily w.r.t to $Q^{\pi}$

But if $\pi(s) = a$, we can not estimate for $Q(s, a)$ where $a \neq \pi(s)$
=> Exploration problem

### Epsilon-greedy policy
If $a = argmax_a' (Q(s, a'))$, then $\pi(a | s) = 1 - \epsilon + \frac{\epsilon}{|A|}$

Otherwise $\pi(a | s) = \frac{\epsilon}{|A|}$

## On/Off-policy Learning
On-policy: learn the same policy as one used in generating actions. An example is SARSA ($s_t, a_t, r_t, s_{t+1}, a_{t+1}$)

Off-policy: learn $\pi$, follow $\pi_k$ for generation. An example is **Q-learning**

## Q-Learning
- Initialize
- Draw $a_t$ from policy $\pi(s_t)$ ($\epsilon$-greedy w.r.t Q)
- Loop until convergence
  - Act $a_t$ to obtain ($r_t, s_{t+1})
  - Draw $a_{t+1}$ from policy $\pi(s_{t+1})$ 
  - $Q(s, a) += \alpha * [r_t + \gamma \max_{a'} Q(s_{t+1}, a') - Q(s_t, a_t)]$ 
  - Update $\pi(s_t)$ based on $\epsilon$-greedy w.r.t Q
  - $t += 1$ 

Convergence guarantee:
- Behaviour policy satisfies Greedy in the Limit of Infinite Expension (GLIE)
  - Every pair $(s, a)$ visited infinitely as time approach infinite
  - Behaviour policy converge to Greedy policy as time approach infinite
- Step size satisfies Robbins-Monro conditions
  - $\sum \alpha_t = \infty$
  - $\sum \alpha^2_t \lt \infty$

But, tables $|S| \times |A|$ does not scale well with high dimensional State or Action spaces

## Deep Q-Learning
### Value function approximation
Replace large table of $|S| \times |A|$ with vector $w$ by approxmiate $\hat{Q}(s, a; w) \approx Q^{\pi}(s,a)$

### SGD
Loss function: $J(w) = E_{\pi}[(Q^{\pi}(s,a) - \hat{Q}(s, a; w))^2]$

Updated weights: $\triangle w = -\frac{1}{2} \alpha \triangledown_w J(w)$ 

### Deep Q-Network
Problem with normal algorithm is (1) consecutive samples are coorelated, which violate i.i.d assumption of SGD; and (2) non-stationary targets (chasing a goal that always changes)

Solution:
- Experience replay: save current steps to replay buffer and take random samples for learning
- Fixed Q-targets: separate another vector $w^-$, which syncs every $C$ steps

Algorithm:
- Initialize
- Loop until convergence:
  - Draw $a_t$ from policy $\pi(s_t)$ ($\epsilon$-greedy w.r.t $\hat{Q}^(s_t, a_t; w)$)
  - Draw $a_{t+1}$ from policy $\pi(s_{t+1})$ 
  - Save $(s_t, a_t, r_t, s_{t+1})$ to replay buffer $D$
  - Sample randomly from $D$
  - For every samples $(s, a, r, s')$:
    - if s' is terminal nodes: $y_j = r_j$
    - else: $y_j = r_j + \gamma \max_{a_{j+1}} \hat{Q}(s_{j+1}, a_{j+1}; w^-)$
  - $\triangle w = \alpha (y_j - \hat{Q}(s_j, a_j; w^)) \triangledown_w \hat{Q}(s_j, a_j; w^)$ 
  - $t += 1$
  - if $t \% C = 0$: $w^- = w$

# What is after Deep Q-Learning
Policy gradient: previous value-based methods is limited by deadly triad, therefore we learn policy-based instead. There are many more benefits:
- $\argmax_a Q(s, a)$ is expensive in continuous and high dimension domain
- Stochastic policies requires random action
- Optimal policy sometimes simpler than optimal $V$
- Better convergence

=> Deep Deterministic Policy Gradients (DDPG)