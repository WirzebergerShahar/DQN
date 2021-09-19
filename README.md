# DQN
A DQN approximates a state-value function in a Q-Learning framework with a neural network to let RL work for complex, high-dimensional environments.

In this exercise you are asked to solve a control problem with continuous state and discrete action space using DQN. The environment is given by the OpenAI Gym CartPole environment
(https://gym.openai.com/envs/CartPole-v0/). The goal of CartPole is to balance a pole connected with
one joint on top of a moving cart (see figure). The state is continuous and given by four real numbers s
= [cart position, cart velocity, pole angle, pole angular velocity] (cart position ∈ [−4.8, 4.8], cart velocity
∈ [−∞, ∞], pole angle ∈ [−0.418, 0.418]rad and pole angular velocity ∈ [−∞, ∞]). The actions are discrete
and given by a = [push cart to the left, push cart to the right ] = [0,1]. The reward is +1 for every step
taken. An episode is terminated if (i) the pole angle goes beyond ±12 degrees (±0.2094 rad), (ii) the cart
position is larger than 2.4. The score is the total reward (= total number of steps made) received in one
episode. The overall performance of the agent is evaluated by using a task score. The task score is defined
as the maximum of the minimum score over five consecutive episodes. For example, if the agent achieves in
10 episodes the following scores: 7, 9, 10, 20, 35, 15, 25, 5, 8, 8, then the task score is min[10, 20, 35, 15, 25]
= 10.
