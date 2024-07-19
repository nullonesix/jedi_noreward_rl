# Jedi No Reward Reinforcement Learning

An AI that learns to be a Jedi on its own.

# Usage

1. Run EternalJK game and join a multiplayer server.
2. python jka_noreward.py

# Based On

- https://github.com/pathak22/noreward-rl for the general concept of self-learning
- https://github.com/seungeunrho/minimalRL for the A3C actor-critic reinforcement learning algorithm
- https://github.com/pytorch/examples/blob/main/mnist/main.py for the convolutional neural network setup
- https://github.com/TeaPearce/Counter-Strike_Behavioural_Cloning for the bucketing of mouse movements

# Example Output

(along with actual unsupervised automated gameplay)
```
reward tensor(60.9559, grad_fn=<MseLossBackward0>)
R tensor(83.9321, grad_fn=<AddBackward0>)
1253
framerate: 11.59739471885585
```

- reward is the quality of the moment
- R is the expected future quality
- 1253 is just the iteration number (how many times the agent just seen the game screen and taken an action and trained its 3 neural networks, all of these are done in lockstep)
- framerate is just the number of iterations the agent performs per second (can be increased by using smaller neural networks, for example)

# How It Works

Here reward is the intrinsic reward as described in figure 2 of https://pathak22.github.io/noreward-rl/resources/icml17.pdf:

![intrinsic agency](https://raw.githubusercontent.com/nullonesix/jedi_noreward_rl/main/noreward.png)

R is the cumulative expected future rewards (with exponential decay factor gamme = 0.99, ie future rewards are less desirable than the same immediate rewards)

so for example if the AI is playing at 10 frames per second then a reward of 100 two seconds into the future is woth gamma^(2*10) * 100 = (0.99)^20 * 100 = 8.17

so the further into the future a reward is, the more it is decayed (with is standard in reinforcement learning)

the key difference here is that the rewards are not external (eg via a game score) but internal (ie "curiosity" as computed by the agent)

to quote the paper: "We formulate curiosity as the error in an agent’s ability to predict the consequence of its own actions in a visual feature space learned by a self-supervised inverse dynamics model."

intuitively this means the agent is drawn towards outcomes it cannot itself predict

theoretically this motivates the agent to not stand still, to explore other areas of the map, and to engage with other players 



