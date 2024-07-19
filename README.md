# Jedi No Reward Reinforcement Learning

An AI that learns to be a Jedi on its own.

# Usage

1. Run EternalJK and join a multiplayer server.
2. python jka_noreward.py

# Based On

- https://github.com/pathak22/noreward-rl for the general concept of self-learning
- https://github.com/seungeunrho/minimalRL for the A3C RL algorithm
- https://github.com/pytorch/examples/blob/main/mnist/main.py for the convolutional neural network setup
- https://github.com/TeaPearce/Counter-Strike_Behavioural_Cloning for the bucketing of mouse movements

# Example Output

```
reward tensor(60.9559, grad_fn=<MseLossBackward0>)
R tensor(83.9321, grad_fn=<AddBackward0>)
1253
framerate: 11.59739471885585
```

Here reward is the intrinsic reward as described in figure 2 of https://pathak22.github.io/noreward-rl/resources/icml17.pdf:

![intrinsic agency](https://raw.githubusercontent.com/nullonesix/jedi_noreward_rl/main/noreward.png)



