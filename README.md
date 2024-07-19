# Jedi No (External) Reward Reinforcement Learning

An AI that learns to be a Jedi on its own. All feedback comes from within the agent itself. Specificaly, it learns to predict the future and rewards itself for finding (image/pixel-based) states that it cannot anticipate -- i.e. "curiosities". 

## Demo

https://youtu.be/2mIrj11kYDM

![curious jedi](https://github.com/nullonesix/jedi_noreward_rl/blob/main/noreward_demo_thumbnail.png?raw=true)

## Usage

first time:
```
1. Run EternalJK game and join a multiplayer server.
3. Configure your game controls to be the same as mine (see first 2 hyperparameters below).
3. python jka_noreward.py new
4. let it play
```
see parameter counts:
```
PS C:\Users\nullo\Documents\jka_noreward> python .\jka_noreward.py show
loading model..
model loaded.
Total number of trainable actor-critic model parameters: 8631893
Total number of trainable inverse model parameters: 14431886
Total number of trainable forward model parameters: 4694116
```
load a saved model and let it play:
```
PS C:\Users\nullo\Documents\jka_noreward> python .\jka_noreward.py
loading model..
model loaded.
Total number of trainable actor-critic model parameters: 8631893
Total number of trainable inverse model parameters: 14431886
Total number of trainable forward model parameters: 4694116
n_iterations: 2
framerate: 46.34974169130039
error_inverse_model: 0.37778469920158386
error_forward_model: 200.05471801757812
reward: 200.05471801757812
mean error actor critic model: 4.398651123046875
n_iterations: 3
framerate: 7.07696755190778
error_inverse_model: 0.46000540256500244
error_forward_model: 150.2513427734375
reward: 150.2513427734375
mean error actor critic model: 8.75518798828125
...
n_iterations: 1110
framerate: 14.039405370448229
error_inverse_model: 0.18923787772655487
error_forward_model: 74.45198822021484
reward: 74.45198822021484
mean error actor critic model: 2.894500732421875
n_iterations: 1111
framerate: 14.039663828893318
error_inverse_model: 0.4495146572589874
error_forward_model: 75.67051696777344
reward: 75.67051696777344
mean error actor critic model: 0.13396330177783966
```
## Hyperparameters

The things that you usually tune by hand. For eaxmple, I'm running this on a laptop GPU, someone with many high-end GPUs might wish to increase the size of their neural networks. Alternatively someone running on CPU might which to decrease their sizes in order to achieve an agent framerate of at least 15 iterations per second.

```py
# Hyperparameters
key_possibles = ['w', 'a', 's', 'd', 'space', 'r', 'e'] # legend: [forward, left, back, right, style, use, center view]
mouse_button_possibles = ['left', 'middle', 'right'] # legend: [attack, crouch, jump]
mouse_x_possibles = [-1000.0,-500.0, -300.0, -200.0, -100.0, -60.0, -30.0, -20.0, -10.0, -4.0, -2.0, -0.0, 2.0, 4.0, 10.0, 20.0, 30.0, 60.0, 100.0, 200.0, 300.0, 500.0,1000.0]
mouse_y_possibles = [-200.0, -100.0, -50.0, -20.0, -10.0, -4.0, -2.0, -0.0, 2.0, 4.0, 10.0, 20.0, 50.0, 100.0, 200.0]
n_actions = len(key_possibles)+len(mouse_button_possibles)+len(mouse_x_possibles)+len(mouse_y_possibles)
n_train_processes = 1 # 3
update_interval = 100 # 10 # 1 # 5
gamma = 0.98 # 0.999 # 0.98
max_train_ep = 10000000000000000000000000000 # 300
max_test_ep = 10000000000000000000000000000 #400
n_filters = 64 # 128 # 256 # 512
input_rescaling_factor = 2
input_height = input_rescaling_factor * 28
input_width = input_rescaling_factor * 28
conv_output_size = 22464 # 44928 # 179712 # 179712 # 86528 # 346112 # 73728
pooling_kernel_size = input_rescaling_factor * 2 # 16
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('using device:', device)
forward_model_width = 4096 #2048
inverse_model_width = 1024 #2048
mouse_rescaling_factor = 10
dim_phi = 100
action_predictability_factor = 100
n_transformer_layers = 1
```

## Based On

- https://github.com/pathak22/noreward-rl for the general concept of curiosity-driven learning and its formalization
- https://github.com/seungeunrho/minimalRL for the A3C actor-critic reinforcement learning algorithm
- https://github.com/pytorch/examples/blob/main/mnist/main.py for the convolutional neural network setup
- https://github.com/TeaPearce/Counter-Strike_Behavioural_Cloning for the non-uniform bucketing of mouse movements

## How It Works

Here reward is the intrinsic reward as described in figure 2 of https://pathak22.github.io/noreward-rl/resources/icml17.pdf:

![intrinsic agency](https://raw.githubusercontent.com/nullonesix/jedi_noreward_rl/main/noreward.png)

- R is the cumulative expected future rewards (with exponential decay factor gamma = 0.99, ie future rewards are less desirable than the same immediate rewards)
- so for example if the AI is playing at 10 frames per second then a reward of 100 two seconds into the future is worth gamma^(2*10) * 100 = (0.99)^20 * 100 = 8.17
- so the further into the future a reward is, the more it is decayed (which is standard in reinforcement learning)
- the key difference here is that the rewards are not external (eg via a game score) but internal (ie "curiosity" as computed by the agent)
- to quote the paper: "We formulate curiosity as the error in an agent’s ability to predict the consequence of its own actions in a visual feature space learned by a self-supervised inverse dynamics model."
- intuitively this means the agent is drawn towards outcomes it cannot itself anticipate
- theoretically this motivates the agent to not stand still, to explore other areas of the map, and to engage with other players 

## Future Work

- stacked frames for better time/motion perception
- a memory of the past via LSTM (as done in the curiosity paper) or transformer
- integrate YOLO-based aimbot: https://github.com/petercunha/Pine

## Agent Views

before resizing:

![full view](https://raw.githubusercontent.com/nullonesix/jedi_noreward_rl/main/view.png)

after resizing (ie true size view) (zoom in to see that this resolution is sufficient), but before grayscaling:

![true size view](https://raw.githubusercontent.com/nullonesix/jedi_noreward_rl/main/agent_view.png)


