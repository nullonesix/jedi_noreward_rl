import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import torch.multiprocessing as mp
import time
import win32gui
from PIL import ImageGrab
# import pyscreenshot as ImageGrab
import PIL
import torchvision.transforms as transforms

grayscale = transforms.Grayscale(num_output_channels=1)

# Environment
import gym
from gym import spaces
import numpy as np

import keyboard
import mouse
import ctypes as cts
import pynput
import sys
import win32gui, win32ui, win32con, win32api
import ctypes
from ctypes import wintypes

DWMWA_EXTENDED_FRAME_BOUNDS = 9
rect = wintypes.RECT()

def set_pos(dx, dy):
    # print('dx', dx, 'dy', dy)
    # pos = queryMousePosition()
    # x, y = pos['x'] + dx, pos['y'] + dy
    # x = 1 + int(x * 65536./Wd)
    # y = 1 + int(y * 65536./Hd)
    extra = cts.c_ulong(0)
    ii_ = pynput._util.win32.INPUT_union()
    # ii_.mi = pynput._util.win32.MOUSEINPUT(x, y, 0, (0x0001 | 0x8000), 0, cts.cast(cts.pointer(extra), cts.c_void_p))
    ii_.mi = pynput._util.win32.MOUSEINPUT(dx, dy, 0, (0x0001), 0, cts.cast(cts.pointer(extra), cts.c_void_p))
    command=pynput._util.win32.INPUT(cts.c_ulong(0), ii_)
    cts.windll.user32.SendInput(1, cts.pointer(command), cts.sizeof(command))

def get_screenshot():
    hwnd = win32gui.FindWindow(None, 'EternalJK')
    win32gui.SetForegroundWindow(hwnd)
    ctypes.windll.dwmapi.DwmGetWindowAttribute(
        hwnd,
        DWMWA_EXTENDED_FRAME_BOUNDS,
        ctypes.byref(rect),
        ctypes.sizeof(rect)
    )
    bbox = (rect.left, rect.top, rect.right, rect.bottom)
    img = ImageGrab.grab(bbox)
    if 'view' in sys.argv:
        img.save('view.png')
    img = img.resize((input_width, input_height), PIL.Image.NEAREST)
    if 'view' in sys.argv:
        img.save('agent_view.png')
        print('agent view size:', input_width, 'x', input_height)
        sys.exit()
    return img

class CustomEnv(gym.Env):
    """Custom Environment that follows gym interface"""
    # metadata = {'render.modes': ['human']}

    def __init__(self):
        super(CustomEnv, self).__init__()
        self.average_state = 0
        self.n_states = 0
        # hwnd = win32gui.FindWindow(None, 'EternalJK')
        # win32gui.SetForegroundWindow(hwnd)
        # # bbox = win32gui.GetWindowRect(hwnd)

        # # Get the window's client area
        # rect = win32gui.GetClientRect(hwnd)
        # left, top = win32gui.ClientToScreen(hwnd, (rect[0], rect[1]))
        # right, bottom = win32gui.ClientToScreen(hwnd, (rect[2], rect[3]))
        # bbox = (left, top, right + 650, bottom + 520)

        # img = ImageGrab.grab(bbox).resize((input_width, input_height), PIL.Image.NEAREST)
        img = get_screenshot()
        
        if 'view' in sys.argv:
            img.save('agent_view.png')
            # full_res = ImageGrab.grab(bbox)
            # full_res.save('full_res_view.png')
            sys.exit()
        # state = torch.tensor(np.array(img)).float().to(device)
        # self.previous_state = state

        frame1 = torch.tensor(np.array(img)).float().to(device)
        frame2 = torch.tensor(np.array(img)).float().to(device)

        state = torch.cat([frame1, frame2], dim=0)
        # print(state.shape)
        self.previous_frame = frame2
        self.previous_state = state
        # sys.exit()

        # self.param1 = param1
        # self.param2 = param2
        # Define action and observation space
        # They must be gym.spaces objects
        # self.action_space = spaces.Discrete(2)  # Example: two discrete actions
        # self.observation_space = spaces.Box(low=0, high=1, shape=(3,), dtype=np.float32)  # Example: 3-dimensional observation

    # def step(self, action):
    def step(self, action):
        
        # print('action.shape', action.shape)
        # Execute one time step within the environment
        # Actions
        # w, a, s, d, space, ctrl, mouse_left, mouse_middle, mouse_right, 19 mouse_deltaX,  13 mouse_deltaY
        # mouse_x_possibles = [-1000.0,-500.0, -300.0, -200.0, -100.0, -60.0, -30.0, -20.0, -10.0, -4.0, -2.0, -0.0, 2.0, 4.0, 10.0, 20.0, 30.0, 60.0, 100.0, 200.0, 300.0, 500.0,1000.0]
        # mouse_y_possibles = [-200.0, -100.0, -50.0, -20.0, -10.0, -4.0, -2.0, -0.0, 2.0, 4.0, 10.0, 20.0, 50.0, 100.0, 200.0]
        # symmetry test
        # mouse_x_possibles = list(reversed(mouse_x_possibles))
        # mouse_y_possibles = list(reversed(mouse_y_possibles))

        # keyboard.release('w, a, s, d, space, ctrl, r, space, ctrl, e')


        # key_possibles = ['w', 'a', 's', 'd', 'space', 'ctrl', 'e']
        time_take_action = time.time()
        keyboard.release(','.join(key_possibles))
        mouse.release(button='left')
        mouse.release(button='middle')
        mouse.release(button='right')
        if keyboard.is_pressed('c'):
            sys.exit()
        pressed_keys = []
        mouse_x = 0.0
        mouse_y = 0.0
        # print('action', action)
        # if action[0].item() == 1:
        #     pressed_keys.append('w')
        # if action[1].item() == 1:
        #     pressed_keys.append('a')
        # if action[2].item() == 1:
        #     pressed_keys.append('s')
        # if action[3].item() == 1:
        #     pressed_keys.append('d')
        # if action[4].item() == 1:
        #     pressed_keys.append('space')
        # if action[5].item() == 1:
        #     pressed_keys.append('ctrl')
        # if action[6].item() == 1:
        #     pressed_keys.append('e')
        for i in range(len(key_possibles)):
            if action[i].item() == 1:
                pressed_keys.append(key_possibles[i])
        if pressed_keys:
            keyboard.press(','.join(pressed_keys))
        for i in range(len(mouse_button_possibles)):
            if action[i+len(key_possibles)].item() == 1:
                mouse.press(button=mouse_button_possibles[i])
        # if action[7].item() == 1:
        #     if not 'runmode' in sys.argv:
        #         mouse.press(button='left')
        # if action[8].item() == 1:
        #     if not 'runmode' in sys.argv:
        #         mouse.press(button='middle')
        # if action[9].item() == 1:
        #     if not 'runmode' in sys.argv:
        #         mouse.press(button='right')
        for i in range(len(mouse_x_possibles)):
            if action[i+len(key_possibles)+len(mouse_button_possibles)].item() == 1:
                mouse_x += mouse_x_possibles[i]
        for i in range(len(mouse_y_possibles)):
            if action[i+len(key_possibles)+len(mouse_button_possibles)+len(mouse_x_possibles)].item() == 1:
                mouse_y += mouse_y_possibles[i]
        # set_pos(int(mouse_x), int(mouse_y))
        # set_pos(int(mouse_x/10), int(mouse_y/10))
        print('mouse_dx:', mouse_x)
        print('mouse_dy:', mouse_y)
        set_pos(int(mouse_x/mouse_rescaling_factor), int(mouse_y/mouse_rescaling_factor))
        print('take action time:', time.time() - time_take_action)

        time_screenshot_start = time.time()
        # hwnd = win32gui.FindWindow(None, 'EternalJK')
        # win32gui.SetForegroundWindow(hwnd)
        # bbox = win32gui.GetWindowRect(hwnd)
        # print(bbox)
        # Get the extended frame bounds
        # ctypes.windll.dwmapi.DwmGetWindowAttribute(
        #     hwnd,
        #     DWMWA_EXTENDED_FRAME_BOUNDS,
        #     ctypes.byref(rect),
        #     ctypes.sizeof(rect)
        # )
        # bbox = (rect.left, rect.top, rect.right, rect.bottom)
        # print('get window time:', time.time() - time_jk_start)
        # time_grab_start = time.time()
        # img = ImageGrab.grab(bbox)
        # img = ImageGrab.grab(backend="mss", childprocess=False)
        img = get_screenshot()
        print('screenshot time:', time.time() - time_screenshot_start)
        # img.save('view.png')
        # sys.exit()
        # time_resize_start = time.time()
        # img = img.resize((input_width,input_height), PIL.Image.NEAREST)
        # print('image resize time:', time.time() - time_resize_start)
        # img.save('input.png')
        # state = torch.tensor(np.array(img)).float().to(device)
        frame1 = self.previous_frame
        frame2 = torch.tensor(np.array(img)).float().to(device)
        # assert frame1.mean() != frame2.mean()
        state = torch.cat([frame1, frame2], dim=0)
        
        time_phi_start = time.time()
        phi_previous_state = phi_model(self.previous_state)
        print('phi model inference time:', time.time() - time_phi_start)
        # phi_previous_state_f = phi_previous_state.clone()
        # print('phi_previous_state.shape', phi_previous_state.shape)
        phi_state = phi_model(state)
        # print('phi_state.shape', phi_state.shape)
        # print('torch.cat([phi_previous_state, phi_state], dim=1).shape', torch.cat([phi_previous_state, phi_state], dim=1).shape)
        time_inverse_start = time.time()
        action_hat = inverse_model(torch.cat([phi_previous_state, phi_state], dim=1))
        print('inverse model inference time:', time.time() - time_inverse_start)
        action = torch.unsqueeze(action, dim=0)
        action = action.float()

        error_inverse_model = torch.nn.functional.mse_loss(action_hat, action, size_average=None, reduce=None, reduction='mean') # input, target
        print('error_inverse_model:', error_inverse_model.item())
        optimizer_inverse.zero_grad()
        # error_inverse_model.backward(retain_graph=True)
        time_inverse_start = time.time()
        error_inverse_model.backward()
        optimizer_inverse.step()
        print('inverse model learn time:', time.time() - time_inverse_start)

        # print('action.shape', action.shape)
        # print('phi_previous_state.shape', phi_previous_state.shape)
        # print('torch.cat([action, phi_previous_state], dim=1).shape', torch.cat([action, phi_previous_state], dim=1).shape)

        # phi_hat_state = forward_model(torch.cat([action, phi_previous_state], dim=1))
        phi_previous_state_f = phi_model(self.previous_state)
        action_f = action.clone()
        phi_state_f = phi_model(state)
        time_forward_start = time.time()
        phi_hat_state_f = forward_model(torch.cat([action_f, phi_previous_state_f], dim=1))
        print('forward model inference time:', time.time() - time_forward_start)

        # print('action_hat.shape', action_hat.shape)
        # print('action.shape', action.shape)
        # action = action.float()
        # error_inverse_model = torch.nn.functional.mse_loss(action_hat, action, size_average=None, reduce=None, reduction='mean') # input, target
        # error_forward_model = torch.nn.functional.mse_loss(phi_hat_state, phi_state, size_average=None, reduce=None, reduction='mean') # input, target

        # torch.autograd.set_detect_anomaly(True)

        # error_inverse_model = torch.nn.functional.mse_loss(action_hat, action, size_average=None, reduce=None, reduction='mean') # input, target
        # optimizer_inverse.zero_grad()
        # error_inverse_model.backward(retain_graph=True)
        # optimizer_inverse.step()

        error_forward_model = torch.nn.functional.mse_loss(phi_hat_state_f, phi_state_f, size_average=None, reduce=None, reduction='mean') # input, target
        print('error_forward_model:', error_forward_model.item())
        optimizer_forward.zero_grad()
        
        time_forwad_start = time.time()
        error_forward_model.backward()
        optimizer_forward.step()
        print('forward model learn time:', time.time() - time_forward_start)

        # reward = 0
        # distance = torch.sqrt(torch.sum(torch.pow(torch.subtract(state, self.average_state), 2), dim=0))
        # distance = dist = (state - self.average_state).pow(2).sum().sqrt()
        # distance = (state - self.average_state).pow(2).sum().sqrt()
        # self.average_state = (self.average_state * self.n_states + state) / (self.n_states + 1)
        # self.n_states += 1
        # reward = (phi_hat_state - phi_state).pow(2).sum().sqrt()
        reward = error_forward_model
        # reward = error_forward_model - action_predictability_factor * error_inverse_model
        print('reward:', reward.item())
        # print('reward.shape', reward.shape)
        done = False
        info = {}
        self.previous_state = state
        self.previous_frame = frame2
        return state, reward, done, info

    def reset(self):
        # Reset the state of the environment to an initial state
        # self.state = np.zeros(3)
        # return self.state
        # hwnd = win32gui.FindWindow(None, 'EternalJK')
        # win32gui.SetForegroundWindow(hwnd)
        # bbox = win32gui.GetWindowRect(hwnd)
        # img = ImageGrab.grab(bbox).resize((input_width,input_height), PIL.Image.NEAREST)
        # img.save('input.png')
        # state = torch.tensor(np.array(img)).float().to(device)
        img = get_screenshot()
        frame1 = self.previous_frame
        frame2 = torch.tensor(np.array(img)).float().to(device)
        state = torch.cat([frame1, frame2], dim=0)
        self.average_state = state
        self.n_states = 1
        return state

    def render(self, mode='human'):
        # Render the environment to the screen
        pass

    def close(self):
        # Perform any necessary cleanup
        pass

# Hyperparameters
key_possibles = ['w', 'a', 's', 'd', 'space', 'ctrl', 'e'] # legend: [forward, left, back, right, style, alt attack, center view]
mouse_button_possibles = ['left', 'middle', 'right'] # legend: [attack, crouch, jump]
mouse_x_possibles = [-1000.0,-500.0, -300.0, -200.0, -100.0, -60.0, -30.0, -20.0, -10.0, -4.0, -2.0, -0.0, 2.0, 4.0, 10.0, 20.0, 30.0, 60.0, 100.0, 200.0, 300.0, 500.0,1000.0]
mouse_y_possibles = [-200.0, -100.0, -50.0, -20.0, -10.0, -4.0, -2.0, -0.0, 2.0, 4.0, 10.0, 20.0, 50.0, 100.0, 200.0]
n_actions = len(key_possibles)+len(mouse_button_possibles)+len(mouse_x_possibles)+len(mouse_y_possibles)
n_train_processes = 1 # 3
# learning_rate = 0.0002
# learning_rate = 1.0/802261.0
update_interval = 10 # 10 # 1 # 5
gamma = 0.98 # 0.999 # 0.98
max_train_ep = 10000000000000000000000000000 # 300
max_test_ep = 10000000000000000000000000000 #400
n_filters = 64 # 128 # 256 # 512
input_rescaling_factor = 2
input_height = input_rescaling_factor * 28
input_width = input_rescaling_factor * 28
# conv_output_size = n_filters
conv_output_size = 22464 # 44928 # 179712 # 179712 # 86528 # 346112 # 73728
# conv_output_size = 64
pooling_kernel_size = input_rescaling_factor * 2 # 16
device = torch.device("cuda")
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
forward_model_width = 4096 #2048
inverse_model_width = 1024 #2048
mouse_rescaling_factor = 10
dim_phi = 100
action_predictability_factor = 100
n_transformer_layers = 1

# Actions
# w, a, s, d, space, ctrl, mouse_left, mouse_middle, mouse_right, mouse_deltaX, mouse_deltaY

# def _get_conv_output_size(height, width, n_filters):
#     # First convolutional layer
#     height = height - 2
#     width = width - 2
    
#     # Second convolutional layer
#     height = height - 2
#     width = width - 2
    
#     # Calculate the flattened size
#     return n_filters * height * width


class ActorCritic(nn.Module):
    def __init__(self):
        super(ActorCritic, self).__init__()
        self.conv1 = nn.Conv2d(1, n_filters, 3, 1)
        self.conv2 = nn.Conv2d(n_filters, n_filters, 3, 1)
        # self.dropout1 = nn.Dropout(0.25)
        # self.dropout2 = nn.Dropout(0.5)
        self.fc_pi_pre = nn.Linear(conv_output_size, dim_phi*2)
        self.fc_v_pre = nn.Linear(conv_output_size, dim_phi*2)
        self.transformer_pi = nn.Transformer(nhead=dim_phi*2, num_encoder_layers=n_transformer_layers, num_decoder_layers=n_transformer_layers, d_model=dim_phi*2, batch_first=True)
        self.transformer_v = nn.Transformer(nhead=dim_phi*2, num_encoder_layers=n_transformer_layers, num_decoder_layers=n_transformer_layers, d_model=dim_phi*2, batch_first=True)
        self.fc_pi_post = nn.Linear(dim_phi*2, n_actions * 2)
        self.fc_v_post = nn.Linear(dim_phi*2, 1)

    def pi(self, x, softmax_dim=-1):
        if len(x.shape) == 3:
            x = torch.unsqueeze(x, dim=0)
        x = x.permute(0, 3, 1, 2)
        x = grayscale(x)
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, pooling_kernel_size)
        # x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc_pi_pre(x)
        x = self.transformer_pi(x, x)
        x = self.fc_pi_post(x)
        x = x.reshape(x.shape[0], n_actions, 2)
        prob = F.softmax(x, dim=softmax_dim)
        return prob

    def v(self, x):
        if len(x.shape) == 3:
            x = torch.unsqueeze(x, dim=0)
        x = x.permute(0, 3, 1, 2)
        x = grayscale(x)
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, pooling_kernel_size)
        # x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc_v_pre(x)
        x = self.transformer_v(x, x)
        v = self.fc_v_post(x)
        return v


class PhiModel(nn.Module):
    def __init__(self):
        super(PhiModel, self).__init__()
        self.conv1 = nn.Conv2d(1, n_filters, 3, 1)
        self.conv2 = nn.Conv2d(n_filters, n_filters, 3, 1)
        # self.dropout1 = nn.Dropout(0.25)
        # self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(conv_output_size, dim_phi)

    def forward(self, x):
        if len(x.shape) == 3:
            x = torch.unsqueeze(x, dim=0)
        x = x.permute(0, 3, 1, 2)
        x = grayscale(x)
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, pooling_kernel_size)
        # x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        return x

# class ForwardModel(nn.Module):
#     def __init__(self):
#         super(ForwardModel, self).__init__()
#         self.fc1 = nn.Linear(n_actions+dim_phi, forward_model_width)
#         self.fc2 = nn.Linear(forward_model_width, forward_model_width)
#         self.fc3 = nn.Linear(forward_model_width, dim_phi)

#     def forward(self, x):
#         # if len(x.shape) == 3:
#             # x = torch.unsqueeze(x, dim=0)
#         # x = x.permute(0, 3, 1, 2)
#         x_init = x # torch.cat([action_f, phi_previous_state_f], dim=1)
#         x = self.fc1(x)
#         x = F.relu(x)
#         x = self.fc2(x)
#         x = F.relu(x)
#         x = self.fc3(x) + torch.narrow(input=x_init, dim=1, start=n_actions, length=dim_phi) # residual skip connection
#         return x
    
class ForwardModel(nn.Module):
    def __init__(self):
        super(ForwardModel, self).__init__()
        self.transformer = nn.Transformer(nhead=n_actions+dim_phi, num_encoder_layers=n_transformer_layers, num_decoder_layers=n_transformer_layers, d_model=n_actions+dim_phi, batch_first=True)
        self.fc1 = nn.Linear(n_actions+dim_phi, dim_phi)

    def forward(self, x):
        x_init = x
        x = self.transformer(x, x)
        x = self.fc1(x) + torch.narrow(input=x_init, dim=1, start=n_actions, length=dim_phi)
        return x

# class InverseModel(nn.Module):
#     def __init__(self):
#         super(InverseModel, self).__init__()
#         self.fc1 = nn.Linear(dim_phi * 2, inverse_model_width)
#         self.fc2 = nn.Linear(inverse_model_width, inverse_model_width)
#         self.fc3 = nn.Linear(inverse_model_width, n_actions)

#     def forward(self, x):
#         # if len(x.shape) == 3:
#             # x = torch.unsqueeze(x, dim=0)
#         # x = x.permute(0, 3, 1, 2)
#         x = self.fc1(x)
#         x = F.relu(x)
#         x = self.fc2(x)
#         x = F.relu(x)
#         x = self.fc3(x)
#         return x
    
class InverseModel(nn.Module):
    def __init__(self):
        super(InverseModel, self).__init__()
        self.transformer = nn.Transformer(nhead=dim_phi*2, num_encoder_layers=n_transformer_layers, num_decoder_layers=n_transformer_layers, d_model=dim_phi*2, batch_first=True)
        self.fc1 = nn.Linear(dim_phi*2, n_actions)

    def forward(self, x):
        x = self.transformer(x, x)
        x = self.fc1(x) #+ torch.narrow(input=x_init, dim=1, start=n_actions, length=dim_phi)
        return x

# def train(global_model, phi_model, inverse_model, forward_model, rank):
def train(rank):
    n_iterations = 1
    local_model = ActorCritic().to(device)
    local_model.load_state_dict(global_model.state_dict())

    # trainable_actorcritic_params = sum(p.numel() for p in global_model.parameters() if p.requires_grad)
    # print(f'Total number of trainable actor-critic model parameters: {trainable_actorcritic_params}')
    # trainable_inverse_params = sum(p.numel() for p in list(phi_model.parameters())+list(inverse_model.parameters()) if p.requires_grad)
    # print(f'Total number of trainable inverse model parameters: {trainable_inverse_params}')
    # trainable_forward_params = sum(p.numel() for p in forward_model.parameters() if p.requires_grad)
    # print(f'Total number of trainable forward model parameters: {trainable_forward_params}')

    # if 'show' in sys.argv:
    #     sys.exit()

    # optimizer = optim.Adam(global_model.parameters(), lr=1.0/float(trainable_actorcritic_params))
    # optimizer_inverse = optim.Adam(list(inverse_model.parameters()) + list(phi_model.parameters()), lr=1.0/float(trainable_inverse_params))
    # optimizer_forward = optim.Adam(forward_model.parameters(), lr=1.0/float(trainable_forward_params))
    

    # env = gym.make('CartPole-v1')
    env = CustomEnv()

    start_time = time.time()

    for n_epi in range(max_train_ep):
        done = False
        s = env.reset()
        while not done:
            # n_iterations +=1
            # print(n_iterations)
            # print('framerate:', update_interval * n_iterations / (time.time() - start_time))
            # if (n_iterations % 1000) == 0:
            #     print('saving model..')
            #     print(n_iterations, ':', 'R:', R, 'framerate:', n_iterations / (time.time() - start_time))
            #     torch.save(global_model, 'jka_noreward_actorcritic_model.pth')
            #     torch.save(phi_model, 'jka_noreward_phi_model.pth')
            #     torch.save(inverse_model, 'jka_noreward_inverse_model.pth')
            #     torch.save(forward_model, 'jka_noreward_forward_model.pth')
            #     print('model saved.')
            s_lst, a_lst, r_lst = [], [], []
            for t in range(update_interval):
                time_iteration_start = time.time()
                n_iterations +=1
                print('--------- n_iterations:', n_iterations)
                print('framerate:', n_iterations / (time.time() - start_time))
                if (n_iterations % 1000) == 0:
                    print('saving model..')
                    print(n_iterations, ':', 'R:', R, 'framerate:', n_iterations / (time.time() - start_time))
                    torch.save(global_model, 'jka_noreward_actorcritic_model.pth')
                    torch.save(phi_model, 'jka_noreward_phi_model.pth')
                    torch.save(inverse_model, 'jka_noreward_inverse_model.pth')
                    torch.save(forward_model, 'jka_noreward_forward_model.pth')
                    print('model saved.')
                # prob = local_model.pi(torch.from_numpy(s).float())
                time_actor_start = time.time()
                prob = local_model.pi(s)
                print('actor policy inference time:', time.time() - time_actor_start)
                m = Categorical(prob)
                # a = m.sample().item()
                a = m.sample().to(device)
                # print('a.shape', a.shape)
                # print('a', a)
                # print('a[0].item()', a[0].item())
                # print('a[0].item()', a[0][0].item())

                # s_prime, r, done, info = env.step(a[0], phi_model, inverse_model, forward_model, optimizer_inverse, optimizer_forward)
                time_env_start = time.time()
                s_prime, r, done, info = env.step(a[0])
                print('environment step time:', time.time() - time_env_start)


                s_lst.append(s)
                # a_lst.append([a])
                a_lst.append(a)
                r_lst.append(r/100.0)

                s = s_prime
                if done:
                    break
                print('iteration time:', time.time() - time_iteration_start)
            


            time_update_start = time.time()


            s_final = torch.tensor(s_prime, dtype=torch.float)
            R = 0.0 if done else local_model.v(s_final).item()
            print('R:', R)
            # R = 0.0
            td_target_lst = []
            for reward in r_lst[::-1]:
                R = gamma * R + reward
                # print('R', R)
                # td_target_lst.append([R])
                # print('R.shape', R.shape)
                td_target_lst.append(torch.tensor(R))
            td_target_lst.reverse()

            # s_batch, a_batch, td_target = torch.tensor(s_lst, dtype=torch.float), torch.tensor(a_lst), torch.tensor(td_target_lst)
            # print('s_lst', s_lst)
            # s_batch = torch.tensor(s_lst, dtype=torch.float)
            # s_batch = torch.stack(s_lst, dtype=torch.float)
            s_batch = torch.stack(s_lst, dim=0)
            # print('s_batch.shape', s_batch.shape)
            # print(a_lst)
            # a_batch = torch.tensor(a_lst, dtype=torch.long)  # Ensure this is a long tensor
            a_batch = torch.stack(a_lst, dim=0)
            # a_batch = torch.cat(a_lst, dim=0)
            # td_target = torch.tensor(td_target_lst, dtype=torch.float)
            td_target = torch.stack(td_target_lst, dim=0)

            # print('s_batch.shape', s_batch.shape)
            # print('HERE')
            # print('local_model.v(s_batch).shape', local_model.v(s_batch).shape)
            # print('td_target.shape', td_target.shape)

            td_target = td_target.reshape(update_interval, 1)

            time_actor_start = time.time()
            advantage = td_target - local_model.v(s_batch)
            print('critic value model inference time:', time.time() - time_actor_start)

            # print('advantage.shape', advantage.shape)
            # print('END HERE')
            # pi = local_model.pi(s_batch, softmax_dim=1)
            pi = local_model.pi(s_batch)
            # print('s_batch.shape (before permutation)', s_batch.shape)
            a_batch = a_batch.permute(0, 2, 1) # f
            # print('s_batch.shape (after permutation)', s_batch.shape)
            pi_a = pi.gather(1, a_batch)
            advantage = advantage.unsqueeze(-1) # f
            # print('pi_a.shape', pi_a.shape)
            # print('s_batch.shape', s_batch.shape)
            # print('td_target.shape', td_target.shape)
            # print('F.smooth_l1_loss(local_model.v(s_batch), td_target.detach()).shape', F.smooth_l1_loss(local_model.v(s_batch), td_target.detach()).shape)
            # print('advantage.detach().shape', advantage.detach().shape)
            # advantage = advantage.view_as(pi_a)
            # advantage = advantage.unsqueeze(-1) 
            loss = -torch.log(pi_a) * advantage.detach() + F.smooth_l1_loss(local_model.v(s_batch), td_target.detach())
            # print('error actor critic model:', loss)
            print('mean error actor critic model:', loss.mean().item())
            optimizer.zero_grad()
            time_actor_start = time.time()
            loss.mean().backward()
            for global_param, local_param in zip(global_model.parameters(), local_model.parameters()):
                global_param._grad = local_param.grad
            optimizer.step()
            print('actor model learn time:', time.time() - time_actor_start)
            local_model.load_state_dict(global_model.state_dict())

            print('update time:', time.time() - time_update_start)

    env.close()
    print("Training process {} reached maximum episode.".format(rank))


# def test(global_model):
#     env = gym.make('CartPole-v1')
#     score = 0.0
#     print_interval = 20

#     for n_epi in range(max_test_ep):
#         done = False
#         s = env.reset()
#         while not done:
#             prob = global_model.pi(torch.from_numpy(s).float())
#             a = Categorical(prob).sample().item()
#             s_prime, r, done, info = env.step(a)
#             s = s_prime
#             score += r

#         if n_epi % print_interval == 0 and n_epi != 0:
#             print("# of episode :{}, avg score : {:.1f}".format(
#                 n_epi, score/print_interval))
#             score = 0.0
#             time.sleep(1)
#     env.close()


if __name__ == '__main__':
    global_model = ActorCritic().to(device)
    phi_model = PhiModel().to(device)
    inverse_model = InverseModel().to(device)
    forward_model = ForwardModel().to(device)
    if not "new" in sys.argv:
        print('loading model..')
        global_model = torch.load('jka_noreward_actorcritic_model.pth')
        phi_model = torch.load('jka_noreward_phi_model.pth')
        inverse_model = torch.load('jka_noreward_inverse_model.pth')
        forward_model = torch.load('jka_noreward_forward_model.pth')
        print('model loaded.')
    trainable_actorcritic_params = sum(p.numel() for p in global_model.parameters() if p.requires_grad)
    print(f'Total number of trainable actor-critic model parameters: {trainable_actorcritic_params}')
    trainable_inverse_params = sum(p.numel() for p in list(phi_model.parameters())+list(inverse_model.parameters()) if p.requires_grad)
    print(f'Total number of trainable inverse model parameters: {trainable_inverse_params}')
    trainable_forward_params = sum(p.numel() for p in forward_model.parameters() if p.requires_grad)
    print(f'Total number of trainable forward model parameters: {trainable_forward_params}')

    if 'show' in sys.argv:
        sys.exit()

    optimizer = optim.Adam(global_model.parameters(), lr=1.0/float(trainable_actorcritic_params))
    optimizer_inverse = optim.Adam(list(inverse_model.parameters()) + list(phi_model.parameters()), lr=1.0/float(trainable_inverse_params))
    optimizer_forward = optim.Adam(forward_model.parameters(), lr=1.0/float(trainable_forward_params))
    # trainable_actorcritic_params = sum(p.numel() for p in global_model.parameters() if p.requires_grad)
    # print(f'Total number of trainable actor-critic model parameters: {trainable_actorcritic_params}')
    # trainable_inverse_params = sum(p.numel() for p in list(phi_model.parameters())+list(inverse_model.parameters()) if p.requires_grad)
    # print(f'Total number of trainable inverse model parameters: {trainable_inverse_params}')
    # trainable_forward_params = sum(p.numel() for p in forward_model.parameters() if p.requires_grad)
    # print(f'Total number of trainable forward model parameters: {trainable_forward_params}')
    # sys.exit()
    global_model.share_memory()

    # train(global_model, phi_model, inverse_model, forward_model, rank=1)
    train(rank=1)
    sys.exit()

    processes = []
    for rank in range(n_train_processes + 1):  # + 1 for test process
        # if rank == 0:
        if False:
            p = mp.Process(target=test, args=(global_model,))
        else:
            p = mp.Process(target=train, args=(global_model, rank,))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
