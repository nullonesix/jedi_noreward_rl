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

import easyocr

reader = easyocr.Reader(['en'])

# import pytesseract

# Path to the Tesseract executable (only needed on Windows)
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# from transformers import TrOCRProcessor, VisionEncoderDecoderModel

# processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
# model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")

# load image from the IAM dataset
# url = "https://fki.tic.heia-fr.ch/static/img/a01-122-02.jpg"
# image = Image.open(requests.get(url, stream=True).raw).convert("RGB")

# pixel_values = processor(image, return_tensors="pt").pixel_values
# generated_ids = model.generate(pixel_values)

# generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

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
    global jka_momentum
    time_screenshot_start = time.time()
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
    if 'view' in sys.argv and n_iterations > 10:
        img.save('view.png')

    # print(img.size) # (1924, 1487)
    crop_img = img.convert("RGB").crop((img.size[0]/4.45, 19*img.size[1]/20, img.size[0]/3.65, img.size[1]))
    crop_img.save('momentum.png')
    # text = pytesseract.image_to_string(crop_img, config='--psm 7 digits')
    text = reader.readtext(np.array(crop_img))
    try:
        jka_momentum = int(text[0][1])
        print('jka_momentum:', jka_momentum)
    except:
        jka_momentum = 0
        print('non-momentum text:', text)
    img = img.resize((input_width, input_height), PIL.Image.NEAREST)
    if 'view' in sys.argv and n_iterations > 10:
        img.save('agent_view.png')
        print('agent view size:', input_width, 'x', input_height)
        sys.exit()
    print('screenshot time:', time.time() - time_screenshot_start)
    return img

def take_action(action):
    time_take_action = time.time()
    if keyboard.is_pressed('c'):
        keyboard.release(','.join(key_possibles))
        mouse.release(button='left')
        mouse.release(button='middle')
        mouse.release(button='right')
        sys.exit()
    mouse_x = 0.0
    mouse_y = 0.0
    for i in range(len(key_possibles)):
        if action[i].item() == 1:
            keyboard.press(key_possibles[i])
        else:
            keyboard.release(key_possibles[i])
    for i in range(len(mouse_button_possibles)):
        if action[i+len(key_possibles)].item() == 1:
            mouse.press(button=mouse_button_possibles[i])
        else:
            mouse.release(button=mouse_button_possibles[i])
    for i in range(len(mouse_x_possibles)):
        if action[i+len(key_possibles)+len(mouse_button_possibles)].item() == 1:
            mouse_x += mouse_x_possibles[i]
    for i in range(len(mouse_y_possibles)):
        if action[i+len(key_possibles)+len(mouse_button_possibles)+len(mouse_x_possibles)].item() == 1:
            mouse_y += mouse_y_possibles[i]
    print('mouse_dx:', mouse_x)
    print('mouse_dy:', mouse_y)
    set_pos(int(mouse_x/mouse_rescaling_factor), int(mouse_y/mouse_rescaling_factor))
    print('take action time:', time.time() - time_take_action)

class CustomEnv(gym.Env):
    """Custom Environment that follows gym interface"""

    def __init__(self):
        super(CustomEnv, self).__init__()
        self.average_state = 0
        self.n_states = 0
        img = get_screenshot()
        frame1 = torch.tensor(np.array(img)).float().to(device)
        frame2 = torch.tensor(np.array(img)).float().to(device)
        self.average_frame = frame2
        state = torch.cat([self.average_frame, frame1, frame2], dim=0)
        self.previous_frame = frame2
        self.previous_state = state
        self.n_frames_seen = 1

    def step(self, action):
        take_action(action)
        img = get_screenshot()
        frame1 = self.previous_frame
        frame2 = torch.tensor(np.array(img)).float().to(device)
        self.average_frame = (self.n_frames_seen * self.average_frame + frame2) / (self.n_frames_seen + 1)
        self.n_frames_seen += 1 
        frame2[0][0][0] = n_iterations / (10 ** 7)
        print('imprecise (Windows) time between frames:', frame2[0][0][0].item() - frame1[0][0][0].item())
        # assert frame1.mean() != frame2.mean()
        state = torch.cat([self.average_frame, frame1, frame2], dim=0)
        phi_previous_state = phi_model(self.previous_state)
        phi_state = phi_model(state)
        action_hat = inverse_model(torch.cat([phi_previous_state, phi_state], dim=1))
        action = torch.unsqueeze(action, dim=0)
        error_inverse_model = inverse_model_loss_rescaling_factor * torch.nn.functional.cross_entropy(action_hat.permute(0, 2, 1), action, size_average=None, reduce=None, reduction='mean') # input, target
        print('error_inverse_model:', error_inverse_model.item())
        optimizer_inverse.zero_grad()
        error_inverse_model.backward()
        if 'sign' in sys.argv:
            for p in list(inverse_model.parameters())+list(phi_model.parameters()):
                # assert p.grad.square().mean() > 0
                p.grad = torch.sign(p.grad)
        optimizer_inverse.step()
        phi_previous_state_f = phi_model(self.previous_state)
        action_f = action.clone()
        phi_state_f = phi_model(state)
        phi_hat_state_f = forward_model(torch.cat([action_f, phi_previous_state_f], dim=1))
        error_forward_model = torch.nn.functional.mse_loss(phi_hat_state_f, phi_state_f, size_average=None, reduce=None, reduction='mean') # input, target
        print('error_forward_model:', error_forward_model.item())
        optimizer_forward.zero_grad()
        error_forward_model.backward()
        if 'sign' in sys.argv:
            for p in forward_model.parameters():
                # assert p.grad.square().mean() > 0
                p.grad = torch.sign(p.grad)
        optimizer_forward.step()
        print('reward components:', error_forward_model, error_inverse_model, jka_momentum)
        # reward = error_forward_model - error_inverse_model + jka_momentum
        reward = error_forward_model + jka_momentum
        print('reward:', reward.item())
        done = False
        info = {}
        self.previous_state = state
        self.previous_frame = frame2
        return state, reward, done, info

    def reset(self):
        img = get_screenshot()
        frame1 = self.previous_frame
        frame2 = torch.tensor(np.array(img)).float().to(device)
        state = torch.cat([self.average_frame, frame1, frame2], dim=0)
        self.average_state = state
        self.n_states = 1
        return state

    def render(self, mode='human'):
        pass

    def close(self):
        pass

# Hyperparameters
key_possibles = ['w', 'a', 's', 'd', 'space', 'r', 'e'] # legend: [forward, left, back, right, style, use, center view]
mouse_button_possibles = ['left', 'middle', 'right'] # legend: [attack, crouch, jump]
mouse_x_possibles = [-1000.0,-500.0, -300.0, -200.0, -100.0, -60.0, -30.0, -20.0, -10.0, -4.0, -2.0, -0.0, 2.0, 4.0, 10.0, 20.0, 30.0, 60.0, 100.0, 200.0, 300.0, 500.0,1000.0]
mouse_y_possibles = [-200.0, -100.0, -50.0, -20.0, -10.0, -4.0, -2.0, -0.0, 2.0, 4.0, 10.0, 20.0, 50.0, 100.0, 200.0]
n_actions = len(key_possibles)+len(mouse_button_possibles)+len(mouse_x_possibles)+len(mouse_y_possibles)
n_train_processes = 1 # 3
update_interval = 10 # 10 # 1 # 5
gamma = 0.98 # 0.999 # 0.98
max_train_ep = 10000000000000000000000000000 # 300
max_test_ep = 10000000000000000000000000000 #400
n_filters = 64 # 128 # 256 # 512
input_rescaling_factor = 2
input_height = input_rescaling_factor * 28
input_width = input_rescaling_factor * 28
conv_output_size = 34112 # 22464 # 44928 # 179712 # 179712 # 86528 # 346112 # 73728
pooling_kernel_size = input_rescaling_factor * 2 # 16
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('using device:', device)
forward_model_width = 4096 #2048
inverse_model_width = 1024 #2048
mouse_rescaling_factor = 10
dim_phi = 100
action_predictability_factor = 100
n_transformer_layers = 1
n_iterations = 1
inverse_model_loss_rescaling_factor = 10
jka_momentum = 0

class ActorCritic(nn.Module):
    def __init__(self):
        super(ActorCritic, self).__init__()
        self.conv1 = nn.Conv2d(1, n_filters, 3, 1)
        self.conv2 = nn.Conv2d(n_filters, n_filters, 3, 1)
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
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        return x

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
    
class InverseModel(nn.Module):
    def __init__(self):
        super(InverseModel, self).__init__()
        self.transformer = nn.Transformer(nhead=dim_phi*2, num_encoder_layers=n_transformer_layers, num_decoder_layers=n_transformer_layers, d_model=dim_phi*2, batch_first=True)
        self.fc1 = nn.Linear(dim_phi*2, 2*n_actions)

    def forward(self, x):
        x = self.transformer(x, x)
        x = self.fc1(x) #+ torch.narrow(input=x_init, dim=1, start=n_actions, length=dim_phi)
        x = x.reshape(x.shape[0], n_actions, 2)
        prob = F.softmax(x, dim=2)
        return prob

def train(rank):
    global n_iterations
    local_model = ActorCritic().to(device)
    local_model.load_state_dict(global_model.state_dict())
    env = CustomEnv()
    start_time = time.time()
    for n_epi in range(max_train_ep):
        done = False
        s = env.reset()
        while not done:
            s_lst, a_lst, r_lst = [], [], []
            for t in range(update_interval):
                n_iterations +=1
                print('--------- n_iterations:', n_iterations)
                print('framerate:', n_iterations / (time.time() - start_time))
                if (n_iterations % 1000) == 0:
                    print('saving model..')
                    torch.save(global_model, 'jka_noreward_actorcritic_model.pth')
                    torch.save(phi_model, 'jka_noreward_phi_model.pth')
                    torch.save(inverse_model, 'jka_noreward_inverse_model.pth')
                    torch.save(forward_model, 'jka_noreward_forward_model.pth')
                    print('model saved.')
                prob = local_model.pi(s)
                m = Categorical(prob)
                a = m.sample().to(device)
                s_prime, r, done, info = env.step(a[0])
                s_lst.append(s)
                # a_lst.append([a])
                a_lst.append(a)
                r_lst.append(r/100.0)
                s = s_prime
                if done:
                    break
            time_update_start = time.time()
            s_final = torch.tensor(s_prime, dtype=torch.float)
            R = 0.0 if done else local_model.v(s_final).item()
            print('R:', R)
            td_target_lst = []
            for reward in r_lst[::-1]:
                R = gamma * R + reward
                td_target_lst.append(torch.tensor(R))
            td_target_lst.reverse()
            s_batch = torch.stack(s_lst, dim=0)
            a_batch = torch.stack(a_lst, dim=0)
            td_target = torch.stack(td_target_lst, dim=0)
            td_target = td_target.reshape(update_interval, 1)
            advantage = td_target - local_model.v(s_batch)
            pi = local_model.pi(s_batch)
            a_batch = a_batch.permute(0, 2, 1) # f
            pi_a = pi.gather(1, a_batch)
            advantage = advantage.unsqueeze(-1) # f 
            loss = ( -torch.log(pi_a) * advantage.detach() + F.smooth_l1_loss(local_model.v(s_batch), td_target.detach()) )
            print('mean error actor critic model:', loss.mean().item())
            optimizer.zero_grad()
            time_actor_start = time.time()
            loss.mean().backward()
            if 'sign' in sys.argv:
                for p in local_model.parameters():
                    # assert p.grad.square().mean() > 0
                    p.grad = torch.sign(p.grad)
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
    if 'sign' in sys.argv:
        print('using sign gradient descent.')
        optimizer = optim.SGD(global_model.parameters(), lr=1.0/float(trainable_actorcritic_params))
        optimizer_inverse = optim.SGD(list(inverse_model.parameters()) + list(phi_model.parameters()), lr=1.0/float(trainable_inverse_params))
        optimizer_forward = optim.SGD(forward_model.parameters(), lr=1.0/float(trainable_forward_params))
    else:
        optimizer = optim.Adam(global_model.parameters(), lr=1.0/float(trainable_actorcritic_params))
        optimizer_inverse = optim.Adam(list(inverse_model.parameters()) + list(phi_model.parameters()), lr=1.0/float(trainable_inverse_params))
        optimizer_forward = optim.Adam(forward_model.parameters(), lr=1.0/float(trainable_forward_params))
    global_model.share_memory()
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
