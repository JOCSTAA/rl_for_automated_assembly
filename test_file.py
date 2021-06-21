import os
from time import sleep
import pyautogui
import numpy as np
from Env import make
import matplotlib.pyplot as plt
from model import create_model
from environment import show
import random

action_style = "discrete"
if action_style.lower() == "discrete":
    model_type = "ppo"
else:
    model_type = "c_ppo"

image_shape = (224,224,1)
lstm_cnt = 3
reward_value = 10
end_reward = 100 * reward_value
memory_size = 1000
batch_size = 5
EPOCHS = 10
save_model_at = 100

model = create_model(model_type=model_type, Image_shape=image_shape, lstm_cnt = lstm_cnt)
actor = model[0]
critic = model[1]

allowable_error = 3
observation_style = "dif_image"
reward_style = "crude"

def clear():
    pyautogui.hotkey('CTRL','ALT', 'SHIFT', ',')

    # or

    # print("\n"*10000)


class work:
    def __init__(self, work):
        self.name = work
        self.env = make(work, allowable_error, action_style=action_style, observation_style=observation_style, reward_style=reward_style)
        self.state = self.env.reset()
        self.input = []
        for i in range(lstm_cnt):
            self.input.append(self.state)

        self.action_space = self.env.action_space()
        self.log_std = -0.5 * np.ones(self.action_space, dtype=np.float32)
        self.std = np.exp(self.log_std)
        self.oldstate = None
        self.currentstate = np.array(self.env.show_state_array(),dtype= 'int')
        self.cnt = 0
        self.done = 0
        self.successes = 0


    def gaussian_likelihood(self, action, pred, log_std):
        pre_sum = -0.5 * (((action - pred) / (np.exp(log_std) + 1e-8)) ** 2 + 2 * log_std + np.log(2 * np.pi))
        return np.sum(pre_sum, axis=1)

    def c_act(self):
        pred = actor.predict([self.input])

        low, high = -1.0, 1.0  # -1 and 1 are boundaries of tanh
        action = pred + np.random.uniform(low, high, size=pred.shape) * self.std
        action = np.clip(action, low, high)

        logp_t = self.gaussian_likelihood(action, pred, self.log_std)

        return action.tolist()[0], logp_t.tolist()

    def d_act(self):
        prediction = actor.predict([self.input])[0]
        action = np.random.choice(12, p=prediction)
        action_onehot = np.zeros([self.action_space])
        action_onehot[action] = 1
        return action_onehot.tolist(), prediction.tolist(), action

    def update(self):
        del self.input[0]
        self.input.append(self.state)

    def act(self):
        self.cnt += 1
        self.oldstate = np.array(self.env.show_state_array(),dtype= 'int')
        if action_style == "discrete":
            action, prediction, a = self.d_act()
            new_state, reward, done, fail = self.env.interact(a)
        else:
            action, prediction = self.c_act()
            new_state, reward, done, fail = self.env.interact(action)

        reward *= reward_value

        self.old_state = self.input
        self.state = new_state
        self.update()
        self.currentstate = np.array(self.env.show_state_array(),dtype= 'int')

        if done:
            self.cnt = 0
            self.done += 1
            if fail:
                reward = -end_reward
                print("EPISODE FAILED")
                self.state = self.env.reset()
                self.input = []
                for i in range(lstm_cnt):
                    self.input.append(self.state)
            else:
                self.successes += 1
                reward = end_reward
                print("EPISODE SUCCESS: MODEL TRAINED SUCCESFULY")
                self.state = self.env.reset()
                self.input = []
                for i in range(lstm_cnt):
                    self.input.append(self.state)
        return self.old_state, reward, action, prediction

work_pieces = ["disk_and_shaft", "house", "ring","hammer","triangle","dumbell","square","chevron","arrow","nut_and_bolt"]

work_array = []
for work_piece in work_pieces:
    work_array.append(work(work_piece))

def update_print():
    pyautogui.hotkey('CTRL', 'ALT', 'SHIFT', ',')
    print("=============================================================================================================")
    print("=============================================================================================================")
    print("   OBJECT   ||          OLD STATE          ||          OLD STATE          ||No of ACTIONS ||  SUCCESSES     ||")
    print("=============================================================================================================")
    for object in work_array:
        print("   {}   ||          {}          ||          {}          ||{} ||{}/{}||".format(object.name, object.oldstate, object.currentstate, object.cnt, object.successes, object.done))
    print("=============================================================================================================")





cnt = 0
while cnt < 20:
    update_print()
    sleep(1)
    cnt+=1

