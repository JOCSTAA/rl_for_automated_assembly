import numpy as np
from Env import make
from keras.models import load_model
from model import create_model, PPO_loss, C_PPO_loss
from model import create_model
import random

# import needed for threading
import tensorflow as tf
from tensorflow.python.keras import backend as K
import threading
from threading import Thread, Lock
lock = Lock()
import time

# configure Keras and TensorFlow sessions and graph
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)
K.set_session(sess)
graph = tf.compat.v1.get_default_graph()

LSTM = True
action_style = "contineous"
if LSTM:
    if action_style.lower() == "discrete":
        model_type = "ppo_l"
    else:
        model_type = "c_ppo_l"

else:
    if action_style.lower() == "discrete":
        model_type = "ppo"
    else:
        model_type = "c_ppo"

image_shape = (224,224,1)
lstm_cnt = 3
reward_value = 10
end_reward = 100 * reward_value
memory_size = 1000
batch_size = 10
EPOCHS = 10
save_model_at = 100

model = create_model(model_type=model_type, Image_shape=image_shape, lstm_cnt = lstm_cnt)
actor = model[0]
critic = model[1]

if LSTM:
    appendage = "l"
else:
    appendage = "_"
# actor = load_model("saved_models/general_{}actor.model".format(appendage),custom_objects={'ppo_loss_continuous': C_PPO_loss})
# critic = load_model("saved_models/general_{}critic.model".format(appendage))

allowable_error = 3
observation_style = "dif_image"
reward_style = "crude"


work_pieces = ["disk_and_shaft", "house", "ring","hammer","triangle","dumbell","square","chevron","arrow","nut_and_bolt"]
# work_pieces = ["nut_and_bolt"]

class work:
    def __init__(self, work):
        self.name = work
        self.env = make(work, allowable_error, action_style=action_style, observation_style=observation_style, reward_style=reward_style)
        self.reset()
        self.action_space = self.env.action_space()
        self.log_std = -0.5 * np.ones(self.action_space, dtype=np.float32)
        self.std = np.exp(self.log_std)
        self.oldstate = None
        self.currentstate = np.array(self.env.show_state_array(),dtype= 'int')
        self.cnt = 0
        self.done = 0
        self.successes = 0

    def reset(self):
        self.state = self.env.reset()
        self.input = []
        if LSTM:
            for i in range(lstm_cnt):
                self.input.append(self.state)

        else:self.input = self.state

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
        if LSTM:
            del self.input[0]
            self.input.append(self.state)

        else:self.input = self.state

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
                self.reset()
            else:
                self.successes += 1
                reward = end_reward
                print("EPISODE SUCCESS: MODEL TRAINED SUCCESFULY")
                self.reset()
        return self.old_state, reward, action, prediction

def remember(seed):
    memory.append(seed)
    if len(memory) > memory_size:
        del memory[0]

def save_model():
    if LSTM: appendage = "l"
    else: appendage = "_"
    if cnt % save_model_at == 0:
        print("saving model...")
        actor.save("saved_models/general_{}actor.model".format(appendage))
        critic.save("saved_models/general_{}critic.model".format(appendage))


def replay():
    if len(memory) < batch_size:
        samples = memory
    else:
        samples = random.sample(memory, batch_size)
        samples.append(memory[-1])

    states, rewards, actions, predictions = [], [], [], []
    for i in range(len(samples)):
        states.append(samples[i][0])
        rewards.append(samples[i][1])
        actions.append(samples[i][2])
        predictions.append(samples[i][3])

    disc_r = np.vstack(rewards)
    values = critic.predict(states)
    advantages = disc_r - np.array(values)

    y_true = np.hstack([advantages.tolist(), actions, predictions])
    y_true = y_true.tolist()

    actor.fit(states, y_true, epochs=EPOCHS, verbose=0, shuffle=True)
    critic.fit(states, disc_r.tolist(), epochs=EPOCHS, verbose=0, shuffle=True)
    save_model()

def update_print(work):
    # pyautogui.hotkey('CTRL', 'ALT', 'SHIFT', ',')
    print("{} LAST OBJECT: {}".format(cnt, work.name))
    print("=============================================================================================================")
    print("=============================================================================================================")
    print("   OBJECT   ||          OLD STATE          ||          NEW STATE          ||No of ACTIONS ||  SUCCESSES     ||")
    print("=============================================================================================================")
    for object in work_array:
        print("{}   ||{}||{}||{} ||{}/{}||".format(object.name, object.oldstate, object.currentstate, object.cnt, object.successes, object.done))
    print("=============================================================================================================")


def train_threading(object, pos):
    steps = 20
    global graph
    with graph.as_default():
        actor.make_predict_function()
        critic.make_predict_function()
        current_state = object.currentstate
        for i in range(steps):
            input, reward, action, prediction = object.act()
            remember([input, reward, action, prediction])

        lock.acquire()
        replay()
        lock.release()

        object.env.change_state_to(current_state)
        remember(object.act())

        update_print(object)
        global cnt
        cnt+=1
        object.env.show_progress()

work_array = []
for work_piece in work_pieces:
    work_array.append(work(work_piece))

cnt = 0
memory = []
threads = [threading.Thread(target=train_threading,daemon=True,args=(work_array[i],i))for i in range(len(work_array))]

for t in threads:
    time.sleep(2)
    t.start()

# while True:
#     object = random.choice(work_array)
#     input, reward, action, prediction = object.act()
#     remember([input, reward, action, prediction])
#     replay()
#
#
#     update_print(object)
#     object.env.show_progress()
#     cnt += 1

# Create threads
