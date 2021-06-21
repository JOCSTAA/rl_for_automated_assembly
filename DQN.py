import numpy as np
from Env import make
import matplotlib.pyplot as plt
from model import create_model
from keras.models import load_model

class DQN():
    def __init__(self,
                work,
                variance = 35,
                gamma=0.975,
                show_state_progress=False,
                action_limit=False,
                max_per_episode=2000,
                show_state_progress_val=100,
                decay_factor=0.999,
                eps_min=0.01,
                eps=0.75,
                tau=0.125,
                shake_up=False,
                shake_up_val=2000,
                allowable_error=0,
                replay_cnt= 1,
                number_of_episodes = 100,
                save_at = 50,
                trial_limit=True,
                trial_limit_val=1000,
                show_progress=False,
                end_reward_value = 100,
                observation_style= "dif_image",#dif_image#stack_image
                reward_style = "crude",#crude#image#sparse
                action_val = 1):#discrete#contineous

        self.env = make(work, allowable_error, action_style= "discrete", observation_style= observation_style, reward_style = reward_style)
        self.work = work
        self.state = self.env.reset()
        self.gamma = gamma
        self.show_state_progress = show_state_progress
        self.action_limit = action_limit
        self.max_per_episode = max_per_episode
        self.show_state_progress_val = show_state_progress_val
        self.allowable_error = float(allowable_error)
        self.replay_cnt = replay_cnt
        self.action_space = self.env.action_space()
        self.number_of_episodes = number_of_episodes
        self.save_at = save_at
        self.memory = []
        self.eps_initial = eps
        self.shake_up = shake_up
        self.shake_up_val = shake_up_val
        self.decay_factor = decay_factor
        self.eps_min = eps_min
        self.tau = tau
        self.variance = variance
        self.trial_limit = trial_limit
        self.trial_limit_val = trial_limit_val
        self.show_progress = show_progress
        self.observation_style = observation_style
        self.reward_value = 1
        self.end_reward = end_reward_value * self.reward_value
        self.action_val = action_val
        self.reward_style = reward_style
        self.log_std = -0.5 * np.ones(self.action_space, dtype=np.float32)
        self.std = np.exp(self.log_std)
        self.batch_size = 200
        self.model_name = "abg" +"_"+ self.work +"_"+ self.reward_style +"_"+ self.observation_style +"_"+ "DQN"

    def plot(self, run = False):
        if run:
            self.model_name.append("_run")
        np.save("graphs/DQN_scores.npy", np.array(self.score))
        np.save("graphs/DQN_steps.npy", np.array(self.step))
        np.save("graphs/DQN_success.npy", np.array(self.successes))
        plt.plot(self.step, self.score)
        plt.xlabel('Number of steps (i)')
        plt.ylabel('Episode steps: speed')
        plt.savefig("graphs/" + self.model_name + ".png")
        plt.savefig('graphs/' + self.model_name + '.pdf')


        avg = np.mean(self.score)
        print("saving model average now:", avg, self.average)
        if avg <= self.average:
            self.predictor.save("saved_models/" + self.model_name + "_dqn.model")
            self.average = avg

        self.predictor = load_model("saved_models/" + self.model_name + "_dqn.model")

    def model_act(self):
        if np.random.random() < self.eps:
            action = np.random.randint(0, 12)
            # print("exploring", action)
        else:
            action = np.argmax(self.predictor.predict(self.state))
            # print("exploiting", action)
        return action

    def target_train(self):
        self.predictor.set_weights(self.fitter.get_weights())

        # weights = self.fitter.get_weights()
        # target_weights = self.predictor.get_weights()
        # for i in range(len(target_weights)):
        #     target_weights[i] = weights[i] * self.tau + target_weights[i] * (1 - self.tau)
        # self.predictor.set_weights(target_weights)

    def replay(self):
        # print("READJUSTING PARAMETERS=========================================================================================")
        samples = np.array(self.memory)

        states, targets = [], []

        for i in range(len(samples)):
            state = samples[i, 0]
            reward = samples[i,1]
            action = samples[i,2]
            new_state = samples[i,3]

            target = self.predictor.predict(state)
            Q_future = max(self.predictor.predict(new_state)[0])
            target[0][action] = reward + self.gamma * Q_future

            states.append(state)
            targets.append(target)

        states = np.vstack(states)
        targets = np.vstack(targets)

        self.fitter.fit(states, targets, epochs=1, verbose=0)
        self.target_train()

    def episode(self):
        self.memory = []
        done, score, cnt, fail, local_maxima_stuck = False, 0, 1, False, False
        print("TRAINING NEW EPISODE: allowable error margin = ", self.allowable_error)

        while not done:
            self.eps = max(self.eps, self.eps_min)
            action = self.model_act()
            new_state, reward, done, fail = self.env.interact(action, increment= self.action_val)
            reward *= self.reward_value

            if done:
                if fail:
                    print("EPISODE FAILED")
                    print(cnt, self.env.show_state_array())
                else:
                    print("EPISODE SUCCESS: MODEL TRAINED SUCCESSFULLY")
                    print(cnt, self.env.show_state_array())

            self.memory.append([self.state, reward, action, new_state])

            self.state = new_state

            if cnt > self.max_per_episode and self.action_limit and not done:
                print("EPISODE FAIL: Max Number of actions reached")
                print("Fail point at :", cnt, ":", self.env.show_state_array())
                fail = True
                done = True

            if cnt % self.show_state_progress_val == 0 and self.show_state_progress:
                print(cnt, ":", np.array(self.env.show_state_array(), dtype='int'), "reward:", reward)

            if done or len(self.memory) % self.replay_cnt == 0:
                self.replay()
                if done:
                    break
                self.memory = []

            if self.shake_up and cnt % self.shake_up_val == 0:
                self.eps = 0.5

            self.eps *= self.decay_factor
            self.env.show_progress()
            cnt += 1

        return fail, cnt

    def train(self,model = []):
        print("TRAINING FOR DQN MODEL====================================================================================")
        if self.observation_style.lower() == "dif_image":
            shape = (224, 224, 1)
        elif self.observation_style.lower() == "stack_image":
            shape = (448, 224, 1)
        elif self.observation_style.lower() == "state_array":
            shape = (6,)
        else:
            print("wrong observation style in train")

        self.fitter = create_model(model_type = "dqn", Image_shape=shape)
        self.predictor = create_model(model_type = "dqn", Image_shape=shape)

        number_of_succesful_episodes = 0
        number_of_failed_episodes = 0
        self.step = []
        self.score = []
        self.successes = []
        self.average = self.max_per_episode
        self.eps = self.eps_initial

        for i in range(0, self.number_of_episodes):
            self.eps = self.eps_initial
            self.state = self.env.reset()
            print("EPISODE ", i, ": Number of succesful episodes:", number_of_succesful_episodes,": Number of failed episodes:", number_of_failed_episodes)
            print("Initial state: ", self.env.show_state_array())
            fail, cnt = self.episode()

            if fail:
                number_of_failed_episodes += 1
            else:
                number_of_succesful_episodes += 1

            if i % self.save_at == 0:
                self.step.append(i)
                self.score.append(cnt)
                self.successes.append(not fail)
                self.plot()

            else:
                self.step.append(i)
                self.score.append(cnt)
                self.successes.append(not fail)

        self.plot()
        return [self.predictor], self.step, self.score, self.successes

    def run(self, model):
        print("BEGINNING MODEL RUN=======================================================================================")

        self.predictor = model
        self.fitter = model
        self.state = self.env.reset()
        self.eps = 0.5
        print("Initial state :", self.env.show_state_array())

        cnt = 0
        fail, done = False, False
        while not done:
            self.memory=[]
            action = self.model_act()
            self.eps *= self.decay_factor
            print(self.predictor.predict(self.state), action)

            new_state, reward, done, fail = self.env.interact(action, increment= self.action_val)
            print(reward)

            if cnt > self.trial_limit_val and self.trial_limit:
                fail = True
                done = True

            if self.show_progress:
                self.env.show_progress()

            self.memory.append([self.state, reward, action, new_state])
            self.replay()

            self.state = new_state
            cnt += 1

        if fail:
            print("FAIL: max amount of actions for trial exceeded")

        else:
            print("SUCCESS: part was succesfully positioned at ", cnt, "steps")
        return self.env.show_state_array(), cnt, fail
