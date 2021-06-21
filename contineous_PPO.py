import numpy as np
from Env import make
import matplotlib.pyplot as plt
from model import create_model
from environment import show
import random

class PPO():
    def __init__(self,
                work,
                variance = 20,
                gamma = 0.95,
                show_state_progress = False,
                action_limit = False,
                max_per_episode = 2000,
                show_state_progress_val = 100,
                allowable_error = 0,
                replay_cnt= 1,
                lstm_cnt = 3,
                number_of_episodes = 100,
                save_at = 50,
                trial_limit=True,
                trial_limit_val=1000,
                show_progress=False,
                end_reward_value = 100,
                observation_style= "dif_image",#dif_image#stack_image
                reward_style = "crude",#crude#image#sparse
                action_style = "discrete",
                action_val=1,
                goal_state= [[]],
                epochs=10,
                show_image = True
                ):#discrete#contineous

        self.env = make(work, allowable_error, action_style=action_style, observation_style=observation_style,
                        goal_state=goal_state,
                        reward_style=reward_style)
        self.work = work
        self.state = self.env.reset()
        self.gamma = gamma
        self.lstm_cnt = lstm_cnt
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
        self.EPOCHS = epochs
        self.variance = variance
        self.trial_limit = trial_limit
        self.trial_limit_val = trial_limit_val
        self.show_progress = show_progress
        self.observation_style = observation_style
        self.reward_value = 10
        self.end_reward = end_reward_value * self.reward_value
        self.action_val = action_val
        self.action_style = action_style
        self.reward_style = reward_style
        self.show_image = show_image
        self.log_std = -0.5 * np.ones(self.action_space, dtype=np.float32)
        self.std = np.exp(self.log_std)
        self.memory_size = 1000
        self.batch_size = 5
        self.model_name = "testing_overfitting_problem" +"_"+ self.work +"_"+ self.reward_style +"_"+ self.action_style +"_"+ self.observation_style +"_"+ "PPO"

    def plot(self):
        np.save("graphs/PPO_"+ self.action_style +"_scores.npy", np.array(self.score))
        np.save("graphs/PPO_" + self.action_style +"_steps.npy", np.array(self.step))
        np.save("graphs/PPO_" + self.action_style + "_success.npy", np.array(self.successes))
        plt.plot(self.step, self.score)
        plt.xlabel('Number of steps (i)')
        plt.ylabel('Episode steps: speed')
        plt.savefig("graphs/" + self.model_name + ".png")
        plt.savefig('graphs/' + self.model_name + '.pdf')

    def _equal(self):
        length = len(self.memory)
        if length > 15:
            for i in range(length-12, length-2, 2):
                temp = np.array(self.memory[i])
                temp2 = np.array(self.memory[i+2])

                if not np.equal(temp[0],temp2[0]).all():
                    return False
                if temp[2] != temp2[2]:
                    return False
            return True
        else: return False

    def explore(self):
        print(self.state)
        action, predictions = self.c_model_act()

        for i in range(self.action_space):
            new_state,rew, d, f = self.env.interact(i)

            if i != action and rew > 0:
                self.memory.append([self.state, rew, predictions])

    def remember(self, reward, action, prediction):
        self.LT_memory.append([self.input, reward, action, prediction])

        if len(self.LT_memory) > self.memory_size:
            del self.LT_memory[0]

    def model_act(self):
        prediction = self.actor.predict([self.input])[0]
        action = np.random.choice(12, p=prediction)
        action_onehot = np.zeros([self.action_space])
        action_onehot[action] = 1
        return action_onehot.tolist(), prediction.tolist(), action

    def gaussian_likelihood(self, action, pred, log_std):
        pre_sum = -0.5 * (((action - pred) / (np.exp(log_std) + 1e-8)) ** 2 + 2 * log_std + np.log(2 * np.pi))
        return np.sum(pre_sum, axis=1)

    def lstm_input(self):
        if len(self.input) <= self.lstm_cnt:
            self.input = []
            for i in range(self.lstm_cnt):
                self.input.append(self.state)
        else:
            del self.input[0]
            self.input.append(self.state)

    def c_model_act(self):
        pred = self.actor.predict([self.input])

        low, high = -1.0, 1.0  # -1 and 1 are boundaries of tanh
        action = pred + np.random.uniform(low, high, size=pred.shape) * self.std
        action = np.clip(action, low, high)

        logp_t = self.gaussian_likelihood(action, pred, self.log_std)

        return action.tolist()[0], logp_t.tolist()

    def discount_rewards(self, reward):
        running_add = 0
        discounted_r = np.zeros_like(reward)

        # if reward[-1] > 0:#good ending
        #     for i in reversed(range(0, len(reward))):
        #         if running_add >= 0:
        #             running_add = (running_add * self.gamma) + reward[i]
        #             discounted_r[i] = running_add
        #         else:
        #             discounted_r[i] = reward[i]
        #
        # else:#bad ending
        #     # for i in reversed(range(0, len(reward))):
        #     #     if running_add <= 0:
        #     #         running_add = (running_add * self.gamma) + reward[i]
        #     #         discounted_r[i] = running_add
        #     #     else:
        #     #         discounted_r[i] = reward[i]
        #     return reward

        for i in reversed(range(0, len(reward))):
            running_add = (running_add * self.gamma) + reward[i]
            discounted_r[i] = running_add


        return discounted_r

    def replay(self, dream = False):
        # print("READJUSTING PARAMETERS=========================================================================================")
        if len(self.LT_memory) < self.batch_size:
            samples = self.LT_memory
        else:
            samples = random.sample(self.LT_memory, self.batch_size)
            samples.append(self.LT_memory[-1])

        states, rewards, actions, predictions = [],[],[],[]
        for i in range(len(samples)):
            states.append(samples[i][0])
            rewards.append(samples[i][1])
            actions.append(samples[i][2])
            predictions.append(samples[i][3])

        disc_r = np.vstack(rewards)
        values = self.critic.predict(states)
        advantages = disc_r - np.array(values)

        y_true = np.hstack([advantages.tolist(), actions, predictions])
        y_true = y_true.tolist()

        self.actor.fit(states, y_true, epochs=self.EPOCHS, verbose=0, shuffle=True)
        self.critic.fit(states, disc_r.tolist(), epochs=self.EPOCHS, verbose=0, shuffle=True)

    def episode(self):
        self.input = []
        done, score, cnt, fail, local_maxima_stuck = False, 0, 1, False, False
        #print("TRAINING NEW EPISODE: allowable error margin = ", self.allowable_error)
        while not done:
            self.lstm_input()
            if self.action_style == "discrete":
                action, prediction, a = self.model_act()
                new_state, reward, done, fail = self.env.interact(a)
            elif self.action_style == "contineous":
                action, prediction = self.c_model_act()
                new_state, reward, done, fail = self.env.interact(action)
            else:print("wrong action entry in episode")

            reward *= self.reward_value

            if done:
                if fail:
                    reward = -self.end_reward
                    print("EPISODE FAILED")
                    print(cnt, self.env.show_state_array())
                else:
                    reward = self.end_reward
                    print("EPISODE SUCCESS: MODEL TRAINED SUCCESFULY")
                    print(cnt, self.env.show_state_array())

            self.remember(reward, action, prediction)
            self.state = new_state

            if cnt > self.max_per_episode and self.action_limit and not done:
                print("EPISODE FAIL: Max Number of actions reached")
                print("Fail point at :" , cnt, ":", self.env.show_state_array())
                fail = True
                done = True

            if cnt % self.show_state_progress_val == 0 and self.show_state_progress:
                print(cnt,":", np.array(self.env.show_state_array(),dtype= 'int'), "reward:",reward)

            self._equal()
            self.replay()
            if done:
                self.replay(dream=True)
                break
            # self.memory = []

            if self.show_image:self.env.show_progress()
            cnt += 1

        return fail, cnt

    def train(self, model = []):
        print("TRAINING FOR PPO MODEL(", self.action_style , "):============================================================")
        if self.observation_style.lower() == "dif_image":
            shape = (224, 224, 1)
        elif self.observation_style.lower() == "stack_image":
            shape = (448, 224, 1)
        elif self.observation_style.lower() == "state_array":
            shape = (6,)
        else:
            print("wrong observation style in train")

        if self.action_style.lower() == "discrete":
            model_type = "ppo_l"

        elif self.action_style.lower() == "contineous":
            model_type = "c_ppo_l"

        else:
            print("wrong action space in train")

        try:
            self.actor, self.critic = model
        except:
            self.actor, self.critic = create_model(model_type=model_type, Image_shape=shape, lstm_cnt = self.lstm_cnt)

        number_of_succesful_episodes = 0
        number_of_failed_episodes = 0
        self.step = []
        self.score = []
        self.successes = []

        for i in range(0, self.number_of_episodes):
            self.state = self.env.reset()
            self.LT_memory = []
            if self.number_of_episodes > 1:
                print("EPISODE: ", i+1, "out of:", self.number_of_episodes+1, ": Number of succesful episodes:", number_of_succesful_episodes,": Number of failed episodes:", number_of_failed_episodes)
            print("Initial state: ", self.env.show_state_array())
            fail, cnt = self.episode()

            if fail:
                number_of_failed_episodes += 1
            else:
                number_of_succesful_episodes += 1

            if i % self.save_at == 0:
                self.actor.save("saved_models/" + self.model_name + "_actor.model")
                self.critic.save("saved_models/" + self.model_name + "_critic.model")
                self.step.append(i)
                self.score.append(cnt)
                self.successes.append(not fail)
                self.plot()

            else:
                self.step.append(i)
                self.score.append(cnt)
                self.successes.append(not fail)

        self.plot()
        return [self.actor, self.critic], self.step, self.score, self.successes

    def run(self, model):
        print("BEGINNING MODEL RUN=======================================================================================")

        self.actor = model[0]
        self.critic = model[1]
        self.state = self.env.reset()
        print("Initial state :", self.env.show_state_array())
        self.LT_memory = []
        fail, cnt = self.episode()
        # cnt,done = 0, False
        # while not done:
        #     if self.action_style == "discrete":
        #         action, prediction, a = self.model_act()
        #         new_state, reward, done, fail = self.env.interact(a)
        #     elif self.action_style == "contineous":
        #         action, prediction = self.c_model_act()
        #         new_state, reward, done, fail = self.env.interact(action)
        #     else:print("wrong action entry in episode")
        #     cnt+=1

        if fail:
            print("FAIL: max amount of actions for trial exceeded")

        else:
            print("SUCCESS: part was succesfully positioned at ", cnt, "steps")

        self.env.show_current_image()
        return self.env.show_state_array(),cnt, fail