import random
import numpy as np
from environment import make_image, imshow, dif_prep
from action import rand_reset,zero_reset, act
from reward import crude_reward
from make_reference_images import collect_work_variables
import threading
from threading import Thread, Lock
import time

class WD_PPO():
    def __init__(self,
                NN_model,
                work,
                state = zero_reset(),
                gamma = 0.99,
                show_state_progress = False,
                action_limit = False,
                max_per_episode = 2000,
                show_state_progress_val = 100,
                allowable_error = 0,
                replay_cnt= 5,
                action_space = 12,
                number_of_episodes = 100,
                save_at = 10,
                variance= 20):

        self.actor = NN_model[0]
        self.critic = NN_model[1]
        self.work = work
        self.initail_state = state
        self.state = state
        self.gamma = gamma
        self.show_state_progress = show_state_progress
        self.action_limit = action_limit
        self.max_per_episode = max_per_episode
        self.show_state_progress_val = show_state_progress_val
        self.allowable_error = float(allowable_error)
        self.replay_cnt = replay_cnt
        self.action_space = action_space
        self.memory = []
        self.number_of_episodes = number_of_episodes
        self.save_at = save_at
        self.goal_state, self.CAD_model_path, self.iso_details = collect_work_variables(work)
        self.goal_image = make_image(self.goal_state, self.CAD_model_path, self.iso_details)
        self.batch_size = replay_cnt
        self.EPOCHS = 10
        self.variance = variance
        self.end_reward = 5

    def _equal(self):
        length = len(self.memory)

        for i in range(length - 12, length-2, 2):

            v = np.array(self.memory[i])
            b = np.array(self.memory[i+2])

            if not np.equal(v[0],b[0]).all():
                return False
            if v[2] != b[2]:
                return False
        return True

    def model_act(self):
        prediction = self.actor.predict(self.current_state_model_input)[0]
        action = np.random.choice(12, p=prediction)
        return action, prediction

    def explore(self):
        initial_state = self.state
        print(initial_state)
        action, predictions = self.model_act()

        for i in range(self.action_space):
            new_state = act(i, self.state)
            new_image = make_image(new_state, self.CAD_model_path, self.iso_details)
            rew, d, f = crude_reward(self.goal_state, self.state, new_state, self.current_image, new_image)

            if i != action and rew > 0:
                action_onehot = np.zeros([self.action_space])
                action_onehot[action] = 1
                self.memory.append([self.state, action_onehot, rew, predictions])


    def discount_rewards(self, reward):
        running_add = 0
        discounted_r = np.zeros_like(reward)
        for i in reversed(range(0, len(reward))):
            running_add = (running_add * self.gamma) + reward[i]
            discounted_r[i] = running_add

        return discounted_r

    def replay(self):
        print("READJUSTING PARAMETERS=========================================================================================")
        length = len(self.memory)
        if length > self.batch_size:
            # samples = random.sample(self.memory, self.batch_size)
            # for i in range(length - self.replay_cnt, length):
            #     samples.append(self.memory[i])
            samples = self.memory[length-self.batch_size:length]

        else:
            samples = np.array(self.memory)

        samples = np.array(samples)
        states, actions, rewards, predictions = [],[],[],[]
        for i in range(len(samples)):
            states.append(dif_prep(self.goal_image, make_image(samples[i, 0], self.CAD_model_path, self.iso_details)))
            actions.append(samples[i, 1])
            rewards.append(samples[i, 2])
            predictions.append(samples[i, 3])

        if abs(rewards[len(rewards) - 1]) > 1:
            disc_r = self.discount_rewards(rewards)
            print(disc_r)
        else:
            disc_r = rewards
        states = np.vstack(states)
        actions = np.vstack(actions)
        predictions = np.vstack(predictions)
        disc_r = np.vstack(disc_r)
        values = self.critic.predict(states)
        advantages = disc_r - values

        # stack everything to numpy array
        y_true = np.hstack([advantages, predictions, actions])
        # training Actor and Critic networks
        self.actor.fit(states, y_true, epochs=self.EPOCHS, verbose=0, shuffle=True)
        self.critic.fit(states, disc_r, epochs=self.EPOCHS, verbose=0, shuffle=True)

    def episode(self):
        done, score, cnt, fail, local_maxima_stuck = False, 0, 1, False, False
        print("allowable error margin = ", self.allowable_error)
        while not done:

            self.current_image = make_image(self.state, self.CAD_model_path, self.iso_details)
            self.current_state_model_input = dif_prep(self.goal_image, self.current_image)
            Action, Prediction = self.model_act()
            new_state = act(Action, self.state)
            new_image = make_image(new_state, self.CAD_model_path, self.iso_details)
            # there no action at any state that would produce the same state except when part is out of frame
            rew, done, fail = crude_reward(self.goal_state, self.state, new_state, self.current_image, new_image, allowable_error=self.allowable_error)

            if done:
                if np.array_equal(self.current_image, new_image):
                    rew += -self.end_reward
                    print(fail, rew)
                else:
                    rew += self.end_reward
                    print(fail, "true", rew)

            print(cnt, rew, new_state)
            action_onehot = np.zeros([self.action_space])
            action_onehot[Action] = 1
            self.memory.append([self.state, action_onehot, rew, Prediction])

            self.state = new_state

            length = len(self.memory)
            if length > 15:
                local_maxima_stuck = self._equal()

            if local_maxima_stuck:
                print("local maxima reached: RELOCATING")
                print(cnt, ":", self.state)

                for i in range(length - 12, length):
                    if self.memory[i][2] == 0:
                        self.memory[i][2] = -1
                self.explore()
                print(Prediction)
                self.replay()

            if cnt > self.max_per_episode and self.action_limit and not done:
                print("EPISODE FAIL: Max Number of actions reached")
                print(cnt, ":", self.state)
                fail = True
                done = True

            if cnt % self.show_state_progress_val == 0 and self.show_state_progress:
                print(cnt, ":", self.state)


            if done or len(self.memory) % self.replay_cnt == 0 :
                print(Prediction)
                self.replay()
                if done:
                    self.replay()
                    break

            imshow(np.vstack((self.current_image, self.goal_image, np.subtract(self.current_image, self.goal_image))))
            cnt += 1

        return fail, cnt

    def train(self):

        number_of_succesful_episodes = 0
        number_of_failed_episodes = 0
        step = []
        score = []

        for i in range(1, self.number_of_episodes+1):
            self.memory = []
            self.state = rand_reset(self.variance)
            print("EPISODE ", i, ": number of succesful episodes:", number_of_succesful_episodes,": number of failed episodes:", number_of_failed_episodes)
            fail, cnt = self.episode()

            if fail:
                number_of_failed_episodes += 1
            else:
                number_of_succesful_episodes += 1

            if i % self.save_at == 0:
                self.actor.save("saved_models/temp_" + self.work + "_PPO_actor.model")
                self.critic.save("saved_models/temp_" + self.work + "_PPO_critic.model")

            step.append(i-1)
            score.append(self.max_per_episode-cnt)

        #plot image here and save plot
        return [self.actor, self.critic], step, score