import numpy as np
from Env import make
import matplotlib.pyplot as plt
from model import create_model

class AC():
    def __init__(self,
                work,
                variance = 20,
                gamma = 0.9,
                show_state_progress = False,
                action_limit = False,
                max_per_episode = 2000,
                show_state_progress_val = 100,
                allowable_error = 0,
                replay_cnt= 1,
                number_of_episodes = 100,
                save_at = 50,
                trial_limit=True,
                trial_limit_val=1000,
                show_progress=False,
                end_reward_value = 100,
                observation_style= "stack_image",#dif_image#stack_image
                reward_style = "crude",#crude#image#sparse
                action_style = "discrete",
                action_val=1,
                epochs=10):#discrete#contineous

        self.env = make(work, allowable_error, action_style= action_style, observation_style= observation_style, reward_style = reward_style)
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
        self.log_std = -0.5 * np.ones(self.action_space, dtype=np.float32)
        self.std = np.exp(self.log_std)
        self.batch_size = 200
        self.model_name = "temp" +"_"+ self.work +"_"+ self.reward_style +"_"+ self.action_style +"_"+ self.observation_style +"_"+ "AC"

    def plot(self):
        np.save("graphs/AC_"+ self.action_style +"scores.npy", np.array(self.score))
        np.save("graphs/AC_"+ self.action_style +"steps.npy", np.array(self.step))
        np.save("graphs/AC_"+ self.action_style +"success.npy", np.array(self.successes))
        plt.plot(self.step, self.score)
        plt.xlabel('Number of steps (i)')
        plt.ylabel('Episode steps: speed')
        plt.savefig("graphs/" + self.model_name + ".png")
        plt.savefig('graphs/' + self.model_name + '.pdf')

    def remember(self, reward, action, new_state):
        self.LT_memory.append([self.state, reward, action, new_state])

        if len(self.LT_memory) > self.batch_size:
            del self.LT_memory[0]

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

    def model_act(self):
        prediction = self.actor.predict(self.state)[0]
        action = np.random.choice(12, p=prediction)
        action_onehot = np.zeros([self.action_space])
        action_onehot[action] = 1
        return action_onehot, action

    def c_model_act(self):
        prediction = self.actor.predict(self.state)[0]
        action = prediction
        return action

    def explore(self):
        print(self.state)
        action, predictions = self.model_act()

        for i in range(self.action_space):
            new_state,rew, d, f = self.env.interact(i)

            if i != action and rew > 0:
                self.memory.append([self.state, rew, predictions])

    def discount_rewards(self, reward):
        running_add = 0
        discounted_r = np.zeros_like(reward)

        for i in reversed(range(0, len(reward))):
            if running_add >= 0:
                running_add = (running_add * self.gamma) + reward[i]
                discounted_r[i] = running_add
            else:
                discounted_r[i] = reward[i]

        return discounted_r

    def replay(self, dream = False):
        # print("READJUSTING PARAMETERS=========================================================================================")
        samples = np.array(self.memory)
        if dream:
            samples = np.array(self.LT_memory)

        states, rewards, actions  = [],[],[]

        for i in range(len(samples)):
            states.append(samples[i, 0])
            rewards.append(samples[i, 1])
            actions.append(samples[i, 2])

        if rewards[len(rewards) - 1] > self.reward_value:
            disc_r = self.discount_rewards(rewards)
        else:
            disc_r = rewards

        states = np.vstack(states)
        actions = np.vstack(actions)
        disc_r = np.vstack(disc_r)
        values = self.critic.predict(states)
        advantages = disc_r - values

        self.actor.fit(states, actions, sample_weight=advantages, epochs=1, verbose=0)
        self.critic.fit(states, disc_r, epochs=1, verbose=0)

    def episode(self):
        self.memory = []
        done, score, cnt, fail, local_maxima_stuck = False, 0, 1, False, False
        print("TRAINING NEW EPISODE: allowable error margin = ", self.allowable_error)
        while not done:
            if self.action_style == "discrete":
                action,a = self.model_act()
                new_state, reward, done, fail = self.env.interact(a)
            elif self.action_style == "contineous":
                action = self.c_model_act()
                new_state, reward, done, fail = self.env.interact(action)
            else:print("wrong action entry in episode")

            reward *= self.reward_value

            if done:
                if fail:
                    reward += -self.end_reward
                    print("EPISODE FAILED")
                    print(cnt, self.env.show_state_array())
                else:
                    reward += self.end_reward
                    print("EPISODE SUCCESS: MODEL TRAINED SUCCESFULY")
                    print(cnt, self.env.show_state_array())

            self.memory.append([self.state, reward, action])

            self.state = new_state

            if cnt > self.max_per_episode and self.action_limit and not done:
                print("EPISODE FAIL: Max Number of actions reached")
                print("Fail point at :" , cnt, ":", self.env.show_state_array())
                fail = True
                done = True

            if cnt % self.show_state_progress_val == 0 and self.show_state_progress:
                print(cnt,":", np.array(self.env.show_state_array(),dtype= 'int'), "reward:",reward)

            if done or len(self.memory) % self.replay_cnt == 0:
                self._equal()
                self.replay()
                if done:
                    self.replay(dream=True)
                    break
                self.memory = []

            self.env.show_progress()
            cnt += 1

        return fail, cnt

    def train(self):
        print("TRAINING FOR ACTOR CRITIC MODEL===========================================================================")
        if self.observation_style.lower() == "dif_image":
            shape = (224, 224, 1)
        elif self.observation_style.lower() == "stack_image":
            shape = (448, 224, 1)
        elif self.observation_style.lower() == "state_array":
            shape = (6,)
        else:
            print("wrong observation style in train")

        if self.action_style.lower() == "discrete":
            model_type = "ac"

        elif self.action_style.lower() == "contineous":
            model_type = "c_ac"

        else:
            print("wrong action space in train")

        self.actor, self.critic = create_model(model_type = model_type, Image_shape=shape)

        number_of_succesful_episodes = 0
        number_of_failed_episodes = 0
        self.step = []
        self.score = []
        self.successes = []

        for i in range(0, self.number_of_episodes):
            self.state = self.env.reset()
            self.LT_memory = []
            print("EPISODE ", i, ": Number of succesful episodes:", number_of_succesful_episodes,": Number of failed episodes:", number_of_failed_episodes)
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

        self.actor = model
        self.state = self.env.reset()
        print("Initial state :", self.env.show_state_array())

        cnt = 0
        fail, done = False, False
        while not done:
            if self.action_style == "discrete":
                action, prediction = self.model_act()
            elif self.action_style == "contineous":
                action, prediction = self.c_model_act()
            else:
                print("wrong action entry in run")
            new_state, reward, done, fail = self.env.interact(action)

            if cnt > self.trial_limit_val and self.trial_limit:
                fail = True
                done = True

            if self.show_progress:
                self.env.show_progress()

            self.state = new_state
            cnt += 1

        if fail:
            print("FAIL: max amount of actions for trial exceeded")

        else:
            print("SUCCESS: part was succesfully positioned at ", cnt, "steps")
            return self.env.show_state_array(), cnt, fail