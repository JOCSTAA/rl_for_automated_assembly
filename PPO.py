import numpy as np
from Env import make
import matplotlib.pyplot as plt
from model import create_model

class D_PPO():
    def __init__(self,
                work,
                variance,
                gamma = 0.99,
                show_state_progress = False,
                action_limit = False,
                max_per_episode = 2000,
                show_state_progress_val = 100,
                allowable_error = 0,
                replay_cnt= 3,
                number_of_episodes = 100,
                save_at = 10,
                trial_limit=True,
                trial_limit_val=1000,
                show_progress=False,
                end_reward_value = 100,
                observation_style= "stack_image",
                epochs = 10):

        self.env = make(work, allowable_error, action_style= "discrete", observation_style= "stack_image", reward_style = "crude")
        self.work = work
        self.state = self.env.reset(variance)
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

        self.model_name = "temp5_" + self.work + "_sparse_reward_discrete_PPO"

    def plot(self):
        plt.plot(self.step, self.score)
        plt.xlabel('Number of steps (i)')
        plt.ylabel('Episode steps: speed')
        plt.savefig("graphs/"+self.model_name+".png")
        plt.savefig('graphs/'+self.model_name+'.pdf')

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
        return action, prediction

    def explore(self):
        print(self.state)
        action, predictions = self.model_act()

        for i in range(self.action_space):
            new_state,rew, d, f = self.env.interact(i)

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
        #print("READJUSTING PARAMETERS=========================================================================================")
        samples = np.array(self.memory)

        states, actions, rewards, predictions = [],[],[],[]

        for i in range(len(samples)):
            states.append(samples[i, 0])
            actions.append(samples[i, 1])
            rewards.append(samples[i, 2])
            predictions.append(samples[i, 3])

        if abs(rewards[len(rewards) - 1]) > self.reward_value:
            disc_r = self.discount_rewards(rewards)
        else:
            disc_r = rewards

        states = np.vstack(states)
        actions = np.vstack(actions)
        predictions = np.vstack(predictions)
        disc_r = np.vstack(disc_r)
        values = self.critic.predict(states)
        advantages = disc_r - values

        y_true = np.hstack([advantages, predictions, actions])
        self.actor.fit(states, y_true, epochs=self.EPOCHS, verbose=0, shuffle=True)
        self.critic.fit(states, disc_r, epochs=self.EPOCHS, verbose=0, shuffle=True)

    def episode(self):
        self.memory = []
        done, score, cnt, fail, local_maxima_stuck = False, 0, 1, False, False
        print("TRAINING NEW EPISODE: allowable error margin = ", self.allowable_error)
        while not done:
            action, prediction = self.model_act()
            new_state, reward, done, fail = self.env.interact(action)
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

            action_onehot = np.zeros([self.action_space])
            action_onehot[action] = 1
            self.memory.append([self.state, action_onehot, reward, prediction])

            self.state = new_state

            if cnt > self.max_per_episode and self.action_limit and not done:
                print("EPISODE FAIL: Max Number of actions reached")
                print("Fail point at :" , cnt, ":", self.env.show_state_array())
                fail = True
                done = True

            if cnt % self.show_state_progress_val == 0 and self.show_state_progress:
                print(cnt,":", self.env.show_state_array(), "reward:",reward)
                #print(prediction)


            if done or len(self.memory) % self.replay_cnt == 0 :
                self._equal()
                self.replay()
                if done:
                    self.replay()
                    break
                self.memory = []

            self.env.show_progress()
            cnt += 1

        return fail, cnt

    def train(self):
        if self.observation_style.lower() == "dif_image": self.actor, self.critic = create_model("ppo")
        elif self.observation_style.lower() == "stack_image":self.actor, self.critic = create_model("ppo", Image_shape= (448,224,1))
        elif self.observation_style.lower() == "state_array": self.actor, self.critic = create_model("ppo", Image_shape= (6,))
        else: print("wrong observation style in train")

        number_of_succesful_episodes = 0
        number_of_failed_episodes = 0
        self.step = []
        self.score = []

        for i in range(1, self.number_of_episodes+1):
            self.state = self.env.reset(self.variance)
            print("EPISODE ", i, ": Number of succesful episodes:", number_of_succesful_episodes,": Number of failed episodes:", number_of_failed_episodes)
            print("Initial state: ", self.env.show_state_array())
            fail, cnt = self.episode()

            if fail:
                number_of_failed_episodes += 1
            else:
                number_of_succesful_episodes += 1

            if i % self.save_at == 0:
                self.actor.save("saved_models/"+self.model_name+"_actor.model")
                self.critic.save("saved_models/"+self.model_name+"_critic.model")
                self.step.append(i - 1)
                self.score.append(self.max_per_episode - cnt)
                self.plot()

            else:
                self.step.append(i)
                self.score.append(self.max_per_episode-cnt)

        self.plot()
        return [self.actor, self.critic], self.step, self.score

    def run(self, model):
        print("BEGINNING MODEL RUN=======================================================================================")

        self.actor = model
        self.state = self.env.reset(self.variance)
        print("Initial state :", self.env.show_state_array())

        cnt = 0
        fail, done = False, False
        while not done:
            action, prediction = self.model_act()
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
            return self.env.show_state_array()
