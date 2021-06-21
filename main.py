import numpy as np
from environment import show, make_image
from action import rand_reset,zero_reset, act
from reward import image_reward, checkifdone
from model import create_model, PPO_loss, C_PPO_loss
from make_reference_images import collect_work_variables
from keras.models import load_model
import keras.backend as K
from DQN import DQN
from Actor_critic import AC
from contineous_PPO import PPO
from math import floor
from matplotlib import pyplot as plt
import random

class work():
    def __init__(self, work):
        goal_state, CAD_path, iso_details = collect_work_variables(work)
        self.goal_images = make_image(goal_state, CAD_path, iso_details, usage="reward")
        self.name = work
        self.CAD_path = CAD_path
        self.iso_details = iso_details
        self.state = zero_reset()

    def show_image(self):
        img = make_image(self.state, self.CAD_path, self.iso_details, usage = "show")
        show(img)

    def action(self, action_value):
        self.new_state = act(action_value, self.state)

        self.prev_image = make_image(self.state, self.CAD_path, self.iso_details)
        self.new_image = make_image(self.new_state, self.CAD_path, self.iso_details)

        rew = image_reward(self.goal_images[1], self.prev_image, self.new_image)
        done = checkifdone(self.goal_images[1], self.new_image, allowable_error=0)

        return self.state,rew, done

    def train(self, number_of_episodes = 500,
              show_state_progress = True,
              method = "ppo",
              action_style = "contineous",
              gamma=0.99,
              variance = 20,
              action_limit=True,
              max_per_episode=6000,
              decay_factor=0.999,
              eps_min=0.01,
              eps=0.75,
              allowable_error = 0,
              reset = "rand",
              show_image = True,
              model=[],
              goal_state = [],
              show_state_progress_val=100,
              save_at = 10):

        if method.lower() == "dqn":
            self.trainer = DQN(self.name,
                               gamma=gamma,
                               action_limit=action_limit,
                               max_per_episode=max_per_episode,
                               show_state_progress=show_state_progress,
                               show_state_progress_val=show_state_progress_val,
                               allowable_error=allowable_error,
                               number_of_episodes=number_of_episodes,
                               save_at=save_at)

        elif method.lower() == "ppo":
            self.trainer = PPO(self.name,
                            action_style= action_style,
                            gamma=gamma,
                            action_limit=action_limit,
                            max_per_episode=max_per_episode,
                            show_state_progress=show_state_progress,
                            show_state_progress_val=show_state_progress_val,
                            allowable_error=allowable_error,
                            number_of_episodes=number_of_episodes,
                            goal_state = goal_state,
                            show_image = show_image,
                            save_at=save_at)

        elif method.lower() == "ac":
            self.trainer = AC(self.name,
                            action_style= action_style,
                            gamma=gamma,
                            action_limit=action_limit,
                            max_per_episode=max_per_episode,
                            show_state_progress=show_state_progress,
                            show_state_progress_val=show_state_progress_val,
                            allowable_error=allowable_error,
                            number_of_episodes=number_of_episodes,
                            save_at=save_at)

        else:print("wrong method entry in work / train")

        self.NN_model, self.steps, self.score, self.successes = self.trainer.train(model = model)
        return self.NN_model, self.steps, self.score, self.successes

    def run(self,
            NN_model,
            variance = 20,
            method = "ppo",
            action_style = "contineous",
            trial_limit = True,
            trial_limit_val = 1000,
            show_progress = False,
            goal_state=[],
            show_progress_val= 100,
            allowable_error=0):

        self.NN_model = NN_model

        if method.lower() == "dqn":
            self.runner = DQN(self.name, show_progress=show_progress, allowable_error=allowable_error, trial_limit = trial_limit, trial_limit_val= trial_limit_val)

        elif method.lower() == "ppo":
            self.runner = PPO(self.name, show_state_progress_val= show_progress_val, action_style=action_style, show_progress=show_progress, allowable_error=allowable_error, trial_limit=trial_limit, trial_limit_val= trial_limit_val,goal_state = goal_state)

        elif method.lower() == "ac":
            self.runner = AC(self.name, action_style=action_style, show_progress=show_progress, allowable_error=allowable_error, trial_limit=trial_limit, trial_limit_val= trial_limit_val)

        else: print("wrong method entry in work / run")

        state, score, success = self.runner.run(NN_model)
        return state, score, success

def sample():
    Steps, Scores = [],[]
    work_piece = "nut_and_bolt"
    action_style = "discrete"
    piece = work(work_piece)
    sample_model, steps, scores  = piece.train(method = "ac",action_style=action_style, number_of_episodes= 500, show_state_progress=True , show_state_progress_val= 1, max_per_episode= 6000, action_limit=False, allowable_error= 5)
    sample_model = sample_model[0]
    Steps.append(steps)
    Scores.append(scores)
    # sample_model = load_model('saved_models/temp4_arrowcont_PPO_actor.model', custom_objects={'ppo_loss_continuous': C_PPO_loss})
    state = piece.run(sample_model,method = "ac",action_style=action_style, trial_limit=False, show_progress=True, allowable_error= 3)
    piece.show_image()
    print("The state array for part is:", state)

def test(conv = False):
    # HOW TO LOAD MODEL
    # model = load_model(modelFile, custom_objects={'ppo_loss': ppo_loss} )
    # sample_model = load_model('saved_models/temp2_arrow_PPO_actor.model', custom_objects={'ppo_loss': PPO_loss})
    # sample_model = load_model('saved_models/temp4_arrowcont_PPO_actor.model', custom_objects={'ppo_loss_continuous': C_PPO_loss})

    # method = "ppo"
    # action_style = "contineous"
    # number_of_episodes = 1000
    # number_of_objects = len(work_pieces)
    # action_limit = 6000
    # number_of_individual_episodes = 5
    # error = 5
    #
    # if conv:
    #     sample_model_actor = load_model("saved_models/CMAIN_actor.model",
    #                                     custom_objects={'ppo_loss_continuous': C_PPO_loss})
    #     sample_model_critic = load_model("saved_models/MAIN_critic.model")
    #     sample_model = [sample_model_actor, sample_model_critic]
    #
    #     steps = np.load("graphs/CMAIN_" + action_style + "_" + method + "_multiobjects_steps.npy")
    #     scores = np.load("graphs/CMAIN_" + action_style + "_" + method + "_multiobjects_scores.npy")
    #     successes = np.load("graphs/CMAIN_" + action_style + "_" + method + "_multiobjects_success.npy")
    #     steps = list(steps)
    #     scores = list(scores)
    #     successes = list(successes)
    #     episodes_done = len(steps)
    #     fail_cnt = 0
    #     for x in successes:
    #         if not x:
    #             fail_cnt += 1
    #
    #     print("USING PREVIOUS MODEL")
    #
    # else:
    #     sample_model_actor = load_model("saved_models/_MAIN_actor.model",
    #                                     custom_objects={'ppo_loss_continuous': C_PPO_loss})
    #     sample_model_critic = load_model("saved_models/_MAIN_critic.model")
    #     sample_model = [sample_model_actor, sample_model_critic]
    #
    #     steps = np.load("graphs/MAIN_" + action_style + "_" + method + "_multiobjects_steps.npy")
    #     scores = np.load("graphs/MAIN_" + action_style + "_" + method + "_multiobjects_scores.npy")
    #     successes = np.load("graphs/MAIN_" + action_style + "_" + method + "_multiobjects_success.npy")
    #     steps = []
    #     scores = []
    #     successes = []
    #     episodes_done = len(steps)
    #     fail_cnt = 0
    #
    #     print("USING PREVIOUS MODEL")
    #
    # for i in range(episodes_done, number_of_episodes - episodes_done):
    #     print("EPISODE ", i, "OUT OF", number_of_episodes)
    #     print("FAILED: ", fail_cnt, "SUCCESSES:", i - fail_cnt)
    #     n = i % number_of_objects
    #     piece = work(work_pieces[n])
    #
    #     if n == 0:
    #         number_of_individual_episodes -= 1
    #     number_of_individual_episodes = max(number_of_individual_episodes, 1)
    #
    #     state, score, success = piece.run(sample_model,method=method, action_style=action_style,
    #                                                      number_of_episodes=1, model=sample_model,
    #                                                      show_state_progress=True, show_state_progress_val=100,
    #                                                      max_per_episode=action_limit,
    #                                                      action_limit=True, allowable_error=error)
    #     if not success:
    #         fail_cnt += 1
    #     print(score, success)
    #
    #     for x in score:
    #         scores.append(x)
    #     steps = list(range(len(scores)))
    #     for x in success:
    #         successes.append(x)
    #
    #     if conv: C = "C"
    #     else: C = ""
    #
    #     np.save("graphs/"+ C +"MAINtest_" + action_style + "_" + method + "_multiobjects_scores.npy", np.array(scores))
    #     np.save("graphs/"+ C +"MAINtest_" + action_style + "_" + method + "_multiobjects_steps.npy", np.array(steps))
    #     np.save("graphs/"+ C +"MAINtest_" + action_style + "_" + method + "_multiobjects_success.npy", np.array(successes))
    #     plt.plot(steps, scores)
    #     plt.xlabel('Number of steps (i)')
    #     plt.ylabel('Episode steps: speed')
    #     plt.savefig("graphs/"+ C +"MAINtest_multiobjects.png")
    #     plt.savefig('graphs/'+ C +'MAINtest_multiobjects.pdf')

    work_piece = "okoso"
    method = "ppo"
    action_style = "contineous"
    steps, scores, labels, successes = [], [], [], []
    piece = work(work_piece)
    number_of_episodes = 200
    sample_model_actor = load_model("saved_models/_MAIN_actor.model",
                                    custom_objects={'C_PPO_loss': C_PPO_loss})
    sample_model_critic = load_model("saved_models/_MAIN_critic.model")
    sample_model = [sample_model_actor, sample_model_critic]

    for i in range(number_of_episodes):
        state, score, success = piece.run(sample_model, method=method, action_style=action_style, show_progress=True,
                                          show_progress_val=1, allowable_error=3, trial_limit=False,
                                          trial_limit_val=1000)
        steps.append(i)
        scores.append(score)
        successes.append(success)
        print(scores, state)

    np.save("graphs/" + action_style + method + "trajectory_length_scores.npy", np.array(scores))
    np.save("graphs/" + action_style + method + "trajectory_length_steps.npy", np.array(steps))
    np.save("graphs/" + action_style + method + "trajectory_length_success.npy", np.array(successes))



def train_for_different_goal_points():
    work_piece = "arrow"
    method = "ppo"
    action_style = "contineous"
    number_of_episodes = 1000
    number_of_goals = number_of_episodes
    action_limit = 1000
    piece = work(work_piece)
    error = 5

    goal_states = []
    for i in range(number_of_goals):
        goal_states.append(rand_reset(20))
    # sample_model_actor = load_model("saved_models/multiple_goals_arrow_crude_contineous_dif_image_PPO_actor.model", custom_objects={'ppo_loss_continuous': C_PPO_loss})
    # sample_model_critic = load_model("saved_models/multiple_goals_arrow_crude_contineous_dif_image_PPO_critic.model")
    # sample_model = [sample_model_actor, sample_model_critic]
    #
    # steps = np.load("graphs/MAIN_contineousppomultipoints_length_steps.npy")
    # scores = np.load("graphs/MAIN_contineousppomultipoints_length_scores.npy")
    # successes = np.load("graphs/MAIN_contineousppomultipoints_length_success.npy")
    # steps = list(steps)
    # scores = list(scores)
    # successes = list(successes)
    # episodes_done = len(steps)
    # fail_cnt = 0
    # for x in successes:
    #     if not x:
    #         fail_cnt+=1

    sample_model, step, score, success = piece.train(method=method, action_style=action_style,
                                                             number_of_episodes=1,
                                                             show_state_progress=True, show_state_progress_val=100,
                                                             max_per_episode=action_limit,
                                                             action_limit=True, allowable_error=error)
    steps, scores, successes, fail_cnt, episodes_done = step, score, success,0,0

    for i in range(episodes_done, number_of_episodes - episodes_done):
        print("EPISODE ", i ,"OUT OF", number_of_episodes)
        print("FAILED: ", fail_cnt ,"SUCCESSES:", i - fail_cnt)
        n = i % number_of_goals
        goal_state = goal_states[n]
        sample_model, step, score, success = piece.train(method=method, action_style=action_style,
                                                             number_of_episodes=1, model = sample_model,
                                                             show_state_progress=True, show_state_progress_val=100,
                                                             max_per_episode=action_limit,
                                                             action_limit=True, allowable_error=error,
                                                             goal_state=goal_state)
        if not success[0]:
            fail_cnt += 1
        print(score, step, success)

        steps.append(i)
        scores.append(score[0])
        successes.append(success[0])

        np.save("graphs/MAIN_" + action_style + method + "multipoints_length_scores.npy", np.array(scores))
        np.save("graphs/MAIN_" + action_style + method + "multipoints_length_steps.npy", np.array(steps))
        np.save("graphs/MAIN_" + action_style + method + "multipoints_length_success.npy", np.array(successes))
        plt.plot(steps,  scores)
        plt.xlabel('Number of steps (i)')
        plt.ylabel('Episode steps: speed')
        plt.savefig("graphs/MAIN.png")
        plt.savefig('graphs/MAIN.pdf')

def train_for_different_objects(conv = False):
    work_pieces = ["disk_and_shaft", "house", "ring","hammer","triangle","dumbell","square","chevron","arrow","nut_and_bolt"]
    method = "ppo"
    action_style = "discrete"
    number_of_objects = len(work_pieces)
    action_limit = 5000
    number_of_individual_episodes = 1000
    number_of_episodes = number_of_objects * 1000
    error = 3

    if conv:
        try:
            sample_model_actor = load_model("saved_models/CMAIN_actor.model",
                                            custom_objects={'ppo_loss_continuous': C_PPO_loss})
            sample_model_critic = load_model("saved_models/CMAIN_critic.model")
            sample_model = [sample_model_actor, sample_model_critic]

            steps = np.load("graphs/CMAIN_" + action_style + "_" + method + "_multiobjects_steps.npy")
            scores = np.load("graphs/CMAIN_" + action_style + "_" + method + "_multiobjects_scores.npy")
            successes = np.load("graphs/CMAIN_" + action_style + "_" + method + "_multiobjects_success.npy")
            steps = list(steps)
            scores = list(scores)
            work_done = floor(len(steps) / number_of_individual_episodes)
            successes = list(successes)
            episodes_done = len(steps)
            fail_cnt = 0
            for x in successes:
                if not x:
                    fail_cnt += 1

            print("USING PREVIOUS MODEL")

        except:
            print("USING NEW MODEL")
            piece = work(work_pieces[0])
            sample_model, step, score, success = piece.train(method=method, action_style=action_style,
                                                             number_of_episodes=1,
                                                             show_state_progress=True, show_state_progress_val=1,
                                                             max_per_episode=action_limit,
                                                             action_limit=False, allowable_error=error)

            steps, scores, successes, fail_cnt, work_done,first_set = step, score, success, 0, 0,1

    else:
        try:
            sample_model_actor = load_model("saved_models/_MAIN_actor.model",
                                            custom_objects={'C_PPO_loss': C_PPO_loss})
            sample_model_critic = load_model("saved_models/_MAIN_critic.model")
            sample_model = [sample_model_actor, sample_model_critic]

            steps = np.load("graphs/_MAIN_" + action_style + "_" + method + "_multiobjects_steps.npy")
            scores = np.load("graphs/_MAIN_" + action_style + "_" + method + "_multiobjects_scores.npy")
            successes = np.load("graphs/_MAIN_" + action_style + "_" + method + "_multiobjects_success.npy")
            steps = list(steps)
            scores = list(scores)
            successes = list(successes)
            work_done = floor(len(steps)/number_of_individual_episodes)
            first_set = (len(steps) % number_of_individual_episodes)
            fail_cnt = 0
            for x in successes:
                if not x:
                    fail_cnt += 1

            print("USING PREVIOUS MODEL")

        except:
            print("USING NEW MODEL")
            piece = work(work_pieces[0])
            sample_model, step, score, success = piece.train(method=method, action_style=action_style,
                                                             number_of_episodes=1,
                                                             show_state_progress=True, show_state_progress_val=100,
                                                             max_per_episode=action_limit,
                                                             action_limit=True, allowable_error=error)

            steps, scores, successes, fail_cnt, work_done,first_set = step, score, success, 0, 0, 1

    for i in range(work_done, number_of_episodes - work_done):
        tot = len(steps)
        print("EPISODE ", i+1, "OUT OF", number_of_episodes)
        print("FAILED: ", fail_cnt, "SUCCESSES:",  tot - fail_cnt)

        n = i % number_of_objects
        piece = work(work_pieces[n])
        print("WORK : ", work_pieces[n])

        # if n == 0:
        #     number_of_individual_episodes -= 1
        # number_of_individual_episodes = max(number_of_individual_episodes, 1)

        episode_number = number_of_individual_episodes
        sample_model, step, score, success = piece.train(method=method, action_style=action_style,
                                                         number_of_episodes=episode_number,
                                                         model=sample_model,
                                                         show_state_progress=True, show_state_progress_val=100,
                                                         max_per_episode=action_limit,
                                                         action_limit=True, allowable_error=error)
        first_set = 0
        for scx in success:
            if not scx:
                fail_cnt += 1

        for x in score:
            scores.append(x)
        steps = list(range(len(scores)))
        for x in success:
            successes.append(x)

        if conv:
            C = "C"
        else:
            C = "_"

        np.save("graphs/" + C + "MAIN_" + action_style + "_" + method + "_multiobjects_scores.npy",
                np.array(scores))
        np.save("graphs/" + C + "MAIN_" + action_style + "_" + method + "_multiobjects_steps.npy", np.array(steps))
        np.save("graphs/" + C + "MAIN_" + action_style + "_" + method + "_multiobjects_success.npy",
                np.array(successes))
        plt.plot(steps, scores)
        plt.xlabel('Number of steps (i)')
        plt.ylabel('Episode steps: speed')
        plt.savefig("graphs/" + C + "MAIN_multiobjects.png")
        plt.savefig('graphs/'+ C +'MAIN_multiobjects.pdf')
        actor = sample_model[0]
        critic = sample_model[1]
        actor.save("saved_models/" + C + "MAIN_actor.model")
        critic.save("saved_models/" + C + "MAIN_critic.model")


train_for_different_objects(conv = True)