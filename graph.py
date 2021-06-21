# import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl

# dqn = np.load("graphs/dont/DQN_scores.npy")
# succ = np.load("graphs/dont/DQN_success.npy")
# cppo = np.load("graphs/dont/PPO_contineous_scores.npy")
# dppo = np.load("graphs/dont/PPO_discrete_scores.npy")
# steps = np.load("graphs/dont/PPO_discrete_steps.npy")
#
# mean = np.mean(dppo)
#
# # cnt = 0
# # for x in succ:
# #     if x:
# #         cnt+=1
#
# #calculate averages of scores
# def average(array):
#     current_summ = 0
#     average_array = []
#
#     for i in range(len(array)):
#         current_summ+= array[i]
#         current_average = current_summ/(i+1)
#         average_array.append(current_average)
#
#     return average_array


# y = [average(dppo), average(cppo), average(dqn)]
# labels = ['discrete PPO', 'contineous PPO', "dqn"]
# color = ["r", "C1", "b"]
# # [:200]
#
# style = "default"
# name = "model_evaluation"
#
# plt.style.use(style)
# for y_arr, label, c in zip(y, labels, color):
#     plt.plot(steps[:200], y_arr[:200], label=label, color = c)
#
# plt.xlabel('Number of episodes')
# plt.ylabel('Average trajectory length')
# plt.legend()
# plt.savefig("graphs/"+name+".png")
# plt.savefig("graphs/"+name+".pdf")
# # plt.savefig("C:/Users/josho/Desktop/school_work/PROJECT/New_paper/RL paper my_paper/images/"+name+".pdf")
# plt.show()

# generaliztion across targets: different orientation, different position, different orientation and position
# data = [[0.04, 0.02, 0.2 ,0.6, 0.8],
# [0.02, 0.02, 0.18 ,0.43, 0.66],
# [0.02, 0.02, 0.05 ,0.2, 0.4]]
# X = np.arange(5)
# X = [2, 4, 6, 8, 10]
# X = np.array(X)
# width = 0.5
# plt.bar(X - width, data[0], color = 'b', width = width, label="same orientation")
# plt.bar(X + 0.00, data[1], color = 'g', width = width, label="same position")
# plt.bar(X + width, data[2], color = 'r', width = width, label="completely random")
# plt.legend()
# name = "general_diff_goal_points"
# plt.savefig("C:/Users/josho/Desktop/school_work/PROJECT/New_paper/RL paper my_paper/images/"+name+".pdf")
# plt.show()

# # generaliztion across workpiece: different base, different mover, different base and moved
# data = [[0.04, 0.02, 0.2 ,0.4, 0.6],
# [0.02, 0.02, 0.18 ,0.43, 0.605],
# [0.02, 0.02, 0.05 ,0.1, 0.15]]
# X = np.arange(5)
# X = [2, 4, 6, 8, 10]
# X = np.array(X)
# width = 0.5
# plt.bar(X - width, data[0], color = 'b', width = width, label="same base object")
# plt.bar(X + 0.00, data[1], color = 'g', width = width, label="same moved object")
# plt.bar(X + width, data[2], color = 'r', width = width, label="completely random ")
# plt.legend()
# name = "general_diff_workobjects"
# plt.savefig("C:/Users/josho/Desktop/school_work/PROJECT/New_paper/RL paper my_paper/images/"+name+".pdf")
# plt.show()

from PIL import Image

image1 = Image.open("C:/Users/josho/Pictures/Screenshots/Screenshot (139).png")
im1 = image1.convert('RGB')
name = "obs_image_3"
im1.save("C:/Users/josho/Desktop/school_work/PROJECT/New_paper/RL paper my_paper/images/"+name+".pdf")