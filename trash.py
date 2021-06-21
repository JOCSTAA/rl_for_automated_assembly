# import numpy as np
#
# def dqn_sample():
#     #TRIANGLE
#
#     goal_image_path = "goal_images/triangle.npy"
#     CAD_model_path = "C:/Users/josho/Desktop/school_work/PROJECT/free_cad_stuffs/assem/imported4.fcstd"
#     iso_details = [18, 2, 5]
#
#     goal_image = np.load(goal_image_path)
#     NN_model = create_model()
#     state= zero_reset()
#     state[0,2] = 20
#     print(state)
#
#     NN_model, new_state, fail = train(NN_model,goal_image, CAD_model_path,iso_details, show_state_progress= True, state = state, show_state_progress_val=1)
#
#     if fail:
#         print("FAIL: max amount of actions for trial exceeded")
#
#     else:
#         print("SUCCESS: model succesfuly converged")
#
# def model_sample():
#     #load images and define learning rate
#     image1 = np.load("goal_images/triangle.npy")
#     image2 = np.load("goal_images/triangle.npy")
#     alpha = 0.0001
#
#     #how to create actor and critic models
#     Amodel = create_actor_model(alpha = 0.001)
#     print(Amodel.summary())
#     Cmodel = create_critic_model(alpha = 0.001)
#     print(Cmodel.summary())
#
#     #how to preprocess data before training/ predicting
#     Atarget_vec = [[0,0,0,0,0,0,0,0,0,0,0,0]] #sample output of actor
#     Atarget_vec = np.array(Atarget_vec)
#     Ctarget_vec = [[0]]#sample output of critic
#     Ctarget_vec = np.array(Ctarget_vec)
#     image1 = prepare_image(image1)
#     image2 = prepare_image(image2)
#
#     #how to train /predict for actor
#     # print("before target vec is" ,Atarget_vec)
#     # Amodel.fit([image1,image2], Atarget_vec.reshape(-1, 12), epochs=1, verbose=0)
#     # Atarget_vec = Amodel.predict([image1,image2])
#     # print("after target vec is" ,Atarget_vec)
#     #
#     # #how to train /predict for actor
#     # print("before target  is", Ctarget_vec)
#     # Cmodel.fit([image1, image2], Ctarget_vec.reshape(-1, 1), epochs=1, verbose=0)
#     # Ctarget_vec = Cmodel.predict([image1, image2])
#     # print("after target vec", Ctarget_vec)
#
#
# def reward_sample():
#     work_variable = collect_work_variables("triangle")
#
#     state = work_variable[0]
#     goal = make_image(state, work_variable[1], work_variable[2])
#
#     state = [[0, 0, 20, 0 , 0 ,0]]
#     prev = make_image(np.array(state), work_variable[1], work_variable[2])
#
#     state = [[0, 0, 21, 0, 0, 0]]
#     next = make_image(np.array(state), work_variable[1], work_variable[2])
#     show(goal)
#
#     r = reward(goal, prev, next)
#     print(r)
#
# def create_actor_model_RESNET(alpha):
#     # define two sets of inputs
#     # the first branch operates on the first input
#     goal_image_base_model = ResNet50(input_shape=image_shape_rgb, include_top=False, weights="imagenet")
#     for layer in goal_image_base_model.layers:
#         layer.trainable = False
#         layer._name = layer.name + str('_1')
#
#     # the second branch opreates on the second input
#     current_image_base_model = ResNet50(input_shape=image_shape_rgb, include_top=False, weights="imagenet")
#     for layer in current_image_base_model.layers:
#         layer.trainable = False
#         layer._name = layer.name + str('_2')
#
#     print(goal_image_base_model.output.shape)
#     # combine the output of the two branches
#     goal_image = Flatten()(goal_image_base_model.output)
#     goal_image = Model(inputs=goal_image_base_model.input, outputs=goal_image)
#     current_image= Flatten()(current_image_base_model.output)
#     current_image= Model(inputs=current_image_base_model.input, outputs=current_image)
#
#     combined = concatenate([goal_image.output, current_image.output])
#     # action = Dense(64, activation="relu")(combined)
#     # action = Dense(32, activation="relu")(action)
#     # action = Dense(12, activation="relu")(action)
#     action = Dense(12, activation="relu")(combined)
#     # our model will accept the inputs of the two branches and
#     # then output a single value
#     model = Model(inputs=[goal_image.input, current_image.input], outputs=action)
#     model.compile(loss='mse', optimizer= Adam (lr=alpha), metrics=['mae'])
#     print(model.summary())
#
#     return model
#
# def create_actor_model_RESNET_embedded(alpha):
#     # define two sets of inputs
#     # the first branch operates on the first input
#
#
#     # define our model that computes embeddings for Pascal VOC images
#     gresnet = ResNet50(input_shape=image_shape_rgb, include_top=False, weights="imagenet")
#
#     ginp = gresnet.layers[0].input
#
#     goutput = gresnet.layers[-5].output
#     goal_image_base_model = Model(inputs=ginp, outputs=goutput)
#     for layer in goal_image_base_model.layers:
#         layer.trainable = False
#         layer._name = layer.name + str('_1')
#
#     # the second branch opreates on the second input
#     cresnet = ResNet50(input_shape=image_shape_rgb, include_top=False, weights="imagenet")
#
#     cinp = cresnet.layers[0].input
#
#     coutput = cresnet.layers[-5].output
#     current_image_base_model = Model(inputs=cinp, outputs=coutput)
#     for layer in current_image_base_model.layers:
#         layer.trainable = False
#         layer._name = layer.name + str('_2')
#
#     print(goal_image_base_model.output.shape)
#     # combine the output of the two branches
#     goal_image = Flatten()(goal_image_base_model.output)
#     goal_image = Model(inputs=goal_image_base_model.input, outputs=goal_image)
#     current_image= Flatten()(current_image_base_model.output)
#     current_image= Model(inputs=current_image_base_model.input, outputs=current_image)
#
#     combined = concatenate([goal_image.output, current_image.output])
#     # action = Dense(64, activation="relu")(combined)
#     # action = Dense(32, activation="relu")(action)
#     # action = Dense(12, activation="relu")(action)
#     action = Dense(12, activation="relu")(combined)
#     # our model will accept the inputs of the two branches and
#     # then output a single value
#     model = Model(inputs=[goal_image_base_model.input, current_image_base_model.input], outputs=action)
#     model.compile(loss='mse', optimizer= Adam (lr=alpha), metrics=['mae'])
#     print(model.summary())
#
#     return model
#
# def create_actor_model(alpha):
#     # define two sets of inputs
#     inputA = Input(shape=image_shape)
#     inputB = Input(shape=image_shape)
#     # the first branch operates on the first input
#     x = Conv2D(16, (3, 3),padding= "same", activation="relu")(inputA)
#     x = MaxPooling2D(pool_size=(2, 2))(x)
#     x = Conv2D(16, (3, 3),padding = "same", activation="relu")(x)
#     x = MaxPooling2D(pool_size=(2, 2))(x)
#     x = Conv2D(32, (3, 3),padding = "same", activation="relu")(x)
#     x = MaxPooling2D(pool_size=(2, 2))(x)
#     x = Flatten()(x)
#     x = Model(inputs=inputA, outputs=x)
#
#
#     # the second branch opreates on the second input
#     y = Conv2D(16, (3, 3), padding = "same",activation="relu")(inputB)
#     y = MaxPooling2D(pool_size=(2, 2))(y)
#     y = Conv2D(16, (3, 3), padding = "same",activation="relu")(y)
#     y = MaxPooling2D(pool_size=(2, 2))(y)
#     y = Conv2D(32, (3, 3), padding = "same",activation="relu")(y)
#     y = MaxPooling2D(pool_size=(2, 2))(y)
#     y = Flatten()(y)
#     y = Model(inputs=inputB, outputs=y)
#     # combine the output of the two branches
#     combined = concatenate([x.output, y.output])
#     # apply a FC layer and then a regression prediction on the
#     # combined outputs
#     # z = Dense(64, activation="relu")(combined)
#     # z = Dense(32, activation="relu")(z)
#     z = Dense(12, activation="relu")(combined)
#     # our model will accept the inputs of the two branches and
#     # then output a single value
#     model = Model(inputs=[x.input, y.input], outputs=z)
#     model.compile(loss='mse', optimizer= Adam (lr=alpha), metrics=['mae'])
#     print(model.summary())
#     return model
#
#
# def create_critic_model(alpha):
#     # define two sets of inputs
#     inputA = Input(shape=image_shape)
#     inputB = Input(shape=image_shape)
#     # the first branch operates on the first input
#     x = Conv2D(16, (3, 3), activation="relu")(inputA)
#     x = MaxPooling2D(pool_size=(2, 2))(x)
#     # x = Conv2D(16, (3, 3), activation="relu")(x)
#     # x = MaxPooling2D(pool_size=(2, 2))(x)
#     x = Flatten()(x)
#     x = Model(inputs=inputA, outputs=x)
#
#     # the second branch opreates on the second input
#     y = Conv2D(16, (3, 3), activation="relu")(inputB)
#     y = MaxPooling2D(pool_size=(2, 2))(y)
#     y = Conv2D(16, (3, 3), activation="relu")(y)
#     y = MaxPooling2D(pool_size=(2, 2))(y)
#     y = Flatten()(y)
#     y = Model(inputs=inputB, outputs=y)
#     # combine the output of the two branches
#     combined = concatenate([x.output, y.output])
#     # apply a FC layer and then a regression prediction on the
#     # combined outputs
#     z = Dense(64, activation="relu")(combined)
#     z = Dense(32, activation="relu")(z)
#     z = Dense(1, activation="tanh")(z)
#     # our model will accept the inputs of the two branches and
#     # then output a single value
#     model = Model(inputs=[x.input, y.input], outputs=z)
#     model.compile(loss='mse', optimizer= Adam (lr=alpha), metrics=['mae'])
#     print(model.summary())
#     return model
#
#
# import numpy as np
# a = [[0.01],
#      [0.01]]
# b = [[0,0, 1,0],
#      [0,0, 1,0]]
# c = [[9, 3, 1, 5],
#      [0, 0, 1, 0]]
# d = np.stack([a,b,c])
#
# print(d)

# def replay(memory, Actor, Critic, gamma):
#     print("READJUSTING PARAMETERS=========================================================================================")
#
#     length = len(memory)
#     memory = np.array(memory)
#     batch_size = 100
#     if length > batch_size and abs(memory[len(memory)-1:2]) < 100:
#         samples = random.sample(memory, batch_size)
#         for i in range(length - 10, length):
#             samples.append(memory[i])
#
#     else:
#         samples = memory
#
#     samples = np.array(samples)
#     states, actions, rewards, predictions = [],[],[],[]
#     for i in range(len(samples)):
#         states.append(prep(goal_image1, make_image(samples[i, 0], CAD_model_path1, iso_details1)))
#         actions.append(samples[i, 1])
#         rewards.append(samples[i, 2])
#         predictions.append(samples[i, 3])
#
#     if abs(rewards[len(rewards)-1] > 100):
#         disc_r = discount_rewards(rewards, gamma)
#     else:disc_r = discount_rewards(rewards, gamma)
#
#     disc_r -= np.mean(disc_r)  # normalizing the result
#     disc_r /= np.std(disc_r)
#
#     for i in range(len(states)):
#         values = Critic.predict(states[i])[:, 0]
#
#         advantages = disc_r[i] - values #THIS WAS A NEGATIVE BROOOOOOOO
#         r_train = np.array(disc_r)
#         advantages = np.array(advantages)
#
#         y_true = np.hstack((advantages, predictions[i], actions[i]))
#         y_true = np.expand_dims(y_true, 0)
#         r_train = np.expand_dims(r_train, 0)
#
#         Actor.fit(states[i], y_true, epochs=EPOCHS, verbose=0, shuffle=True)
#         Critic.fit(states[i], r_train, epochs=EPOCHS, verbose=0, shuffle = True)
#
# ###WORKING PPO#########################################################################################################
# import random
# import numpy as np
# from environment import make_image, imshow, dif_prep
# from action import rand_reset,zero_reset, act
# from reward import crude_reward
# from make_reference_images import collect_work_variables
# import threading
# from threading import Thread, Lock
# import time
#
# ##COPY OF MODEL
# # def ppo_model(0.00001):
# #     initializer = "he_uniform"
# #     inputA = Input(shape=image_shape)
# #
# #     xx = Flatten(input_shape=image_shape)(inputA)
# #
# #     X = Dense(32, activation="elu", kernel_initializer=initializer)(xx)
# #     #X = Dense(16, activation="elu", kernel_initializer=initializer)(X)
# #
# #     #y = Dense(16, activation="elu", kernel_initializer=initializer)(xx)
# #     #y = Dense(32, activation="elu", kernel_initializer=initializer)(y)
# #
# #     action = Dense(12, activation="softmax", kernel_initializer=initializer)(X)
# #     value = Dense(1, kernel_initializer=initializer)(X)
# #
# #     def ppo_loss(y_true, y_pred):
# #         advantages, prediction_picks, actions = y_true[:, :1], y_true[:, 1:1 + 12], y_true[:,1 + 12:]
# #         LOSS_CLIPPING = 0.2
# #         ENTROPY_LOSS = 5e-3
# #
# #
# #         prob = y_pred * actions
# #
# #         old_prob = actions * prediction_picks
# #
# #         r = prob / (old_prob + 1e-10)
# #         p1 = r * advantages
# #         p2 = K.clip(r, min_value=1 - LOSS_CLIPPING, max_value=1 + LOSS_CLIPPING) * advantages
# #         loss = -K.mean(K.minimum(p1, p2) + ENTROPY_LOSS * -(prob * K.log(prob + 1e-10)))
# #         return loss
# #
# #     Actor = Model(inputs=inputA, outputs=action)
# #     Actor.compile(loss=ppo_loss, optimizer=Adam(lr=alpha))
# #     print(Actor.summary())
# #
# #     Critic = Model(inputs=inputA, outputs=value)
# #     Critic.compile(loss='mse', optimizer=Adam(lr=alpha))
# #     print(Critic.summary())
# #
# #     return Actor, Critic
#
#
#
# class D_PPO():
#     def __init__(self,
#                 NN_model,
#                 work,
#                 state = zero_reset(),
#                 gamma = 0.99,
#                 show_state_progress = False,
#                 action_limit = False,
#                 max_per_episode = 2000,
#                 show_state_progress_val = 100,
#                 allowable_error = 0,
#                 replay_cnt= 5,
#                 action_space = 12,
#                 number_of_episodes = 100,
#                 save_at = 10,
#                 variance= 20):
#
#         self.actor = NN_model[0]
#         self.critic = NN_model[1]
#         self.work = work
#         self.initail_state = state
#         self.state = state
#         self.gamma = gamma
#         self.show_state_progress = show_state_progress
#         self.action_limit = action_limit
#         self.max_per_episode = max_per_episode
#         self.show_state_progress_val = show_state_progress_val
#         self.allowable_error = float(allowable_error)
#         self.replay_cnt = replay_cnt
#         self.action_space = action_space
#         self.memory = []
#         self.number_of_episodes = number_of_episodes
#         self.save_at = save_at
#         self.goal_state, self.CAD_model_path, self.iso_details = collect_work_variables(work)
#         self.goal_image = make_image(self.goal_state, self.CAD_model_path, self.iso_details)
#         self.batch_size = replay_cnt
#         self.EPOCHS = 10
#         self.variance = variance
#         self.end_reward = 5
#
#     def _equal(self):
#         length = len(self.memory)
#
#         for i in range(length - 12, length-2, 2):
#
#             v = np.array(self.memory[i])
#             b = np.array(self.memory[i+2])
#
#             if not np.equal(v[0],b[0]).all():
#                 return False
#             if v[2] != b[2]:
#                 return False
#         return True
#
#     def model_act(self):
#         prediction = self.actor.predict(self.current_state_model_input)[0]
#         action = np.random.choice(12, p=prediction)
#         return action, prediction
#
#     def explore(self):
#         initial_state = self.state
#         print(initial_state)
#         action, predictions = self.model_act()
#
#         for i in range(self.action_space):
#             new_state = act(i, self.state)
#             new_image = make_image(new_state, self.CAD_model_path, self.iso_details)
#             rew, d, f = crude_reward(self.goal_state, self.state, new_state, self.current_image, new_image)
#
#             if i != action and rew > 0:
#                 action_onehot = np.zeros([self.action_space])
#                 action_onehot[action] = 1
#                 self.memory.append([self.state, action_onehot, rew, predictions])
#
#
#     def discount_rewards(self, reward):
#         running_add = 0
#         discounted_r = np.zeros_like(reward)
#         for i in reversed(range(0, len(reward))):
#             running_add = (running_add * self.gamma) + reward[i]
#             discounted_r[i] = running_add
#
#         return discounted_r
#
#     def replay(self):
#         print("READJUSTING PARAMETERS=========================================================================================")
#         length = len(self.memory)
#         if length > self.batch_size:
#             # samples = random.sample(self.memory, self.batch_size)
#             # for i in range(length - self.replay_cnt, length):
#             #     samples.append(self.memory[i])
#             samples = self.memory[length-self.batch_size:length]
#
#         else:
#             samples = np.array(self.memory)
#
#         samples = np.array(samples)
#         states, actions, rewards, predictions = [],[],[],[]
#         for i in range(len(samples)):
#             states.append(dif_prep(self.goal_image, make_image(samples[i, 0], self.CAD_model_path, self.iso_details)))
#             actions.append(samples[i, 1])
#             rewards.append(samples[i, 2])
#             predictions.append(samples[i, 3])
#
#         if abs(rewards[len(rewards) - 1]) > 1:
#             disc_r = self.discount_rewards(rewards)
#             print(disc_r)
#         else:
#             disc_r = rewards
#         states = np.vstack(states)
#         actions = np.vstack(actions)
#         predictions = np.vstack(predictions)
#         disc_r = np.vstack(disc_r)
#         values = self.critic.predict(states)
#         advantages = disc_r - values
#
#         # stack everything to numpy array
#         y_true = np.hstack([advantages, predictions, actions])
#         # training Actor and Critic networks
#         self.actor.fit(states, y_true, epochs=self.EPOCHS, verbose=0, shuffle=True)
#         self.critic.fit(states, disc_r, epochs=self.EPOCHS, verbose=0, shuffle=True)
#
#     def episode(self):
#         done, score, cnt, fail, local_maxima_stuck = False, 0, 1, False, False
#         print("allowable error margin = ", self.allowable_error)
#         while not done:
#             self.memory = []
#
#             self.current_image = make_image(self.state, self.CAD_model_path, self.iso_details)
#             self.current_state_model_input = dif_prep(self.goal_image, self.current_image)
#             Action, Prediction = self.model_act()
#             new_state = act(Action, self.state)
#             new_image = make_image(new_state, self.CAD_model_path, self.iso_details)
#             # there no action at any state that would produce the same state except when part is out of frame
#             rew, done, fail = crude_reward(self.goal_state, self.state, new_state, self.current_image, new_image, allowable_error=self.allowable_error)
#
#             if done:
#                 if np.array_equal(self.current_image, new_image):
#                     rew += -self.end_reward
#                     print(fail, rew)
#                 else:
#                     rew += self.end_reward
#                     print(fail, "true", rew)
#
#             print(cnt, rew, new_state)
#             action_onehot = np.zeros([self.action_space])
#             action_onehot[Action] = 1
#             self.memory.append([self.state, action_onehot, rew, Prediction])
#
#             self.state = new_state
#
#             length = len(self.memory)
#             if length > 15:
#                 local_maxima_stuck = self._equal()
#
#             if local_maxima_stuck:
#                 print("local maxima reached: RELOCATING")
#                 print(cnt, ":", self.state)
#
#                 for i in range(length - 12, length):
#                     if self.memory[i][2] == 0:
#                         self.memory[i][2] = -1
#                 self.explore()
#                 print(Prediction)
#                 self.replay()
#
#             if cnt > self.max_per_episode and self.action_limit and not done:
#                 print("EPISODE FAIL: Max Number of actions reached")
#                 print(cnt, ":", self.state)
#                 fail = True
#                 done = True
#
#             if cnt % self.show_state_progress_val == 0 and self.show_state_progress:
#                 print(cnt, ":", self.state)
#
#
#             if done or len(self.memory) % self.replay_cnt == 0 :
#                 print(Prediction)
#                 self.replay()
#                 if done:
#                     self.replay()
#                     break
#
#             imshow(np.vstack((self.current_image, self.goal_image)))
#             cnt += 1
#
#         return fail, cnt
#
#     def train(self):
#         number_of_succesful_episodes = 0
#         number_of_failed_episodes = 0
#         step = []
#         score = []
#
#         for i in range(1, self.number_of_episodes+1):
#             self.state = rand_reset(self.variance)
#             print("EPISODE ", i, ": number of succesful episodes:", number_of_succesful_episodes,": number of failed episodes:", number_of_failed_episodes)
#             fail, cnt = self.episode()
#
#             if fail:
#                 number_of_failed_episodes += 1
#             else:
#                 number_of_succesful_episodes += 1
#
#             if i % self.save_at == 0:
#                 self.actor.save("saved_models/temp_" + self.work + "_PPO_actor.model")
#                 self.critic.save("saved_models/temp_" + self.work + "_PPO_critic.model")
#
#             step.append(i-1)
#             score.append(self.max_per_episode-cnt)
#
#         #plot image here and save plot
#         return [self.actor, self.critic], step, score
######################################################################################################################
# from environment import show
# import numpy as np
#
# x, y = (dif > 0).nonzero()
#     X, Y = (dif < 0).nonzero()
#     sum = 0
#     cnt = 0
#
#     for i in range(len(x)):
#         positive_y_value = y[i]
#         positive_x_value = x[i]
#         for j in range(len(X)):
#             cnt +=1
#             negative_y_value = Y[j]
#             negative_x_value = X[j]
#
#             x_distance = (positive_x_value - negative_x_value)**2
#             y_distance = (positive_y_value - negative_y_value)**2
#             distance = math.sqrt(x_distance + y_distance)
#             sum += distance
#
#     avg_dif = sum / cnt
#
#     return avg_dif

# OldMax, OldMin = 1, 0
# NewMax, NewMin = 1, -1
# OldValue = 0
# OldRange = (OldMax - OldMin)
# NewRange = (NewMax - NewMin)
# NewValue = (((OldValue - OldMin) * NewRange) / OldRange) + NewMin
# print(NewValue)
#
# if method.lower() == "dqn":
#     self.NN_model = create_model(method.lower())
#
#     for i in range(number_of_episodes):
#         print("EPISODE ", i + 1, ": number of succesful episodes:", number_of_succesful_episodes,
#               ": number of failed episodes:", number_of_failed_episodes)
#
#         if reset.lower() == "rand":
#             self.state = rand_reset(20)
#         elif reset.lower() == "zero":
#             self.state = zero_reset()
#         else:
#             print("wrong state reset type")
#
#         self.NN_model, self.new_state, fail = DQN.train(self.NN_model,
#                                                         self.name,
#                                                         state=self.state,
#                                                         gamma=gamma,
#                                                         action_limit=action_limit,
#                                                         max_per_episode=max_per_episode,
#                                                         show_state_progress=show_state_progress,
#                                                         show_state_progress_val=show_state_progress_val,
#                                                         decay_factor=decay_factor,
#                                                         eps_min=eps_min,
#                                                         eps=eps,
#                                                         allowable_error=allowable_error)
#
#         if fail:
#             number_of_failed_episodes += 1
#         else:
#             number_of_succesful_episodes += 1
#         # todo: do the tensor board thing here
#     return self.NN_model
#
# elif method.lower() == "ac":
#     self.NN_model = create_model(method.lower())
#     scores = []
#     episodes = []
#     average = []
#
#     for i in range(number_of_episodes):
#         print("EPISODE ", i + 1, ": number of succesful episodes:", number_of_succesful_episodes,
#               ": number of failed episodes:", number_of_failed_episodes)
#
#         if reset.lower() == "rand":
#             self.state = rand_reset(20)
#         elif reset.lower() == "zero":
#             self.state = zero_reset()
#         else:
#             print("wrong state reset type")
#
#         self.state = np.array([[0, 20, 0, 1, -4, 9]])
#
#         self.NN_model, self.new_state, fail = Actor_critic.train(self.NN_model,
#                                                                  self.name,
#                                                                  i,
#                                                                  scores,
#                                                                  episodes,
#                                                                  average,
#                                                                  state=self.state,
#                                                                  gamma=gamma,
#                                                                  action_limit=action_limit,
#                                                                  max_per_episode=max_per_episode,
#                                                                  show_state_progress=show_state_progress,
#                                                                  show_state_progress_val=show_state_progress_val,
#                                                                  allowable_error=allowable_error)
#
#         if fail:
#             number_of_failed_episodes += 1
#         else:
#             number_of_succesful_episodes += 1
#         # todo: do the tensor board thing here
#         if i % 100 == 0:
#             self.NN_model[0].save("saved_models/temp_" + self.name + "_actor.model")
#             self.NN_model[1].save("saved_models/temp_" + self.name + "_critic.model")
#     return self.NN_model
#
# if method.lower() == "dqn":
#     self.NN_model = NN_model
#     done = checkifdone(self.goal_images[1], self.current_images[1], allowable_error=0)
#
#     cnt = 0
#     fail = False
#     while not done:
#         self.current_images = make_image(self.state, self.CAD_path, self.iso_details, usage="train")
#         image1 = prepare_image(self.goal_images[0])
#         image2 = prepare_image(self.current_images[0])
#         model_input = [image1, image2]
#
#         a = NN_model.predict(model_input)
#
#         self.state, rew, done = self.action(a)
#
#         if cnt > trial_limit_val and trial_limit:
#             fail = True
#             done = True
#
#         if show_progress and cnt % show_progress_val == 0:
#             self.show_image()
#
#         cnt += 1
#
#     if fail:
#         print("FAIL: max amount of actions for trial exceeded")
#
#     else:
#         print("SUCCESS: part was succesfully positioned")
#         return self.state
#
# elif method.lower() == "ac":
#     self.NN_model = NN_model
#     done = checkifdone(self.goal_images[1], self.current_images[1], allowable_error=0)
#
#     cnt = 0
#     fail = False
#     while not done:
#         self.current_images = make_image(self.state, self.CAD_path, self.iso_details, usage="train")
#         image1 = prepare_image(self.goal_images[0])
#         image2 = prepare_image(self.current_images[0])
#         model_input = [image1, image2]
#
#         prediction = NN_model.predict(model_input)[0]
#         a = np.random.choice(12, p=prediction)
#
#         self.state, rew, done = self.action(a)
#
#         if cnt > trial_limit_val and trial_limit:
#             fail = True
#             done = True
#
#         if show_progress and cnt % show_progress_val == 0:
#             self.show_image()
#
#         cnt += 1
#
#     if fail:
#         print("FAIL: max amount of actions for trial exceeded")
#
#     else:
#         print("SUCCESS: part was succesfully positioned")
#         return self.state

#======================================DQN===============================================================================
# import numpy as np
# from environment import make_image, imshow, prep
# from action import rand_reset,zero_reset, act
# from reward import image_reward, checkifdone, crude_reward
# from model import create_model, prepare_image
# from make_reference_images import collect_work_variables
# import random
#
#
# def remember(state, action, reward, new_state, done, fail, memory):
#     memory.append([state, action, reward, new_state, done, fail])
#     return memory
#
#
# def replay(memory, predictor, fitter, gamma):
#
#     for sample in memory:
#         state, action, reward, new_state, done, fail = sample
#
#         target = predictor.predict(state)
#         if done:
#             if not fail:
#                 target[0][action] = reward * 10000
#             else:
#                 target[0][action] = reward * -10000
#         else:
#             Q_future = max(predictor.predict(new_state)[0])
#             target[0][action] = reward + Q_future * gamma
#         fitter.fit(state, target.reshape(-1, 12), epochs=1, verbose=0)
#
#
# def target_train(predictor, fitter, tau):
#     weights = fitter.get_weights()
#     target_weights = predictor.get_weights()
#     for i in range(len(target_weights)):
#         target_weights[i] = weights[i] * tau + target_weights[i] * (1 - tau)
#     predictor.set_weights(target_weights)
#
#
# #TODO : SPEED UP TRAINING
# def train(NN_model,
#           work,
#           state = zero_reset(),
#           gamma = 0.975,
#           show_state_progress = False,
#           action_limit = False,
#           max_per_episode = 2000,
#           show_state_progress_val = 100,
#           decay_factor = 0.999,
#           eps_min = 0.01,
#           eps = 0.75,
#           tau = 0.125,
#           explore_range = 10,
#           shake_up = False,
#           shake_up_val = 2000,
#           allowable_error = 0):
#
#     # =======================================================================================================================
#     # =====================================================MAIN==============================================================
#     # PRIME MODEL
#     print("Training====================================================================================================")
#     goal_state, CAD_model_path, iso_details = collect_work_variables(work)
#     goal_image = make_image(goal_state, CAD_model_path, iso_details)
#     current_images = make_image(state, CAD_model_path, iso_details)
#
#     predictor = NN_model
#     fitter = NN_model
#
#     memory = []
#
#     cnt = 0
#     out_of_bounds_cnt = 0
#     # done = checkifdone(goal_images[1], current_images[1], allowable_error=allowable_error)
#     done = False
#     fail = False
#     while not done:
#         eps = max(eps, eps_min)
#
#         current_image = make_image(state, CAD_model_path, iso_details)
#
#         # image1 = prepare_image(goal_image )
#         # image2 = prepare_image(current_image )
#         current_state_model_input = prep(goal_image ,current_image )
#
#         if np.random.random() < eps:
#             a, increment = np.random.randint(0, 12), np.random.randint(1, explore_range)
#             print(cnt, "exploring")
#         else:
#             a, increment = np.argmax(predictor.predict(current_state_model_input)), 1
#             print(cnt, "exploiting")
#
#         new_state = act(a,state)
#         new_image = make_image(new_state, CAD_model_path, iso_details)
#         # image3 = prepare_image(new_image )
#         future_state_model_input = prep(goal_image ,new_image )
#
#         rew, done, fail = crude_reward(goal_state, state, new_state, allowable_error = allowable_error)
#         memory = []
#         memory = remember(current_state_model_input, a, rew, future_state_model_input, done, fail, memory)
#         if a % 2 == 0:
#             b = a + 1
#         else: b = a - 1
#         temp = crude_reward(goal_state, state, act(b,state))
#         temp_image = make_image(state, CAD_model_path, iso_details)
#         temp_state_model_input = prep(goal_image , temp_image)
#         memory = remember(current_state_model_input, a, temp[0], temp_state_model_input, temp[1], temp[2], memory)
#         replay(memory, predictor, fitter, gamma)
#
#         # target = rew + gamma * np.max(predictor.predict(future_state_model_input))
#         # target_vec = predictor.predict(current_state_model_input)
#         # target_vec[0, a] = target
#         #
#         # fitter.fit(current_state_model_input, target_vec.reshape(-1, 12), epochs=1, verbose=0)
#
#         predictor.set_weights(fitter.get_weights())
#
#         state = new_state  # add randomness to new state
#         print(state, rew)
#         eps *= decay_factor
#
#         if cnt % show_state_progress_val == 0 and show_state_progress:
#             print(cnt, ":" , state)
#
#         if cnt > max_per_episode and action_limit:
#             print("EPISODE FAIL: Max Number of actions reached")
#             print(cnt, ":", state)
#             fail = True
#             done = True
#             continue
#
#         if shake_up and cnt%shake_up_val == 0:
#             eps = 0.5
#
#         if done:
#             return predictor, state, fail
#         cnt+=1
#
#         imshow(np.vstack((current_image,goal_image)))

# from matplotlib import pyplot as plt
import numpy as np
# x = [0, 1, 2, 3, 4]
# y = [ [0, 1, 2, 3, 4], [5, 6, 7, 8, 9], [9, 8, 7, 6, 5] ]
# labels = ['dqn', 'ac', 'ppo']
#
# for y_arr, label in zip(y, labels):
#     plt.plot(x, y_arr, label=label)
#
# plt.legend()
# plt.savefig("graphs/test.png")
# plt.savefig('graphs/test.pdf')
# plt.show()

a = np.load("graphs/MAIN_contineousppomultipoints_length_success.npy")
a = list(a)
a.append(False)
print(a)