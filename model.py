from keras.models import Sequential,load_model, Model
from keras.layers import Dense, Input, Conv2D, MaxPooling2D,Flatten, concatenate, Lambda, Reshape,Permute, LSTM
from keras.utils.vis_utils import plot_model
from keras.optimizers import Adam, RMSprop
import numpy as np
from keras.applications import ResNet50
from environment import show
from keras import backend as K
import tensorflow as tf

image_size = (224,224)
# image_shape = model_image_shape()
image_shape_rgb = (image_size[0],image_size[1], 3)
action_space= 6


def C_PPO_loss(y_true, y_pred):
    advantages, actions, logp_old_ph, = y_true[:, :1], y_true[:, 1:1 + 6], y_true[:, 1 + 6]
    LOSS_CLIPPING = 0.2
    logp = Gaussian_likelihood(actions, y_pred)

    ratio = K.exp(logp - logp_old_ph)

    p1 = ratio * advantages
    p2 = tf.where(advantages > 0, (1.0 + LOSS_CLIPPING) * advantages,
                  (1.0 - LOSS_CLIPPING) * advantages)  # minimum advantage

    actor_loss = -K.mean(K.minimum(p1, p2))

    return actor_loss


def Gaussian_likelihood(actions, pred):
    log_std = -0.5 * np.ones(6, dtype=np.float32)
    pre_sum = -0.5 * (((actions - pred) / (K.exp(log_std) + 1e-8)) ** 2 + 2 * log_std + K.log(2 * np.pi))
    return K.sum(pre_sum, axis=1)

def PPO_loss(y_true, y_pred):
    advantages, actions, prediction_picks = y_true[:, :1], y_true[:, 1:1 + 12], y_true[:, 1 + 12:]
    LOSS_CLIPPING = 0.2
    ENTROPY_LOSS = 5e-3

    prob = y_pred * actions

    old_prob = actions * prediction_picks

    r = prob / (old_prob + 1e-10)
    p1 = r * advantages
    p2 = K.clip(r, min_value=1 - LOSS_CLIPPING, max_value=1 + LOSS_CLIPPING) * advantages
    loss = -K.mean(K.minimum(p1, p2) + ENTROPY_LOSS * -(prob * K.log(prob + 1e-10)))
    return loss

def create_dqn_model(alpha, Image_shape):
    inputA = Input(shape=Image_shape)
    initializer = "zeros"
    y = Conv2D(16, (3, 3), activation="relu", kernel_initializer=initializer)(inputA)
    y = MaxPooling2D(pool_size=(2, 2))(y)
    y = Conv2D(16, (3, 3), activation="relu", kernel_initializer=initializer)(y)
    y = MaxPooling2D(pool_size=(2, 2))(y)
    X = Flatten()(y)
    X = Dense(12, activation="linear", kernel_initializer=initializer)(X)
    model = Model(inputs = inputA, outputs = X)
    model.compile(loss='mse', optimizer=Adam(lr=0.0001), metrics=['mae'])
    print(model.summary())

    return model

def actor_critic_model(alpha, Image_shape):
    lr = alpha
    lr = 0.000003
    initializer = "he_uniform"
    inputA = Input(shape=Image_shape)

    # y = Conv2D(16, (3, 3), activation="relu")(inputA)
    # y = MaxPooling2D(pool_size=(2, 2))(y)
    # y = Conv2D(16, (3, 3), activation="relu")(y)
    # y = MaxPooling2D(pool_size=(2, 2))(y)
    # xx = Flatten()(y)
    # # y = Dense(64, activation="tanh", kernel_initializer=initializer)(xx)
    # # y = Dense(64, activation="tanh", kernel_initializer=initializer)(y)
    # action = Dense(12, activation="softmax", kernel_initializer=initializer)(xx)
    # # X = Dense(124, activation="tanh", kernel_initializer=initializer)(xx)
    # # X = Dense(124, activation="tanh", kernel_initializer=initializer)(X)
    # value = Dense(1, kernel_initializer=initializer)(xx)

    xx = Flatten(input_shape=Image_shape)(inputA)

    X = Dense(124, activation="tanh", kernel_initializer=initializer)(xx)
    X = Dense(124, activation="tanh", kernel_initializer=initializer)(X)

    y = Dense(64, activation="tanh", kernel_initializer=initializer)(xx)
    y = Dense(64, activation="tanh", kernel_initializer=initializer)(y)

    action = Dense(12, activation="softmax", kernel_initializer=initializer)(y)
    value = Dense(1, kernel_initializer=initializer)(X)

    Actor = Model(inputs=inputA, outputs=action)
    Actor.compile(loss='categorical_crossentropy', optimizer=Adam(lr=lr))
    print(Actor.summary())

    Critic = Model(inputs=inputA, outputs=value)
    Critic.compile(loss='mse', optimizer=Adam(lr=lr))
    print(Critic.summary())

    return Actor, Critic

def C_actor_critic_model(alpha, Image_shape):
    initializer = tf.random_normal_initializer(stddev=0.01)
    inputA = Input(shape=Image_shape)

    # y = Conv2D(16, (3, 3), activation="relu")(inputA)
    # y = MaxPooling2D(pool_size=(2, 2))(y)
    # y = Conv2D(16, (3, 3), activation="relu")(y)
    # y = MaxPooling2D(pool_size=(2, 2))(y)
    # xx = Flatten()(y)
    xx = Flatten(input_shape=Image_shape)(inputA)

    X = Dense(64, activation="tanh", kernel_initializer=initializer)(xx)
    X = Dense(64, activation="tanh", kernel_initializer=initializer)(X)

    y = Dense(32, activation="tanh", kernel_initializer=initializer)(xx)
    y = Dense(32, activation="tanh", kernel_initializer=initializer)(y)

    action = Dense(6, activation="tanh", kernel_initializer=initializer)(y)
    value = Dense(1, kernel_initializer=initializer)(X)

    Actor = Model(inputs=inputA, outputs=action)
    Actor.compile(loss='categorical_crossentropy', optimizer=Adam(lr=alpha))
    print(Actor.summary())

    Critic = Model(inputs=inputA, outputs=value)
    Critic.compile(loss='mse', optimizer=Adam(lr=alpha))
    print(Critic.summary())

    return Actor, Critic

def ppo_model_l(alpha, Image_shape, lstm_cnt):
    initializer = tf.random_normal_initializer(stddev=0.01)
    lstm_cnt = lstm_cnt
    input = Input(shape=(lstm_cnt,224, 224,1))

    # conv_input = []
    # for x in range(lstm_cnt):
    #     temp = input[:,x]
    #     conv_input.append(temp)
    #
    # conv_input = tf.concat(conv_input, axis = 1)
    #
    # # # ==========================================================================
    # xx = Conv2D(16, 3, activation="relu", padding="same")(conv_input)
    # xx = MaxPooling2D(pool_size=(2, 2), padding="same")(xx)
    # xx = Conv2D(16, 3, activation="relu", padding="same")(xx)
    # xx = MaxPooling2D(pool_size=(2, 2), padding="same")(xx)
    #
    # lstm_input = []
    # for x in range(lstm_cnt):
    #     temp = xx[:, (x*56):((x+1)*56)]
    #     temp = Flatten()(temp)
    #     lstm_input.append(temp)
    #
    # lstm_input = tf.stack(lstm_input, axis=1)
    #
    # xx = LSTM(32)(lstm_input)
    lstm_input = []
    for x in range(lstm_cnt):
        temp = input[:,x]
        temp = Flatten()(temp)
        lstm_input.append(temp)

    # conv_input = tf.concat(conv_input, axis = 1)
    #     #
    #     # # # ==========================================================================
    #     # xx = Conv2D(16, 3, activation="relu", padding="same")(conv_input)
    #     # xx = MaxPooling2D(pool_size=(2, 2), padding="same")(xx)
    #     # xx = Conv2D(16, 3, activation="relu", padding="same")(xx)
    #     # xx = MaxPooling2D(pool_size=(2, 2), padding="same")(xx)
    #     #
    #     # lstm_input = []
    #     # for x in range(lstm_cnt):
    #     #     temp = xx[:, (x*56):((x+1)*56)]
    #     #     temp = Flatten()(temp)
    #     #     lstm_input.append(temp)

    lstm_input = tf.stack(lstm_input, axis=1)

    xx = LSTM(32)(lstm_input)

    X = Dense(64, activation="tanh", kernel_initializer=initializer)(xx)
    X = Dense(64, activation="tanh", kernel_initializer=initializer)(X)

    y = Dense(32, activation="tanh", kernel_initializer=initializer)(xx)
    y = Dense(32, activation="tanh", kernel_initializer=initializer)(y)

    action = Dense(12, activation="softmax", kernel_initializer=initializer)(y)
    value = Dense(1, kernel_initializer=initializer)(X)

    def ppo_loss(y_true, y_pred):
        advantages, actions, prediction_picks = y_true[:, :1], y_true[:, 1:1 + 12], y_true[:,1 + 12:]
        LOSS_CLIPPING = 0.2
        ENTROPY_LOSS = 5e-3


        prob = y_pred * actions

        old_prob = actions * prediction_picks

        r = prob / (old_prob + 1e-10)
        p1 = r * advantages
        p2 = K.clip(r, min_value=1 - LOSS_CLIPPING, max_value=1 + LOSS_CLIPPING) * advantages
        loss = -K.mean(K.minimum(p1, p2) + ENTROPY_LOSS * -(prob * K.log(prob + 1e-10)))
        return loss

    Actor = Model(inputs=input, outputs=action)
    Actor.compile(loss=ppo_loss, optimizer=Adam(lr=alpha))
    print(Actor.summary())

    Critic = Model(inputs=input, outputs=value)
    Critic.compile(loss='mse', optimizer=Adam(lr=alpha))
    return Actor, Critic

def c_ppo_model_l(alpha, Image_shape, lstm_cnt):
    initializer = tf.random_normal_initializer(stddev=0.01)
    lstm_cnt = lstm_cnt
    input = Input(shape=(lstm_cnt,224, 224,1))

    conv_input = []
    for x in range(lstm_cnt):
        temp = input[:,x]
        conv_input.append(temp)

    conv_input = tf.concat(conv_input, axis = 1)

    # # ==========================================================================
    #original = 16, new = 32
    xx = Conv2D(16, 3, activation="relu", padding="same")(conv_input)
    xx = MaxPooling2D(pool_size=(2, 2), padding="same")(xx)
    xx = Conv2D(16, 3, activation="relu", padding="same")(xx)
    xx = MaxPooling2D(pool_size=(2, 2), padding="same")(xx)

    lstm_input = []
    for x in range(lstm_cnt):
        temp = xx[:, (x*56):((x+1)*56)]
        temp = Flatten()(temp)
        lstm_input.append(temp)

    lstm_input = tf.stack(lstm_input, axis=1)

    # xx = Reshape((16, -1))(xx)
    # lstm_input = Permute((2, 1))(lstm_input)
    xx = LSTM(32)(lstm_input)

    # xx = Reshape((16, -1))(xx)
    # xx = Permute((2, 1))(xx)
    # xx = LSTM(32)(xx)
    # xx = Flatten()(xx)

    #==========================================================================

    # y = Conv2D(16, (3, 3), activation="relu", kernel_initializer=initializer)(inputA)
    # y = MaxPooling2D(pool_size=(2, 2))(y)
    # y = Conv2D(16, (3, 3), activation="relu", kernel_initializer=initializer)(y)
    # y = MaxPooling2D(pool_size=(2, 2))(y)
    # xx = Flatten()(y)

    # xx = Flatten(input_shape=Image_shape)(inputA)

    X = Dense(128, activation="tanh", kernel_initializer=initializer)(xx)
    X = Dense(128, activation="tanh", kernel_initializer=initializer)(X)
    value = Dense(1, kernel_initializer=initializer)(X)

    # original = 64, new = 128
    y = Dense(64, activation="tanh", kernel_initializer=initializer)(xx)
    y = Dense(64, activation="tanh", kernel_initializer=initializer)(y)
    action = Dense(6, activation="tanh", kernel_initializer=initializer)(y)

    def ppo_loss_continuous(y_true, y_pred):
        advantages, actions, logp_old_ph, = y_true[:, :1], y_true[:, 1:1 + 6], y_true[:,1 + 6]
        LOSS_CLIPPING = 0.2
        logp = gaussian_likelihood(actions, y_pred)

        ratio = K.exp(logp - logp_old_ph)

        p1 = ratio * advantages
        p2 = tf.where(advantages > 0, (1.0 + LOSS_CLIPPING) * advantages,(1.0 - LOSS_CLIPPING) * advantages)  # minimum advantage

        actor_loss = -K.mean(K.minimum(p1, p2))

        return actor_loss

    def gaussian_likelihood(actions, pred):
        log_std = -0.5 * np.ones(6, dtype=np.float32)
        pre_sum = -0.5 * (((actions - pred) / (K.exp(log_std) + 1e-8)) ** 2 + 2 * log_std + K.log(2 * np.pi))
        return K.sum(pre_sum, axis=1)

    Actor = Model(inputs=input, outputs=action)
    Actor.compile(loss=ppo_loss_continuous, optimizer=Adam(lr=alpha))
    print(Actor.summary())

    Critic = Model(inputs=input, outputs=value)
    Critic.compile(loss='mse', optimizer=Adam(lr=alpha))

    return Actor, Critic

def ppo_model(alpha, Image_shape):
    initializer = tf.random_normal_initializer(stddev=0.01)
    input = Input(shape=(224, 224,1))

    conv_input = input

    # # ==========================================================================
    xx = Conv2D(16, 3, activation="relu", padding="same")(conv_input)
    xx = MaxPooling2D(pool_size=(2, 2), padding="same")(xx)
    xx = Conv2D(16, 3, activation="relu", padding="same")(xx)
    xx = MaxPooling2D(pool_size=(2, 2), padding="same")(xx)

    xx = Flatten()(xx)

    X = Dense(64, activation="tanh", kernel_initializer=initializer)(xx)
    X = Dense(64, activation="tanh", kernel_initializer=initializer)(X)

    y = Dense(32, activation="tanh", kernel_initializer=initializer)(xx)
    y = Dense(32, activation="tanh", kernel_initializer=initializer)(y)

    action = Dense(12, activation="softmax", kernel_initializer=initializer)(y)
    value = Dense(1, kernel_initializer=initializer)(X)

    def ppo_loss(y_true, y_pred):
        advantages, actions, prediction_picks = y_true[:, :1], y_true[:, 1:1 + 12], y_true[:,1 + 12:]
        LOSS_CLIPPING = 0.2
        ENTROPY_LOSS = 5e-3


        prob = y_pred * actions

        old_prob = actions * prediction_picks

        r = prob / (old_prob + 1e-10)
        p1 = r * advantages
        p2 = K.clip(r, min_value=1 - LOSS_CLIPPING, max_value=1 + LOSS_CLIPPING) * advantages
        loss = -K.mean(K.minimum(p1, p2) + ENTROPY_LOSS * -(prob * K.log(prob + 1e-10)))
        return loss

    Actor = Model(inputs=input, outputs=action)
    Actor.compile(loss=ppo_loss, optimizer=Adam(lr=alpha))
    print(Actor.summary())

    Critic = Model(inputs=input, outputs=value)
    Critic.compile(loss='mse', optimizer=Adam(lr=alpha))
    return Actor, Critic

def c_ppo_model(alpha, Image_shape):
    initializer = tf.random_normal_initializer(stddev=0.01)
    input = Input(shape=(224, 224,1))

    conv_input = input

    # # ==========================================================================
    xx = Conv2D(16, 3, activation="relu", padding="same")(conv_input)
    xx = MaxPooling2D(pool_size=(2, 2), padding="same")(xx)
    xx = Conv2D(16, 3, activation="relu", padding="same")(xx)
    xx = MaxPooling2D(pool_size=(2, 2), padding="same")(xx)

    xx = Flatten()(xx)

    X = Dense(128, activation="tanh", kernel_initializer=initializer)(xx)
    X = Dense(128, activation="tanh", kernel_initializer=initializer)(X)
    value = Dense(1, kernel_initializer=initializer)(X)

    y = Dense(64, activation="tanh", kernel_initializer=initializer)(xx)
    y = Dense(64, activation="tanh", kernel_initializer=initializer)(y)
    action = Dense(6, activation="tanh", kernel_initializer=initializer)(y)

    def ppo_loss_continuous(y_true, y_pred):
        advantages, actions, logp_old_ph, = y_true[:, :1], y_true[:, 1:1 + 6], y_true[:,1 + 6]
        LOSS_CLIPPING = 0.2
        logp = gaussian_likelihood(actions, y_pred)

        ratio = K.exp(logp - logp_old_ph)

        p1 = ratio * advantages
        p2 = tf.where(advantages > 0, (1.0 + LOSS_CLIPPING) * advantages,(1.0 - LOSS_CLIPPING) * advantages)  # minimum advantage

        actor_loss = -K.mean(K.minimum(p1, p2))

        return actor_loss

    def gaussian_likelihood(actions, pred):
        log_std = -0.5 * np.ones(6, dtype=np.float32)
        pre_sum = -0.5 * (((actions - pred) / (K.exp(log_std) + 1e-8)) ** 2 + 2 * log_std + K.log(2 * np.pi))
        return K.sum(pre_sum, axis=1)

    Actor = Model(inputs=input, outputs=action)
    Actor.compile(loss=ppo_loss_continuous, optimizer=Adam(lr=alpha))
    print(Actor.summary())

    Critic = Model(inputs=input, outputs=value)
    Critic.compile(loss='mse', optimizer=Adam(lr=alpha))

    return Actor, Critic

def create_model(model_type = "ppo", alpha = 0.00001, Image_shape = (224,224), lstm_cnt = 3):
    if model_type.lower() == "dqn":
        return create_dqn_model(alpha, Image_shape)

    elif model_type.lower() == "ac":
        return actor_critic_model(alpha, Image_shape)

    elif model_type.lower() == "c_ac":
        return C_actor_critic_model(alpha, Image_shape)

    elif model_type.lower() == "ppo_l":
        return ppo_model_l(alpha, Image_shape, lstm_cnt)

    elif model_type.lower() == "c_ppo_l":
        return c_ppo_model_l(alpha, Image_shape, lstm_cnt)

    elif model_type.lower() == "ppo":
        return ppo_model(alpha, Image_shape)

    elif model_type.lower() == "c_ppo":
        return c_ppo_model(alpha, Image_shape)

    else:
        print("incorrect data input for model")