from keras.models import Sequential,load_model, Model
from keras.layers import Dense, Input, Conv2D, MaxPooling2D,Flatten, concatenate, Lambda, Dropout,LSTM, SimpleRNN, Reshape,Permute
from make_reference_images import collect_work_variables
from environment import make_image, scale_array, unscale_array, show
from keras.models import load_model
from action import rand_reset_full
import tensorflow as tf
import numpy as np

def create_model():
    Image_shape = (224,224, 1)
    inputA = Input(shape=Image_shape)
    # x = Conv2D(64, 7, activation="relu", padding= "same")(inputA)
    # x = MaxPooling2D(2)(x)
    # x = Conv2D(128, 3, activation="relu", padding="same")(x)
    # x = Conv2D(128, 3, activation="relu", padding="same")(x)
    # x = MaxPooling2D(2)(x)
    # x = Conv2D(256, 3, activation="relu", padding="same")(x)
    # x = Conv2D(256, 3, activation="relu", padding="same")(x)
    # x = MaxPooling2D(2)(x)
    # x = Flatten()(x)
    # x = Dense(128, activation="relu")(x)
    # x = Dropout(0.5)(x)
    # x = Dense(64, activation="relu")(x)
    # x = Dropout(0.5)(x)


    # x = Conv2D(16, (3, 3), activation="relu")(inputA)
    # x = Conv2D(16, (5, 5), activation="relu")(x)
    # x = MaxPooling2D(pool_size=(2, 2))(x)
    # x = Flatten()(x)
    # x = Dense(32, activation="relu")(x)
    # x = Dense(6, activation="tanh")(x)

    model = Sequential()
    model.add(Conv2D(16, 3, activation="relu", padding="same",input_shape=(224,224,1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(16, 3, activation="relu", padding="same"))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Reshape((16, -1)))
    # model.add(Permute((2, 1)))
    # model.add(LSTM(32))
    model.add(Flatten())
    model.add(Dense(64, activation="relu"))
    model.add(Dense(6, activation="tanh"))
    # x = Conv2D(16, (3, 3), activation="relu")((inputA))
    # x = MaxPooling2D(pool_size=(2, 2))(x)
    # x = Conv2D(16, (3, 3), activation="relu")(x)
    # x = MaxPooling2D(pool_size=(2, 2))(x)
    # x = Flatten()(x)
    #
    # x = Dense(128, activation="tanh")(x)
    # x = Dense(64, activation="tanh")(x)
    #
    # x = Dense(6, activation="tanh")(x)


    # model = Model(inputs=inputA, outputs=x)
    model.compile(loss='mse', optimizer="adam", metrics=["accuracy"])
    print(model.summary())
    return model

def make_random_images(work, number_of_images = 10000):
    state, CAD_path, iso_details = collect_work_variables(work)
    image = make_image(state, CAD_path, iso_details, image_size=(224, 224))
    variance = 20

    images = []
    states = []

    #to make neww actual image and state list
    real = scale_array(state, variance=variance)
    image = image.tolist()
    images.append(image)
    states.append(real)
    np.save("test_files/images"+work+"", np.array(images))
    np.save("test_files/states"+work+"", np.array(states))


    for i in range(number_of_images+1):

        state = rand_reset_full(variance)
        image = make_image(state, CAD_path, iso_details, image_size=(224,224))

        real = scale_array(state, variance = variance)
        image = image.tolist()

        images.append(image)
        states.append(real)

        if i % 1000 == 0:
            actual_image_list = np.load("test_files/images"+work+".npy")
            actual_state_list = np.load("test_files/states"+work+".npy")
            actual_image_list = actual_image_list.tolist()
            actual_state_list = actual_state_list.tolist()
            for x in images:
                actual_image_list.append(x)

            for y in states:
                actual_state_list.append(y)
            np.save("test_files/images"+work+"", np.array(actual_image_list))
            np.save("test_files/states"+work+"", np.array(actual_state_list))
            images = []
            states = []
            print("states updated at " + str(i) + "steps")
    return images, states

def train_random_images():
    model = create_model()
    images = np.load("test_files/images"+work+".npy")
    states = np.load("test_files/states"+work+".npy")

    lenn = int(len(images)/10)
    lenn = len(images) - lenn
    print(lenn, len(images))
    X_train, X_valid = images[:lenn], images[lenn:]
    y_train, y_valid = states[:lenn], states[lenn:]
    X_train = X_train.reshape(list(X_train.shape) + [1])
    X_valid = X_valid.reshape(list(X_valid.shape) + [1])

    history = model.fit(X_train,
                        y_train,
                        epochs = 100,
                        shuffle=True,
                        validation_data = (X_valid, y_valid))

    model.save("saved_models/image_test_model"+work+".model")
    return model

work = "house"
variance = 20
temp = make_random_images(work, number_of_images= 5000)
model = train_random_images()
model = load_model('saved_models/image_test_model'+work+'.model')
goal_state, CAD_path, iso_details = collect_work_variables(work)
goal_state = np.array([[0,20,15,0,0,0]])
goal_image = make_image(goal_state, CAD_path, iso_details, image_size=(224, 224))
goal_image = np.array(goal_image)
goal_image = goal_image.reshape(list(goal_image.shape) + [1])
show(goal_image)
goal_image = goal_image.reshape((-1,224,224,1))

print(goal_image)
#
state = model.predict(goal_image)
print(state)
real = unscale_array(state, variance = variance)
print(real)
#
# real = []
# for x in state[0][:3]:
#     real.append(unscale(-variance,variance,x))
# for x in state[0][3:]:
#     real.append(unscale(-180,180,x))
#
# print(real)