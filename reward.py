import numpy as np
import math
from skimage.metrics import structural_similarity as ssim
import cv2
import scipy
from environment import show

def mse(imageA, imageB):
    # the 'Mean Squared Error' between the two images is the
    # sum of the squared difference between the two images;
    # NOTE: the two images must have the same dimension
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])

    # return the MSE, the lower the error, the more "similar"
    # the two images are
    return err

def image_dif(target, current):
    dif = np.subtract(current,target)

    neg_array = []
    pos_array = []
    for y_cordinate in range(len(dif)):
        row = dif[y_cordinate]
        for x_cordinate in range(len(row)):
            pixel_val = row[x_cordinate]
            if pixel_val < 0:
                neg_array.append([y_cordinate,x_cordinate])
            elif pixel_val > 0:
                pos_array.append([y_cordinate, x_cordinate])

    cnt = 0
    sum = 0
    for neg_cordinate in neg_array:
        for pos_cordinate in pos_array:
            cnt +=1
            negative_y_value = neg_cordinate[0]
            negative_x_value = neg_cordinate[1]

            positive_y_value = pos_cordinate[0]
            positive_x_value = pos_cordinate[1]


            x_distance = (positive_x_value - negative_x_value)**2
            y_distance = (positive_y_value - negative_y_value)**2
            distance = math.sqrt(x_distance + y_distance)
            sum += distance

    avg_dif = sum / cnt

    return avg_dif

#todo perfect reward normalizer. make it 10 or 1000 and see difference(see if you can use how close it is to goal state as a factor(next_difference))
def image_reward(goal, prev, next, allowable_error = 0):
    prev_difference = image_dif(goal,prev)*1000
    next_difference = image_dif(goal,next)*1000

    reward = prev_difference - next_difference
    # reward = max(0, reward)
    # reward = min(1, reward)
    print(prev_difference, next_difference, reward)

    dif = next_difference
    if dif <= allowable_error:
        return reward, True, False

    if np.array_equal(prev, next):
        return reward, True, True

    return reward, False, False

def checkifdone(cur, new, allowable_error = 0):
    if image_dif(cur, new) <= allowable_error:
        return True
    return False



def crude_reward(goal, prev, next, cur_image, new_image,allowable_error = 0):
    prev_difference = np.abs(np.subtract(goal, prev))[0]
    next_difference = np.abs(np.subtract(goal, next))[0]

    rew = prev_difference - next_difference
    reward = np.sum(rew)

    dif = next_difference

    if (dif[0] <= allowable_error) and (dif[1] <= allowable_error) and (dif[2] <= allowable_error) and (dif[3] <= allowable_error) and (dif[4] <= allowable_error) and (dif[5] <= allowable_error):
        return reward, True, False

    if np.array_equal(cur_image, new_image):
        return reward, True, True

    return reward, False, False

def sparse_reward(goal, prev, next, cur_image, new_image,allowable_error = 0):
    prev_difference = np.abs(np.subtract(goal, prev))
    next_difference = np.abs(np.subtract(goal, next))

    difa = np.sum(prev_difference)

    difb = np.sum(next_difference)

    reward = difa - difb
    reward = 0

    if difb <= allowable_error:
        return reward, True, False

    if np.array_equal(cur_image, new_image):
        return reward, True, True

    return reward, False, False
