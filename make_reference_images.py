from environment import make_image, show
import numpy as np

def collect_work_variables(work):
    if work.lower() == "table":
        actual_state = [[-206, 233, 20, 0, 0, 0]]
        name = "table"
        model_path = "C:/Users/josho/Desktop/school_work/PROJECT/free_cad_stuffs/assem/simported.fcstd"
        iso_details = [0.75, 1, 2.5]
        goal_image_path = "goal_images/" + name + ".npy"

    elif work.lower() == "arrow":
        actual_state = [[0, 0, 0, 0, 0, 0]]
        name = "arrow"
        model_path = "C:/Users/josho/Desktop/school_work/PROJECT/free_cad_stuffs/assem/imported2.fcstd"
        iso_details = [5, 2, 25]
        goal_image_path = "goal_images/" + name + ".npy"

    elif work.lower() == "nut_and_bolt":
        actual_state = [[0, 0, 0, 0, 0, 0]]
        name = "nut_and_bolt"
        model_path = "C:/Users/josho/Desktop/school_work/PROJECT/free_cad_stuffs/assem/imported3.fcstd"
        iso_details = [5, 2, 50]
        goal_image_path = "goal_images/" + name + ".npy"

    elif work.lower() == "triangle":
        actual_state = [[0, 0, 0, 0, 0, 0]]
        name = "triangle"
        model_path = "C:/Users/josho/Desktop/school_work/PROJECT/free_cad_stuffs/assem/imported4.fcstd"
        iso_details = [18, 2, 150]
        goal_image_path = "goal_images/" + name + ".npy"

    elif work.lower() == "disk_and_shaft":
        actual_state = [[0, 0, 0, 0, 0, 0]]
        name = "disk_and_shaft"
        model_path = "C:/Users/josho/Desktop/school_work/PROJECT/free_cad_stuffs/assem/imported5.fcstd"
        iso_details = [9, 2, 100]
        goal_image_path = "goal_images/" + name + ".npy"

    elif work.lower() == "house":
        actual_state = [[0, 0, 0, 0, 0, 0]]
        name = "house"
        model_path = "C:/Users/josho/Desktop/school_work/PROJECT/free_cad_stuffs/assem/imported6.fcstd"
        iso_details = [9, 2, 100]
        goal_image_path = "goal_images/" + name + ".npy"

    elif work.lower() == "ring":
        actual_state = [[0, 0, 0, 0, 0, 0]]
        name = "ring"
        model_path = "C:/Users/josho/Desktop/school_work/PROJECT/free_cad_stuffs/assem/imported7.fcstd"
        iso_details = [9, 2, 100]
        goal_image_path = "goal_images/" + name + ".npy"

    elif work.lower() == "hammer":
        actual_state = [[0, 0, 0, 0, 0, 0]]
        name = "hammer"
        model_path = "C:/Users/josho/Desktop/school_work/PROJECT/free_cad_stuffs/assem/imported8.fcstd"
        iso_details = [9, 2, 100]
        goal_image_path = "goal_images/" + name + ".npy"


    elif work.lower() == "dumbell":
        actual_state = [[0, 0, 0, 0, 0, 0]]
        name = "dumbell"
        model_path = "C:/Users/josho/Desktop/school_work/PROJECT/free_cad_stuffs/assem/imported9.fcstd"
        iso_details = [3, 2, 50]
        goal_image_path = "goal_images/" + name + ".npy"

    elif work.lower() == "square":
        actual_state = [[0, 0, 0, 0, 0, 0]]
        name = "square"
        model_path = "C:/Users/josho/Desktop/school_work/PROJECT/free_cad_stuffs/assem/imported10.fcstd"
        iso_details = [6, 2, 60]
        goal_image_path = "goal_images/" + name + ".npy"

    elif work.lower() == "chevron":
        actual_state = [[0, 0, 0, 0, 0, 0]]
        name = "chevron"
        model_path = "C:/Users/josho/Desktop/school_work/PROJECT/free_cad_stuffs/assem/imported11.fcstd"
        iso_details = [9, 2, 100]
        goal_image_path = "goal_images/" + name + ".npy"

    elif work.lower() == "okoso":
        actual_state = [[0, 0, 0, 0, 0, 0]]
        name = "okoso"
        model_path = "C:/Users/josho/Desktop/school_work/PROJECT/free_cad_stuffs/assem/imported12.fcstd"
        iso_details = [9, 2, 100]
        goal_image_path = "goal_images/" + name + ".npy"


    else:
        print("incorrect work name")

    return np.array(actual_state), model_path, iso_details


def make_goal_image(work, color):
    if color.lower() == "rgb":
        name = work + "_rgb"
        method = "model"

    elif color.lower() == "grayscale":
        method = "reward"

    else:
        print("incorrect color format")

    work_variable = collect_work_variables(work)
    image = make_image(work_variable[0],work_variable[1], work_variable[2], usage= method)
    show(image)
    np.save("goal_images/" + work + ".npy", image)

def make_goal_images():
    for work in ["table", "arrow","nut_and_bolt","triangle","disk_and_shaft", "house", "ring","hammer","dumbell","square","chevron","okoso"]:
        for color in ["rgb", "grayscale"]:
            make_goal_image(work, color)

def make_sample_image(work, state = np.array([[0,  0 , 0, 0,  0,  0]])):
    work_variable = collect_work_variables(work)
    image = make_image(state, work_variable[1], work_variable[2], usage="show")
    show(image)
    np.save("goal_images/target_image.npy", image)

#make_sample_image("square", state = np.array([[0, 0, 50 , 0 , 0, 0]]))