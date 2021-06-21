import sys
sys.path.append("C:/Users/josho/Desktop/school_work/PROJECT/free_cad_stuffs/FreeCAD_0.19.22284-Win-Conda_vc14.x-x86_64/bin")
sys.path.append("C:/Users/josho/Desktop/school_work/PROJECT/model/environment_tools/")
from svglib.svglib import svg2rlg
from reportlab.graphics import renderPM
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import FreeCAD
from export_svg import export_svg
from keras_preprocessing.image import load_img
import cv2

def scale(OldMin, OldMax, OldValue):
    NewMax = 1
    NewMin = -1

    OldRange = (OldMax - OldMin)
    if OldRange == 0:
        NewValue = NewMin
    else:
        NewRange = (NewMax - NewMin)
        NewValue = (((OldValue - OldMin) * NewRange) / OldRange) + NewMin

    return NewValue

def scale_array(array, variance = 20):
    real = []
    for x in array[0][:3]:
        real.append(scale(-variance,variance,x))
    for x in array[0][3:]:
        real.append(scale(-180,180,x))

    return real


def unscale(NewMin, NewMax, OldValue):
    OldMax = 1
    OldMin = -1

    OldRange = (OldMax - OldMin)
    if OldRange == 0:
        NewValue = NewMin
    else:
        NewRange = (NewMax - NewMin)
        NewValue = (((OldValue - OldMin) * NewRange) / OldRange) + NewMin

    return NewValue

def unscale_array(array, variance = 20):
    real = []
    for x in array[0][:3]:
        real.append(unscale(-variance,variance,x))
    for x in array[0][3:]:
        real.append(unscale(-180,180,x))

    return real

def dif_prep(target, current):
    dif = np.subtract(current, target)

    # dif = dif.reshape(-1, dif.shape[0], dif.shape[1])
    dif = dif.reshape(list(dif.shape) + [1])

    return dif.tolist()

def stack_prep(target, current):
    stack =   np.vstack((target,current))

    # image = stack.reshape(-1, stack.shape[0], stack.shape[1])
    stack.reshape(list(stack.shape) + [1])
    return stack

def imshow(image, work = "default"):
    cv2.imshow(work, image)
    if cv2.waitKey(25) & 0xFF == ord("q"):
        cv2.destroyAllWindows()

def show(img):
    plt.imshow(img)
    plt.show()

def preprocess_for_reward(img_path, im_size):
    img = load_img(img_path, color_mode="grayscale")
    img_array = np.asarray(img)

    img_array = cv2.resize(img_array, (im_size[0], im_size[1]), interpolation = cv2.INTER_AREA)

    img_array = img_array / 255
    img_array = np.subtract(1, img_array)

    return img_array

def preprocess_for_model(img_path, im_size):
    img = load_img(img_path, color_mode="rgb")
    img_array = np.asarray(img)

    img_array = cv2.resize(img_array, (im_size[0], im_size[1]), interpolation = cv2.INTER_AREA)

    return img_array

def preprocess_for_show(img_path, image_size):
    img = Image.open(img_path)
    img = img.resize(image_size)

    return img

#todo: fix revolute stuff in freeCAD so we can be able to use arrow object i.e FreeCAD.activeDocument().Revolution should be FreeCAD.activeDocument().Body001
def make_image(state, model_path, iso_details, image_size = (224, 224), usage = "reward"):
    #state = state array containing the postion and orientation of part
    #model_path = file location part where CAD models are stored
    #scale = size of the model in the result image
    #view_box = position of camera in the image
    scale = iso_details[0]
    view_box = iso_details[1]
    line_width = iso_details[2]

    # OPEN FREECAD APPLICATION
    FreeCAD.newDocument("doc")

    # IMPORT PARTS
    FreeCAD.ActiveDocument.mergeProject(model_path)

    # MOVE PARTS TO DESIRED LOCATION
    #Using euler angles
    FreeCAD.getDocument("doc").Body001.Placement = FreeCAD.Placement(FreeCAD.Vector(state[0,0], state[0,1],state[0,2]), FreeCAD.Rotation(state[0,3],state[0,4],state[0,5]), FreeCAD.Vector(0, 0, 0))
    # FUSE PARTS TO SINGLE ENTITY
    tablea = FreeCAD.activeDocument().addObject("Part::MultiFuse", "Fusion")
    FreeCAD.activeDocument().Fusion.Shapes = [FreeCAD.activeDocument().Body, FreeCAD.activeDocument().Body001]
    FreeCAD.ActiveDocument.recompute()

    # CONVERT 3D MODEL TO SVG DRAWING
    path = "temps/svg.svg"
    export_svg([tablea], file_path=path, scale = scale, view_box = view_box, line_width = line_width)

    # CLOSE FREECAD APPLICATION
    FreeCAD.closeDocument("doc")

    #CONVERT DRAWING TO IMAGE
    drawing = svg2rlg("temps/svg.svg")
    renderPM.drawToFile(drawing, "temps/PNG_image.png", fmt="PNG")

    # load the image
    img_path = 'temps/PNG_image.png'

    if usage.lower() == "model":
        return preprocess_for_model(img_path, image_size)

    elif usage.lower() == "show":
        return preprocess_for_show(img_path, image_size)

    elif usage.lower() == "reward" or usage.lower() == "done":
        return preprocess_for_reward(img_path, image_size)

    else:
        print("enter correct string for image use:model, reward, done, show ")

