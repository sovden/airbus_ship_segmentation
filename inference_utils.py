import matplotlib.pyplot as plt
import PIL
import numpy as np
import math
from keras import backend as K

def compute_dice_for_inference(y_mask, yp_mask, smooth = 1.):
    y_true_f = K.flatten(y_mask.astype('float32'))
    y_pred_f = K.flatten(yp_mask.astype('float32'))
    intersection = K.sum(y_true_f * y_pred_f)
    dice = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return dice

def img_by_path(path, target_size=(256, 256)):
    img = PIL.Image.open(path)
    img = img.resize(target_size)
    c_img = np.array(img)
    print(f"image shape after resize: {c_img.shape}")
    return c_img

def show_images_segmentation(x, y, yp, dice_coefs = None, batch_size=8):
    columns = 3
    rows = min(batch_size, 8)
    fig=plt.figure(figsize=(columns*2, rows*2))
    for i in range(rows):
        fig.add_subplot(rows, columns, 3*i+1)
        plt.axis('off')
        plt.imshow(x[i])
        plt.title("origin")
        fig.add_subplot(rows, columns, 3*i+2)
        plt.axis('off')
        plt.imshow(y[i])
        plt.title("truth")
        fig.add_subplot(rows, columns, 3*i+3)
        plt.axis('off')
        plt.imshow(yp[i])
        if dice_coefs is not None:
            plt.title(f"predict, dice: {round(dice_coefs[i],2)}")
        else:
            plt.title("predict")
    plt.show()

def show_one_image_segmentation(img, yp):
    columns = 2
    rows = 1
    fig=plt.figure(figsize=(columns*4, rows*4))
    fig.add_subplot(rows, columns, 1)
    plt.axis('off')
    plt.imshow(img)
    plt.title("original img")
    fig.add_subplot(rows, columns, 2)
    plt.axis('off')
    plt.imshow(yp)
    plt.title("predicted mask")
    plt.show()


def show_images_classifier(x, y, yp, batch_size=None):
    columns = 4
    rows = math.ceil(batch_size/columns)
    fig = plt.figure(figsize=(columns*3, rows*3))
    for i in range(batch_size):
        fig.add_subplot(rows, columns, i+1)
        plt.imshow(x[i]/255.0)
        plt.axis('off')
        plt.title(f"truth:{int(y[i])}, pred: {yp[i]}")

    plt.show()

def show_one_image_classifier(img, yp):
    plt.imshow(img / 255.0)
    plt.axis('off')
    plt.title(f"predict: {bool(yp)}")
    plt.show()