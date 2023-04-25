import os
import json
from flask import Flask, request, jsonify, flash, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename
from flask_cors import CORS, cross_origin
import pylibjpeg
from pydicom import dcmread
import scipy.ndimage as ndi
import pandas as pd  
import pydicom, numpy as np
import matplotlib.pylab as plt
import os
import zipfile
import seaborn as sns
from random import randrange
from tqdm.notebook import tqdm
import pickle
import cv2
from PIL import Image
import tensorflow as tf
from autokeras.keras_layers import CastToFloat32
from tensorflow import keras
from skimage import morphology

from keras.preprocessing import image
from numpy import asarray
import matplotlib.cm as cm
from PIL import Image as im
import platform

from lime import lime_image
from lime.wrappers.scikit_image import SegmentationAlgorithm
from skimage.segmentation import mark_boundaries, slic

from Utilities.xai_functions import *



##########################################################################################
########################### Preprocessing ################################################
##########################################################################################

prefix = '/' if platform.system() == 'Windows' else ''

def transform_to_hu(img, intercept, slope):
    hu_image = img * slope + intercept

    return hu_image

def window_image(img, window_center, window_width, intercept, slope, rescale=True):
    img = transform_to_hu(img, intercept, slope)
    is_zero = img == 0
    img_min = window_center - window_width//2
    img_max = window_center + window_width//2
    img[img<img_min] = img_min
    img[img>img_max] = img_max

    if rescale:
        # Extra rescaling to 0-1, not in the original notebook
        img = (img - img_min) / (img_max - img_min)
    
    img[is_zero] = 0
    
    return img
    
def get_first_of_dicom_field_as_int(x):
    #get x[0] as in int is x is a 'pydicom.multival.MultiValue', otherwise get int(x)
    if type(x) == pydicom.multival.MultiValue:
        return int(x[0])
    else:
        return int(x)

def get_windowing(data):
    dicom_fields = [data[('0028','1050')].value, #window center
                    data[('0028','1051')].value, #window width
                    data[('0028','1052')].value, #intercept
                    data[('0028','1053')].value] #slope
    return [get_first_of_dicom_field_as_int(x) for x in dicom_fields]

    
    
def view_images(images, title = '', aug = None):
    width = 5
    height = 2
    fig, axs = plt.subplots(height, width, figsize=(15,5))
    
    for im in range(0, height * width):
        data = pydicom.read_file(os.path.join(TRAIN_IMG_PATH,'ID_'+images[im]+ '.dcm'))
        image = data.pixel_array
        window_center , window_width, intercept, slope = get_windowing(data)
        image_windowed = window_image(image, window_center, window_width, intercept, slope)


        i = im // width
        j = im % width
        axs[i,j].imshow(image_windowed, cmap=plt.cm.bone) 
        axs[i,j].axis('off')
        
    plt.suptitle(title)
    plt.show()
    
def convert_pixelarray_to_rgb(image1, image2, image3):
    image = np.zeros((image1.shape[0], image1.shape[1], 3))
    
    image[:,:,0] = image1
    image[:,:,1] = image2
    image[:,:,2] = image3

    return image

def mask_image(brain_image):
    
    segmentation = morphology.dilation(brain_image, np.ones((1, 1)))
    labels, label_nb = ndi.label(segmentation)
    
    label_count = np.bincount(labels.ravel().astype(np.int))
    label_count[0] = 0

    mask = labels == label_count.argmax()
 
    mask = morphology.dilation(mask, np.ones((1, 1)))
    mask = ndi.morphology.binary_fill_holes(mask)
    mask = morphology.dilation(mask, np.ones((3, 3)))
    return mask

def align_image(image):
    img=np.uint8(image)
    contours, hier =cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # find the biggest contour (c) by the area
    c = max(contours, key = cv2.contourArea)

    (x,y),(MA,ma),angle = cv2.fitEllipse(c)

    cv2.ellipse(img, ((x,y), (MA,ma), angle), color=(0, 255, 0), thickness=2)

    rmajor = max(MA,ma)/2
    if angle > 90:
        angle -= 90
    else:
        angle += 96
    xtop = x + math.cos(math.radians(angle))*rmajor
    ytop = y + math.sin(math.radians(angle))*rmajor
    xbot = x + math.cos(math.radians(angle+180))*rmajor
    ybot = y + math.sin(math.radians(angle+180))*rmajor
    cv2.line(img, (int(xtop),int(ytop)), (int(xbot),int(ybot)), (0, 255, 0), 3)

    M = cv2.getRotationMatrix2D((x, y), angle-90, 1)  #transformation matrix

    img = cv2.warpAffine(image, M, (img.shape[1], img.shape[0]), cv2.INTER_CUBIC)
    return img

def center_image(image, com, dim):
    if dim == 3:
        height, width, _ = image.shape
        shift = (height/2-com[0], width/2-com[1], 0)
    else:
        height, width = image.shape
        shift = (height/2-com[0], width/2-com[1])
    res_image = ndi.shift(image, shift)
    return res_image

def scale_image(image, mask):
    height, width = image.shape
    coords = np.array(np.nonzero(mask))
    top_left = np.min(coords, axis=1)
    bottom_right = np.max(coords, axis=1)
    
    cropped_image = image[top_left[0]:bottom_right[0],
                            top_left[1]:bottom_right[1]]
    hc, wc = cropped_image.shape
    
    size = np.max([hc, wc])

    top_left = int((size - hc) / 2)
    bottom_right = int((size - wc) / 2)

    cropped_img_sqr = np.zeros((size, size))
    cropped_img_sqr[top_left:top_left+hc, bottom_right:bottom_right+wc] = cropped_image
    cropped_img_sqr = cv2.resize(cropped_img_sqr, (height,width), interpolation=cv2.INTER_LINEAR)
    
    return cropped_img_sqr
    
def apply_windowing(data, plot = False):
    data = pydicom.dcmread(data)
    img = data.pixel_array
    window_center , window_width, intercept, slope = get_windowing(data)
    if plot:
        plt.imshow(img)
        plt.title("original")
        plt.show()
    
    brain_image = window_image(img, 40, 80, intercept, slope) #bone windowing
    image_mask = mask_image(brain_image)
    
    img = image_mask * img
    if plot:
        plt.imshow(image_mask)
        plt.title("mask")
        plt.show()

        plt.imshow(img)
        plt.title("after denoise")
        plt.show()

    #img = align_image(img)
    
    if plot:
        plt.imshow(img)
        plt.title("after align mask")
        plt.show()
    img = scale_image(img, image_mask)
    if plot:
        plt.imshow(img)
        plt.title("after scale")
        plt.show()

    # windowing: brain, subdural, bonea
    img2 = window_image(img, 40, 80, intercept, slope)
    img3 = window_image(img, 80, 200, intercept, slope)
    img4 = window_image(img, 600, 2800, intercept, slope)

    # combine 3 images into RGB channels
    img = convert_pixelarray_to_rgb(img2, img3, img4)
        
    if plot:
        plt.imshow(img)
        plt.title("after combine to rgb")
        plt.show()

    img2 = window_image(img, 40, 80, intercept, slope)
    com = np.average(np.nonzero(img2), axis=1)
    # img = center_image(img, com, img.ndim)

    if plot:
        plt.imshow(img)
        plt.title("after centering")
        plt.show()

    img = cv2.resize(img, (224,224), interpolation=cv2.INTER_LINEAR)
    
    if plot:
        plt.imshow(img)
        plt.title("after resizing")
        plt.show()

    return img


##########################################################################################
############################ AI Execution ################################################
##########################################################################################

def execute_AI(file, id_model, local_model, layer, predict_id):
    
    path = (os.path.join(prefix +'static/' + str(predict_id)))
    os.mkdir(path)
    data = apply_windowing(file, False)
    data = np.array(data).astype(np.float32)
    plt.figure(frameon=False)
    imgplot = plt.imshow(data)
    plt.axis('off')
    plt.savefig(prefix + 'static/' + predict_id + '/scan.png', bbox_inches = 'tight')
    plt.close()

    prediction = local_model.predict(np.expand_dims(data, axis=0))
    
    print(np.round(prediction))

    #plotLIME(local_model, data, prediction, predict_id)

    
    #heatmap = make_gradcam_heatmap(np.expand_dims(data, axis=0), local_model, layer)
    #calculate_heatmap(heatmap, np.expand_dims(data, axis=0), predict_id)

    if id_model == 1:
        if np.round(prediction).max() > 0:
            x = {
                "result": "Hemorrhage Stroke Detected",
                "prediction": str(prediction)
                }
        else:
            x = {
                "result": "No Hemorrhage Stroke Detected",
                "prediction": str(prediction)
                }
    elif id_model == 2:
        if np.round(prediction).max() > 0:
            x = {
                "result": "Ischemic Stroke Detected",
                "prediction": str(prediction)
                }
        else:
            x = {
                "result": "No Ischemic Stroke Detected",
                "prediction": str(prediction)
                }
    else:
        if np.argmax(prediction) == 0:
            x = {
                "result": "No Stroke Detected",
                "prediction": str(prediction)
                }
        elif np.argmax(prediction) == 1:
            x = {
                "result": "Hemorrhage Stroke Detected",
                "prediction": str(prediction)
                }
        else:
            x = {
                "result": "Ischemic Stroke Detected",
                "prediction": str(prediction)
                }

    json_string = json.dumps(x)
    with open(prefix + 'static/' + predict_id + '/result.json', 'w') as outfile:
        outfile.write(json_string)
    return prefix + 'static/' + predict_id

def execute_AI_zip(files, id_model, local_model, layer, predict_id):
    
    path = (os.path.join(prefix +'static/' + str(predict_id)))
    os.mkdir(path)
    predictions = {}
    with zipfile.ZipFile(files, 'r') as zip_file:
        # Loop through all the files in the zip file
        for num, filename in enumerate(zip_file.namelist()):
            # Check if the file is a DICOM file
            if filename.endswith('.dcm'):
                # Read the DICOM file from the zip archive
                with zip_file.open(filename) as file:
                    # Load the DICOM file using pydicom
                    #print(pydicom.dcmread(files))
                    patient_id = 'Test'#pydicom.dcmread(files)[0x0010, 0x0020].value
                    id_scan = patient_id + "_"+str(num)

                    data, dicom_file_old = apply_windowing(file, False)
                    data = np.array(data).astype(np.float32)
                    plt.figure(frameon=False)
                    plt.imshow(data)
                    plt.axis('off')
                    plt.savefig(prefix + 'static/' + predict_id + '/scan.png', bbox_inches = 'tight')
                    plt.close()

                    local_pred = local_model.predict(np.expand_dims(data, axis=0))
                    predictions[id_scan] = local_pred
    print(predictions)
    max_id = max(predictions, key=predictions.get)
    max_value = predictions[max_id]
    print(max_value, max_id)
    prediction = max_value.flatten()
    #print(np.round(max_value))
    if id_model == 1:
        if np.round(prediction).max() > 0:
            x = {
                "result": "Hemorrhage Stroke Detected",
                "prediction": str(prediction),
                "layer": str(max_id)
                }
        else:
            x = {
                "result": "No Hemorrhage Stroke Detected",
                "prediction": str(prediction),
                "layer": str(max_id)
                }
    elif id_model == 2:
        if np.round(prediction).max() > 0:
            x = {
                "result": "Ischemic Stroke Detected",
                "prediction": str(prediction),
                "layer": str(max_id)
                }
        else:
            x = {
                "result": "No Ischemic Stroke Detected",
                "prediction": str(prediction),
                "layer": str(max_id)
                }
    else:
        if np.argmax(prediction) == 0:
            x = {
                "result": "No Stroke Detected",
                "prediction": str(prediction),
                "layer": str(max_id)
                }
        elif np.argmax(prediction) == 1:
            x = {
                "result": "Hemorrhage Stroke Detected",
                "prediction": str(prediction),
                "layer": str(max_id)
                }
        else:
            x = {
                "result": "Ischemic Stroke Detected",
                "prediction": str(prediction),
                "layer": str(max_id)
                }

    json_string = json.dumps(x)
    with open(prefix + 'static/' + predict_id + '/result.json', 'w') as outfile:
        outfile.write(json_string)
    print("Done")
    return prefix + 'static/' + predict_id



def explain_AI(file, id_model, local_model, layer, predict_id):
    
    path = (os.path.join(prefix +'static/' + str(predict_id)))
    os.mkdir(path)
    data = apply_windowing(file, False)
    data = np.array(data).astype(np.float32)
    plt.figure(frameon=False)
    imgplot = plt.imshow(data)
    plt.axis('off')
    plt.savefig(prefix + 'static/' + predict_id + '/scan.png', bbox_inches = 'tight')
    plt.close()

    prediction = local_model.predict(np.expand_dims(data, axis=0))
    
    print(np.round(prediction))

    #plotLIME(local_model, data, prediction, predict_id)

    
    #heatmap = make_gradcam_heatmap(np.expand_dims(data, axis=0), local_model, layer)
    #calculate_heatmap(heatmap, np.expand_dims(data, axis=0), predict_id)



def explain_AI(file, id_xai, id_model, complexity, local_model, layer, predict_id):
    path = (os.path.join(prefix +'static/' + str(predict_id)))
    os.mkdir(path)
    file_copy = file
    data = apply_windowing(file_copy, False)
    data = np.array(data).astype(np.float32)
    plt.figure(frameon=False)
    plt.imshow(data)
    plt.axis('off')
    plt.savefig(prefix + 'static/' + predict_id + '/scan.png', bbox_inches = 'tight')
    plt.close()

    prediction = local_model.predict(np.expand_dims(data, axis=0))
    
    print(np.round(prediction))

    if id_xai == 'lime':
        plotLIME(local_model, data, prediction, complexity, predict_id)
    else:
        heatmap = make_gradcam_heatmap(np.expand_dims(data, axis=0), local_model, layer)
        calculate_heatmap(heatmap, np.expand_dims(data, axis=0), predict_id)

    if id_model == 1:
        if np.round(prediction).max() > 0:
            x = {
                "result": "Hemorrhage Stroke Detected",
                "prediction": str(prediction)
            }
        else:
            x = {
                "result": "No Hemorrhage Stroke Detected",
                "prediction": str(prediction)
                }
    elif id_model == 2:
        if np.round(prediction).max() > 0:
            x = {
                "result": "Ischemic Stroke Detected",
                "prediction": str(prediction)
                }
        else:
            x = {
                "result": "No Ischemic Stroke Detected",
                "prediction": str(prediction)
                }
    else:
        if np.argmax(prediction) == 0:
            x = {
                "result": "No Stroke Detected",
                "prediction": str(prediction)
                }
        elif np.argmax(prediction) == 1:
            x = {
                "result": "Hemorrhage Stroke Detected",
                "prediction": str(prediction)
                }
        else:
            x = {
                "result": "Ischemic Stroke Detected",
                "prediction": str(prediction)
                }

    json_string = json.dumps(x)
    with open(prefix + 'static/' + predict_id + '/result.json', 'w') as outfile:
        outfile.write(json_string)
    return prefix + 'static/' + predict_id

