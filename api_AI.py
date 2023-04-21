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
################################## XAI ###################################################
##########################################################################################

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    # First, we create a model that maps the input image to the activations
    # of the last conv layer as well as the output predictions
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )
    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        print(preds)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # This is the gradient of the output neuron (top predicted or chosen)
    # with regard to the output feature map of the last conv layer
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    # then sum all the channels to obtain the heatmap class activation
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def calculate_heatmap(cam, img, predict_id):
    img = (img*255)
    img = keras.preprocessing.image.img_to_array(img[0])
    
    #img = img[0]
    #img = keras.preprocessing.image.img_to_array(img[0])
    # Rescale heatmap to a range 0-255
    heatmap = np.uint8(255 * cam)

    # Use jet colormap to colorize heatmap
    jet = cm.get_cmap("jet")

    # Use RGB values of the colormap
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    # Create an image with RGB colorized heatmap
    jet_heatmap = keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = keras.preprocessing.image.img_to_array(jet_heatmap)

    # Superimpose the heatmap on original image
    superimposed_img = jet_heatmap * 0.4 + img
    superimposed_img = keras.preprocessing.image.array_to_img(superimposed_img)

    # Save the superimposed image
    # Display Grad CAM
    #print(np.array(superimposed_img))#
    plt.figure(frameon=False)
    imgplot = plt.imshow(superimposed_img)

    plt.axis('off')
    plt.savefig(prefix + 'static/' + predict_id + '/cam.png', bbox_inches = 'tight')
    plt.close()
    
    return imgplot


def visualize_heatmap(heatmap, img, path):
    
    #print(heatmap.shape)
    heatmap = np.array(Image.fromarray(np.uint8(heatmap * 255) , 'L'))
    #print(heatmap.shape)
    plt.figure(frameon=False)
    plt.imshow(heatmap)
    plt.axis('off')
    #plt.savefig('cam.png')
    
    plt.show()
    plt.close()


def plotLIME(model, data, prediction, complexity, predict_id):
    explainer = lime_image.LimeImageExplainer() 

    segmenter = SegmentationAlgorithm('slic', n_segments=200, compactness=8, sigma=1,
                     start_label=1)
    explanation_1 = explainer.explain_instance(data, 
                                         classifier_fn = model, 
                                         top_labels=3, 
                                         hide_color=0, # 0 - gray 
                                         num_samples=complexity,
                                         segmentation_fn=segmenter
                                        )
    temp, mask = explanation_1.get_image_and_mask(explanation_1.top_labels[np.argmax(prediction)], 
                                                positive_only=True, 
                                                num_features=5, 
                                                hide_rest=False)
    lime_res = mark_boundaries(temp, mask, mode = "thick", color = (255,0,0))
    #print(np.max(temp))
    
    imgplot = plt.imshow(lime_res)
    plt.axis('off')
    plt.savefig(prefix + 'static/' + predict_id + '/lime.png', bbox_inches = 'tight')
    plt.close()

def get_XAI_info(xai_id):
    xai_id_method, xai_id_complexity = xai_id.split('_')

    if xai_id_complexity == "low":
        xai_id_complexity = 100
    elif xai_id_complexity == "medium":
        xai_id_complexity = 500
    else:
        xai_id_complexity = 1000

    return xai_id_method, xai_id_complexity

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

