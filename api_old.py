import os
import json
import uuid
from flask import Flask, request, jsonify, flash, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename
from flask_cors import CORS, cross_origin
import pylibjpeg
from pydicom import dcmread

import pandas as pd  
import pydicom, numpy as np
import matplotlib.pylab as plt
import os
import seaborn as sns
from random import randrange
from tqdm.notebook import tqdm
import pickle
import cv2
from PIL import Image
import tensorflow as tf
from autokeras.keras_layers import CastToFloat32
from tensorflow import keras

from keras.preprocessing import image
from numpy import asarray
import matplotlib.cm as cm
from PIL import Image as im

from lime import lime_image
from lime.wrappers.scikit_image import SegmentationAlgorithm
from skimage.segmentation import mark_boundaries, slic


##########################################################################################
########################### Preprocessing ################################################
##########################################################################################

def window_image(img, window_center,window_width, intercept, slope, rescale=True):
    
    img = (img*slope +intercept)
    img_min = window_center - window_width//2
    img_max = window_center + window_width//2
    img[img<img_min] = img_min
    img[img>img_max] = img_max
    
    if rescale:
        # Extra rescaling to 0-1, not in the original notebook
        img = (img - img_min) / (img_max - img_min)
    
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
    # set the first channel to the pixel array
    image[:,:,0] = image1
    image[:,:,1] = image2
    image[:,:,2] = image3
    
    return image

def apply_windowing(data, plot = True):
    data = pydicom.dcmread(data)
    img = data.pixel_array
    window_center , window_width, intercept, slope = get_windowing(data)
    img2 = window_image(img, 40, 80, intercept, slope)
    img3 = window_image(img, 80, 200, intercept, slope)
    img4 = window_image(img, 600, 2800, intercept, slope)

    res_img = convert_pixelarray_to_rgb(img2, img3, img4)
    
    res_img = cv2.resize(res_img, (224,224), interpolation=cv2.INTER_LINEAR)    
        
    if plot:
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(1,4, sharex='col', figsize=(10,24), gridspec_kw={'hspace': 0.1, 'wspace': 0})

        ax1.set_title('Default window')
        im1 = ax1.imshow(img,  cmap=plt.cm.bone)

        ax2.set_title('Brain window')
        img2 = window_image(img, 40, 80, intercept, slope)
        im2 = ax2.imshow(img2, cmap=plt.cm.bone)

        ax3.set_title('Subdural window')
        img3 = window_image(img, 80, 200, intercept, slope)
        im3 = ax3.imshow(img3, cmap=plt.cm.bone)

        ax4.set_title('Bone window')
        img4 = window_image(img, 600, 2800, intercept, slope)
        im4 = plt.imshow(img4, cmap=plt.cm.bone)

    
    return res_img

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
    #print(np.array(superimposed_img))
    imgplot = plt.imshow(superimposed_img)

    plt.axis('off')
    plt.savefig('static/' + predict_id + '/cam.png')
    plt.close()
    
    return imgplot


def visualize_heatmap(heatmap, img, path):
    
    #print(heatmap.shape)
    heatmap = np.array(Image.fromarray(np.uint8(heatmap * 255) , 'L'))
    #print(heatmap.shape)
    imgplot = plt.imshow(heatmap)
    plt.axis('off')
    #plt.savefig('cam.png')
    
    plt.show()
    plt.close()


def plotLIME(model, data, prediction, predict_id):
    explainer = lime_image.LimeImageExplainer() 

    segmenter = SegmentationAlgorithm('slic', n_segments=150, compactness=8, sigma=1,
                     start_label=1)
    explanation_1 = explainer.explain_instance(data, 
                                         classifier_fn = model, 
                                         top_labels=3, 
                                         hide_color=0, # 0 - gray 
                                         num_samples=1000,
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
    plt.savefig('static/' + predict_id + '/lime.png')
    plt.close()

##########################################################################################
############################ AI Execution ################################################
##########################################################################################

def execute_AI(file, id_model, local_model, predict_id):
    
    path = (os.path.join('static/' + str(predict_id)))
    os.mkdir(path)
    data = apply_windowing(file, False)
    data = np.array(data).astype(np.float32)
    
    imgplot = plt.imshow(data)
    plt.axis('off')
    plt.savefig('static/' + predict_id + '/scan.png')
    plt.close()

    prediction = local_model.predict(np.expand_dims(data, axis=0))
    
    print(np.round(prediction))

    plotLIME(local_model, data, prediction, predict_id)

    
    heatmap = make_gradcam_heatmap(np.expand_dims(data, axis=0), local_model, "conv2d_3")
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
    with open('static/' + predict_id + '/result.json', 'w') as outfile:
        outfile.write(json_string)
