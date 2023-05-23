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
import torch
import torch.nn as nn
from pydicom.uid import ExplicitVRLittleEndian
from scipy import interpolate
import shutil
import pydicom_seg
import SimpleITK as sitk

from keras.preprocessing import image
from numpy import asarray
import matplotlib.cm as cm
from PIL import Image as im
import platform

from lime import lime_image
from lime.wrappers.scikit_image import SegmentationAlgorithm
from skimage.segmentation import mark_boundaries, slic

from Utilities.xai_functions import *
from Utilities.db_functions import *

##########################################################################################
########################### Preprocessing ################################################
##########################################################################################

prefix = '/' if platform.system() == 'Windows' else ''

def transform_to_hu(img, intercept, slope):
    """Transform raw pixel values in CT scans to the Hounsfield Units (HU)
        HU: A standardized scale used to measure tissue density.

    Args:
        img (numpy array): Waw pixel values of CT scan
        intercept (int): Linear attenuation coefficients of the tissues.
        slope (int): Linear attenuation coefficients of the tissues.

    Returns:
        numpy array: Converted HU values from raw pixel values of img
    """
    hu_image = img * slope + intercept

    return hu_image

def window_image(img, window_center, window_width, intercept, slope, rescale=True):
    """Apply windowing to enhance specific regions of interest (brain, bone, subdural)

    Args:
        img (numpy array): Raw pixel values of CT scan
        window_center (int): Midpoint or central value of the selected range of pixel values
        window_width (int): Determine the range of pixel values that will be displayed.
        intercept (int): Linear attenuation coefficient (intercept) of the tissues.
        slope (int): Linear attenuation coefficient (slope) of the tissues.
        rescale (bool, optional): Extra rescaling to 0-1. Defaults to True.

    Returns:
        numpy array: windowed image
    """
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
    """Get default windowing value

    Args:
        data (dicom format): Input image for getting windowing values

    Returns:
        int, int, int, int: window_center , window_width, intercept, slope
    """
    dicom_fields = [data[('0028','1050')].value, #window center
                    data[('0028','1051')].value, #window width
                    data[('0028','1052')].value, #intercept
                    data[('0028','1053')].value] #slope
    return [get_first_of_dicom_field_as_int(x) for x in dicom_fields]
    
def convert_pixelarray_to_rgb(image1, image2, image3):
    """Combine image1, image2, image3 into 3-channels image

    Args:
        image1 (numpy array): Windowed image (Brain)
        image2 (numpy array): Windowed image (Subdural)
        image3 (numpy array): Windowed image (Bone)

    Returns:
        numpy array: Combined 3-channels image
    """

    image = np.zeros((image1.shape[0], image1.shape[1], 3))
    
    image[:,:,0] = image1
    image[:,:,1] = image2
    image[:,:,2] = image3

    return image

def mask_image(brain_image, dilation = 12):
    """Find brain mask

    Args:
        brain_image (numpy array): Windowed image (Brain)
        dilation (int, optional): Dilation distance. Defaults to 12.

    Returns:
        numpy array: Mask image for brain_image
    """
    
    segmentation = morphology.dilation(brain_image, np.ones((dilation, dilation)))
    labels, label_nb = ndi.label(segmentation)

    label_count = np.bincount(labels.ravel().astype(np.int))
    label_count[0] = 0
    mask = labels == label_count.argmax()
    mask = morphology.dilation(mask, np.ones((3, 3)))
    mask = ndi.morphology.binary_fill_holes(mask)
    mask = morphology.dilation(mask, np.ones((3, 3)))
        
    return mask

def align_image(image):
    """Align brain image symmetrically

    Args:
        image (numpy array): Mask image of brain image

    Returns:
        numpy array: Aligned brain image
    """

    img=np.uint8(image)
    contours, hier =cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # find the biggest contour (c) by the area
    c = max(contours, key = cv2.contourArea)

    try:
        (x,y),(MA,ma),angle = cv2.fitEllipse(c)
    except:
        return img

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
    # image centering

    if dim == 3:
        height, width, _ = image.shape
        shift = (height/2-com[0], width/2-com[1], 0)
    else:
        height, width = image.shape
        shift = (height/2-com[0], width/2-com[1])
    res_image = ndi.shift(image, shift)
    return res_image

def scale_image(image, mask):
    """Scale the brain image to its full size within the image.

    Args:
        image (numpy array): Input brain image
        mask (numpy array): Mask image of brain image

    Returns:
        numpy array: Scaled brain image
    """

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

def is_small_mask(mask, image, threshold=0.1):
    """
    Check the brain covers less than a threshold portion of the original image.
    Args:
        mask (numpy array): Binary mask of the image.
        images (numpy array): Input image for checking brain size.
        threshold (float): Minimum portion of the image that the mask must cover.
    Returns:
        Boolean. True, if the mask is smaller than the threshold.
                 False, otherwise.
    """
    isSmallMask = False
    image_size = np.prod(image.shape)
    mask_size = np.count_nonzero(mask)
    mask_coverage = mask_size / image_size
    if mask_coverage < threshold:
        isSmallMask = True
    
    return isSmallMask
    
def apply_windowing(data):

    data = pydicom.read_file(data)
    """
    Apply data preprocessing steps:
        Image Denoising
        Image Alignment
        Image Scaling
        Image Centering
        Image Windowing
        Image Resizing
    
    If brain size is too small only proceed with:
        Image Centering
        Image Windowing
        Image Resizing
    Args:
        Dicom: input dicom image for preprocessing
    Returns:
        Dicom: preprocessed image before windowing and resizing
        Numpy Array: preprocessed image (224, 224, 3)
    """
    
    img = data.pixel_array
    window_center , window_width, intercept, slope = get_windowing(data)
    
    dicom_preprocess = data # for display preprocessed image after prediction
 
    brain_image = window_image(img, 40, 80, intercept, slope) #bone windowing
    image_mask = mask_image(brain_image)
    
    # exclude small brain image - too small for detecting stroke
    # need additional process for handling a label and training
    if not is_small_mask(image_mask, img):
        img = image_mask * img
        img = align_image(img)
        img = scale_image(img, image_mask)
        
    else:
        com = np.average(np.nonzero(brain_image), axis=1)
        img = center_image(img, com, img.ndim) # no needs for centering, applied in scaling step
    
    # save preprocessed dicom image for displaying after prediction (without windowing, resizing)
    img_uint16 = img.astype(np.uint16)
    dicom_preprocess.PixelData = img_uint16.tobytes()
    print(f"shape: {img.shape}, type: "+ str(type(img[0][0])))
    
    # windowing: brain, subdural, bone
    img2 = window_image(img, 40, 80, intercept, slope)
    img3 = window_image(img, 80, 200, intercept, slope)
    img4 = window_image(img, 600, 2800, intercept, slope)

    # combine 3 images into RGB channels
    img = convert_pixelarray_to_rgb(img2, img3, img4)

    img = cv2.resize(img, (224,224), interpolation=cv2.INTER_LINEAR)
    img_preprocess = img 
    
    return dicom_preprocess, img_preprocess

def saveDICOM(data_array, old_dicom, name):
    with open("rgb_array.pkl", "wb") as f:
        pickle.dump(data_array, f)
    #pic = Image.fromarray(data_array, 'RGB')

    #pic = pic.convert('L')
    #pic = np.array(pic)
    
    #img = Image.fromarray(data_array)
    #upscaled_img = np.array(img.resize((512, 512), resample=Image.BILINEAR))
    #pic = np.dot(upscaled_img[..., :3], [0.21, 0.72, 0.07]).astype(np.uint8)

    old_dicom = pydicom.dcmread(old_dicom)
    orig_size = data_array.shape
    #old_data = old_dicom.pixel_array
    desired_size = (512,512)

    x_orig = np.arange(0, orig_size[1])
    y_orig = np.arange(0, orig_size[0])
    x_desired = np.linspace(0, orig_size[1]-1, desired_size[1])
    y_desired = np.linspace(0, orig_size[0]-1, desired_size[0])

    # create a 2D interpolation function
    interp_func = interpolate.interp2d(x_orig, y_orig, data_array, kind='linear')

    # use the interpolation function to resample the array
    resampled_arr = interp_func(x_desired, y_desired)

    with open("rgb_array_res.pkl", "wb") as f:
        pickle.dump(resampled_arr, f)
    
    #min_val = old_data.min()
    #max_val = old_data.max()

    # calculate the scaling factor based on the input range and output range
    #scale_factor = int((max_val - min_val) / (pic.max() - pic.min()))

    # rescale the array using the scaling factor and the minimum value
    #pic = (pic - pic.min()) * scale_factor + min_val
    data = data_array #* resampled_arr
    #data = resampled_arr*1000
    old_dicom.PixelData = data.tobytes() 
    old_dicom.Rows, old_dicom.Columns = data.shape
    old_dicom.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

    old_dicom.save_as(name+'_.dcm')
    return old_dicom

def saveDICOM_MaskCOM(data, dicom, name):
    
    orig_size = data.shape
    #old_data = old_dicom.pixel_array
    desired_size = (512,512)

    x_orig = np.arange(0, orig_size[1])
    y_orig = np.arange(0, orig_size[0])
    x_desired = np.linspace(0, orig_size[1]-1, desired_size[1])
    y_desired = np.linspace(0, orig_size[0]-1, desired_size[0])

    # create a 2D interpolation function
    interp_func = interpolate.interp2d(x_orig, y_orig, data, kind='linear')

    # use the interpolation function to resample the array
    resampled_arr = interp_func(x_desired, y_desired)

    #with open("resampled.pickle", "wb") as f:
    #    pickle.dump(resampled_arr, f)
    image = np.ceil(resampled_arr)

    # Get the dimensions of the image
    height, width = resampled_arr.shape

    output_image = np.zeros_like(resampled_arr)

    # Iterate over each pixel in the image
    for i in range(height):
        for j in range(width):
            # Get the pixel value at the current position
            current_pixel = image[i, j]
            
            # Get the neighboring pixel values
            neighbors = []
            if i > 0:
                neighbors.append(image[i-1, j])
            if i < height-1:
                neighbors.append(image[i+1, j])
            if j > 0:
                neighbors.append(image[i, j-1])
            if j < width-1:
                neighbors.append(image[i, j+1])
            
            # Check if there are exactly two different pixel values among the neighbors
            unique_neighbors = set(neighbors)
            if len(unique_neighbors) == 2:
                output_image[i, j] = 0
            else:
                output_image[i, j] = 1

    dicom.PixelData = (dicom.pixel_array*output_image).astype(np.uint16).tobytes()
    dicom.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    dicom.save_as(name+'_segmentation.dcm')

def saveDICOM_Mask(data, dicom, name):

    data = np.expand_dims(data, axis=0).astype(int)
    segmentation = sitk.GetImageFromArray(data)
    segmentation = sitk.Cast(segmentation,sitk.sitkUInt16)

    source_images = [
        pydicom.dcmread(dicom, stop_before_pixels=True)
    ]
    template = pydicom_seg.template.from_dcmqi_metainfo('template.json')
    writer = pydicom_seg.MultiClassWriter(
        template=template,
        inplane_cropping=False,  # Crop image slices to the minimum bounding box on
                                # x and y axes. Maybe not supported by other frameworks.
        skip_empty_slices=True,  # Don't encode slices with only zeros
        skip_missing_segment=False,  # If a segment definition is missing in the
                                    # template, then raise an error instead of
                                    # skipping it.
    )
    dcm = writer.write(segmentation, source_images)
    dcm.save_as(name+'_segmentation.dcm')

##########################################################################################
############################ AI Execution ################################################
##########################################################################################

def execute_torch_AI(file, id_model, local_model, predict_id):
    
    path = (os.path.join(prefix +'static/' + str(predict_id)))
    os.mkdir(path)
    dicom_preprocess, data = apply_windowing(file)
    dicom_preprocess.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

    dicom_preprocess.save_as(prefix + 'static/temp/' + predict_id + '/dicom_scan.dcm')
    data = np.array(data).astype(np.float32)
    plt.figure(frameon=False)
    imgplot = plt.imshow(data)
    plt.axis('off')
    plt.savefig(prefix + 'static/' + predict_id + '/scan.png', bbox_inches = 'tight')
    plt.close()

    data_torch = torch.from_numpy(np.expand_dims(np.moveaxis(data, -1, 0), axis=0)).to('cpu')
    print(data_torch.shape)
    prediction_torch = local_model(data_torch)
    m = nn.Softmax()
    prediction = m(prediction_torch).detach().numpy()
    
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


def execute_AI(file, id_model, local_model, predict_id):
    
    path = (os.path.join(prefix +'static/' + str(predict_id)))
    os.mkdir(path)
    dicom_preprocess, data = apply_windowing(file)
    dicom_preprocess.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

    dicom_preprocess.save_as(prefix + 'static/temp/' + predict_id + '/dicom_scan.dcm')
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

                    dicom_preprocess, data = apply_windowing(file)
                    dicom_preprocess.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

                    dicom_preprocess.save_as(prefix + 'static/temp/' + predict_id + '/dicom_scan.dcm')
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

    json_string = json.dumps(createJson(prediction, id_model, max_id))
    with open(prefix + 'static/' + predict_id + '/result.json', 'w') as outfile:
        outfile.write(json_string)
    print("Done")
    return prefix + 'static/' + predict_id


def predict_AI(file, model_com, model_hem, model_isch, predict_id):
    path = (os.path.join(prefix +'static/' + str(predict_id)))
    os.mkdir(path)
    dicom_preprocess, data = apply_windowing(file)
    dicom_preprocess.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

    dicom_preprocess.save_as(prefix + 'static/temp/' + predict_id + '/dicom_scan.dcm')
    data = np.array(data).astype(np.float32)
    plt.figure(frameon=False)
    imgplot = plt.imshow(data)
    plt.axis('off')
    plt.savefig(prefix + 'static/' + predict_id + '/scan.png', bbox_inches = 'tight')
    plt.close() 

    prediction_com = model_com.predict(np.expand_dims(data, axis=0))
    prediction_hem = model_hem.predict(np.expand_dims(data, axis=0))    
    prediction_isch = model_isch.predict(np.expand_dims(data, axis=0))
    
    res_com = np.round(prediction_com[0])
    res_hem = np.round(prediction_hem)[0]
    res_isch = np.round(prediction_isch)[0]

    if (res_com[0] == 1 and res_hem == 0 and res_isch == 0).all() :
        print_result = "No Stroke Detected"
    elif (res_com[1] == 1 and res_hem == 0 and res_isch == 1).all() :
        print_result = "Ischemic Stroke Detected"
    elif (res_com[2] == 1 and res_hem == 1 and res_isch == 0).all() :
        print_result = "Ischemic Stroke Detected"
    else:
        print_result = "Indication of Stroke"

    x = {
        "result": print_result,
       "prediction_combined": str(prediction_com[0]),
       "prediction_hemorrhage": str(prediction_hem[0]),
       "prediction_ischemic": str(prediction_isch[0])
        }

    json_string = json.dumps(x)
    with open(prefix + 'static/' + predict_id + '/result.json', 'w') as outfile:
        outfile.write(json_string)
    return prefix + 'static/' + predict_id



def explain_AI_old(file, id_model, local_model, layer, predict_id):
    
    path = (os.path.join(prefix +'static/' + str(predict_id)))
    os.mkdir(path)
    dicom_preprocess, data = apply_windowing(file)
    dicom_preprocess.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

    dicom_preprocess.save_as(prefix + 'static/temp/' + predict_id + '/dicom_scan.dcm')
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

def explain_AI_Simple(file, model_com, model_hem, model_isch, model_torch, layer, predict_id, file_name):

    path = (os.path.join(prefix +'static/temp/' + str(predict_id)+'/'))
    os.mkdir(path)
    dicom_preprocess, data = apply_windowing(file)
    dicom_preprocess.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

    dicom_preprocess.save_as(prefix + 'static/temp/'+str(predict_id)+'_dicom_scan.dcm')


    data = np.array(data).astype(np.float32)

    data_torch = torch.from_numpy(np.expand_dims(np.moveaxis(data, -1, 0), axis=0)).to('cpu')
    print(data_torch.shape)
    prediction_torch = model_torch(data_torch)
    m = nn.Softmax()

    prediction_com = model_com.predict(np.expand_dims(data, axis=0))
    prediction_hem = model_hem.predict(np.expand_dims(data, axis=0))    
    prediction_isch = model_isch.predict(np.expand_dims(data, axis=0))

    lime_array = plotLIME(model_torch, data_torch[0], prediction_torch, 50, True)
    prediction_torch = m(prediction_torch).detach().numpy()
    with open ('res.pickle', 'wb') as f:
        pickle.dump(lime_array, f)

    res_com = np.round(prediction_com[0])
    res_hem = np.round(prediction_hem)[0]
    res_isch = np.round(prediction_isch)[0]
    res_com = np.round(prediction_torch)[0]
    
    #lime_array = plotLIME(model_com, data, prediction_com, 100)

    #heatmap = make_gradcam_heatmap(np.expand_dims(data, axis=0), model_com, layer)

    # Grad-Cam will be replaced by SHAP
    # To fully integrate SHAP, we need the final models to create the explainer.
    # SHAP takes a lot of time to compute, meaning only execute it when explicitly wanted!
    
    #cam_array = calculate_heatmap(heatmap, np.expand_dims(data, axis=0), predict_id)

    if (res_com[0] == 1 and res_hem == 0 and res_isch == 0).all() :
        print_result = "No Stroke Detected"
    elif (res_com[1] == 1 and res_hem == 0 and res_isch == 1).all() :
        print_result = "Ischemic Stroke Detected"
    elif (res_com[2] == 1 and res_hem == 1 and res_isch == 0).all() :
        print_result = "Ischemic Stroke Detected"
    else:
        print_result = "Indication of Stroke"

    x = {
        "result": print_result,
       "prediction_combined": str(prediction_com[0]),
       "prediction_hemorrhage": str(prediction_hem[0]),
       "prediction_ischemic": str(prediction_isch[0])
        }

    json_string = json.dumps(x)
    with open(prefix + 'static/temp/' + predict_id + '/result.json', 'w') as outfile:
        outfile.write(json_string)

    #print(lime_array.shape)

    saveDICOM_MaskCOM(lime_array, dicom_preprocess, prefix + 'static/temp/'+str(predict_id))
    return prefix + 'static/' + predict_id

def explain_AI_torch(file, id_xai, id_model, complexity, local_model, predict_id):

    path = (os.path.join(prefix +'static/' + str(predict_id)))
    os.mkdir(path)
    dicom_preprocess, data = apply_windowing(file)
    dicom_preprocess.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

    dicom_preprocess.save_as(prefix + 'static/temp/' + predict_id + '/dicom_scan.dcm')
    data = np.array(data).astype(np.float32)
    plt.figure(frameon=False)
    imgplot = plt.imshow(data)
    plt.axis('off')
    plt.savefig(prefix + 'static/' + predict_id + '/scan.png', bbox_inches = 'tight')
    plt.close()

    data_torch = torch.from_numpy(np.expand_dims(np.moveaxis(data, -1, 0), axis=0)).to('cpu')
    prediction_torch = local_model(data_torch)
    m = nn.Softmax()
    prediction = m(prediction_torch)
    
    #print(np.round(prediction))

    if id_xai == 'lime':
        plotLIME(local_model, data_torch[0], prediction_torch, complexity, predict_id, True)
    # else:
    #     heatmap = make_gradcam_heatmap(np.expand_dims(data, axis=0), local_model, layer)
    #     calculate_heatmap(heatmap, np.expand_dims(data, axis=0), predict_id)
    prediction = prediction.detach().numpy()

    json_string = json.dumps(createJson(prediction, id_model))
    with open(prefix + 'static/' + predict_id + '/result.json', 'w') as outfile:
        outfile.write(json_string)
    return prefix + 'static/' + predict_id


def explain_AI(file, id_xai, id_model, complexity, local_model, layer, predict_id):
    path = (os.path.join(prefix +'static/' + str(predict_id)))
    os.mkdir(path)
    file_copy = file
    dicom_preprocess, data = apply_windowing(file)
    dicom_preprocess.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

    dicom_preprocess.save_as(prefix + 'static/temp/' + predict_id + '/dicom_scan.dcm')
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


    json_string = json.dumps(createJson(prediction, id_model))
    with open(prefix + 'static/' + predict_id + '/result.json', 'w') as outfile:
        outfile.write(json_string)
    return prefix + 'static/' + predict_id


def explain_case_simple(case_id, model_com, model_hem, model_isch, model_torch):
    #runs a simple explanation on all scans in a given case

    dicom_names = load_dicoms(case_id)
    static_path = "static/temp/"
    print(dicom_names)
    for name in dicom_names:
        file = (static_path+name+'.dcm')
        print(file)
        print(name)

        explain_AI_Simple(file, model_com, model_hem, model_isch, model_torch, 'separable_conv2d_2', name, case_id)

        store_results(static_path, case_id, 'segmentation', 'Cases/'+case_id+'/results/combined_lime_low/')
        store_results(static_path, case_id, 'dicom_scan', ('Cases/'+case_id+'/scans_preprocessed/'))
    #shutil.rmtree(path = static_path)

def predict_case_simple(case_id, model_com, model_hem, model_isch, model_torch):
    #runs a simple prediction on all scans in a given case

    dicom_names = load_dicoms(case_id)
    static_path = "static/temp/"
    print(dicom_names)
    for name in dicom_names:
        file = (static_path+name+'.dcm')
        print(file)

        predict_AI_single(file, model_com, model_hem, model_isch, model_torch, name, case_id)
        store_results(static_path, case_id, 'dicom_scan', ('Cases/'+case_id+'/scans_preprocessed/'))
        

    shutil.rmtree(path = static_path)

    return None

def predict_AI_single (file, model_com, model_hem, model_isch, model_torch, predict_id, case_id):
    path = (os.path.join(prefix +'static/temp/' + str(predict_id)+'/'))
    os.mkdir(path)
    dicom_preprocess, data = apply_windowing(file)
    dicom_preprocess.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

    dicom_preprocess.save_as(prefix + 'static/temp/'+str(predict_id)+'_dicom_scan.dcm')
    data = np.array(data).astype(np.float32)

    data_torch = torch.from_numpy(np.expand_dims(np.moveaxis(data, -1, 0), axis=0)).to('cpu')
    print(data_torch.shape)
    prediction_torch = model_torch(data_torch)
    m = nn.Softmax()

    prediction_com = model_com.predict(np.expand_dims(data, axis=0))
    prediction_hem = model_hem.predict(np.expand_dims(data, axis=0))    
    prediction_isch = model_isch.predict(np.expand_dims(data, axis=0))

    prediction_torch = prediction_torch.detach().numpy()
    
    #res_com = np.round(prediction_com[0])
    res_hem = np.round(prediction_hem)[0]
    res_isch = np.round(prediction_isch)[0]
    res_com = np.round(prediction_torch)[0]

    if (res_com[0] == 1 and res_hem == 0 and res_isch == 0).all() :
        print_result = "No Stroke Detected"
    elif (res_com[1] == 1 and res_hem == 0 and res_isch == 1).all() :
        print_result = "Ischemic Stroke Detected"
    elif (res_com[2] == 1 and res_hem == 1 and res_isch == 0).all() :
        print_result = "Ischemic Stroke Detected"
    else:
        print_result = "Indication of Stroke"

    x = {
        "result": print_result,
        "prediction_combined": str(prediction_com[0]),
        "prediction_hemorrhage": str(prediction_hem[0]),
        "prediction_ischemic": str(prediction_isch[0])
        }
    
    prediction = {'combined': {'prediction': {'result': print_result, 'predictions': str(prediction_com[0])}},
              'ischemic': {'prediction': {'result': print_result, 'predictions': str(prediction_isch[0])}},
              'hemorrhage': {'prediction': {'result': print_result, 'predictions': str(prediction_hem[0])}}}
    
    print(prediction)
    meta_info = { 'filename': 'test123', 'key': '374784t2764'}


    save_prediction(case_id, predict_id, prediction, meta_info)

    json_string = json.dumps(x)
    with open(prefix + 'static/temp/' + predict_id + '/result.json', 'w') as outfile:
        outfile.write(json_string)
    return prefix + 'static/temp/' + predict_id
        

def createJson(prediction, id_model, max_id = None):

    if id_model == 1:
        if np.round(prediction).max() > 0:
            x = {
                "result": "Hemorrhage Stroke Detected",                
                "prediction": str(prediction[0]),
                "layer": str(max_id)
                }
        else:
            x = {
                "result": "No Hemorrhage Stroke Detected",
                "prediction": str(prediction[0]),
                "layer": str(max_id)
                }
    elif id_model == 2:
        if np.round(prediction).max() > 0:
            x = {
                "result": "Ischemic Stroke Detected",
                "prediction": str(prediction[0]),
                "layer": str(max_id)
                }
        else:
            x = {
                "result": "No Ischemic Stroke Detected",
                "prediction": str(prediction[0]),
                "layer": str(max_id)
                }
    else:
        if np.argmax(prediction) == 0:
            x = {
                "result": "No Stroke Detected",
                "prediction": str(prediction[0]),
                "layer": str(max_id)
                }
        elif np.argmax(prediction) == 1:
            x = {
                "result": "Hemorrhage Stroke Detected",
                "prediction": str(prediction[0]),
                "layer": str(max_id)
                }
        else:
            x = {
                "result": "Ischemic Stroke Detected",
                "prediction": str(prediction[0]),
                "layer": str(max_id)
                }
            
    return x

