import os
import json
import base64
import uuid
from flask import Flask, request, jsonify, flash, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename

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

from keras.preprocessing import image
from numpy import asarray
import matplotlib.cm as cm
from PIL import Image as im

from lime import lime_image
from lime.wrappers.scikit_image import SegmentationAlgorithm
from skimage.segmentation import mark_boundaries

sns.set_theme(style="whitegrid", palette="viridis")

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
    data = pydicom.read_file(data)
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

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, multi, pred_index=None):
    # First, we create a model that maps the input image to the activations
    # of the last conv layer as well as the output predictions
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )
    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            if multi:
                pred_index = tf.argmax(preds[:, np.argmax(preds)])
            else:
                pred_index = tf.argmax(preds[0])
        # if self.pred_index is None:
        #        class_channel = preds
        #    else:
        #        class_channel = preds[:, self.pred_index]
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

def calculate_heatmap(cam, img):
    cam = cv2.resize(cam, (224, 224))
    cam = np.maximum(cam, 0)
    heatmap = cam / np.max(cam)
    #print(heatmap)
    
    ##Return to BGR [0..255] from the preprocessed image
    img = asarray(img).astype(np.uint8)
    img = img[0, :]
    img -= np.min(img)
    img = np.minimum(img, 255)
#
    cam = cv2.applyColorMap(np.uint8(255*heatmap), cv2.COLORMAP_JET)
    cam = np.float32(cam) + np.float32(img)
    cam = 255 * cam / np.max(cam)
    
    imgplot = plt.imshow(cam)
    plt.show()
    
    return cam


def visualize_heatmap(heatmap, img, path):
    
    #data = im.fromarray(heatmap*255)
    #opencvImage = cv2.cvtColor(np.array(heatmap), cv2.COLOR_RGB2BGR)
    #print(img[0].shape)
    #super_imposed_img = cv2.addWeighted(heatmap, 0.5, img[0], 0.5, 0)
    #
    #heatmap = cv2.resize(heatmap, (224, 224))
    imgplot = plt.imshow(heatmap)
    plt.axis('off')
    plt.savefig(path + '/cam.png')
    
    #plt.show()



def execute_AI(file, id_model, predict_id):
    
    os.mkdir(os.path.join('/static/' + predict_id))
    data = apply_windowing(file, False)
    data = np.array(data).astype(np.float32)
    
    combined = False
    
    if id_model == 1:
        local_model = model_hem
    elif id_model == 2:
        local_model = model_ischemic
    else:
        local_model = model_combined
        combined = True
    
    prediction = local_model.predict(np.expand_dims(data, axis=0))
    
    print(np.round(prediction))
    
    heatmap = make_gradcam_heatmap(np.expand_dims(data, axis=0), local_model, "conv2d_3", combined)
    #cam_heatmap = calculate_heatmap(heatmap, np.expand_dims(data, axis=0))
    visualize_heatmap(heatmap, data, ('/static/' + predict_id))
    
    explainer = lime_image.LimeImageExplainer() 

    segmenter = SegmentationAlgorithm('quickshift', kernel_size=2, max_dist=500, ratio=0.4)
    explanation_1 = explainer.explain_instance(data, 
                                         classifier_fn = local_model, 
                                         top_labels=3, 
                                         hide_color=0, # 0 - gray 
                                         num_samples=1000,
                                         segmentation_fn=segmenter
                                        )

    temp, mask = explanation_1.get_image_and_mask(explanation_1.top_labels[np.argmax(prediction)], 
                                                positive_only=True, 
                                                num_features=5, 
                                                hide_rest=False)

    imgplot = plt.imshow(mark_boundaries(temp / 2 + 0.5, mask))
    plt.axis('off')
    filePath = os.path.join('/static/' + predict_id + '/', "lime.png")
    plt.savefig(filePath)
    print(filePath)
    
    
model_hem = tf.keras.models.load_model('hemorrhage_clf', compile=False)
model_ischemic = tf.keras.models.load_model('ischemic', compile=False)
model_combined = tf.keras.models.load_model('Combined', compile=False)

ALLOWED_EXTENSIONS = {'dcm'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.',1)[1].lower() in ALLOWED_EXTENSIONS

# ---------------- Flask API -----------------------

app = Flask(__name__, static_folder=os.path.join('/static'))
# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/hemorrhage/predict', methods=['POST'])
def hemorrhagePredict():
    if 'file' not in request.files:
        return jsonify({'error':'media not provided'}), 400
    file = request.files['file']
    
    id = str(uuid.uuid1())
    
    execute_AI(file, 1, id)
    
    if file.filename == '':
        return jsonify({'error': 'no file uploaded'}), 400
    elif file and allowed_file(file.filename):
        return jsonify({"predictionId": id})

@app.route('/ischemic/predict', methods=['POST'])
def ischemicPredict():
    if 'file' not in request.files:
        return jsonify({'error':'media not provided'}), 400
    file = request.files['file']
    
    print(file)
    
    id = str(uuid.uuid1())
        
    execute_AI(file, 2, id)
    
    if file.filename == '':
        return jsonify({'error': 'no file uploaded'}), 400
    elif file and allowed_file(file.filename):
        return jsonify({"predictionId": id})

@app.route('/combined/predict', methods=['POST'])
def combinedPredict():
    if 'file' not in request.files:
        return jsonify({'error':'media not provided'}), 400
    file = request.files['file']
    
    id = str(uuid.uuid1())
    
    execute_AI(file, 3, id)
    
    if file.filename == '':
        return jsonify({'error': 'no file uploaded'}), 400
    elif file and allowed_file(file.filename):
        return jsonify({"predictionId": id})

# @app.route('/reports/<path:path>')
# def send_report(metadata): 
#     id = uuid.uuid1()
#     json_str = json.dumps(metadata)
#     return send_from_directory('reports',path)

if __name__ == '__main__':
    app.run(debug=True, port=5000)


