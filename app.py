import os
import json
import uuid
from flask import Flask, request, jsonify, flash, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename
from flask_cors import CORS, cross_origin
import pylibjpeg
import scipy.ndimage as ndi
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
from skimage import morphology
from joblib import load

from keras.preprocessing import image
from numpy import asarray
import matplotlib.cm as cm
from PIL import Image as im

from lime import lime_image
from lime.wrappers.scikit_image import SegmentationAlgorithm
from skimage.segmentation import mark_boundaries, slic
from api_new import *

sns.set_theme(style="whitegrid", palette="viridis")
    
model_hem = tf.keras.models.load_model('hemorrhage_clf', compile=False)
model_ischemic = tf.keras.models.load_model('New_models/Ischemic', compile=False)
model_combined = tf.keras.models.load_model('New_models/Combined', compile=False)

pipeline = load('filename.joblib') 

ALLOWED_EXTENSIONS = {'dcm'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.',1)[1].lower() in ALLOWED_EXTENSIONS

# ---------------- Flask API -----------------------

app = Flask(__name__, static_folder=os.path.join('/static'))
# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
cors = CORS(app, resources={r"*": {"origins": "*"}})
app.config['CORS_HEADERS'] = 'Content-Type'

@app.route('/hemorrhage/predict', methods=['POST'])
def hemorrhagePredict():
    if 'file' not in request.files:
        return jsonify({'error':'media not provided'}), 400
    file = request.files['file']
    
    id = str(uuid.uuid1())
    
    execute_AI(file, 1, model_hem, "conv2d_3", id)
    
    if file.filename == '':
        response = jsonify({'error': 'no file uploaded'}), 400
    elif file and allowed_file(file.filename):
         return jsonify({"predictionId": id})

@app.route('/ischemic/predict', methods=['POST'])
def ischemicPredict():
    if 'file' not in request.files:
        return jsonify({'error':'media not provided'}), 400
    file = request.files['file']
    
    print(file)
    
    id = str(uuid.uuid1())
        
    execute_AI(file, 2, model_ischemic, 'conv2d_3', id)
    
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
    
    execute_AI(file, 3, model_combined, 'separable_conv2d_2',id)
    
    if file.filename == '':
        return jsonify({'error': 'no file uploaded'}), 400
    elif file and allowed_file(file.filename):
        return jsonify({"predictionId": id})
    

@app.route('/tabular/predict', methods=['POST'])
def combinedPredict():
    if 'file' not in request.files:
        return jsonify({'error':'media not provided'}), 400
    file = request.files['file']
    
    id = str(uuid.uuid1())
    
    evaluation = execute_tabular_AI(file, pipeline)
    
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
