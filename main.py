import os
import uuid
from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import os
import seaborn as sns
import tensorflow as tf
import joblib
from api_AI import *

sns.set_theme(style="whitegrid", palette="viridis")
    
model_hem = tf.keras.models.load_model('hemorrhage_clf', compile=False)
model_ischemic = tf.keras.models.load_model('New_models/Ischemic', compile=False)
model_combined = tf.keras.models.load_model('New_models/Combined', compile=False)

pipeline = joblib.load('stroke-prediction/model.joblib') 

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
    

@app.route('/ischemic/explain/<xai_id>/', methods=['POST'])
def ischemicExplain(xai_id):
    if 'file' not in request.files:
        return jsonify({'error':'media not provided'}), 400
    file = request.files['file']
        
    id = str(uuid.uuid1())
    print('ischemic')
        
    #explain_AI(file, 2, model_ischemic, id, "lime")

    xai_id_method, xai_id_complexity = get_XAI_info(xai_id)

    if xai_id_method == "lime":
        explain_AI(file, 'lime', 2, xai_id_complexity, model_ischemic, 'conv2d_3',id)
    else:
        explain_AI(file, 'grad-cam', 2, xai_id_complexity, model_ischemic, 'conv2d_3',id)
    
    
    if file.filename == '':
        return jsonify({'error': 'no file uploaded'}), 400
    elif file and allowed_file(file.filename):
         return jsonify({"predictionId": id})
    
@app.route('/combined/explain/<xai_id>/', methods=['POST'])
def combinedExplain(xai_id):
    if 'file' not in request.files:
        return jsonify({'error':'media not provided'}), 400
    file = request.files['file']
        
    id = str(uuid.uuid1())
    print(xai_id)

    xai_id_method, xai_id_complexity = get_XAI_info(xai_id)

    if xai_id_method == "lime":
        explain_AI(file, 'lime', 0, xai_id_complexity, model_combined, 'separable_conv2d_2',id)
    else:
        explain_AI(file, 'grad-cam', 0, xai_id_complexity, model_combined, 'separable_conv2d_2',id)
    
    if file.filename == '':
        return jsonify({'error': 'no file uploaded'}), 400
    elif file and allowed_file(file.filename):
         return jsonify({"predictionId": id})

@app.route('/hemorrhage/explain/<xai_id>/', methods=['POST'])
def hemorrhageExplain(xai_id):
    if 'file' not in request.files:
        return jsonify({'error':'media not provided'}), 400
    file = request.files['file']
        
    id = str(uuid.uuid1())
    print('hemmo')

    xai_id_method, xai_id_complexity = get_XAI_info(xai_id)

    if xai_id_method == "lime":
        explain_AI(file, 'lime', 1, xai_id_complexity, model_hem, 'conv2d_3',id)
    else:
        explain_AI(file, 'grad-cam', 1, xai_id_complexity, model_hem, 'conv2d_3',id)
    
        
    #explain_AI(file, 2, model_ischemic, id, "lime")
    
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
def tabularPredict():
    
    X = pd.DataFrame(request.json)
      
    result: list = pipeline.predict(X)
    
    return jsonify({"result": int(result[0])})

if __name__ == '__main__':
    # This is used when running locally only. When deploying to Google App
    # Engine, a webserver process such as Gunicorn will serve the app. This
    # can be configured by adding an `entrypoint` to app.yaml.
    # Flask's development server will automatically serve static files in
    # the "static" directory. See:
    # http://flask.pocoo.org/docs/1.0/quickstart/#static-files. Once deployed,
    # App Engine itself will serve those files as configured in app.yaml.
    app.run(host='127.0.0.1', port=8080, debug=True)