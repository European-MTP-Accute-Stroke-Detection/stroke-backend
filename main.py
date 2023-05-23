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
from Utilities.db_functions import *
from Utilities.xai_functions import *
import torch
from catboost import CatBoostClassifier

sns.set_theme(style="whitegrid", palette="viridis")
    
model_hem = tf.keras.models.load_model('hemorrhage_clf', compile=False)
model_ischemic = tf.keras.models.load_model('New_models/Ischemic', compile=False)
model_combined = tf.keras.models.load_model('New_models/Combined', compile=False)
model_torch = torch.load("New_models/torch_test/efficientnet_v2_l.ckpt", map_location='cpu')

pipeline = joblib.load('stroke-prediction/model.joblib') 

with open ("preprocessors_catboost.pickle", "rb") as f:
    preprocessors = pickle.load(f)

with open ("hypertuned-catboost.pickle", "rb") as f:
    tabular_model = pickle.load(f)

with open ("shap_explainer.pickle", "rb") as f:
    shap_explainer = pickle.load(f)

ALLOWED_EXTENSIONS = {'dcm'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.',1)[1].lower() in ALLOWED_EXTENSIONS

# ---------------- Flask API -----------------------

print(os.path.join('/static'))
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
    
    #static_path = predict_AI(file, model_combined, model_hem, model_ischemic, id)


    static_path = execute_AI(file, 1, model_hem, id)

    store_results(static_path, id)
    
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

    #static_path = predict_AI(file, model_combined, model_hem, model_ischemic, id)   
    static_path = execute_AI(file, 2, model_ischemic, id)
    
    store_results(static_path, id)

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
        static_path = explain_AI(file, 'lime', 2, xai_id_complexity, model_ischemic, 'conv2d_3',id)
    else:
        static_path = explain_AI(file, 'grad-cam', 2, xai_id_complexity, model_ischemic, 'conv2d_3',id)
    store_results(static_path, id)
    
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
        static_path = explain_AI(file, 'lime', 0, xai_id_complexity, model_combined, 'separable_conv2d_2',id)
    else:
        static_path = explain_AI(file, 'grad-cam', 0, xai_id_complexity, model_combined, 'separable_conv2d_2',id)
    store_results(static_path, id)

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
        static_path = explain_AI(file, 'lime', 1, xai_id_complexity, model_hem, 'conv2d_3',id)
    else:
        static_path = explain_AI(file, 'grad-cam', 1, xai_id_complexity, model_hem, 'conv2d_3',id)
    store_results(static_path, id)
        
    #explain_AI(file, 2, model_ischemic, id, "lime")
    
    if file.filename == '':
        return jsonify({'error': 'no file uploaded'}), 400
    elif file and allowed_file(file.filename):
         return jsonify({"predictionId": id})
    
@app.route('/explain', methods=['POST'])
def explain():
    if 'file' not in request.files:
        return jsonify({'error':'media not provided'}), 400
    file = request.files['file']
        
    file_name = 'test'
    id = str(uuid.uuid1())
    static_path = explain_AI_Simple(file, model_combined, model_hem, model_ischemic, 'separable_conv2d_2',id, file_name)

    store_results(static_path, id, file_name)
            
    if file.filename == '':
        return jsonify({'error': 'no file uploaded'}), 400
    elif file and allowed_file(file.filename):
         return jsonify({"predictionId": id})
    
@app.route('/torch/explain/<xai_id>/', methods=['POST'])
def torch_explain(xai_id):
    if 'file' not in request.files:
        return jsonify({'error':'media not provided'}), 400
    file = request.files['file']
        
    id = str(uuid.uuid1())
    print('hemmo')

    xai_id_method, xai_id_complexity = get_XAI_info(xai_id)

    if xai_id_method == "lime":
        static_path = explain_AI_torch(file, 'lime', 0, xai_id_complexity, model_torch,id)
    else:
        static_path = explain_AI_torch(file, 'grad-cam', 0, xai_id_complexity, model_hem,id)
    store_results(static_path, id)
        
    #explain_AI(file, 2, model_ischemic, id, "lime")
    
    if file.filename == '':
        return jsonify({'error': 'no file uploaded'}), 400
    elif file and allowed_file(file.filename):
         return jsonify({"predictionId": id})
    

@app.route('/predict_complete/<uid>', methods=['POST'])
def predict_ai_complete(uid):
    print(uid)
        
    predict_case_simple(uid, model_combined, model_hem, model_ischemic, model_torch)

    return jsonify({"Done": 'done'})

@app.route('/explain_complete/<uid>', methods=['POST'])
def explain_ai_complete(uid):
    print(uid)
        
    explain_case_simple(uid, model_combined, model_hem, model_ischemic, model_torch)

    return jsonify({"Done": 'done'})

    
@app.route('/predict', methods=['POST'])
def predict_ai():
    if 'file' not in request.files:
        return jsonify({'error':'media not provided'}), 400
    file = request.files['file']
    
    id = str(uuid.uuid1())
    
    static_path = predict_AI(file, model_combined, model_hem, model_ischemic, id)
    #static_path = execute_AI(file, 3, model_combined, 'separable_conv2d_2',id)
    
    store_results(static_path, id)

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
    
    #static_path = predict_AI(file, model_combined, model_hem, model_ischemic, id)
    static_path = execute_AI(file, 3, model_combined,id)
    
    store_results(static_path, id)

    if file.filename == '':
        return jsonify({'error': 'no file uploaded'}), 400
    elif file and allowed_file(file.filename):
        return jsonify({"predictionId": id})
    

@app.route('/torch/predict', methods=['POST'])
def torch_combined_Predict():
    if 'file' not in request.files:
        return jsonify({'error':'media not provided'}), 400
    file = request.files['file']
    
    id = str(uuid.uuid1())
    
    #static_path = predict_AI(file, model_combined, model_hem, model_ischemic, id)
    static_path = execute_torch_AI(file, 3, model_torch,id)
    
    store_results(static_path, id)

    if file.filename == '':
        return jsonify({'error': 'no file uploaded'}), 400
    elif file and allowed_file(file.filename):
        return jsonify({"predictionId": id})
    

@app.route('/tabular/predict', methods=['POST'])
def tabularPredict():
    
    X = pd.DataFrame(request.json)

    with open ('testX.pickle', "wb") as f:
        pickle.dump(X, f)

    numerical_features = ['age', 'avg_glucose_level', 'bmi']

    for p in preprocessors:
        X[numerical_features] = p.transform(X[numerical_features])

    shap_values = shap_explainer(X)

    df = pd.DataFrame(shap_values.values, columns=X.columns)
    sorted_columns = df.abs().mean().sort_values(ascending=False).index

    result = tabular_model.predict(X)
    
    df_sorted = df[sorted_columns]
    df_sorted['result'] = result[0]
    df_json = df_sorted.to_json(orient='records')


    #return_json = jsonify({"result": int(result[0])})

    return df_json


if __name__ == '__main__':
    # This is used when running locally only. When deploying to Google App
    # Engine, a webserver process such as Gunicorn will serve the app. This
    # can be configured by adding an `entrypoint` to app.yaml.
    # Flask's development server will automatically serve static files in
    # the "static" directory. See:
    # http://flask.pocoo.org/docs/1.0/quickstart/#static-files. Once deployed,
    # App Engine itself will serve those files as configured in app.yaml.
    app.run(host='127.0.0.1', port=8080, debug=True)