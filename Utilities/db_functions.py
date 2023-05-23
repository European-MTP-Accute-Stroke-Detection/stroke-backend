import firebase_admin
from firebase_admin import credentials, storage
from firebase_admin import firestore
import os
import json


firebase_key = 'ServiceAccountKey/brainwatch-14583-firebase-adminsdk-67n85-bd47564c3e.json'
cred = credentials.Certificate(firebase_key)
app_firebase = firebase_admin.initialize_app(cred,{'storageBucket': 'brainwatch-14583.appspot.com'}) # connecting to firebase
db = firestore.client()

def store_results(path_to_save, id_patient, name, remote_directory_path):
    bucket = storage.bucket() # storage bucket
    #remote_directory_path = 'Cases/'+id_patient+'/results/combined_lime_low/'

    for root, dirs, files in os.walk(path_to_save):
        for file in files:
        # Construct the path to the local file and the remote path in the bucket
            local_file_path = os.path.join(root, file)
            remote_file_path = os.path.join(remote_directory_path, local_file_path[len(path_to_save)+1:])

            if "json" in file:
                print(local_file_path)
                X = json.load(open(local_file_path))
                db.collection(u'model-results').document(id_patient).set(X)

            # Upload the file to Firebase Storage
            if name in file:
                blob = bucket.blob(remote_file_path)
                blob.upload_from_filename(local_file_path)

def save_tabular_data_patient(X, id):

    db.collection(u'tabular_patients').document(id).set(X)
    print("data saved")


def save_prediction(case_id, scan_id, prediction, meta_info):
    # saves under the given case_id the prediction of the scan_id by creating a new document with its own id

    # prediction_format:
    # model: {stroke_hem, stroke_isch, no_stroke, uncertainty_score}
    #
    # meta_info = {filename, key}

    case_ref = db.collection(u'cases').document(case_id).collection(u'scans').document(scan_id)
    if case_ref.get().exists:
        prediction_dict = case_ref.get().to_dict()

        if 'combined' in prediction:
            prediction_dict['results_combined'] = prediction['combined']
        if 'ischemic' in prediction:
            prediction_dict['results_ischemic'] = prediction['ischemic']
        if 'hemorrhage' in prediction:
            prediction_dict['results_hemorrhage'] = prediction['hemorrhage']

    else:

        prediction_dict = {'filename' : meta_info['filename'], 'key' : meta_info['key'], 
                        'results_combined' : prediction['combined'], 
                        'results_ischemic' : prediction['ischemic'], 
                        'results_hemorrhage' : prediction['hemorrhage']}
        
    case_ref.set(prediction_dict)

def read_cases_scan(case_id, scan_id):
    case_ref = db.collection(u'cases').document(case_id).collection(u'scans').document(scan_id)

    doc = case_ref.get()
    return doc.to_dict()


def load_dicoms(case_id): 
    #returns the files with their names

    dicom_scans, dicom_names = [], []

    bucket = storage.bucket() # storage bucket
    remote_directory_path = 'Cases/'+case_id+'/scans/'
    blobs = bucket.list_blobs(prefix=remote_directory_path)

    for local_blob in blobs:
        # Get the file name and URL
        if len(local_blob.name.replace(remote_directory_path, "")) > 0:
            filename = local_blob.name.replace(remote_directory_path, "")
            dicom_names.append(filename.replace('.dcm', ''))
            if not os.path.exists("static/temp/"):
                os.makedirs("static/temp/")
            dicom_scans.append(local_blob.download_to_filename("static/temp/"+filename))

    return dicom_names