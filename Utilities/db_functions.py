import firebase_admin
from firebase_admin import credentials, storage
from firebase_admin import firestore
import os
import json
import numpy as np


# IMPORTANT: Some changes may be necessary before using the following code snippet.

# Provide the correct path to your Firebase Service Account Key JSON file.
firebase_key = 'ServiceAccountKey/brainwatch-14583-firebase-adminsdk-67n85-57a14dcb73.json'
cred = credentials.Certificate(firebase_key)

# Initialize the Firebase Admin SDK with the provided credentials and specify the storage bucket.
app_firebase = firebase_admin.initialize_app(cred, {'storageBucket': 'brainwatch-14583.appspot.com'})

# Connect to the Firestore database using the initialized Firebase Admin SDK.
db = firestore.client()


def store_results(path_to_save, id_patient, name, remote_directory_path):
    """
    Store results in Firebase Storage and Firestore.

    This function takes the local path where the results are saved, and uploads
    the files to Firebase Storage. Additionally, it extracts data from JSON files
    and stores it in Firestore under the collection 'model-results', associated
    with the given 'id_patient'.

    Parameters:
        path_to_save (str): The local directory path where the results are saved.
        id_patient (str): The unique identifier of the patient, used as a document ID in Firestore.
        name (str): The name used to filter specific files for uploading to Firebase Storage.
        remote_directory_path (str): The remote directory path in the Firebase Storage bucket
                                     where the files will be stored.

    """
    bucket = storage.bucket()  # Get the Firebase Storage bucket

    # Traverse through the local directory and its subdirectories
    for root, dirs, files in os.walk(path_to_save):
        for file in files:
            # Construct the path to the local file and the remote path in the bucket
            local_file_path = os.path.join(root, file)
            remote_file_path = os.path.join(remote_directory_path, local_file_path[len(path_to_save) + 1:])

            # If the file is in JSON format, store the data in Firestore
            if "json" in file:
                X = json.load(open(local_file_path))
                db.collection(u'model-results').document(id_patient).set(X)

            # Upload the file to Firebase Storage if it contains the specified 'name'
            if name in file:
                blob = bucket.blob(remote_file_path)
                blob.upload_from_filename(local_file_path)

def save_tabular_data_patient(X, id):
    """
    Save Tabular Data for a Patient in Firestore.

    This function takes tabular data 'X' and saves it in Firestore under the collection 'tabular_patients'
    with the specified 'id' as the document identifier.

    Parameters:
        X (dict): A dictionary representing the tabular data for the patient.
                  The keys represent column names, and the values represent row data.
        id (str): The unique identifier of the patient, used as the document ID in Firestore.

    """
    db.collection(u'tabular_patients').document(id).set(X)
    print("Data saved successfully.")

def save_prediction(case_id, scan_id, prediction, meta_info):
    """
    Save Prediction Results in Firestore.

    This function saves the prediction results of a given 'scan_id' under the 'case_id' in Firestore.
    If the document for the given 'case_id' and 'scan_id' exists, the prediction results are updated.
    If not, a new document with its own ID is created.

    Parameters:
        case_id (str): The unique identifier of the case where the scan belongs.
        scan_id (str): The unique identifier of the scan for which the prediction is made.
        prediction (dict): A dictionary containing the prediction results.
                           The keys should be 'combined', 'ischemic', 'hemorrhage', and 'uncertainty_score'.
        meta_info (dict): A dictionary containing meta-information about the prediction.
                          It should include 'filename' and 'key'.

    Example Data:
        prediction_data = {
            'combined': { 'model': 'stroke_isch', 'uncertainty_score': 0.75, 'other_data': ... },
            'ischemic': { 'model': 'stroke_isch', 'uncertainty_score': 0.8, 'other_data': ... },
            'hemorrhage': { 'model': 'stroke_hem', 'uncertainty_score': 0.6, 'other_data': ... },
            'uncertainty_score': 0.85
        }

        meta_info = {
            'filename': 'scan001.dcm',
            'key': 'unique_key_001'
        }

    """
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
        prediction_dict = {
            'filename': meta_info['filename'],
            'key': meta_info['key'],
            'results_combined': prediction['combined'],
            'results_ischemic': prediction['ischemic'],
            'results_hemorrhage': prediction['hemorrhage']
        }

    case_ref.set(prediction_dict)


def read_cases_scan(case_id, scan_id):
    """
    Read Scan Information for a Case from Firestore.

    This function retrieves the scan information associated with a given 'scan_id' under the 'case_id'
    from Firestore and returns it as a Python dictionary.

    Parameters:
        case_id (str): The unique identifier of the case where the scan belongs.
        scan_id (str): The unique identifier of the scan for which information is to be retrieved.

    Returns:
        dict: A dictionary containing the scan information retrieved from Firestore.

    """
    case_ref = db.collection(u'cases').document(case_id).collection(u'scans').document(scan_id)
    doc = case_ref.get()

    # If the document exists, return its data as a dictionary
    if doc.exists:
        return doc.to_dict()
    else:
        print(f"Scan with ID '{scan_id}' not found under Case ID '{case_id}'.")
        return {}

def load_dicoms(case_id):
    """
    Load DICOM Files from Firebase Storage.

    This function retrieves DICOM files associated with a given 'case_id' from Firebase Storage
    and saves them to a temporary local directory. It returns a list of DICOM file names.

    Parameters:
        case_id (str): The unique identifier of the case for which DICOM files are to be loaded.

    Returns:
        list: A list containing DICOM file names.

    """
    dicom_scans, dicom_names = [], []

    bucket = storage.bucket()  # Get the Firebase Storage bucket
    remote_directory_path = 'Cases/' + case_id + '/scans/'
    blobs = bucket.list_blobs(prefix=remote_directory_path)

    for local_blob in blobs:
        # Get the file name and URL
        if len(local_blob.name.replace(remote_directory_path, "")) > 0:
            filename = local_blob.name.replace(remote_directory_path, "")
            dicom_names.append(filename.replace('.dcm', ''))
            if not os.path.exists("static/temp/"):
                os.makedirs("static/temp/")
            dicom_scans.append(local_blob.download_to_filename("static/temp/" + filename))

    return dicom_names


import numpy as np

def createJson(prediction, id_model, max_id=None):
    """
    Create JSON Result Object for a Stroke Prediction.

    This function takes the prediction results, the 'id_model', and an optional 'max_id' and creates
    a JSON object with relevant information based on the model type and prediction values.

    Parameters:
        prediction (numpy.ndarray): The prediction results generated by the model.
        id_model (int): The identifier for the type of model used (1, 2, or any other integer). The number indicates the ML model.
        max_id (int, optional): The maximum value index from the prediction (default: None).

    Returns:
        dict: A JSON dictionary containing the prediction result details.

    """

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
