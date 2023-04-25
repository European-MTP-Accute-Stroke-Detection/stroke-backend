import firebase_admin
from firebase_admin import credentials, storage
from firebase_admin import firestore
import os


firebase_key = 'ServiceAccountKey/brainwatch-14583-firebase-adminsdk-67n85-bd47564c3e.json'
cred = credentials.Certificate(firebase_key)
app_firebase = firebase_admin.initialize_app(cred,{'storageBucket': 'brainwatch-14583.appspot.com'}) # connecting to firebase
db = firestore.client()

def store_results(path_to_save, id_patient):
    bucket = storage.bucket() # storage bucket
    remote_directory_path = 'Dicom/'+id_patient

    for root, dirs, files in os.walk(path_to_save):
        for file in files:
        # Construct the path to the local file and the remote path in the bucket
            local_file_path = os.path.join(root, file)
            remote_file_path = os.path.join(remote_directory_path, local_file_path[len(path_to_save)+1:])

            # Upload the file to Firebase Storage
            blob = bucket.blob(remote_file_path)
            blob.upload_from_filename(local_file_path)

def save_tabular_data_patient(X, id):
    db.collection(u'tabular_patients').document(id).set(X)
    print("data saved")