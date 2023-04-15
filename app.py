import firebase_admin
import tempfile
import os
import face_recognition
import cv2
import datetime
from firebase_admin import credentials, db
from google.cloud import storage
from flask import Flask, request, redirect, render_template, jsonify, Response
import numpy as np
import io
import base64
from geopy.geocoders import Nominatim

import geocoder

g = geocoder.ip('me')
def locationa():
    g = geocoder.ip('me')
    if g.latlng:
        lat, lng = g.latlng
        loca= f'{lat}+{lng}'
        return loca
        
    else:
        return 'no location'
    




# Initialize Firebase App
cred = credentials.Certificate("accountKey.json")
firebase_admin.initialize_app(cred, {
    "storageBucket": "alpha-74d30.appspot.com",
    "databaseURL":"https://alpha-74d30-default-rtdb.asia-southeast1.firebasedatabase.app/"
})

app = Flask(__name__)

# Define the database schema
root = db.reference()
individuals_ref = root.child("individuals")
done_ref = root.child("done")


# Initialize Google Cloud Storage client
client = storage.Client.from_service_account_json('accountKey.json')
bucket = client.get_bucket('alpha-74d30.appspot.com')

def gen():
    IMAGE_FILES = []
    images = {}
    filename = []

    # Create a temporary directory to store the downloaded images
    with tempfile.TemporaryDirectory() as tmpdir:
        # Download all images from the Google Cloud Storage bucket to the temporary directory
        for blob in bucket.list_blobs():
            file_extension = blob.name.split('.')[-1]
            if file_extension.lower() in ['jpg', 'jpeg', 'png', 'gif', 'bmp', 'webp']:
                filename = os.path.join(tmpdir, os.path.basename(blob.name))
                blob.download_to_filename(filename)
                image_data = open(filename, 'rb').read()
                images[blob.name] = image_data


        def encoding_img(images, individuals_ref):
            encodeList = []
            for img in images.values():
                # Convert image bytes to numpy array
                img = cv2.imdecode(np.frombuffer(img, np.uint8), cv2.IMREAD_COLOR)
                if img is not None:  # Check if the image is properly loaded
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = cv2.resize(img, (0, 0), None, 0.25, 0.25)
                    face_encodings = face_recognition.face_encodings(img)
                    if face_encodings:
                        encode = face_encodings[0]
                        encodeList.append(encode)
            return encodeList

        encodeListknown = encoding_img(images, individuals_ref)


    cap = cv2.VideoCapture(0)
    while True:
        success, img = cap.read()
        imgc = cv2.resize(img, (0, 0), None, 0.25, 0.25)
        # converting image to RGB from BGR
        imgc = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        fasescurrent = face_recognition.face_locations(imgc)
        encode_fasescurrent = face_recognition.face_encodings(imgc, fasescurrent)

        # faceloc- one by one it grab one face location from fasescurrent
        # than encodeFace grab encoding from encode_fasescurrent
        # we want them all in same loop so we are using zip
        name = "Unknown"  # add this line to declare a default value for the name variable
        individual_refa = None

        for encodeFace, faceloc in zip(encode_fasescurrent, fasescurrent):
            matches_face = face_recognition.compare_faces(encodeListknown, encodeFace)
            face_distence = face_recognition.face_distance(encodeListknown, encodeFace)
            # print(face_distence)
            # finding minimum distence index that will return best match
            matchindex = np.argmin(face_distence)
            time_now = datetime.datetime.now()

            if matches_face[matchindex]:
                filename = list(images.keys())[matchindex]
                name = os.path.splitext(os.path.basename(filename))[0].upper()

                y1, x2, y2, x1 = faceloc
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (255, 0, 0), 2, cv2.FILLED)
                cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

                location = locationa()  # replace with the actual location name

                    # If the individual's name is known, add their details to the "done" database
                individuals_refa = done_ref.child(name)
                individuals = individuals_refa.get()
                
                
                found_existing_individual = False

            # Get the latest record for the individual and extract the count from the key
               
                count = 0
                if individuals is not None:
                    count = individuals.get('count')
                
                latest_record = None
                if individuals is not None:
                    for key, record in individuals.items():
                        if isinstance(record, int):
                                continue
                        record_time = datetime.datetime.strptime(record.get('time'), '%Y-%m-%d %H:%M:%S.%f')
                        if (time_now - record_time).total_seconds() < 3600 and record.get('location') == location:
                            print((time_now - record_time).total_seconds())
                            latest_record = record
                            break

                
                if latest_record is not None:
                    found_existing_individual = True

                if not found_existing_individual and name != "Unknown":
                    individual_refa = done_ref.child(name)
                    individuals = individual_refa.get()

                    if individuals is None:
                        count = 1
                        individual_refa.update({
                            'count': count
                        })
                    else:
                        if isinstance(individuals, list):
                            count = 1
                        else:
                            count = individuals.get('count', 0) + 1
                            individual_refa.update({
                                'count': count
                            })

                    record_ref = individual_refa.child(str(count))
                    record_ref.set({
                        'time': str(time_now),
                        'location': location
                    })


            # cv2.imshow("campare", img)
        # cv2.waitKey(0)
        frame = cv2.imencode('.jpg', img)[1].tobytes()
        yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        key = cv2.waitKey(20)
        if key == 27:
            break


@app.route('/video_feed')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(gen(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    app.run(debug=True,port=5001)
