import cv2
import os
import csv
import pandas as pd

""""
import urllib.request as urlreq
import matplotlib.pyplot as plt
from pylab import rcParams

# save facial landmark detection model's url in LBFmodel_url variable
LBFmodel_url = "https://github.com/kurnianggoro/GSOC2017/raw/master/data/lbfmodel.yaml"

# save facial landmark detection model's name as LBFmodel
LBFmodel = "LFBmodel.yaml"

# check if file is in working directory
if (LBFmodel in os.listdir(os.curdir)):
    print("File exists")
else:
    # download picture from url and save locally as lbfmodel.yaml
    urlreq.urlretrieve(LBFmodel_url, LBFmodel)
    print("File downloaded")
# save face detection algorithm's url in haarcascade_url variable
haarcascade_url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_alt2.xml"

# save face detection algorithm's name as haarcascade
haarcascade = "haarcascade_frontalface_alt2.xml"

# chech if file is in working directory
if (haarcascade in os.listdir(os.curdir)):
    print("File exists")
else:
    # download file from url and save locally as haarcascade_frontalface_alt2.xml
    urlreq.urlretrieve(haarcascade_url, haarcascade)
    print("File downloaded")

"""

haarcascade = "haarcascade_frontalface_alt2.xml"
LBFmodel = "LFBmodel.yaml"

detector = cv2.CascadeClassifier(haarcascade)
landmark_detector = cv2.face.createFacemarkLBF()
landmark_detector.loadModel(LBFmodel)

def write_to_csv(csv_file, time_counter, landmarks):
    with open(csv_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        # Zaman bilgisi için 'time' sütunu ekle
        row = ['time']
        for i in range(68):
            row += ['x{}'.format(i+1), 'y{}'.format(i+1)]
        writer.writerow(row)

        for time, landmark_points in zip(time_counter, landmarks):
            row = [time]
            for landmark in landmark_points:
                for x, y in landmark:
                    row.extend([x, y])
            writer.writerow(row)

def process_video(video_path, detector, landmark_detector, output_csv):
    cap = cv2.VideoCapture(video_path)
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    frame_period = int(1000 / frame_rate) if frame_rate != 0 else 1000 
    current_frame = 0
    time_counter = []
    landmarks_data = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        current_frame += 1
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = detector.detectMultiScale(frame_rgb)
        if len(faces) > 0:
            _, landmarks = landmark_detector.fit(frame_rgb, faces)
            if len(landmarks) > 0:
                time_counter.append(current_frame * frame_period / 1000) 
                landmarks_data.append(landmarks[0])  

    if len(landmarks_data) > 0:
        if os.path.exists(output_csv):
            os.remove(output_csv)
        write_to_csv(output_csv, time_counter, landmarks_data)

    cap.release()
    cv2.destroyAllWindows()


video_path = "video.mp4"
output_csv = "landmark_data.csv"

process_video(video_path, detector, landmark_detector, output_csv)

path="/Users/canavar/Documents/datasets/BagOfLies/"
annotations_path = f"{path}annotations.csv"
annotations_df = pd.read_csv(annotations_path)
selected_columns = annotations_df[["run", "usernum"]]
# Her bir satırı işle
for index, row in selected_columns.iterrows():
    run_folder = f"{path}/Finalised/User_{row['usernum']}/run_{row['run']}"
    video_path = os.path.join(run_folder, "video.mp4")
    output_csv = os.path.join(run_folder, "face.csv")
    process_video(video_path, detector, landmark_detector, output_csv)
    print("succes")
    print(run_folder)
