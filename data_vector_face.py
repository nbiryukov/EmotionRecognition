import os
import numpy as np
import pandas as pd
from PreprocessingImage import PreprocessingImage

import dlib
import cv2
from imutils import face_utils


databasePath = 'resources/learn/'
preprocessing = PreprocessingImage(databasePath)
imageList = os.listdir(databasePath)
result = list()
for image in imageList:
    data, emotion = preprocessing.preprocessing(image)
    data.append(emotion)
    result.append(data)
df = pd.DataFrame(np.array(result))
df.to_csv('data_for_learn.csv', sep=';', index=False, header=True)



def viewImage(img, name_of_window):
    cv2.namedWindow(name_of_window, cv2.WINDOW_NORMAL)
    cv2.imshow(name_of_window, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

face_dir = 'resources/db/KA.HA1.tiff'
landmarks_predictor_model = 'resources/shape_predictor_68_face_landmarks.dat'
face_img = cv2.imread(face_dir, cv2.IMREAD_COLOR)
# face_img = cv2.resize(face_img, (int(face_img.shape[1] * 1.5), int(face_img.shape[0] * 1.5)), interpolation = cv2.INTER_AREA)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(landmarks_predictor_model)

rectangles = detector(face_img, 0)
points = predictor(face_img, rectangles[0])
points = face_utils.shape_to_np(points)

face_img_tmp = face_img.copy()
for x, y in points:
    cv2.circle(face_img_tmp, (x, y), 1, (0, 0, 255), -1)
central_point = points[30]
for i in points[:30]:
    cv2.line(face_img_tmp, (i[0], i[1]), (central_point[0], central_point[1]), (0, 255, 0), 1)
for i in points[31:]:
    cv2.line(face_img_tmp, (i[0], i[1]), (central_point[0], central_point[1]), (0, 255, 0), 1)

viewImage(face_img_tmp, "исходное лицо")
