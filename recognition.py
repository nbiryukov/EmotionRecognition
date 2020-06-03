import os
import cv2
from keras import models
import numpy as np
import pandas as pd
from PreprocessingImage import PreprocessingImage
import matplotlib.pyplot as plt
import util


databasePath = 'resources/test/'
imageList = os.listdir(databasePath)
preprocessing = PreprocessingImage(databasePath)
features = list()
emotions = list()
images = list()
for imageName in imageList:
    image = cv2.imread(databasePath + imageName, cv2.IMREAD_COLOR)
    images.append(image)
    data, emotion = preprocessing.preprocessing(imageName)
    features.append(data)
    emotions.append(emotion)
features = np.array(features)
emotions = np.array(emotions)
print(features)
print(emotions)

emotionString = ['NE', 'HA', 'AN', 'DI', 'FE', 'SA', 'SU']
statistic = {emotion: [] for emotion in emotionString}
model = models.load_model('network/model.h5')
model.summary()
fig, axes = plt.subplots(1, 2)
for i in range(len(features)):
    axes[0].imshow(images[i])
    axes[0].set_title(util.emotionName[emotions[i]])
    classes = model.predict(np.array([features[i]]))
    classes = classes[0].tolist()
    classes = [round(num, 4) for num in classes]
    classes = [num * 100 for num in classes]
    classes = [round(num, 4) for num in classes]
    axes[1].bar(util.emotionString, classes, color='b')
    axes[1].set_ylabel('Процент эмоции')
    axes[1].set_xlabel('Эмоции')

    statistic[emotions[i]].append(classes)

    print(emotions[i])
    print(classes)
    print("--------------------")
    plt.draw()
    # plt.pause(0.05)
    fig.waitforbuttonpress()
    plt.cla()

statistic = util.calculateStatistic(statistic)
print(statistic)
df = pd.DataFrame.from_dict(statistic, orient='index', columns=util.emotionString)
print(df)
df.to_csv('statistic.csv', sep=';', header=True)