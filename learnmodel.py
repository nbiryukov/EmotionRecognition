import numpy as np
import pandas as pd
from keras import models
from keras import layers
from keras.utils import np_utils
from keras.utils.vis_utils import plot_model

emotionString = ['NE', 'HA', 'AN', 'DI', 'FE', 'SA', 'SU']

inputData = pd.read_csv('data_for_learn.csv', sep=';').to_numpy()
features = inputData[:, 0:67]
emotions = inputData[:, 67]

features = features.astype('float64')
scale = lambda e: emotionString.index(e)
emotions = np.array([scale(e) for e in emotions], dtype='uint8')

numberClasses = 7
emotionsCategorical = np_utils.to_categorical(emotions, numberClasses)

hiddenSize = 1024
inputShapeSize = 67
inputLayer = layers.Input(shape=(inputShapeSize,))
hidden1 = layers.Dense(hiddenSize, activation='relu')(inputLayer)
hidden2= layers.Dense(hiddenSize, activation='relu')(hidden1)
hidden3= layers.Dense(hiddenSize, activation='relu')(hidden2)
outLayer = layers.Dense(numberClasses, activation='softmax')(hidden3)
model = models.Model(input=inputLayer, output=outLayer)

model = models.Sequential()
model.add(layers.Dense(hiddenSize, activation='relu', input_shape=(inputShapeSize,))) # hidden1
model.add(layers.Dense(hiddenSize, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(hiddenSize, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(numberClasses, activation='softmax')) # out layer

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

batchSize = 756 # берем общее количество изображений из базы
numEpochs = 350
history = model.fit(features, emotionsCategorical,
          batch_size=batchSize, nb_epoch=numEpochs,
          verbose=1)

model.save('network/model.h5')
histDf = pd.DataFrame(history.history)
with open('network/history.json', mode='w') as f:
    histDf.to_json(f)
plot_model(model, to_file='network/model.png')
print(history.history)