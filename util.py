import numpy as np

emotionString = ['Neutral', 'Happiness', 'Anger', 'Disgust', 'Fear', 'Sadness', 'Surprise']
emotionCode = ['NE', 'HA', 'AN', 'DI', 'FE', 'SA', 'SU']
emotionName = {
    'NE': 'Neutral',  # нейтральное
    'HA': 'Happiness',  # стастье
    'AN': 'Anger',  # злость
    'DI': 'Disgust',  # отвращение
    'FE': 'Fear',  # страх
    'SA': 'Sadness',  # печаль
    'SU': 'Surprise'  # удивление
}


def calculateStatistic(dataEmotions):
    data = dataEmotions.copy()
    for emotion in dataEmotions:
        arr = data[emotion]
        length = len(data[emotion])
        data[emotion] = [sum(x) for x in zip(*arr)]
        data[emotion] = np.array(data[emotion]) / length
        data[emotion] = np.around(data[emotion], decimals=4)
    for emotion in emotionCode:
        data[emotionName[emotion]] = data.pop(emotion)

    return data

