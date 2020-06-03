import os
import cv2

pathDB = 'resources/learn/'
imageList = os.listdir(pathDB)
result = list()
for imageName in imageList:
    # расстягиваем в длину
    originalImg = cv2.imread(pathDB + imageName, cv2.IMREAD_COLOR)
    imgH = originalImg.copy()
    imgH = cv2.resize(imgH, (imgH.shape[1], int(imgH.shape[0] * 1.1)), interpolation = cv2.INTER_AREA)
    cv2.imwrite('resources/learn_new/' + imageName[:6] + '_1' + '.tiff', imgH)

    # в ширину
    imgW = originalImg.copy()
    imgW = cv2.resize(imgW, (int(imgW.shape[1] * 1.1), imgW.shape[0]), interpolation=cv2.INTER_AREA)
    cv2.imwrite('resources/learn_new/' + imageName[:6] + '_2' + '.tiff', imgW)

    # общее расширение в длину и ширину но в разных пропорциях
    imgAll = originalImg.copy()
    imgAll = cv2.resize(imgAll, (int(imgAll.shape[1] * 1.08), int(imgAll.shape[0] * 1.12)), interpolation=cv2.INTER_AREA)
    cv2.imwrite('resources/learn_new/' + imageName[:6] + '_3' + '.tiff', imgAll)