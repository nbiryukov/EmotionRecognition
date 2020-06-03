import dlib
import cv2
from imutils import face_utils
from NormalizeVectors import NormalizeVectors
from VectorData import VectorData

class PreprocessingImage:

    def __init__(self, databasePath):
        self.landmarks_predictor_model = 'resources/shape_predictor_68_face_landmarks.dat'
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(self.landmarks_predictor_model)
        self.databasePath = databasePath
        self.emotionString = ['NE', 'HA', 'AN', 'DI', 'FE', 'SA', 'SU']
        self.emotionCode = [0, 1, 2, 3, 4, 5, 6]


    def preprocessing(self, imageName):
        # чтение изображения
        image = cv2.imread(self.databasePath + imageName, cv2.IMREAD_COLOR)
        emotion = imageName[3:5]

        # детекция лица и 68 точек
        rectangles = self.detector(image, 0)
        points = self.predictor(image, rectangles[0])
        points = face_utils.shape_to_np(points)
        central_point = points[30]

        # отрисовка
        print("длинна нормализации(" + imageName + "): " + str(abs(rectangles[0].bottom() - rectangles[0].top())))
        face_img_tmp = image.copy()
        for x, y in points:
            cv2.circle(face_img_tmp, (x, y), 1, (0, 0, 255), -1)
        for i in points[:30]:
            cv2.line(face_img_tmp, (i[0], i[1]), (central_point[0], central_point[1]), (0, 255, 0), 1)
        for i in points[31:]:
            cv2.line(face_img_tmp, (i[0], i[1]), (central_point[0], central_point[1]), (0, 255, 0), 1)
        cv2.rectangle(face_img_tmp, (rectangles[0].left(), rectangles[0].top()),
                      (rectangles[0].right(), rectangles[0].bottom()),
                      (0, 255, 0), 1)
        # self.viewImage(face_img_tmp, imageName)

        # нахождение отрезков от центральной точки
        vectorData = VectorData()
        vectors = []
        for i in points[:30]:
            vectors.append(vectorData.getVectorLength(central_point, i))
        for i in points[31:]:
            vectors.append(vectorData.getVectorLength(central_point, i))

        # нормализация отрезков
        height = abs(rectangles[0].bottom() - rectangles[0].top())
        normalize = NormalizeVectors(vectors, height, points)
        normalizeVectors = normalize.getNormalizeVectorsLength()

        # вывод нормализованных отрезков и эмоции(в строковом виде)
        return normalizeVectors, emotion


    def viewImage(self, img, name_of_window):
        cv2.namedWindow(name_of_window, cv2.WINDOW_NORMAL)
        cv2.imshow(name_of_window, img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
