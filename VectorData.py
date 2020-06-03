import math


class VectorData:

    def __init__(self):
        pass

    def getVectorLength(self, point, pointOther):
        x1 = point[0]
        y1 = point[1]
        x2 = pointOther[0]
        y2 = pointOther[1]
        return math.sqrt(pow(x2 - x1, 2) + pow(y2 - y1, 2))
