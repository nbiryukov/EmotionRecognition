
class NormalizeVectors:

    def __init__(self, vectors=[], height=1, points=[]):
        self.vectors = vectors
        self.height = height
        self.points = points


    def getNormalizeVectorsLength(self):
        normalizeVectors = []
        for v in self.vectors:
            normalizeVectors.append(self.normalizeVectorLengthByHeight(v))

        return normalizeVectors


    def normalizeVectorLengthByHeight(self, vectorLen):
        return round(vectorLen / self.height, 10)
