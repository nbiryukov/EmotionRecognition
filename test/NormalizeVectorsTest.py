from NormalizeVectors import NormalizeVectors
import unittest

class NormalizeVectorsTest(unittest.TestCase):

    def setUp(self):
        self.normalizeVectors = NormalizeVectors(vectors=[78.42871964, 8.54400374531753], height=125)

    def testNormalizeVectorLengthByHeightBigLen(self):
        len = 78.42871964
        lenExp = 0.6274297571
        self.assertAlmostEqual(self.normalizeVectors.normalizeVectorLengthByHeight(len), lenExp, 0.0000000001)

    def testNormalizeVectorLengthByHeightSmalLen(self):
        len = 8.54400374531753
        lenExp = 0.0683520300
        self.assertAlmostEqual(self.normalizeVectors.normalizeVectorLengthByHeight(len), lenExp, 0.0000000001)

    def testGetNormalizeVectorsLength(self):
        vectorsLenExp = [0.6274297571, 0.0683520300]
        self.assertEqual(self.normalizeVectors.getNormalizeVectorsLength(), vectorsLenExp)