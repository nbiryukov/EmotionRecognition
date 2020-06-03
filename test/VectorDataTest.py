from VectorData import VectorData
import unittest


class VectorDataTest(unittest.TestCase):

    def setUp(self):
        self.vectorData = VectorData()

    def testIntReturn(self):
        point = (2, 1)
        pointOther = (5, 5)
        self.assertEqual(self.vectorData.getVectorLength(point, pointOther), 5)

    def testIntReturnReverse(self):
        point = (2, 1)
        pointOther = (5, 5)
        self.assertEqual(self.vectorData.getVectorLength(pointOther, point), 5)

    def testDoubleResult(self):
        point = (2, 1)
        pointOther = (10, 4)
        self.assertAlmostEqual(self.vectorData.getVectorLength(pointOther, point), 8.544, delta=0.0001)

    def testZero(self):
        point = (10, 4)
        pointOther = (10, 4)
        self.assertEqual(self.vectorData.getVectorLength(pointOther, point), 0)


if __name__ == "__main__":
    unittest.main()
