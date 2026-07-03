import math
import unittest

from shapes import Circle, Rectangle, Triangle


class TestShapes(unittest.TestCase):
    def test_rectangle(self):
        self.assertAlmostEqual(Rectangle(3, 4).area(), 12.0)

    def test_triangle(self):
        self.assertAlmostEqual(Triangle(3, 4).area(), 6.0)

    def test_circle_unit(self):
        self.assertAlmostEqual(Circle(1).area(), math.pi)

    def test_circle_r2(self):
        self.assertAlmostEqual(Circle(2).area(), math.pi * 4)


if __name__ == "__main__":
    unittest.main()
