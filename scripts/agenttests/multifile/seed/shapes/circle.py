import math

from .base import Shape


class Circle(Shape):
    def __init__(self, radius: float):
        self.radius = radius

    def area(self) -> float:
        # BUG: area of a circle is pi * r**2, not pi * r
        return math.pi * self.radius
