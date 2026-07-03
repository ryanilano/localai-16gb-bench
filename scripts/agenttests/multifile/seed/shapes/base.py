from abc import ABC, abstractmethod


class Shape(ABC):
    """Base class for all shapes. Subclasses must implement area()."""

    @abstractmethod
    def area(self) -> float:
        raise NotImplementedError

    def describe(self) -> str:
        return f"{type(self).__name__} with area {self.area():.4f}"
