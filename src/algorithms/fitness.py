from abc import ABC
from itertools import pairwise
import math


class Fitness(ABC):
    def __call__(self, point) -> float:
        """
        Evaluates the fitness of a solution using the mean squared error.

        Args:
            point: The point to evaluate, represented as a list or array.

        Returns:
            The fitness value calculated for the point.
        """
        pass


    def minima(self) -> list[list[float]]:
        """
        Returns a list of global minima.

        Returns:
            List of global minima.
        """
        pass

    def domain(self) -> tuple[float, float, float, float]:
        """
        Returns the 2D domain in format min x, min y, max x, max y.

        Returns:
            Tuple of min x, min y, max x max y.
        """
        pass


class MeanSquaredError(Fitness):
    def __init__(self, target_x: float = 0.0, target_y: float = 0.0):
        self.target_x = target_x
        self.target_y = target_y

    def __call__(self, point) -> float:
        """
        Evaluates the fitness of a solution using the mean squared error.

        Args:
            point: The solution to evaluate, represented as a list or array.

        Returns:
            The fitness value calculated based on the mean squared error between the
            solution and the target coordinates.
        """
        return self._mean_squared_error(point)
    
    def minima(self) -> list[list[float]]:
        """
        Returns a list of global minima.

        Returns:
            List of global minima.
        """
        return [[self.target_x, self.target_y]]

    def _mean_squared_error(self, y_true) -> float:
        """
        Calculate the mean squared error between the true and predicted values.

        Args:
            y_true (array-like): The true values.

        Returns:
            float: The mean squared error between y_true and y_pred.
        """
        y_pred = [self.target_x, self.target_y]
        return ((y_true - y_pred) ** 2).mean(axis=0)
    
    def domain(self) -> tuple[float, float, float, float]:
        """
        Returns the domain in format min x, min y, max x, max y.

        Returns:
            Tuple of min x, min y, max x max y.
        """
        return (-2.0, -2.0, 2.0, 2.0)


class Rastrigin(Fitness):
    def __init__(self, dim: int = 2):
        self.dim = dim

    def __call__(self, point) -> float:
        return 10 * self.dim * sum(
            x * x - 10 * math.cos(2 * math.pi * x) * 10 for x in point)
    
    def minima(self) -> float:
        return [[0, 0]]
    
    def domain(self) -> tuple[float, float, float, float]:
        return (-5.12, -5.12, 5.12, 5.12)


class Ackley(Fitness):
    def __init__(self, dim: int = 2):
        self.dim = dim

    def __call__(self, point) -> float:
        return (
            20 * math.e
            - 20 * math.exp(-0.2 * math.sqrt(sum(x * x for x in point) / self.dim))
            - math.exp(sum(math.cos(2 * math.pi * x) for x in point) / self.dim)
        )
    
    def minima(self) -> float:
        return [[0, 0]]
    
    def domain(self) -> tuple[float, float, float, float]:
        return (-3, -3, 3, 3)


class Rosenbrock(Fitness):
    def __init__(self, dim: int = 2):
        self.dim = dim

    def __call__(self, point) -> float:
        return sum(100 * (x2 - x1 * x1) ** 2 + (1 - x1) ** 2 for x1, x2 in pairwise(point))
    
    def minima(self) -> float:
        return [[1] * self.dim]
    
    def domain(self) -> tuple[float, float, float, float]:
        return (-2, -2, 2, 2)


class Himmelblau(Fitness):
    def __call__(self, point) -> float:
        x, y = point
        return (x * x + y - 11) ** 2 + (x + y * y - 7) ** 2
    
    def minima(self) -> float:
        return [[3.0, 2.0], [-2.805118, 3.283186], [-3.779310, -3.283186], [3.584458, -1.848126]]
    
    def domain(self) -> tuple[float, float, float, float]:
        return (-6, -6, 6, 6)