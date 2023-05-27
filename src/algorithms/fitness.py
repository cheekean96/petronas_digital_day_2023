from abc import ABC


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


    def minima(self) -> list[float]:
        """
        Return a list of global minima.

        Returns:
            list of global minima.
        """
        pass


class MeanSquaredError(Fitness):
    def __init__(self, target_x, target_y):
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
    
    def minima(self) -> list[float]:
        """
        Return a list of global minima.

        Returns:
            list of global minima.
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
