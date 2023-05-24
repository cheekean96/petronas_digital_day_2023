class Fitness:
    def __init__(self, target_x, target_y):
        self.target_x = target_x
        self.target_y = target_y

    def problem_(self, soln):
        """
        Evaluates the fitness of a solution using the mean squared error.

        Args:
            soln: The solution to evaluate, represented as a list or array.

        Returns:
            The fitness value calculated based on the mean squared error between the
            solution and the target coordinates.

        Note:
            This function uses global variables `target_x` and `target_y` to represent the target coordinates.
            These variables can be linked to a click event later.
        """
        return self.mean_squared_error_(soln, [self.target_x, self.target_y])

    def assess_fitness_(self, individual, problem):
        """
        Determines the fitness of an individual using the given problem.

        Args:
            individual: The individual to evaluate.
            problem: The fitness evaluation function to use.

        Returns:
            The fitness value of the individual calculated using the provided problem.
        """
        return problem(individual)

    def mean_squared_error_(self, y_true, y_pred):
        """
        Calculate the mean squared error between the true and predicted values.

        Args:
            y_true (array-like): The true values.
            y_pred (array-like): The predicted values.

        Returns:
            float: The mean squared error between y_true and y_pred.
        """
        return ((y_true - y_pred) ** 2).mean(axis=0)
