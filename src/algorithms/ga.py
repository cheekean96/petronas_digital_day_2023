import numpy as np
import random
from .fitness import Fitness

class GeneticAlgorithm:
    """
    A class representing a Genetic Algorithm.

    Attributes:
        fitness (Fitness): The `Fitness` object representing the fitness evaluation function.
        current_population (numpy.ndarray): A 2-dimensional array representing the current population,
                                            where each row represents an individual and each column represents a gene.
        current_best (float): The fitness value of the current best individual in the population.
    """

    def __init__(self, population_size, vector_length, fitness):
        self.fitness = fitness
        self.current_population = self.create_population(population_size, vector_length)
        self.current_best = self.find_current_best(self.current_population)

    def next_generation(self, mrate, mscale, should_mutate):
        """
        Update the current population to the next generation based on the GeneticAlgorithm.

        Args:
            mrate (float): Mutation rate to be applied during population update.
            mscale (float): Mutation scale to be applied during population update.
            should_mutate (bool): Flag indicating whether mutation should be applied.

        Returns:
            None

        Side Effects:
            - Updates the current_population attribute of the object.
            - Updates the current_best attribute of the object.
        """
        self.current_population = self.update_population(
            self.current_population, should_mutate, mrate, mscale
        )
        self.current_best = self.find_current_best(self.current_population)

    def find_current_best(self, population):
        """
        Evaluate a given swarm and return the fittest particle based on their best previous position.

        Args:
            swarm (list): List of particles in the swarm.

        Returns:
            Particle: The fittest particle in the swarm based on their best previous position.

        Note:
            This function can be optimized to loop over the swarm only once, but for the sake of simplicity
            in this tutorial, it is implemented in three lines.
        """
        fitnesses = [self.fitness(x) for x in population]
        best_value = min(fitnesses)
        best_index = fitnesses.index(best_value)
        return population[best_index]

    def create_population(self, population_size, vector_length):
        """
        Create a population matrix with random values between 0 and 1.

        Args:
            population_size (int): The number of individuals in the population.
            vector_length (int): The length of each individual vector.

        Returns:
            numpy.ndarray: A matrix representing the population, where each row is an individual vector.

        Note:
            The generated values are uniformly distributed between 0 and 1.

        """
        return np.random.rand(population_size, vector_length)

    def tournament_select_with_replacement(self, population, tournament_size):
        """
        Select an individual from the population using tournament selection with replacement.

        Args:
            population (numpy.ndarray): A 2-dimensional array of individuals where
                                        each row represents an individual and each column represents a gene.
            tournament_size (int): The number of individuals in each tournament.

        Returns:
            The fittest individual among the challengers, as determined by the fitness function.

        """
        challengers_indexes = np.random.choice(population.shape[0], tournament_size, replace=True)
        challengers = population[challengers_indexes]
        return self.find_current_best(challengers)

    def crossover(self, parent_a, parent_b):
        """
        Perform two-point crossover on two parent individuals.

        Args:
            parent_a (numpy.ndarray): The first parent individual.
            parent_b (numpy.ndarray): The second parent individual.

        Returns:
            tuple: A tuple containing two child individuals resulting from the crossover operation.

        """
        l = parent_a.shape[0]  # noqa E741
        c, d = random.randint(0, l), random.randint(0, l)

        # Flip if c greater than d
        if c > d:
            d, c = c, d  # noqa E701
        if c == d:
            d += 1  # noqa E701
        temp = np.copy(parent_a)
        child_a = np.concatenate([parent_a[0:c], parent_b[c:d], parent_a[d:]])
        child_b = np.concatenate([parent_b[0:c], temp[c:d], parent_b[d:]])
        return child_a, child_b

    def mutate(self, child, mutation_rate, mutation_scale):
        """
        Mutate a child using Gaussian convolution.

        Args:
            child (numpy.ndarray): A 1-dimensional array representing an individual.
            mutation_rate (float): The probability of mutation occurring for each gene.
            mutation_scale (float): The standard deviation of the Gaussian distribution used for mutation.

        Returns:
            The mutated child as a 1-dimensional array.

        """
        if mutation_rate >= random.uniform(0, 1):
            size = child.shape[0]
            mutation_value = np.random.normal(0, mutation_scale, size)
            child = child + mutation_value
        return child

    def update_population(self, current_population, should_mutate, mutation_rate, mutation_scale):
        """
        Perform one generational update of the Genetic Algorithm.

        Args:
            current_population (numpy.ndarray): A 2-dimensional array representing the current population, where each
                                                row represents an individual and each column represents a gene.
            should_mutate (bool): Flag indicating whether mutation should be applied to the offspring.
            mutation_rate (float): The probability of mutation occurring for each gene.
            mutation_scale (float): The standard deviation of the Gaussian distribution used for mutation.

        Returns:
            The next generation population as a 2-dimensional array.
        """
        pop_size = len(current_population)
        next_population = np.empty((pop_size, 2))
        tournament_size = 2
        for i in range(int(pop_size / 2)):
            parent_a = self.tournament_select_with_replacement(
                current_population, tournament_size
            )
            parent_b = self.tournament_select_with_replacement(
                current_population, tournament_size
            )
            child_a, child_b = self.crossover(parent_a, parent_b)
            next_population[i] = self.mutate(child_a, mutation_rate, mutation_scale)\
                if should_mutate else child_a
            position_child_b = i + (pop_size / 2)
            next_population[int(position_child_b)] = self.mutate(child_b, mutation_rate, mutation_scale)\
                if should_mutate else child_b
        return next_population
