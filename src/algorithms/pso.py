import numpy as np
import random
from .fitness import Fitness

class Particle:
    """
    A Particle used in PSO.

    Attributes
    ----------
    problem : function
        The problem to minimize.
    velocity : np.array
        The current velocity of the particle.
    position : np.array
        The current position of the particle, used as the solution for the given problem.
    id : int
        The unique id of the particle.

    Methods
    -------
    assess_fitness()
        Determines the fitness of the particle using the given problem.
    update(fittest_informant, global_fittest,
           follow_current, follow_personal_best,
           follow_social_best, follow_global_best,
           scale_update_step)
        Updates the velocity and position of the particle using the PSO update algorithm.
    """

    def __init__(self, problem, velocity, position, index, target_x, target_y):
        self.velocity = velocity
        self.position = position
        self.fittest_position = position
        self.problem = problem
        self.id = index
        self.previous_fitness = 1e7
        self.target_x = target_x
        self.target_y = target_y
        self.fitness = Fitness(self.target_x, self.target_y)

    def assess_fitness(self):
        """
        Determines the fitness of the particle using the given problem.

        Returns:
            The fitness value of the particle.

        """
        return self.fitness.assess_fitness_(self.position, self.problem)

    def update(self, fittest_informant, global_fittest, follow_current, follow_personal_best,
               follow_social_best, follow_global_best, scale_update_step):
        """
        Updates the velocity and position of the particle using the PSO update algorithm.

        Args:
            fittest_informant: The fittest particle among the informants.
            global_fittest: The fittest particle in the global population.
            follow_current: The weight for the current velocity.
            follow_personal_best: The weight for the personal best position.
            follow_social_best: The weight for the social best position.
            follow_global_best: The weight for the global best position.
            scale_update_step: The scaling factor for the update step.

        Returns:
            None
        """
        self.position += self.velocity * scale_update_step
        cognitive = random.uniform(0, follow_personal_best)
        social = random.uniform(0, follow_social_best)
        glob = random.uniform(0, follow_global_best)
        self.velocity = (follow_current * self.velocity
                         + cognitive * (self.fittest_position - self.position)
                         + social * (fittest_informant.fittest_position - self.position)
                         + glob * (global_fittest.fittest_position - self.position))
        current_fitness = self.assess_fitness()
        if current_fitness < self.previous_fitness:
            self.fittest_position = self.position
        self.previous_fitness = current_fitness

class PSO:
    """
    An implementation of Particle Swarm Optimization (PSO) pioneered by Kennedy, Eberhart, and Shi.

    The swarm consists of particles with fixed-length vectors for velocity and position.
    Position is initialized with a uniform distribution between 0 and 1, while velocity is initialized with zeros.
    Each particle has a given number of informants, which are randomly chosen at each iteration.

    Attributes:
        swarm_size (int): The size of the swarm.
        vector_length (int): The dimensions of the problem. Should be the same
        as the one used when creating the problem object.
        num_informants (int): The number of informants used for the social component in particle velocity update.

    Public Methods:
        improve(follow_current, follow_personal_best, follow_social_best, follow_global_best, scale_update_step)
            Update each particle in the swarm and update the global fitness.
        update_swarm(follow_current, follow_personal_best, follow_social_best, follow_global_best, scale_update_step)
            Update each particle, randomly choosing informants for each particle's update.
        update_global_fittest()
            Update the `global_fittest` variable to be the current fittest particle in the swarm.
    """

    def __init__(self, problem, swarm_size, vector_length, target_x, target_y, num_informants=2):
        """
        Initialize a PSO object.

        Args:
            problem: The problem object representing the fitness evaluation function.
            swarm_size (int): The size of the swarm.
            vector_length (int): The dimensions of the problem. Should be the same as
            the one used when creating the problem object.
            num_informants (int): The number of informants used for the social component in particle velocity update.
        """
        self.swarm_size = swarm_size
        self.num_informants = num_informants
        self.problem = problem
        self.target_x = target_x
        self.target_y = target_y
        self.swarm = [Particle(self.problem, np.zeros(vector_length), np.random.rand(vector_length),
                               self.target_x, self.target_y, i)
                      for i in range(swarm_size)]
        self.global_fittest = np.random.choice(self.swarm, 1)[0]
        self.fitness = Fitness(self.target_x, self.target_y)

    def update_swarm(self, follow_current, follow_personal_best, follow_social_best,
                     follow_global_best, scale_update_step):
        """
        Update each particle in the swarm.

        Args:
            follow_current: The weight for the current velocity.
            follow_personal_best: The weight for the personal best position.
            follow_social_best: The weight for the social best position.
            follow_global_best: The weight for the global best position.
            scale_update_step: The scaling factor for the velocity update.

        Note:
            This method randomly selects informants for each particle's
            update and ensures each particle is its own informant.
        """
        for particle in self.swarm:
            informants = np.random.choice(self.swarm, self.num_informants)
            if particle not in informants:
                np.append(informants, particle)
            fittest_informant = self.find_current_best(informants, self.problem)
            particle.update(fittest_informant,
                            self.global_fittest,
                            follow_current,
                            follow_personal_best,
                            follow_social_best,
                            follow_global_best,
                            scale_update_step)

    def update_global_fittest(self):
        """
        Update the `global_fittest` variable to be the current fittest particle in the swarm.

        Note:
            This method compares the fitness of each particle in the swarm and updates
            `global_fittest` if a fitter particle is found.
        """
        fittest = self.find_current_best(self.swarm, self.problem)
        global_fittest_fitness = self.global_fittest.assess_fitness()
        if (fittest.assess_fitness() < global_fittest_fitness):
            self.global_fittest = fittest

    def improve(self, follow_current, follow_personal_best, follow_social_best, follow_global_best, scale_update_step):
        """
        Improve the population for one iteration.

        Args:
            follow_current: The weight for the current velocity.
            follow_personal_best: The weight for the personal best position.
            follow_social_best: The weight for the social best position.
            follow_global_best: The weight for the global best position.
            scale_update_step: The scaling factor for the velocity update.

        Note:
            This method updates the swarm by calling `update_swarm` to update each particle and
            `update_global_fittest` to update the global fitness.
        """
        self.update_swarm(
            follow_current, follow_personal_best, follow_social_best, follow_global_best, scale_update_step
        )
        self.update_global_fittest()

    # Find the fittest Partle in the swarm
    def find_current_best(self, swarm, problem):
        """
        Evaluate a given swarm and return the fittest particle based on their best previous position.

        Args:
            swarm (list): List of particles in the swarm.
            problem (object): The problem instance used for assessing fitness.

        Returns:
            Particle: The fittest particle in the swarm based on their best previous position.

        Note:
            This function can be optimized to loop over the swarm only once, but for the sake of simplicity
            in this tutorial, it is implemented in three lines.
        """
        fitnesses = [self.fitness.assess_fitness_(x.fittest_position, problem) for x in swarm]
        best_value = min(fitnesses)
        best_index = fitnesses.index(best_value)
        return swarm[best_index]
