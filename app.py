import panel as pn
# from pso_panel import CreatePSOPanel
# from ga_panel import CreateGAPanel

pn.extension(sizing_mode="stretch_width")

############################# fitness.py #############################

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


############################# pso.py #############################

import numpy as np
import random
# from .fitness import Fitness

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

############################# pso_panel.py #############################

import numpy as np
# from algorithms.pso import Particle, PSO
# from algorithms.fitness import Fitness
from holoviews import opts, dim
import holoviews as hv
import panel as pn
from holoviews.streams import Stream
hv.extension('bokeh', logo=False)


class CreatePSOPanel:

    def __init__(self):
        # Creating a swarm of particles
        self.vector_length = 2
        self.size = 25
        self.swarm_size = 50
        self.num_informants = 2

        # Value initialisation
        self.target_x = 0.5
        self.target_y = 0.5
        self.fitness = Fitness(self.target_x, self.target_y)
        self.swarm = [
            Particle(self.fitness.problem_, np.random.uniform(-2, 2, self.vector_length),
                     np.random.rand(self.vector_length), self.target_x, self.target_y, i)
            for i, x in enumerate(range(self.swarm_size))
        ]
        self.vect_data = self.get_vectorfield_data(self.swarm)
        self.vectorfield = hv.VectorField(self.vect_data, vdims=['Angle', 'Magnitude', 'Index'])
        self.particles = [
            np.array([self.vect_data[0], self.vect_data[1], self.vect_data[4]])
            for i, particle in enumerate(self.swarm)
        ]
        self.points = hv.Points(self.particles, vdims=['Index'])
        self.layout = self.vectorfield * self.points
        self.layout.opts(
            opts.VectorField(color='Index', cmap='tab20c', magnitude=dim('Magnitude').norm() * 10, pivot='tail'),
            opts.Points(color='Index', cmap='tab20c', size=5)
        )
        self.target_tap = hv.Points(
            (self.target_x, self.target_y, 1), label='Target').opts(color='r', marker='^', size=15)

        # Widget default values
        self.default_pop_size = 25
        self.default_time = 5
        self.default_num_informants = 6
        self.default_current = 0.7
        self.default_personal_best = 2.0
        self.default_social_best = 0.9
        self.default_global_best = 0.0
        self.default_scale_update_step = 0.7

    def run(self):
        self.pso = PSO(self.fitness.problem_, self.size, self.vector_length, self.target_x, self.target_y)

        # Sliders & defaults
        self.target_x_slider = pn.widgets.FloatSlider(
            name="Target (X-Coordinate)", width=550, start=0.0, end=1.0, value=self.target_x
        )
        self.target_y_slider = pn.widgets.FloatSlider(
            name="Target (Y-Coordinate)", width=550, start=0.0, end=1.0, value=self.target_y
        )
        self.population_size_slider = pn.widgets.IntSlider(
            name='Population Size', width=550, start=10, end=50, value=self.default_pop_size
        )
        self.time_slider = pn.widgets.IntSlider(name='Time Evolving (s)', width=550, start=0, end=15,
                                                value=self.default_time)
        self.num_informants_slider = pn.widgets.IntSlider(
            name='Number of Informants', width=550, start=0, end=20, value=self.default_num_informants
        )
        self.follow_current_slider = pn.widgets.FloatSlider(
            name='Follow Current', width=550, start=0.0, end=5, value=self.default_current
        )
        self.follow_personal_best_slider = pn.widgets.FloatSlider(
            name='Follow Personal Best', width=550, start=0, end=5, value=self.default_personal_best
        )
        self.follow_social_best_slider = pn.widgets.FloatSlider(
            name='Follow Social Best', width=550, start=0.0, end=5, value=self.default_social_best
        )
        self.follow_global_best_slider = pn.widgets.FloatSlider(
            name='Follow Global Best', width=550, start=0.0, end=1, value=self.default_global_best
        )
        self.scale_update_step_slider = pn.widgets.FloatSlider(name='Scale Update Step', width=550,
                                                               start=0.0, end=1, value=0.7)

        # Reset params button
        self.reset_params_button = pn.widgets.Button(name='Reset Parameters', width=50)
        self.reset_params_button.on_click(self.reset_event)

        # # Set the target (Note: Interactive tap not working, replaced with slider instead)
        # self.tap_stream = hv.streams.SingleTap(transient=True, x=self.target_x, y=self.target_y)
        # # Place the target indicator
        # self.target_tap = hv.DynamicMap(self.tap_event, streams=[self.tap_stream])

        # Create button events
        self.vector_field = hv.DynamicMap(self.update, streams=[Stream.define('Next')()])

        # Run button
        self.run_button = pn.widgets.Button(name='\u25b6 Begin Improving', width=75)
        self.run_button.on_click(self.b)

        # New population button
        self.new_pop_button = pn.widgets.Button(name='New Population', width=75)
        self.new_pop_button.on_click(self.new_pop_event)

        # Next generation button
        self.next_generation_button = pn.widgets.Button(name='Next Generation', width=75)
        self.next_generation_button.on_click(self.next_gen_event)

        # Layout everything together
        self.instructions = pn.pane.Markdown(
            """
            # Partical Swarm Optimisation Dashboard
            ## Instructions:
            1. **Adjust the (x, y) coordinate to place the target.**
            2. Click '\u25b6 Begin Improving' button to begin improving for the time on the Time Evolving slider.
            3. Experiment with the sliders.
            """
        )

        self.dashboard = pn.Column(self.instructions,
                                   pn.Column(self.vector_field.opts(width=600, height=600),
                                             pn.Column(pn.Row(self.run_button, pn.Spacer(width=75),
                                                              self.new_pop_button, pn.Spacer(width=75),
                                                              self.next_generation_button),
                                                       """## Place Your Target Here:""",
                                                       self.target_x_slider,
                                                       self.target_y_slider,
                                                       """## Adjust Hyperparameters Here:""",
                                                       self.time_slider,
                                                       self.num_informants_slider,
                                                       self.population_size_slider,
                                                       self.follow_current_slider,
                                                       self.follow_personal_best_slider,
                                                       self.follow_social_best_slider,
                                                       self.follow_global_best_slider,
                                                       self.scale_update_step_slider,
                                                       self.reset_params_button)))

        # dashboard.servable()
        return self.dashboard

    def b(self, event):
        """
        Perform PSO initialization and start the vector field animation.

        Args:
            event: The event triggering the function.

        Returns:
            None

        Note:
            This function initializes the Particle Swarm Optimization (PSO) algorithm by
            setting global variables and creating a PSO instance.
            It also starts the animation of the vector field using the specified parameters.
        """
        self.size = self.population_size_slider.value
        self.num_informants = self.num_informants_slider.value
        self.target_x = self.target_x_slider.value
        self.target_y = self.target_y_slider.value
        self.fitness = Fitness(self.target_x, self.target_y)
        self.pso_fitnesses = []
        self.pso = PSO(self.fitness.problem_, self.size, self.vector_length,
                       self.target_x, self.target_y, self.num_informants)
        self.swarm = [
            Particle(self.fitness.problem_, np.random.uniform(-2, 2, self.vector_length),
                     np.random.rand(self.vector_length), self.target_x, self.target_y, i)
            for i, x in enumerate(range(self.swarm_size))
        ]
        self.vect_data = self.get_vectorfield_data(self.swarm)
        self.vectorfield = hv.VectorField(self.vect_data, vdims=['Angle', 'Magnitude', 'Index'])
        self.particles = [
            np.array([self.vect_data[0], self.vect_data[1], self.vect_data[4]])
            for i, particle in enumerate(self.swarm)
        ]
        self.points = hv.Points(self.particles, vdims=['Index'])
        self.layout = self.vectorfield * self.points
        self.layout.opts(
            opts.VectorField(color='Index', cmap='tab20c', magnitude=dim('Magnitude').norm() * 10, pivot='tail'),
            opts.Points(color='Index', cmap='tab20c', size=5)
        )

        self.vector_field.periodic(0.005, timeout=self.time_slider.value)

    def reset_event(self, event):
        """
        Reset the global variables to their default values.

        Args:
            event: The event triggering the reset.

        Returns:
            None

        Note:
            This function resets the values of several global variables to their default values.
            It is typically used as an event handler for a reset button or similar functionality.
        """
        self.target_x_slider.value, self.target_y_slider.value = 0.5, 0.5
        self.follow_current_slider.value, self.follow_personal_best_slider.value = \
            self.default_current, self.default_personal_best
        self.follow_social_best_slider.value, self.follow_global_best_slider.value = \
            self.default_social_best, self.default_global_best
        self.scale_update_step_slider.value, self.population_size_slider.value = \
            self.default_scale_update_step, self.default_pop_size
        self.time_slider.value, self.num_informants_slider.value = \
            self.default_time, self.default_num_informants

    # def tap_event(self, x, y):
    #     """
    #     Update the global target coordinates and return a visualization of the target.

    #     # Args:
    #     #     x (float): The x-coordinate of the target.
    #     #     y (float): The y-coordinate of the target.

    #     Returns:
    #         hv.Points: Visualization of the target as a Holoviews Points object.

    #     Note:
    #         This function updates the global target coordinates based on the provided x and y values.
    #         It also returns a visualization of the target as a Holoviews Points object.
    #     """
    #     # Note: interactive tap not functioning properly, replaced with slider value instead.
    #     x, y = self.target_x, self.target_y

    #     return hv.Points((x, y, 1), label='Target').opts(color='r', marker='^', size=15)

    def new_pop_event(self, event):
        """
        Create a new population for the Particle Swarm Optimization (PSO) algorithm and trigger vector field streams.

        Args:
            event: The event triggering the function.

        Returns:
            None

        Note:
            This function creates a new population for the PSO algorithm by setting
            the global PSO instance with updated parameters.
            It also triggers the vector field streams, initiating the visualization update.
        """
        self.size = self.population_size_slider.value
        self.num_informants = self.num_informants_slider.value
        self.pso = PSO(self.fitness.problem_, self.size, vector_length=2,
                       target_x=self.target_x, target_y=self.target_y, num_informants=self.num_informants)
        hv.streams.Stream.trigger(self.vector_field.streams)

    def next_gen_event(self, event):
        """
        Trigger the vector field streams to update the visualization for the next generation.

        Args:
            event: The event triggering the function.

        Returns:
            None

        Note:
            This function triggers the vector field streams, initiating the update of the
            visualization for the next generation.
        """
        hv.streams.Stream.trigger(self.vector_field.streams)

    def update(self):
        """
        Perform an update step in the Particle Swarm Optimization (PSO) algorithm and return a visualization layout.

        Returns:
            hv.Layout: Visualization layout containing a vector field, particle
            scatter plot, and current fittest point.

        Note:
            This function performs an update step in the PSO algorithm using the slider values for various parameters.
            It generates a vector field, particle scatter plot, and current fittest point visualization using
            HoloViews. The resulting visualizations are combined into a layout and returned.
        """
        self.pso.improve(self.follow_current_slider.value, self.follow_personal_best_slider.value,
                         self.follow_social_best_slider.value, self.follow_global_best_slider.value,
                         self.scale_update_step_slider.value)
        self.vect_data = self.get_vectorfield_data(self.pso.swarm)
        self.vectorfield = hv.VectorField(
            self.vect_data, vdims=['Angle', 'Magnitude', 'Index']
        ).opts(color='Index', cmap='tab20c', magnitude=dim('Magnitude').norm() * 10, pivot='tail')
        self.particles = [
            np.array([self.vect_data[0], self.vect_data[1], self.vect_data[4]])
            for i, particle in enumerate(self.swarm)
        ]
        # Place the target indicator
        self.scatter = hv.Points(
            self.particles, vdims=['Index'], group='Particles'
        ).opts(color='Index', cmap='tab20c', size=5, xlim=(0, 1), ylim=(0, 1))
        self.fittest = hv.Points(
            (self.pso.global_fittest.fittest_position[0],
             self.pso.global_fittest.fittest_position[1], 1), label='Current Fittest'
        ).opts(color='b', fill_alpha=0.1, line_width=1, size=10)
        self.target_tap = hv.Points(
            (self.target_x, self.target_y, 1), label='Target'
        ).opts(color='r', marker='^', size=15)
        self.layout = self.vectorfield * self.scatter * self.fittest * self.target_tap
        # self.layout.opts(
        #     opts.Points(color='b', fill_alpha=0.1, line_width=1, size=10),
        #     opts.VectorField(color='Index', cmap='tab20c', magnitude=dim('Magnitude').norm() * 10, pivot='tail'),
        #     opts.Points('Particles', color='Index', cmap='tab20c', size=5, xlim=(0, 1), ylim=(0, 1)),
        #     opts.Points(color='r', marker='^', size=15),
        # )
        return self.layout

    def to_angle(self, vector):
        """
        Calculate the magnitude and angle of a vector.

        Args:
            vector (list or tuple): A 2-dimensional vector represented as [x, y].

        Returns:
            tuple: A tuple containing the magnitude and angle of the vector.
                - magnitude (float): The length or magnitude of the vector.
                - angle (float): The angle of the vector in radians, measured counterclockwise
                                from the positive y-axis.
        """
        x = vector[0]
        y = vector[1]
        mag = np.sqrt(x ** 2 + y ** 2)
        angle = (np.pi / 2.) - np.arctan2(x / mag, y / mag)
        return mag, angle

    def get_vectorfield_data(self, swarm):
        """
        Retrieve data for vector field visualization.

        Args:
            swarm (list): List of particles in the swarm.

        Returns:
            tuple: A tuple containing the following lists:
                - xs (list): X-coordinates of particle positions.
                - ys (list): Y-coordinates of particle positions.
                - angles (list): Angles of particle velocities.
                - mags (list): Magnitudes of particle velocities.
                - ids (list): Identifiers of particles.
        """
        xs, ys, angles, mags, ids = [], [], [], [], []
        for particle in swarm:
            xs.append(particle.position[0])
            ys.append(particle.position[1])
            mag, angle = self.to_angle(particle.velocity)
            mags.append(mag)
            angles.append(angle)
            ids.append(particle.id)
        return xs, ys, angles, mags, ids

############################# app.py #############################

pso = CreatePSOPanel()
# ga = CreateGAPanel()

pages = {
    "PSO": pso.run(),
    # "GA": ga.run(),
}

def show(page):
    return pages[page]


starting_page = pn.state.session_args.get("page", [b"PSO"])[0].decode()

welcome_message = pn.pane.Markdown(
    """
    # Welcome to Petronas Digital Day (PDD)!

    We are thrilled to present to you an exciting simulation dashboard that takes you on a journey \
    into the world of optimization through Particle Swarm Optimization and Genetic Algorithm.

    Step into the shoes of an optimizer as you explore the power of these cutting-edge techniques. \
    Interact with our simulation by adjusting the hyperparameters and witness firsthand how these \
    algorithms efficiently solve complex problems.

    Unleash your creativity and curiosity as you dive deep into the realm of optimization. \
    Whether you are a seasoned expert or new to the field, our user-friendly interface ensures an \
    engaging and informative experience for all.

    Prepare to be amazed as you discover how Particle Swarm Optimization and Genetic Algorithm revolutionize \
    decision-making processes across various industries. Get ready to optimize your understanding and embark on \
    an interactive adventure like no other. Enjoy your exploration!
    """
)

menu = pn.widgets.RadioButtonGroup(
    value=starting_page,
    options=list(pages.keys()),
    name="Page",
    sizing_mode="fixed",
    button_type="success",
)

page = pn.Column(pn.Row(menu), welcome_message)

ishow = pn.bind(show, page=menu)
pn.state.location.sync(menu, {"value": "page"})

ACCENT_COLOR = "#007D79"  # "#0072B5"
DEFAULT_PARAMS = {
    "site": "Petronas Digital Day",
    "accent_base_color": ACCENT_COLOR,
    "header_background": ACCENT_COLOR,
}
pn.template.FastListTemplate(
    title="Optimization: Metaheuristics Algorithm",
    sidebar=[page],
    main=[ishow],
    **DEFAULT_PARAMS,
).servable()
