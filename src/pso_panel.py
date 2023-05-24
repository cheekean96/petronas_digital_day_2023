import numpy as np
from algorithms.pso import Particle, PSO
from algorithms.fitness import Fitness
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
