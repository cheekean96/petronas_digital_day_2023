import numpy as np
from src.algorithms.ga import GeneticAlgorithm
from src.algorithms.fitness import MeanSquaredError
from holoviews import opts, dim
import holoviews as hv
from holoviews.plotting.util import process_cmap
import panel as pn
from holoviews.streams import Stream
hv.extension('bokeh', logo=False)


Fitness = MeanSquaredError


class CreateGAPanel:

    def __init__(self):
        self.population_size = 100
        self.vector_length = 2

        # Value initialisation
        self.fitness = Fitness()

        # Widget default values
        self.default_mutate_status = True
        self.default_niters = 5
        self.default_mutation_rate = 0.3
        self.default_mutation_scale = 1

        # Cached objects
        self._contours = None

    def run(self):
        self.ga = GeneticAlgorithm(self.population_size, self.vector_length, self.fitness)

        self.mutate_checkbox = pn.widgets.Checkbox(
            name='Mutate', width=550, value=self.default_mutate_status
        )
        self.niters_slider = pn.widgets.IntSlider(
            name='Time Evolving (s)', width=550, start=0, end=50, value=self.default_niters
        )
        self.mutation_rate_slider = pn.widgets.FloatSlider(
            name='Mutation Rate', width=550, start=0.0, end=1.0, value=self.default_mutation_rate
        )
        self.mutation_scale_slider = pn.widgets.IntSlider(
            name='Mutation Scale', width=550, start=0, end=50, value=self.default_mutation_scale
        )

        # Reset params button
        self.reset_params_button = pn.widgets.Button(name='Reset Parameters', width=75)
        self.reset_params_button.on_click(self.reset_event)

        # Create button events
        self.dmap = hv.DynamicMap(self.update, streams=[Stream.define('Next')()])

        # Run button
        self.run_button = pn.widgets.Button(name='\u25b6 Begin Evolving', width=75)
        self.run_button.on_click(self.b)

        # New population button
        self.new_pop_button = pn.widgets.Button(name='New Population', width=75)
        self.new_pop_button.on_click(self.e)

        # Next generation button
        self.next_generation_button = pn.widgets.Button(name='Next Generation', width=75)
        self.next_generation_button.on_click(self.next_gen_event)

        # Layout everything together
        self.instructions = pn.pane.Markdown(
            """
            # Genetic Algorithm: Simulation
            ## Instructions:
            1. **Adjust the (x, y) coordinate to place the target.**
            2. Click '\u25b6 Begin Evolution' button to begin evolving for the time on the Time Evolving slider.
            3. Experiment with the Mutation Rate (the probability of an individual in the next generation mutating)
            4. Experiment with the Mutation Scale (the size of the mutation,
               tip: zoom out using the Wheel Zoom on the right of the plot).
            """
        )
        self.dashboard = pn.Column(self.instructions,
                                   pn.Column(self.dmap.opts(width=600, height=600),
                                             pn.Row(self.run_button, pn.Spacer(width=75),
                                                    self.new_pop_button, pn.Spacer(width=75),
                                                    self.next_generation_button),
                                             """## Adjust Hyperparameters Here:""",
                                             self.mutate_checkbox,
                                             self.niters_slider,
                                             self.mutation_rate_slider,
                                             self.mutation_scale_slider,
                                             self.reset_params_button))

        return self.dashboard

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
        self.mutate_checkbox.value = self.default_mutate_status
        self.niters_slider.value = self.default_niters
        self.mutation_rate_slider.value = self.default_mutation_rate
        self.mutation_scale_slider.value = self.default_mutation_scale

    def next_gen_event(self, event):
        """
        Trigger the streams associated with the DynamicMap object.

        Args:
            event: The event object associated with the next generation event.

        Returns:
            None

        """
        hv.streams.Stream.trigger(self.dmap.streams)

    def update(self):
        """
        Update the GeneticAlgorithm object and return a visualization of the current population.

        Returns:
            hv.Scatter: A Holoviews Scatter plot representing the current population,
                        along with the current fittest individual.

        """
        self.ga.next_generation(
            self.mutation_rate_slider.value, self.mutation_scale_slider.value, self.mutate_checkbox.value
        )
        min_x, min_y, max_x, max_y = self.fitness.domain()
        self.scatter = hv.Scatter(
            self.ga.current_population, label='Population'
        ).opts(color='b', xlim=(min_x, max_x), ylim=(min_y, max_y))
        self.fittest = hv.Points(
            (self.ga.current_best[0], self.ga.current_best[1], 1), label='Current Fittest'
        ).opts(color='c', size=10)
        self.target_tap = hv.Points(
            self.fitness.minima(), label='Minima'
        ).opts(color='r', marker='^', size=15)
        self.contours = self._contour_plot()
        self.layout = self.scatter * self.contours * self.target_tap * self.fittest

        return self.layout
    
    def _contour_plot(self) -> hv.Contours:
        """
        Create contour plot.
        This plot is static for each fitness function and set of bounds,
        so it is cached.
        To regenerate, set `self._contours` to None.

        Returns:
            hv.Contours: The Contour plot.
        """
        if self._contours is None:
            bounds = self.fitness.domain()
            min_x, min_y, max_x, max_y = bounds
            x = np.linspace(min_x, max_x, 1000)
            y = np.linspace(min_y, max_y, 1000)
            X, Y = np.meshgrid(x, y)
            Z = np.apply_along_axis(self.fitness, 1, np.c_[X.ravel(), Y.ravel()]).reshape(X.shape)
            img = hv.Image(Z, bounds=bounds) 
            self._contours = hv.operation.contours(img, levels=20)
            cmap = process_cmap(['#f0f0f0', '#e0e0e0'])
            self._contours.opts(opts.Contours(cmap=cmap, show_legend=False))
        return self._contours


    def e(self, event):
        """
        Event handler for initializing the GeneticAlgorithm object and triggering associated streams.

        Args:
            event: The event object associated with the initialization event.

        Returns:
            None

        Side Effects:
            - Sets the population size and vector length attributes.
            - Initializes a new GeneticAlgorithm object.
            - Triggers the streams associated with the DynamicMap object.

        """
        self.ga = GeneticAlgorithm(self.population_size, self.vector_length, self.fitness)
        hv.streams.Stream.trigger(self.dmap.streams)

    def b(self, event):
        """
        Event handler for running a simulation using the DynamicMap object.

        Args:
            event: The event object associated with the simulation event.

        Returns:
            None

        Side Effects:
            - Runs the simulation for a specified duration.
            - The duration is determined by the value of the `niters_slider` attribute.

        """
        self.fitness = Fitness()
        self.ga = GeneticAlgorithm(self.population_size, self.vector_length, self.fitness)
        if self.dmap.periodic.instance is not None:
            self.dmap.periodic.stop()
        self.dmap.periodic(0.1, timeout=self.niters_slider.value, block=False)
