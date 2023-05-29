import panel as pn
from src.pso_panel import CreatePSOPanel
from src.ga_panel import CreateGAPanel

pn.extension(sizing_mode="stretch_width")

pso = CreatePSOPanel()
ga = CreateGAPanel()

welcome_page = pn.Column(
    # pn.pane.Video("docs/vid/pdd_opt.mp4", width=600, autoplay=True, loop=True, align="center"),
    pn.pane.Markdown(
        """
        # Particle Swarm Optimization (PSO)
        ## Unleashing the Power of Swarm Intelligence

        Welcome to the world of Particle Swarm Optimization (PSO), where nature-inspired algorithms meet \
        computational power. PSO mimics the behavior of a flock of birds or a school of fish to solve \
        complex optimization problems. Let's dive into the concept of PSO and explore its real-life significance.

        In PSO, particles represent potential solutions that navigate a problem space. Each particle adjusts its \
        position and velocity based on its own best-known solution and the global best solution found by the swarm. \
        Through communication and cooperation, the particles converge towards the optimal solution, balancing \
        exploration and exploitation.
        """
    ),
    pn.Row(
        pn.pane.GIF("docs/img/pso_birds.gif", width=300, height=180, align="center"),
        pn.pane.GIF("docs/img/pso_animation.gif", width=300, height=180, align="center"),
        align="center"
    ),
    pn.pane.Markdown(
        """
        PSO's significance extends across various domains. In engineering, it aids in designing efficient systems \
        like aircraft wings and energy-efficient buildings. In finance, it optimizes investment portfolios to \
        maximize returns while minimizing risks. PSO also improves traffic flow by optimizing signal timings and \
        route planning, reducing congestion and fuel consumption. Additionally, it plays a crucial role in training \
        artificial neural networks, enhancing predictive models in fields such as image recognition and \
        natural language processing.
        """
    ),
    pn.layout.Divider(),
    pn.pane.Markdown(
        """
        # Genetic Algorithm (GA)
        ## Unleashing Evolutionary Power

        Welcome to the world of Genetic Algorithms (GA), where nature's principles guide computational optimization. \
        GA draws inspiration from the process of natural selection to solve complex problems. Let's explore the \
        concept of GA and discover its significance in real-life applications.

        In GA, a population of potential solutions evolves over generations, emulating the survival of the fittest. \
        Each solution is represented as a chromosome, composed of genes encoding problem-specific attributes. \
        Through genetic operators like crossover and mutation, new generations emerge, inheriting desirable traits \
        from their predecessors.
        """
    ),
    pn.Row(
        pn.pane.GIF("docs/img/ga_crossover.gif", width=300, height=180, align="center"),
        pn.pane.GIF("docs/img/ga_animation.gif", width=300, height=180, align="center"),
        align="center"
    ),
    pn.pane.Markdown(
        """
        GA's significance lies in its ability to tackle challenging real-life problems. In engineering, GA \
        optimizes the design of structures, such as creating aerodynamic shapes or enhancing energy efficiency. \
        In logistics and scheduling, GA finds optimal routes, allocating resources effectively and minimizing costs. \
        GA is also prominent in machine learning, assisting in feature selection and parameter optimization for \
        complex models, leading to better performance and generalization.

        Furthermore, GA revolutionizes the field of bioinformatics by analyzing DNA sequences, identifying genetic \
        markers, and understanding evolutionary relationships. In financial modeling, GA optimizes investment \
        strategies, adapting to changing market conditions and maximizing returns.
        """
    ), align="center"
)

pages = {
    "Welcome": welcome_page,
    "PSO": pso.run(),
    "GA": ga.run(),
}

def show(page):
    return pages[page]


starting_page = pn.state.session_args.get("page", [b"Welcome"])[0].decode()

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

page = pn.Column(pn.Column(
    pn.Row(menu),
    # pn.pane.GIF(r".\docs\img\pso_animation.gif", width=300, align="center"),
), welcome_message,)

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
