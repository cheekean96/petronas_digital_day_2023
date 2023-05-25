import panel as pn
from src.pso_panel import CreatePSOPanel
from src.ga_panel import CreateGAPanel

pn.extension(sizing_mode="stretch_width")

pso = CreatePSOPanel()
ga = CreateGAPanel()

pages = {
    "PSO": pso.run(),
    "GA": ga.run(),
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
