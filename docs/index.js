importScripts("https://cdn.jsdelivr.net/pyodide/v0.22.1/full/pyodide.js");

function sendPatch(patch, buffers, msg_id) {
  self.postMessage({
    type: 'patch',
    patch: patch,
    buffers: buffers
  })
}

async function startApplication() {
  console.log("Loading pyodide!");
  self.postMessage({type: 'status', msg: 'Loading pyodide'})
  self.pyodide = await loadPyodide();
  self.pyodide.globals.set("sendPatch", sendPatch);
  console.log("Loaded!");
  await self.pyodide.loadPackage("micropip");
  const env_spec = ['https://cdn.holoviz.org/panel/0.14.4/dist/wheels/bokeh-2.4.3-py3-none-any.whl', 'https://cdn.holoviz.org/panel/0.14.4/dist/wheels/panel-0.14.4-py3-none-any.whl', 'pyodide-http==0.1.0', 'ga_panel', 'pso_panel']
  for (const pkg of env_spec) {
    let pkg_name;
    if (pkg.endsWith('.whl')) {
      pkg_name = pkg.split('/').slice(-1)[0].split('-')[0]
    } else {
      pkg_name = pkg
    }
    self.postMessage({type: 'status', msg: `Installing ${pkg_name}`})
    try {
      await self.pyodide.runPythonAsync(`
        import micropip
        await micropip.install('${pkg}');
      `);
    } catch(e) {
      console.log(e)
      self.postMessage({
	type: 'status',
	msg: `Error while installing ${pkg_name}`
      });
    }
  }
  console.log("Packages loaded!");
  self.postMessage({type: 'status', msg: 'Executing code'})
  const code = `
  
import asyncio

from panel.io.pyodide import init_doc, write_doc

init_doc()

import panel as pn
from pso_panel import CreatePSOPanel
from ga_panel import CreateGAPanel

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


await write_doc()
  `

  try {
    const [docs_json, render_items, root_ids] = await self.pyodide.runPythonAsync(code)
    self.postMessage({
      type: 'render',
      docs_json: docs_json,
      render_items: render_items,
      root_ids: root_ids
    })
  } catch(e) {
    const traceback = `${e}`
    const tblines = traceback.split('\n')
    self.postMessage({
      type: 'status',
      msg: tblines[tblines.length-2]
    });
    throw e
  }
}

self.onmessage = async (event) => {
  const msg = event.data
  if (msg.type === 'rendered') {
    self.pyodide.runPythonAsync(`
    from panel.io.state import state
    from panel.io.pyodide import _link_docs_worker

    _link_docs_worker(state.curdoc, sendPatch, setter='js')
    `)
  } else if (msg.type === 'patch') {
    self.pyodide.runPythonAsync(`
    import json

    state.curdoc.apply_json_patch(json.loads('${msg.patch}'), setter='js')
    `)
    self.postMessage({type: 'idle'})
  } else if (msg.type === 'location') {
    self.pyodide.runPythonAsync(`
    import json
    from panel.io.state import state
    from panel.util import edit_readonly
    if state.location:
        loc_data = json.loads("""${msg.location}""")
        with edit_readonly(state.location):
            state.location.param.update({
                k: v for k, v in loc_data.items() if k in state.location.param
            })
    `)
  }
}

startApplication()