from OMPython import OMCSessionZMQ
import zipfile
from pathlib import Path
import casadi as ca
import rockit
import pandas as pd
import os
import numpy as np
import shutil
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

def plot_from_def(plot_def, df, show=True, save_to_file=False, filename='plot.html'):
    fig = make_subplots(rows=len(plot_def)-1, cols=1, shared_xaxes=True, vertical_spacing=0.02)
    for i, p in enumerate(p for p in plot_def.keys() if p != 'x'):
        fig.update_yaxes(title_text=p, row=i+1, col=1)
        offset = plot_def[p]['offset'] if 'offset' in plot_def[p].keys() is not None else 0
        factor = plot_def[p]['factor'] if 'factor' in plot_def[p].keys() is not None else 1
        for v in plot_def[p]['vars']:
            fig.add_trace(go.Scatter(x=plot_def['x']['values'], name=v,
                                     y=df[v]*factor+offset), row=i+1, col=1)
            
    # Show x-axis title and ticks only on the last subplot
    fig.update_xaxes(title_text=plot_def['x']['title'], row=i+1, col=1, showticklabels=True)
    # Update layout to show a shared vertical line across all subplots when hovering
    fig.update_layout(hovermode="x unified", hoversubplots='axis')
    
    fig.update_layout(height=800)
    if show: 
        fig.show()
    if save_to_file:
        pio.write_html(fig, file=filename, auto_open=False)

def load_modelica_files(omc, modelica_files=[]):
    """
    Load required Modelica files and packages.

    :param mo_files: The list of paths to the involved Modelica models and libraries.

    """

        
    # Load needed model files and libraries.
    for f in modelica_files:
        print('Loading {} ...'.format(f))
        if omc.loadFile(f).startswith('false'):
          raise Exception('Modelica compilation failed: {}'.format(omc.sendExpression('getErrorString()')))
        # omc.sendExpression('loadFile(\"{}\")'.format(f))

    print('List of defined Modelica class names: {}'.format(omc.sendExpression("getClassNames()")))

def build_model_fmu(omc, mo_class, commandLineOptions=None):
    """
    Compile an FMU from a Modelica model.

    :param mo_class: The Modelica class name to be compiled.
    :return: Full path of the generated FMU.

    """

    ## set commandLineOptions if provided by users
    if commandLineOptions is not None:
        exp = "".join(["setCommandLineOptions(", "\"", commandLineOptions, "\"", ")"])
        print(exp)
        # self.omc.sendExpression('setCommandLineOptions(\"+g=Modelica\")')
        print(omc.sendExpression(exp))

    # Translate model to FMU.
    fmu_path = omc.sendExpression('buildModelFMU({0}, version=\"{1}\")'.format(mo_class, fmu_version))
    flag = omc.sendExpression('getErrorString()')
    if not fmu_path.endswith('.fmu'): raise Exception(f'FMU generation failed: {flag}')
    print(f"translateModelFMU warnings:\n{flag}")

    # fmu_path = self.omc.sendExpression('buildModelFMU({0}, version=\"{1}\", fmuType=\"cs\")'.format(mo_class, self.fmu_version))

    return fmu_path

def unpack_fmu(fmu_file):
  """
  Unpack the contents of an FMU file
  """
  # To create a directory, strip the .fmu ending from the fmu_file and add a timestamp
  from datetime import datetime
  suffix = datetime.now().strftime("_%Y-%m-%d_%H-%M-%S")
  unpacked_fmu = os.path.join(os.getcwd(),fmu_file[:fmu_file.find('.')] + suffix)
  # Unzip
  import zipfile
  with zipfile.ZipFile(fmu_file, 'r') as zip_ref: zip_ref.extractall(unpacked_fmu)
  print(f'Unpacked {fmu_file} into {unpacked_fmu}')
  return unpacked_fmu

def cleanup_fmu(unpacked_fmu):
  shutil.rmtree(unpacked_fmu)

def explore_dae(dae):
    print(f"\nParameters and initial guesses ({dae.np()}):")

    for p_name in dae.p():
        print(f" * {p_name}: initial guess {dae.get(p_name)}, nominal {dae.nominal(p_name)}")

    print(f"\nStates and initial values ({dae.nx()}):")

    for x_name in dae.x():
        print(f' * {x_name}: initial value {dae.get(x_name)}, nominal {dae.nominal(x_name)}')

    print(f"\nControls and initial values ({dae.nu()}):")

    for u_name in dae.u():
        print(f' * {u_name}: initial values {dae.get(u_name)}, nominal {dae.nominal(u_name)}')

    print(f"\nOutputs and initial values ({dae.ny()}):")

    for y_name in dae.y():
        print(f' * {y_name}: initial values {dae.get(y_name)}, nominal {dae.nominal(y_name)}')

def get_dae_results(tgrid, dae, x_sim: np.array, y_sim: np.array, u_sim: np.array, t0: int) -> dict:
    """
    Return the simulation results as a dictionary with the right keywords.

    Parameters
    ----------
    x_sim: np.array
        Model state trajectories.
    y_sim: np.array
        Model output trajectories.
    u_sim: np.array
        Model input trajectories.
    t0: integer
        Initial time. Used to return the right time vector.

    Returns
    -------
    res: dictionary
        Result trajectories as {res_variable: values}
    """
    res = {}

    # Move the time to the right starting point
    time_array = np.array(t0 + tgrid)
    res["time"] = time_array

    # Get the states
    for i, xi in enumerate(dae.x()):
        x_name = str(xi)
        x_array = np.array(x_sim[i, :].T)
        res[x_name] = x_array

    for i, yi in enumerate(dae.y()):
        y_name = str(yi)
        y_array = np.array(y_sim[i, :].T)
        res[y_name] = y_array

    for i, ui in enumerate(dae.u()):
        u_name = str(ui)
        u_array = np.array(u_sim[i, :].T)
        res[u_name] = u_array

    return res

####
# Start of the transfer_model method. 
# Returns an optimo.opt object with attributes like the dae. 
# Arguments:
fmu_version = 2.0
model='vdp'
force_recompile=False
omc = OMCSessionZMQ()

T_horizon = 10 # prediction horizon in seconds
N = 100  # number of integration steps in the prediction horizon
M = 1  # number of integrations steps per control interval

t0 = 0
tgrid = np.asarray([T_horizon / N * k for k in range(N + 1)])

# u_ext_sim = np.cos(tgrid).reshape(1, N+1)*0.2
# u_ext_sim = np.ones((1, N+1))*0.2
# x_ext_0 = np.array([2, 0])
u_ext_sim = None
x_ext_0 = None

objective_terms = ['objectiveIntegrand']

####

# Compile the FMU if needed
if not os.path.exists(f'{model}.fmu') or force_recompile:
    load_modelica_files(omc, modelica_files=[f'{model}.mo'])
    build_model_fmu(omc, f'{model}')
fmu_path = str(Path(f'{model}.fmu').resolve())

# Parse FMU to dae object
dae = ca.DaeBuilder("model", unpack_fmu(fmu_path), {"debug": False})

explore_dae(dae)

# Extract symbols for states and inputs
x = ca.vcat([dae.var(name) for name in dae.x()])
u = ca.vcat([dae.var(name) for name in dae.u()])
y = ca.vcat([dae.var(name) for name in dae.y()])

# Get external values if provided. Otherwise use those from the model
x_ext_0   = x_ext_0 if x_ext_0 is not None else dae.get(dae.x())
u_ext_sim = u_ext_sim if u_ext_sim is not None else dae.get(dae.u())*np.ones((1, N+1))

# Define symbolic function from states and inputs to the system outputs and states
# The inputs are also returned to read them as they are perceived by the model
f_xu_xyu = dae.create("f_xu_xyu", ["x", "u"], ["ode", "ydef", "u"])

plot_def = {}
plot_def['x'] = {}
plot_def['x']['values'] = tgrid
plot_def['x']['title'] = 'Time (s)'
plot_def['Angle (rad)'] = {}
plot_def['Angle (rad)']['vars'] = ['x1', 'x2']
plot_def['Outputs'] = {}
plot_def['Outputs']['vars'] = ['objectiveIntegrand']
plot_def['Inputs'] = {}
plot_def['Inputs']['vars'] = ['u']

simulate = True
if simulate:
    dae_dict = {}
    dae_dict["x"] = x
    dae_dict["u"] = u
    dae_dict["ode"]  = f_xu_xyu(x=x, u=u)["ode"]
    opts = {}
    opts["print_stats"] = False

    f_sim = ca.integrator("simulator", "cvodes", dae_dict, 0, tgrid, opts)

    # Simuilate the model dynamics
    res_x_sim = f_sim(x0=x_ext_0, u=u_ext_sim)
    x_sim = res_x_sim["xf"].full()

    # Now evaluate the outputs from inputs and computed dynamics
    res_xyu_sim = f_xu_xyu(x=x_sim, u=u_ext_sim)
    y_sim = res_xyu_sim["ydef"].full()
    u_sim = res_xyu_sim["u"].full()

    res = get_dae_results(tgrid, dae, x_sim, y_sim, u_sim, t0)

    df = pd.DataFrame(res)

    plot_from_def(plot_def, df, show=False, save_to_file=True, filename='plot_sim.html')


optimize = True 
if optimize:

    # Define rockit ocp problem
    ocp = rockit.Ocp(T=T_horizon)

    # Choose transcription method
    # ocp.method(rockit.SingleShooting(N=N, M=M))
    # ocp.method(rockit.MultipleShooting(N=N, M=M))
    ocp.method(rockit.DirectCollocation(N=N, M=M))

    ipopt_options = {
        "ipopt.max_iter": 500,
        "ipopt.tol": 1e-6
    }

    ocp.solver("ipopt", ipopt_options)

    # Loop over all states
    for x_name in dae.x():
        # Let rockit know that this symbol is a state
        ocp.register_state(dae.var(x_name), scale=dae.nominal(x_name))

    # Loop over all inputs to optimize
    for u_name in dae.u():
        # Let rockit know that this symbol is a control
        ocp.register_control(dae.var(u_name), scale=dae.nominal(u_name))

    # Let rockit know what the state dynamics are
    ocp.set_der(x, f_xu_xyu(x=x, u=u)["ode"])  

    for i, x_name in enumerate(dae.x()):
        ocp.subject_to(ocp.at_t0(dae.var(x_name)) == x_ext_0[i])

    ocp.set_t0(0)

    # Formulate objective from objective terms list
    for term in objective_terms:
        ocp.add_objective(ocp.integral(f_xu_xyu(x=x, u=u)['ydef'][dae.y().index(term)]))

    # Set constraints
    ocp.subject_to(-1 <= (dae.var('u') <= 0.75))

    # Solve the optimization problem
    sol = ocp.solve()

    # Extract the results
    x_ocp = np.atleast_2d(sol.sample(ocp.x, grid="integrator")[1].T)
    u_ocp = np.atleast_2d(sol.sample(ocp.u, grid="integrator")[1].T)
    res_xyu_ocp = f_xu_xyu(x=x_ocp, u=u_ocp)
    y_ocp = res_xyu_ocp["ydef"].full()
    u_ocp = res_xyu_ocp["u"].full()
    res = get_dae_results(tgrid, dae, x_ocp, y_ocp, u_ocp, t0)
    df = pd.DataFrame(res)

    plot_from_def(plot_def, df, show=False, save_to_file=True, filename='plot_ocp.html')
