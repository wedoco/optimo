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
    fig = make_subplots(rows=len(plot_def), cols=1, shared_xaxes=True, vertical_spacing=0.05)
    for i, p in enumerate(plot_def.keys()):
        fig.update_yaxes(title_text=p, row=i+1, col=1)
        offset = plot_def[p]['offset'] if 'offset' in plot_def[p].keys() is not None else 0
        factor = plot_def[p]['factor'] if 'factor' in plot_def[p].keys() is not None else 1
        for v in plot_def[p]['vars']:
            fig.add_trace(go.Scatter(x=df.index, name=v,
                                    y=df[v]*factor+offset), row=i+1, col=1)
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

def get_dae_results(tgrid, dae, x_sim: np.array, t0: int) -> dict:
    """
    Return the simulation results as a dictionary with the right keywords.

    Parameters
    ----------
    x_sim: np.array
        Model state trajectories.
    y_sim: np.array
        Model output trajectories.
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

    # for i, yi in enumerate(dae.y()):
    #     y_name = str(yi)
    #     y_array = np.array(y_sim[i, :].T)
    #     res[y_name] = y_array

    return res

####
# Start of the transfer_model method. 
# Returns an optimo.opt object with attributes like the dae. 
# Arguments:
fmu_version = 2.0
model='vdp'
force_recompile=False
omc = OMCSessionZMQ()

T_horizon = 20 # prediction horizon in seconds
N = 100  # number of integration steps in the prediction horizon
M = 1  # number of integrations steps per control interval

u_ext_sim = np.zeros((1, N+1))
x_ext_0 = np.array([2, 0])

####

# Compile the FMU if needed
if force_recompile:
    load_modelica_files(omc, modelica_files=[f'{model}.mo'])
    build_model_fmu(omc, f'{model}')
fmu_path = str(Path(f'{model}.fmu').resolve())

# Parse FMU to dae object
dae = ca.DaeBuilder("model", unpack_fmu(fmu_path), {"debug": False})

explore_dae(dae)

# Get external values if provided. Otherwise use those from the model
p0 = dae.get(dae.p())
x_ext_0   = x_ext_0 if x_ext_0 is not None else dae.get(dae.x())
u_ext_sim = u_ext_sim if u_ext_sim is not None else dae.get(dae.u())*np.ones((1, N+1))

# Extract symbols for states and inputs
x = ca.vcat([dae.var(name) for name in dae.x()])
u = ca.vcat([dae.var(name) for name in dae.u()])

# Perform a symbolic call to the system dynamics and output Function
f_ode = dae.create("f_ode", ["x", "u"], ['ode'])
out_ode = f_ode(x=x, u=u)  

dae_dict = {}
dae_dict["x"] = x
dae_dict["u"] = u
dae_dict["ode"]  = out_ode["ode"]
opts = {}
opts["print_stats"] = False

t0 = 0
dt_input = T_horizon / N
dt_output = dt_input / (N - 1) * N
tgrid = np.asarray([T_horizon / N * k for k in range(N + 1)])

sim_function = ca.integrator("simulator", "cvodes", dae_dict, 0, tgrid, opts)

res_sim = sim_function(x0=x_ext_0, u=u_ext_sim)
x_sim = res_sim["xf"].full()
# y_sim = res_sim["yf"].full()
res = get_dae_results(tgrid, dae, x_sim, t0)

df = pd.DataFrame(res)

plot_def = {}
plot_def['x1'] = {}
plot_def['x1']['vars'] = ['x1']
plot_def['x2'] = {}
plot_def['x2']['vars'] = ['x2']
# plot_def['u'] = {}
# plot_def['u']['vars'] = ['u']

plot_from_def(plot_def, df, show=False, save_to_file=True, filename='plot.html')





# Define rockit ocp problem
ocp = rockit.Ocp(T=T_horizon)

# Choose transcription method
# ocp.method(rockit.SingleShooting(N=N, M=M))
ocp.method(rockit.MultipleShooting(N=N, M=M))
# ocp.method(rockit.DirectCollocation(N=N, M=M))

# To make mumps behave more like ma57: "ipopt.mumps_permuting_scaling":0,"ipopt.mumps_scaling":0
# After scaling manually, it as good idea to turn off ipopt scaling: "ipopt.nlp_scaling_method": none
ocp.solver("ipopt", {"ipopt.hessian_approximation": "limited-memory"})

# Loop over all states
for x_name in dae.x():
    # Let rockit know that this symbol is a state
    ocp.register_state(dae.var(x_name), scale=dae.nominal(x_name))

# Loop over all inputs to optimize
for u_name in dae.u():
    # Let rockit know that this symbol is a control
    ocp.register_parameter(dae.var(u_name), scale=dae.nominal(u_name))

# Loop over all outputs
for y_name in dae.y():
    # Let rockit know that this symbol is a control
    ocp.register_variable(dae.var(y_name), scale=dae.nominal(y_name))

# Perform a symbolic call to the system dynamics and output Function
f = dae.create("f", ["x", "u"], ['ode', 'ydef'])
out = f(x=ocp.x, u=ocp.u)  

# Let rockit know what the state dynamics are
ocp.set_der(ocp.x, out["ode"])  # out['ode'] gives the derivative of the states.

# Store all symbolic expressions for outputs 
y_sym = {}
for y_name, expression in zip(dae.y(), ca.vertsplit(out["ydef"])):
    y_sym[y_name] = expression

# Set initial guesses for unknowns
for i, u_name in enumerate(dae.u()):
    ocp.set_value(dae(u_name), u0[i])
for i, x_name in enumerate(dae.x()):
    ocp.set_initial(dae(x_name), x0[i])

ocp.set_t0(0)

sol = ocp.solve()

sol_dict = {'time': sol.sample(ocp.t, grid="integrator")[1]}
for var_name in dae.x()+dae.u()+dae.y():
    sol_dict[var_name] = sol.sample(dae(var_name), grid="integrator")[1]

df = pd.DataFrame(sol_dict)

# Formulate objective
ocp.add_objective(ocp.integral(y_sym['objectiveIntegrand']))

# Formulate objective
# ocp.add_objective(ocp.integral(dae('objectiveIntegrand')))

# Set constraints
ocp.subject_to(-1 <= (dae('u') <= 0.75))

# Solve ocp
def check_iterations(iter, sol):
    print(iter)
    print(sol.value(dae('u')))

ocp.callback(check_iterations)

try:
    sol = ocp.solve()
except:
    print("Solution not converged!!")
    ocp.show_infeasibilities(1e-7)
    sol = ocp.non_converged_solution
