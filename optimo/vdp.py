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


### PSEUDOCODE OF WHAT WE WANT TO ACHIEVE ###
# mo = transfer_model('vdp', force_recompile=True)

# t  = np.linspace(0.,10.,100) # Create one hundred evenly spaced points 
# u1 = np.sin(t)               # Create the first input vector 
# u2 = np.cos(t)               # Create the second input vector 
# u_traj = np.transpose(np.vstack((t,u1,u2))) # Create the data matrix and # transpose it to the correct form 
# input_object = (['u1','u2'], u_traj) # Define the input object

# res_sim = mo.simulate(final_time=10, input=input_object)

# mo.set_init_traj(res_sim)   # Set the initial trajectory for the optimization
# d = np.tan(t)               # Create the disturbance vector
# mo.set_external_data(d)     # Set the disturbance vector

# res_opt = mo.optimize()

# mpc = mo.build_mpc(m_map={'m1_model': 'm1_sys'})        # Build the MPC controller
# sys = initialize_sys()                                  # Initialize the system

# m = sys.reset()
# for i in range(100):
#     d = sys.get_disturbance_forecast()
#     u = mpc.get_control(m, d)
#     m = sys.step(u)

# mpc.show_stats()


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
N = 500  # number of integration steps in the prediction horizon
M = 1  # number of integrations steps per control interval

t0 = 0
tgrid = np.asarray([T_horizon / N * k for k in range(N + 1)])

# u_ext_sim = np.cos(tgrid).reshape(1, N+1)*0.2
# u_ext_sim = np.ones((1, N+1))*0.2
# x_ext_0 = np.array([2, 0])
u_ext_sim = None
x_ext_0 = None

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

# Define symbolic function from states and inputs to the system outputs and states
# The inputs are also returned to read them as they are perceived by the model
f_xu_xyu = dae.create("f_y", ["x", "u"], ["ode", "ydef", "u"])

out_f = f_xu_xyu(x=x, u=u)  

dae_dict = {}
dae_dict["x"] = x
dae_dict["u"] = u
dae_dict["ode"]  = out_f["ode"]
opts = {}
opts["print_stats"] = False

sim_function = ca.integrator("simulator", "cvodes", dae_dict, 0, tgrid, opts)

# Get external values if provided. Otherwise use those from the model
x_ext_0   = x_ext_0 if x_ext_0 is not None else dae.get(dae.x())
u_ext_sim = u_ext_sim if u_ext_sim is not None else dae.get(dae.u())*np.ones((1, N+1))

# Simuilate the model dynamics
res_x_sim = sim_function(x0=x_ext_0, u=u_ext_sim)
x_sim = res_x_sim["xf"].full()

# Now evaluate the outputs from inputs and computed dynamics
res_xyu_sim = f_xu_xyu(x=x_sim, u=u_ext_sim)
y_sim = res_xyu_sim["ydef"].full()
u_sim = res_xyu_sim["u"].full()

# y_sim = res_sim["yf"].full()
res = get_dae_results(tgrid, dae, x_sim, y_sim, u_sim, t0)

df = pd.DataFrame(res)

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

plot_from_def(plot_def, df, show=False, save_to_file=True, filename='plot_sim.html')





T_horizon = 20 # prediction horizon in seconds
N = 500  # number of integration steps in the prediction horizon
M = 1  # number of integrations steps per control interval

t0 = 0
tgrid = np.asarray([T_horizon / N * k for k in range(N + 1)])

# Define rockit ocp problem
ocp = rockit.Ocp(T=T_horizon)

# Choose transcription method
ocp.method(rockit.SingleShooting(N=N, M=M))
# ocp.method(rockit.MultipleShooting(N=N, M=M))
# ocp.method(rockit.DirectCollocation(N=N, M=M))

ipopt_options = {
    "ipopt.hessian_approximation": "limited-memory",
    "ipopt.max_iter": 500,
    "ipopt.tol": 1e-6,
    "ipopt.hsllib": str((Path(__file__).parent / "solvers" / "libhsl.so").resolve()),
}

# To make mumps behave more like ma57: "ipopt.mumps_permuting_scaling":0,"ipopt.mumps_scaling":0
# After scaling manually, it as good idea to turn off ipopt scaling: "ipopt.nlp_scaling_method": none
ocp.solver("ipopt", ipopt_options)

# Loop over all states
for x_name in dae.x():
    # Let rockit know that this symbol is a state
    ocp.register_state(dae.var(x_name), scale=dae.nominal(x_name))

# Loop over all inputs to optimize
for u_name in dae.u():
    # Let rockit know that this symbol is a control
    ocp.register_control(dae.var(u_name), scale=dae.nominal(u_name))

# Loop over all outputs
for y_name in dae.y():
    # Let rockit know that this symbol is a control
    ocp.register_variable(dae.var(y_name), scale=dae.nominal(y_name))

# Perform a symbolic call to the system dynamics and output Function
out_ocp = f_xu_xyu(x=ocp.x, u=ocp.u)  

# Let rockit know what the state dynamics are
ocp.set_der(ocp.x, out_ocp["ode"])  # out_ocp['ode'] gives the derivative of the states.

# Store all symbolic expressions for outputs 
y_sym = {}
for y_name, expression in zip(dae.y(), ca.vertsplit(out_ocp["ydef"])):
    y_sym[y_name] = expression

# Set initial guesses for unknowns
# for i, u_name in enumerate(dae.u()):
#     ocp.set_value(dae(u_name), u_ext_sim[i])
for i, x_name in enumerate(dae.x()):
    ocp.set_initial(dae(x_name), x_ext_0[i])

ocp.set_t0(0)


# Formulate objective
ocp.add_objective(ocp.integral(y_sym['objectiveIntegrand']))

# sol = ocp.solve()

# Formulate objective
# ocp.add_objective(ocp.integral(dae('objectiveIntegrand')))

# Set constraints
ocp.subject_to(-1 <= (dae('u') <= 0.75))

# Solve ocp
def check_iterations(iter, sol):
    print(iter)
    print(sol.value(dae('u')))

# ocp.callback(check_iterations)

try:
    sol = ocp.solve()
except:
    print("Solution not converged!!")
    ocp.show_infeasibilities(1e-7)
    sol = ocp.non_converged_solution

sol_dict = {'time': sol.sample(ocp.t, grid="integrator")[1]}
for var_name in dae.x()+dae.u()+dae.y():
    sol_dict[var_name] = sol.sample(dae(var_name), grid="integrator")[1]

# x_sim = sol.sample(ocp.x, grid="integrator")
# u_sim = sol.sample(ocp.u, grid="integrator")
# y_sim = sol.sample(ocp.v, grid="integrator")
# res = get_dae_results(tgrid, dae, x_sim, y_sim, u_sim, t0)

df = pd.DataFrame(sol_dict)


plot_from_def(plot_def, df, show=False, save_to_file=True, filename='plot_ocp.html')
