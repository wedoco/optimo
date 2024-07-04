from OMPython import OMCSessionZMQ
import zipfile
from pathlib import Path
import casadi as ca
import rockit
import pandas as pd
import os
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


####
# Start of the transfer_model method. 
# Returns an optimo.opt object with attributes like the dae. 
# Arguments:
fmu_version = 2.0
model='vdp'
force_recompile=False
omc = OMCSessionZMQ()

T_horizon = 20
N = 100  # integration horizon
M = 1  # integrations steps per control interval
####

# Compile the FMU if needed
if force_recompile:
    load_modelica_files(omc, modelica_files=[f'{model}.mo'])
    build_model_fmu(omc, f'{model}')
fmu_path = str(Path(f'{model}.fmu').resolve())

# Parse FMU to dae object
dae = ca.DaeBuilder("model", unpack_fmu(fmu_path), {"debug": False})

explore_dae(dae)

# Extracting initial values for the model
p0 = dae.get(dae.p())
x0 = dae.get(dae.x())
u0 = dae.get(dae.u())

# Redefine the API of f
x = ca.vcat([dae.var(name) for name in dae.x()])
u = ca.vcat([dae.var(name) for name in dae.u()])

f = dae.create("f", ["x", "u"], ['ode', 'ydef'])

# Define rockit ocp problem
ocp = rockit.Ocp(T=T_horizon)

# Choose transcription method
# ocp.method(rockit.SingleShooting(N=N, M=M))
ocp.method(rockit.MultipleShooting(N=N, M=M))
# ocp.method(rockit.DirectCollocation(N=N, M=M))

# To make mumps behave more like ma57: "ipopt.mumps_permuting_scaling":0,"ipopt.mumps_scaling":0
# After scaling manually, it as good idea to turn off ipopt scaling: "ipopt.nlp_scaling_method": none
ocp.solver("ipopt", {"ipopt.hessian_approximation": "limited-memory"})

# Collection of CasADi symbols and symbolic expressions that capture variables of the model
vars = {}

# Loop over all states
for x_name in dae.x():
    # Pull symbol out of dae
    vars[x_name] = dae.var(x_name)

    # Let rockit know that this symbol is a state
    ocp.register_state(dae.var(x_name), scale=dae.nominal(x_name))

# Loop over all inputs to optimize
for u_name in dae.u():
    # Pull symbol out of dae
    vars[u_name] = dae.var(u_name)

    # Let rockit know that this symbol is a control
    ocp.register_variable(dae.var(u_name), scale=dae.nominal(u_name))

# Perform a symbolic call to the system dynamics and output Function
out = f(x=ocp.x, u=ocp.v)  

# Let rockit know what the state dynamics are
ocp.set_der(ocp.x, out["ode"])  # out['ode'] gives the derivative of the states.

# Store all symbolic expressions for outputs into vars
for y_name, e in zip(dae.y(), ca.vertsplit(out["ydef"])):
    vars[y_name] = e

# Formulate objective
ocp.add_objective(ocp.integral(vars['objectiveIntegrand']))

# Set constraints
ocp.subject_to(-1 <= (vars['u'] <= 0.75))

# Set initial guesses for unknowns
for i, u_name in enumerate(dae.u()):
    ocp.set_initial(vars[u_name], u0[i])
for i, x_name in enumerate(dae.x()):
    ocp.set_initial(vars[x_name], x0[i])

# Solve ocp
def check_iterations(iter, sol):
    print(iter)
    print(sol.value(vars['u']))

ocp.callback(check_iterations)

try:
    sol = ocp.solve()
except:
    print("Solution not converged!!")
    ocp.show_infeasibilities(1e-7)
    sol = ocp.non_converged_solution

sol_dict = {'time': sol.sample(ocp.t, grid="integrator")[1]}
for var_name, var_value in vars.items():
    sol_dict[var_name] = sol.sample(var_value, grid="integrator")[1]

df = pd.DataFrame(sol_dict)

plot_def = {}
plot_def['x1'] = {}
plot_def['x1']['vars'] = ['x1']
plot_def['x2'] = {}
plot_def['x2']['vars'] = ['x2']
plot_def['u'] = {}
plot_def['u']['vars'] = ['u']

plot_from_def(plot_def, df, show=False, save_to_file=True, filename='plot.html')
