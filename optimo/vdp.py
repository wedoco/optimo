from OMPython import OMCSessionZMQ
import zipfile
from pathlib import Path
import casadi as cs
import rockit
import numpy as np



def loadFiles(omc, mo_files=[]):
    """
    Load required Modelica files and packages.

    :param mo_files: The list of paths to the involved Modelica models and libraries.

    """

    # Load needed model files and libraries.
    for f in mo_files:
        print('Loading {} ...'.format(f))
        omc.sendExpression('loadFile(\"{}\")'.format(f))

    print('List of defined Modelica class names: {}'.format(omc.sendExpression("getClassNames()")))

def buildModelFMU(omc, mo_class, commandLineOptions=None):
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
    # fmu_path = self.omc.sendExpression('buildModelFMU({0}, version=\"{1}\", fmuType=\"cs\")'.format(mo_class, self.fmu_version))

    return fmu_path

def explore_dae_builder(dae_builder):
    print(f"\nParameters and initial guesses ({dae_builder.np()}):")

    for p_name in dae_builder.p():
        print(f" * {p_name}: initial guess {dae_builder.get(p_name)}, nominal {dae_builder.nominal(p_name)}")

    print(f"\nStates and initial values ({dae_builder.nx()}):")

    for x_name in dae_builder.x():
        print(f' * {x_name}: initial value {dae_builder.get(x_name)}, nominal {dae_builder.nominal(x_name)}')

    print(f"\nControls and initial values ({dae_builder.nu()}):")

    for u_name in dae_builder.u():
        print(f' * {u_name}: initial values {dae_builder.get(u_name)}, nominal {dae_builder.nominal(u_name)}')

    print(f"\nOutputs and initial values ({dae_builder.ny()}):")

    for y_name in dae_builder.y():
        print(f' * {y_name}: initial values {dae_builder.get(y_name)}, nominal {dae_builder.nominal(y_name)}')


####
# Start of the transfer_model method. 
# Returns an optimo.opt object with attributes like the dae_builder. 
# Arguments:
fmu_version = 2.0
model='vdp'
omc = OMCSessionZMQ()
u_optimize = ['u']
T_horizon = 20
N = 100  # integration horizon
M = 1  # integrations steps per control interval
####

loadFiles(omc, mo_files=[f'{model}.mo'])
fmu_path = buildModelFMU(omc, f'{model}')
fmu_path = Path(f'{model}.fmu').resolve()

# Unzip FMU (we like to avoid a dependency on libz in CasADi)
unzipped_path = fmu_path.parent / fmu_path.stem
with zipfile.ZipFile(fmu_path, 'r') as zip_ref:
    zip_ref.extractall(unzipped_path)

# Parse FMU
dae_builder = cs.DaeBuilder('model', str(unzipped_path), {"debug": False})

explore_dae_builder(dae_builder)

# Check that all u_optimize exist in the model
for ui in u_optimize:
    assert ui in dae_builder.u()

# Extracting initial values for the model
p0 = dae_builder.get(dae_builder.p())
x0 = dae_builder.get(dae_builder.x())
u0 = dae_builder.get(dae_builder.u())
u_optimize0 = dae_builder.get(u_optimize)

# For debugging: inspect system
dae_builder.disp(True)

# Define 
f = dae_builder.create('f', ['x', 'u'], ['ode'])

nom = f(x=x0, u=u0)
print("Nominal: ", nom)

# Adding an extra output: outputs defined in the FMU
f = dae_builder.create('f', ['x', 'u'], ['ode', 'ydef'])

# Redefine the API of f
x = cs.vcat([dae_builder.var(name) for name in dae_builder.x()])
u = cs.vcat([dae_builder.var(name) for name in dae_builder.u()])
optimize = cs.vcat([dae_builder.var(name) for name in u_optimize])

f = cs.Function('f', [x, optimize], f(x, u), ['x', 'optimize'], ['ode', 'ydef'])

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
for x_label in dae_builder.x():
    # Pull symbol out of dae_builder
    vars[x_label] = dae_builder.var(x_label)

    # Let rockit know that this symbol is a state
    ocp.register_state(dae_builder.var(x_label), scale=dae_builder.nominal(x_label))

# Loop over all inputs to optimize
for u_label in u_optimize:
    # Pull symbol out of dae_builder
    vars[u_label] = dae_builder.var(u_label)

    # Let rockit know that this symbol is a control
    ocp.register_variable(dae_builder.var(u_label), scale=dae_builder.nominal(u_label))

# Perform a symbolic call to the system dynamics and output Function
#  - ocp.x: flattened vector of all states
#  - ocp.u: flattened vector of all controls
out = f(x=ocp.x, optimize=ocp.v)  # here p=ocp.v because we registered the inputs as variables

# Let rockit know what the state dynamics are
ocp.set_der(ocp.x, out["ode"])  # out['ode'] gives the derivative of the states.

# Store all symbolic expressions for outputs into vars
for y_label, e in zip(dae_builder.y(), cs.vertsplit(out["ydef"])):
    print(y_label)
    print(e)
    vars[y_label] = e

# Formulate objective
ocp.add_objective(ocp.integral(vars['objectiveIntegrand']))

# Set constraints
ocp.subject_to(-1 <= (vars['u'] <= 0.75))

# Set initial guesses for unknowns
for i, u_name in enumerate(u_optimize):
    ocp.set_initial(vars[u_name], u_optimize0[i])
for i, x_name in enumerate(dae_builder.x()):
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

sol_dict = {}
for var_name, var_value in vars.items():
    sol_dict[var_name] = sol.sample(var_value, grid="integrator")

    