import pandas as pd
import os
import numpy as np
import shutil
import numpy as np

def load_modelica_files(omc, modelica_files=[]):
    """
    Load required Modelica files and packages.

    :param mo_files: The list of paths to the involved Modelica models and libraries.

    """

        
    # Load needed model files and libraries.
    for f in modelica_files:
        print('Loading {} ...'.format(f))
        if not os.path.exists(f):
            f = find_file_in_modelicapath(f)
        file_loaded = omc.loadFile(f)
        if file_loaded.startswith('false'):
          raise Exception('Modelica compilation failed: {}'.format(omc.sendExpression('getErrorString()')))
        # omc.sendExpression('loadFile(\"{}\")'.format(f))

    print('List of defined Modelica class names: {}'.format(omc.sendExpression("getClassNames()")))

def find_file_in_modelicapath(filename):
    """
    Find a file in the MODELICAPATH environment variable.

    """
    modelicapath = os.environ.get('MODELICAPATH', '')
    directories = modelicapath.split(os.pathsep)
    
    for directory in directories:
        for root, _, files in os.walk(directory):
            if filename in files:
                return os.path.abspath(os.path.join(root, filename))
    
    return None

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
    fmu_version = 2.0
    # The following actually enables directional derivatives
    omc.sendExpression('setDebugFlags("-disableDirectionalDerivatives")')  
    # Compile the FMU
    fmu_path = omc.sendExpression('buildModelFMU({0}, version=\"{1}\")'.format(mo_class, fmu_version))
    flag = omc.sendExpression('getErrorString()')
    if not fmu_path.endswith('.fmu'): raise Exception(f'FMU generation failed: {flag}')
    print(f"translateModelFMU warnings:\n{flag}")

    # fmu_path = self.omc.sendExpression('buildModelFMU({0}, version=\"{1}\", fmuType=\"cs\")'.format(mo_class, self.fmu_version))

    return fmu_path

def explore_dae(dae):
    print(f"\nParameters and initial guesses ({dae.np()}):")

    for p_name in dae.p():
        print(f" * {p_name}: initial guess {dae.start(p_name)}, minimum {dae.min(p_name)}, maximum {dae.max(p_name)}, nominal {dae.nominal(p_name)}")

    print(f"\nStates and initial values ({dae.nx()}):")

    for x_name in dae.x():
        print(f' * {x_name}: initial value {dae.start(x_name)}, minimum {dae.min(x_name)}, maximum {dae.max(x_name)}, nominal {dae.nominal(x_name)}')

    print(f"\nControls and initial values ({dae.nu()}):")

    for u_name in dae.u():
        print(f' * {u_name}: initial values {dae.start(u_name)}, minimum {dae.min(u_name)}, maximum {dae.max(u_name)}, nominal {dae.nominal(u_name)}')

    print(f"\nOutputs and initial values ({dae.ny()}):")

    for y_name in dae.y():
        print(f' * {y_name}: initial values {dae.start(y_name)}, minimum {dae.min(y_name)}, maximum {dae.max(y_name)}, nominal {dae.nominal(y_name)}')

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

    res_df = pd.DataFrame(res)

    return res_df
