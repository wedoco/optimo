import pandas as pd
import os
import numpy as np

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
