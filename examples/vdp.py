import os
import numpy as np
import matplotlib.pyplot as plt
from wedoco_optimo.model import OptimoModel


def run_vdp(force_recompile=True, plot=True):
    # Compile and transfer the Modelica model
    mo = OptimoModel()
    model_file = os.path.join(os.path.dirname(__file__), "vdp.mo")
    mo.transfer_model(model="vdp", modelica_files=[model_file], force_recompile=force_recompile)

    # Simulate
    res_sim_df = mo.simulate()

    # Plot 
    if plot:
        _, axs = plt.subplots(3, 1)
        res_sim_df[["x1","x2"]].plot(ax=axs[0], title="States", legend=True)
        res_sim_df["objectiveIntegrand"].plot(ax=axs[1], title="Objective", legend=True)
        res_sim_df["u"].plot(ax=axs[2], title="Input", legend=True)
        plt.show()

    # Optimize 
    mo.define_optimization(constraints={"u":(-1, 0.75)}, 
                           objective_terms=["objectiveIntegrand"])
    res_ocp_df = mo.optimize()

    # Plot 
    if plot:
        _, axs = plt.subplots(3, 1)
        res_ocp_df[["x1","x2"]].plot(ax=axs[0], title="States", legend=True)
        res_ocp_df["objectiveIntegrand"].plot(ax=axs[1], title="Objective", legend=True)
        res_ocp_df["u"].plot(ax=axs[2], title="Input", legend=True)
        plt.show()

    return res_sim_df, res_ocp_df

def run_vdp_simulate_with_input(force_recompile=False, plot=True):
    # Compile and transfer the Modelica model
    mo = OptimoModel()
    model_file = os.path.join(os.path.dirname(__file__), "vdp.mo")
    mo.transfer_model(model="vdp", modelica_files=[model_file], force_recompile=force_recompile)

    # Simulate
    u_sim = np.sin(mo.tgrid)
    res_sim_df = mo.simulate(u_sim=u_sim)

    # Plot 
    if plot:
        _, axs = plt.subplots(3, 1)
        res_sim_df[["x1","x2"]].plot(ax=axs[0], title="States", legend=True)
        res_sim_df["objectiveIntegrand"].plot(ax=axs[1], title="Objective", legend=True)
        res_sim_df["u"].plot(ax=axs[2], title="Input", legend=True)
        plt.show()

    return res_sim_df

if __name__ == "__main__":
    run_vdp()