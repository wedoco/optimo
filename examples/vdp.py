import matplotlib.pyplot as plt
from optimo.model import OptimoModel

# Compile and transfer the Modelica model
mo = OptimoModel()
mo.transfer_model(model="vdp")

# Simulate
res_sim_df = mo.simulate()

# Plot 
fig, axs = plt.subplots(3, 1)
res_sim_df[["x1","x2"]].plot(ax=axs[0], title="States", legend=True)
res_sim_df["objectiveIntegrand"].plot(ax=axs[1], title="Objective", legend=True)
res_sim_df["u"].plot(ax=axs[2], title="Input", legend=True)
plt.show()

# Optimize 
mo.define_optimization(constraints={"u":(-1, 0.75)}, 
                       objective_terms=["objectiveIntegrand"])
res_ocp_df = mo.optimize()

# Plot 
fig, axs = plt.subplots(3, 1)
res_ocp_df[["x1","x2"]].plot(ax=axs[0], title="States", legend=True)
res_ocp_df["objectiveIntegrand"].plot(ax=axs[1], title="Objective", legend=True)
res_ocp_df["u"].plot(ax=axs[2], title="Input", legend=True)
plt.show()

