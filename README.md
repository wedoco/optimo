# optimo
Optimization of Modelica models.

`optimo` uses [OpenModelica](https://openmodelica.org/) for compilation of Modelica models into FMUs. It then relies on [CasADi](https://web.casadi.org/) and [Rockit](https://gitlab.kuleuven.be/meco-software/rockit) to simulate and optimize from those models in Python. 

## Example
We first define our Modelica model in a `.mo` script.

```Modelica
model vdp
  "Van der Pol oscillator model."
  
  Real x1(start=0) "The first state";  
  Real x2(start=1) "The second state"; 
  input Real u(start=0) "The control signal"; 
  output Real objectiveIntegrand(start=0) "The objective signal"; 

equation
  der(x1) = (1 - x2^2) * x1 - x2 + u; 
  der(x2) = x1; 
  objectiveIntegrand = x1^2 + x2^2 + u^2;

end vdp;
```

Now we use this model for simulation and optimization in a Python script:


```Python
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
mo.initialize_optimization()
mo.define_optimization(constraints={"u":(-1, 0.75)}, 
                       objective_terms=["objectiveIntegrand"])
res_ocp_df = mo.optimize()

# Plot 
fig, axs = plt.subplots(3, 1)
res_ocp_df[["x1","x2"]].plot(ax=axs[0], title="States", legend=True)
res_ocp_df["objectiveIntegrand"].plot(ax=axs[1], title="Objective", legend=True)
res_ocp_df["u"].plot(ax=axs[2], title="Input", legend=True)
plt.show()
```

If no input trajectories are provided, the simulation runs with the initial input values as defined in the model. 
The output of this script is the following graphs:

### Simulation results
![Simulation Results](assets/vdp_sim.png)

### Optimization results
![Optimization Results](assets/vdp_ocp.png)