import casadi as ca
import os
from OMPython import OMCSessionZMQ
from pathlib import Path
import casadi as ca
import rockit
import os
import numpy as np

from wedoco_optimo.helpers import load_modelica_files, build_model_fmu, unpack_fmu, explore_dae, get_dae_results

class OptimoModel:

    def transfer_model(self, model: str, force_recompile: bool=True):

        # Store the model name
        self.model = model

        # Compile the FMU if needed
        if not os.path.exists(f'{self.model}.fmu') or force_recompile:
            omc = OMCSessionZMQ()
            load_modelica_files(omc, modelica_files=[f'{model}.mo'])
            build_model_fmu(omc, f'{model}')
        self.fmu_path = str(Path(f'{model}.fmu').resolve())

        # Parse FMU to dae object
        self.dae = ca.DaeBuilder("model", unpack_fmu(self.fmu_path), {"debug": False})
        explore_dae(self.dae)

        # Extract symbols for states, inputs and outputs
        self.x = ca.vcat([self.dae.var(name) for name in self.dae.x()])
        self.u = ca.vcat([self.dae.var(name) for name in self.dae.u()])
        self.y = ca.vcat([self.dae.var(name) for name in self.dae.y()])

        # Define symbolic function from states and inputs to the system outputs and states
        # The inputs are also returned to read them as they are perceived by the model
        self.f_xu_xyu = self.dae.create("f_xu_xyu", ["x", "u"], ["ode", "ydef", "u"])

        # Define the time grid
        self.define_time_grid(t_horizon=10, N=100, M=1)

        # Define the simulator
        dae_dict = {}
        dae_dict["x"] = self.x
        dae_dict["u"] = self.u
        dae_dict["ode"]  = self.f_xu_xyu(x=self.x, u=self.u)["ode"]
        opts = {}
        opts["print_stats"] = False
        self.f_sim = ca.integrator("simulator", "cvodes", dae_dict, 0, self.tgrid, opts)

    def define_time_grid(self, t_horizon: int, N: int, M: int, t_0: int=0):
        """
        Define the time grid for the simulation and optimization problems
        Args:
            t_horizon: The total time horizon in seconds
            N: Number of integration steps in the prediction horizon
            M: Number of integration steps per control interval
        """
        self.t_horizon = t_horizon
        self.N = N
        self.M = M
        self.tgrid = np.asarray([t_horizon / N * k for k in range(N + 1)])

    def get_default_x0(self):
        return self.dae.get(self.dae.x())

    def get_default_u(self):
        if self.dae.u() == []:
            return np.zeros((1, self.N+1))
        else:
            return self.dae.get(self.dae.u())*np.ones((1, self.N+1))

    def simulate(self, 
                 x0: np.array=None, 
                 u_sim: np.array=None):
        
        # Get external values if provided. Otherwise use those from the model
        x0   = x0 if x0 is not None else self.get_default_x0()
        u_sim = u_sim if u_sim is not None else self.get_default_u()

        # Simuilate the model dynamics
        res_x_sim = self.f_sim(x0=x0, u=u_sim)
        x_sim = res_x_sim["xf"].full()

        # Now evaluate the outputs from inputs and computed dynamics
        res_xyu_sim = self.f_xu_xyu(x=x_sim, u=u_sim)
        y_sim = res_xyu_sim["ydef"].full()
        u_sim = res_xyu_sim["u"].full()
        
        t0 = 0
        res_df = get_dae_results(self.tgrid, self.dae, x_sim, y_sim, u_sim, t0)

        return res_df

    def initialize_optimization(self):
        # Define rockit ocp problem
        self.ocp = rockit.Ocp(T=self.t_horizon)

        # Choose a transcription method
        # ocp.method(rockit.SingleShooting(N=self.N, M=self.M))
        # ocp.method(rockit.MultipleShooting(N=self.N, M=self.M))
        self.ocp.method(rockit.DirectCollocation(N=self.N, M=self.M))

        # Define options and choose a solver for the optimization problem
        ipopt_options = {
            "ipopt.max_iter": 500,
            "ipopt.tol": 1e-6
        }
        self.ocp.solver("ipopt", ipopt_options)

        # Let rockit know that the symbols that are states
        for x_name in self.dae.x():
            self.ocp.register_state(self.dae.var(x_name), scale=self.dae.nominal(x_name))

        # Let rockit know the symbols that are controls
        for u_name in self.dae.u():
            self.ocp.register_control(self.dae.var(u_name), scale=self.dae.nominal(u_name))

        # Let rockit know what the state dynamics are
        self.ocp.set_der(self.x, self.f_xu_xyu(x=self.x, u=self.u)["ode"])  

    def define_optimization(self,
                            x0: np.array=None,
                            constraints: dict=None,
                            objective_terms: list=None):
        
        # Initialize the optimization problem if not already done. 
        if not hasattr(self, "ocp"):
            self.initialize_optimization()

        # Get external values if provided. Otherwise use those from the model
        x0 = x0 if x0 is not None else self.get_default_x0()

        # Set initial state values
        for i, x_name in enumerate(self.dae.x()):
            self.ocp.subject_to(self.ocp.at_t0(self.dae.var(x_name)) == x0[i])

        # Set initial time
        t0 = 0
        self.ocp.set_t0(0)

        # Formulate objective from provided list of objective terms
        for term in objective_terms:
            self.ocp.add_objective(self.ocp.integral(self.f_xu_xyu(x=self.x, u=self.u)['ydef'][self.dae.y().index(term)]))

        # Set constraints
        for constrained_var in constraints.keys():
            self.ocp.subject_to(constraints[constrained_var][0] <= (self.dae.var(constrained_var) <= constraints[constrained_var][1]))

    def optimize(self):

        # Solve the optimization problem
        sol = self.ocp.solve()

        # Extract the results
        x_ocp = np.atleast_2d(sol.sample(self.ocp.x, grid="integrator")[1].T)
        u_ocp = np.atleast_2d(sol.sample(self.ocp.u, grid="integrator")[1].T)
        res_xyu_ocp = self.f_xu_xyu(x=x_ocp, u=u_ocp)
        y_ocp = res_xyu_ocp["ydef"].full()
        u_ocp = res_xyu_ocp["u"].full()

        t0 = 0
        res_df = get_dae_results(self.tgrid, self.dae, x_ocp, y_ocp, u_ocp, t0)

        return res_df