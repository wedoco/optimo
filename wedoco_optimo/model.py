import casadi as ca
import os
from pathlib import Path
import rockit
import numpy as np
import glob
import shutil
import subprocess
import tempfile
import time

from wedoco_optimo.helpers import explore_dae, get_dae_results

class OptimoModel:

    def check_model(self, model: str, modelica_files=list[str], pedantic: bool=False):
        """
        Check a Modelica model using command-line omc (avoids ZMQ hanging issues).
        
        Args:
            model: The Modelica class name to check
            modelica_files: List of Modelica files to load
            pedantic: Enable strict checking
        
        Returns:
            Dictionary with check_result and errors
        """
        # Ensure modelica files are in the right format
        modelica_files = [str(Path(path)) for path in modelica_files]
        
        # Create .mos script for model checking
        mos_script = tempfile.NamedTemporaryFile(mode='w', suffix='.mos', delete=False, dir=os.getcwd())
        
        try:
            # Write checking script
            mos_script.write(f'cd("{os.getcwd()}");\n')
            mos_script.write(f'print("Working directory: " + cd() + "\\n");\n')
            
            # Load Modelica packages
            for mf in modelica_files:
                mos_script.write(f'print("Loading {os.path.basename(mf)}...\\n");\n')
                mos_script.write(f'loadFile("{mf}");\n')
            
            mos_script.write('print("Loaded: " + String(getClassNames()) + "\\n");\n')
            
            # Enable pedantic checking if requested
            if pedantic:
                mos_script.write('setCommandLineOptions("--showErrorMessages");\n')
                mos_script.write('setCommandLineOptions("+d=initialization");\n')
            
            # Check the model
            mos_script.write(f'print("\\nChecking model: {model}\\n");\n')
            mos_script.write(f'check_result := checkModel({model});\n')
            mos_script.write('print("\\n" + "=" * 80 + "\\n");\n')
            mos_script.write('print("Check Result:\\n");\n')
            mos_script.write('print(check_result + "\\n");\n')
            mos_script.write('print("\\nMessages:\\n");\n')
            mos_script.write('print(getErrorString() + "\\n");\n')
            mos_script.write('print("=" * 80 + "\\n");\n')
            mos_script.close()
            
            print(f"🔍 Checking model: {model}")
            
            # Run omc without timeout (model checking is usually fast)
            result = subprocess.run(
                ['omc', mos_script.name],
                cwd=os.getcwd(),
                capture_output=True,
                text=True,
                timeout=None  # No timeout for checking
            )
            
            # Parse and display output
            check_result = ""
            error_string = ""
            in_check_result = False
            in_messages = False
            
            for line in result.stdout.split('\n'):
                if line.strip():
                    print(f"   {line}")
                    
                    if "Check Result:" in line:
                        in_check_result = True
                        in_messages = False
                    elif "Messages:" in line:
                        in_check_result = False
                        in_messages = True
                    elif in_check_result and not line.startswith("="):
                        check_result += line + "\n"
                    elif in_messages and not line.startswith("="):
                        error_string += line + "\n"
            
            return {
                "check_result": check_result.strip(),
                "errors": error_string.strip()
            }
            
        except subprocess.TimeoutExpired:
            raise Exception(f'Model check timed out')
        finally:
            try:
                os.unlink(mos_script.name)
            except:
                pass

    def transfer_model(self, model: str, modelica_files=list[str], force_recompile: bool=True, enable_directional_derivatives: bool=True) -> str:
        """
        Compile and transfer a Modelica model to FMU using command-line omc.
        Avoids OMCSessionZMQ which has hanging issues with buildModelFMU.
        
        Args:
            model: The Modelica class name to compile
            modelica_files: List of Modelica files to load
            force_recompile: Force recompilation even if FMU exists
            enable_directional_derivatives: Enable directional derivatives in FMU
        
        Returns:
            Path to the compiled FMU file
        """
        model_name = model.split(".")[-1]
        fmu_file_path = str(Path(os.path.join(os.getcwd(), f"{model_name}.fmu")))
        
        # Compile the FMU if needed
        if not os.path.exists(fmu_file_path) or force_recompile:
            # CRITICAL: Clean up stale .fmutmp directories that cause buildModelFMU to hang
            print("\n🧹 Cleaning up stale build directories...")
            for stale_dir in glob.glob("*.fmutmp"):
                try:
                    shutil.rmtree(stale_dir)
                    print(f"   Removed: {stale_dir}")
                except Exception as e:
                    raise Exception(f"Cannot remove {stale_dir}: {e}. Please remove manually.")
            
            # Ensure modelica files are in the right format
            modelica_files = [str(Path(path)) for path in modelica_files]
            
            # Build FMU using command-line omc (ZMQ interface hangs with buildModelFMU)
            mos_script = tempfile.NamedTemporaryFile(mode='w', suffix='.mos', delete=False, dir=os.getcwd())
            try:
                # Write compilation script
                mos_script.write(f'cd("{os.getcwd()}");\n')
                
                # Load Modelica packages
                for mf in modelica_files:
                    mos_script.write(f'print("Loading {os.path.basename(mf)}...\\n");\n')
                    mos_script.write(f'loadFile("{mf}");\n')
                
                mos_script.write('print("Loaded: " + String(getClassNames()) + "\\n");\n')
                mos_script.write('setCommandLineOptions("--fmuCMakeBuild=false");\n')
                
                if not enable_directional_derivatives:
                    mos_script.write('setDebugFlags("disableDirectionalDerivatives");\n')
                
                # Build FMU
                mos_script.write('print("Building FMU...\\n");\n')
                mos_script.write(f'result := buildModelFMU({model}, version="2.0", fmuType="me", platforms={{"static"}});\n')
                mos_script.write('print("Result: " + result + "\\n");\n')
                mos_script.write('print("Messages: " + getErrorString() + "\\n");\n')
                mos_script.close()
                
                print(f"🔨 Compiling FMU (time varies by model complexity)...")
                start = time.time()
                
                # Run omc without strict timeout (compilation time varies by model)
                result = subprocess.run(
                    ['omc', mos_script.name],
                    cwd=os.getcwd(),
                    capture_output=True,
                    text=True,
                    timeout=None  # No timeout - let compilation finish
                )
                elapsed = time.time() - start
                
                # Show output
                if result.stdout:
                    for line in result.stdout.split('\n'):
                        if line.strip() and line.strip() not in ['true', '""', '']:
                            print(f"   {line}")
                
                if result.stderr:
                    print("   STDERR:")
                    for line in result.stderr.split('\n'):
                        if line.strip():
                            print(f"     {line}")
                
                # Verify FMU was created
                if not os.path.exists(fmu_file_path):
                    raise Exception(f'FMU compilation failed - file not found: {fmu_file_path}')
                
                print(f"✅ FMU compiled successfully in {elapsed:.1f}s")
                
            except subprocess.TimeoutExpired:
                raise Exception(f'FMU compilation timed out. This should not happen with timeout=None.')
            finally:
                try:
                    os.unlink(mos_script.name)
                except:
                    pass

        # Parse FMU to dae object
        self.dae = ca.DaeBuilder("model", fmu_file_path, {"debug": False})
        explore_dae(self.dae)

        # Extract symbols for states, inputs and outputs
        self.x = ca.vcat([self.dae.var(name) for name in self.dae.x()])
        self.u = ca.vcat([self.dae.var(name) for name in self.dae.u()])
        self.y = ca.vcat([self.dae.var(name) for name in self.dae.y()])

        # Define the time grid
        self.define_time_grid(start_time=0.0, end_time=10.0, dt=0.1)

        # Create the symbolic functions and integrator
        self._create_functions()

        return fmu_file_path

    def _create_functions(self):
        """
        Create CasADi symbolic functions and integrator.
        This is called internally when the model is loaded or when parameters are changed.
        """
        # Define symbolic function from states and inputs to the system outputs and states
        # The inputs are also returned to read them as they are perceived by the model
        self.f_xu_xyu = self.dae.create("f_xu_xyu", ["x", "u"], ["ode", "y", "u"])

        # Create the simulator if time grid is defined
        if hasattr(self, 'tgrid'):
            dae_dict = {}
            dae_dict["x"] = self.x
            dae_dict["u"] = self.u
            dae_dict["ode"]  = self.f_xu_xyu(x=self.x, u=self.u)["ode"]
            opts = {}
            opts["print_stats"] = False
            self.f_sim = ca.integrator("simulator", "cvodes", dae_dict, 0, self.tgrid, opts)

    def set_parameter_values(self, parameters: dict):
        """
        Set parameter values in the DAE and recreate CasADi functions.
        
        Args:
            parameters: Dictionary of parameter names and values to set
                       Example: {'param1': value1, 'param2': value2}
        """
        # Set each parameter value in the DAE
        for param_name, param_value in parameters.items():
            self.dae.set(param_name, param_value)
        
        # Recreate the CasADi functions with new parameter values
        self._create_functions()

    def define_time_grid(self, start_time: float, end_time: float, dt: float, M: int=1):
        """
        Define the time grid for the simulation and optimization problems.
        Args:
            start_time: The starting time (float, e.g., 0.0)
            end_time: The ending time (float, e.g., 10.0)
            dt: Time interval between points (float, e.g., 0.1)
            M: Number of time steps per control step (int, default=1)
        """
        self.start_time = start_time
        self.end_time = end_time
        self.dt = dt
        self.M = M 
        # Use np.arange to ensure the last point is included if possible
        self.tgrid = np.arange(start_time, end_time + dt, dt)
        self.N = len(self.tgrid) - 1 # Number of control steps
        self.t_horizon = end_time - start_time

    def get_default_x0(self):
        return {x_name: self.dae.start(x_name) for x_name in self.dae.x()}

    def get_default_u(self):
        u_start = np.array(self.dae.start(self.dae.u()))
        if self.dae.u() == []:
            return np.zeros((len(u_start), self.N+1))
        else:
            return u_start.reshape(-1, 1)*np.ones((1, self.N+1))
    
    def get_default_p(self):
        return {p_name: self.dae.start(p_name) for p_name in self.dae.p()}

    def simulate(self, x0: dict = None, u_sim: np.array = None,
                 start_time: float = None, end_time: float = None, dt: float = None):
        # Optionally override time grid if parameters are provided
        if start_time is not None and end_time is not None and dt is not None:
            self.define_time_grid(start_time=start_time, end_time=end_time, dt=dt)
            # Re-create the integrator with the new time grid
            self._create_functions()
        
        # Get external values if provided. Otherwise use those from the model
        x0 = x0 if x0 is not None else self.get_default_x0()
        # Convert x0 to a numpy array as required by casadi
        x0 = np.atleast_2d([x0[name] for name in self.dae.x()]) 
        u_sim = np.atleast_2d(u_sim) if u_sim is not None else self.get_default_u()

        # Simulate the model dynamics
        res_x_sim = self.f_sim(x0=x0, u=u_sim)
        x_sim = res_x_sim["xf"].full()

        # Now evaluate the outputs from inputs and computed dynamics
        res_xyu_sim = self.f_xu_xyu(x=x_sim, u=u_sim)
        y_sim = res_xyu_sim["y"].full()
        u_sim = res_xyu_sim["u"].full()
        
        t0 = 0
        res_df = get_dae_results(self.tgrid, self.dae, x_sim, y_sim, u_sim, t0)

        return res_df

    def initialize_optimization(self, ipopt_options: dict=None, prescribed_inputs: dict=None):
        # Define rockit ocp problem
        self.ocp = rockit.Ocp(T=self.t_horizon)

        # Choose a transcription method
        # self.ocp.method(rockit.SingleShooting(N=self.N, M=self.M))
        # self.ocp.method(rockit.MultipleShooting(N=self.N, M=self.M))
        self.ocp.method(rockit.DirectCollocation(N=self.N, M=self.M))

        # Define options and choose a solver for the optimization problem
        ipopt_options_default = {
            "ipopt.max_iter": 500,
            "ipopt.tol": 1e-6
        }
        ipopt_options = ipopt_options_default | ipopt_options if ipopt_options else ipopt_options_default
        
        self.ocp.solver("ipopt", ipopt_options)

        # Let rockit know that the symbols that are states
        for x_name in self.dae.x():
            self.ocp.register_state(self.dae.var(x_name), scale=self.dae.nominal(x_name))

        # Let rockit know which symbols that are prescribed inputs and which are controls
        for u_name in self.dae.u():
            if prescribed_inputs and u_name in prescribed_inputs:
                self.ocp.register_parameter(self.dae.var(u_name), grid="control+", scale=self.dae.nominal(u_name))
            else:
                self.ocp.register_control(self.dae.var(u_name), scale=self.dae.nominal(u_name))

        # Let rockit know what the state dynamics are
        derivatives = self.f_xu_xyu(x=self.x, u=self.u)["ode"]
        for idx, x_name in enumerate(self.dae.x()):
            self.ocp.set_der(self.dae.var(x_name), derivatives[idx], scale=self.dae.nominal(x_name))

    def define_optimization(self,
                            x0: np.array=None,
                            initial_guess: dict=None,
                            constraints: dict=None,
                            objective_terms: list=None, 
                            ipopt_options: dict=None,
                            prescribed_inputs: dict=None):
        
        # Assign objective terms 
        self.objective_terms = objective_terms if objective_terms is not None else []

        # Initialize the optimization problem if not already done. 
        if not hasattr(self, "ocp"):
            self.initialize_optimization(ipopt_options=ipopt_options, prescribed_inputs=prescribed_inputs)

        # Get external values if provided. Otherwise use those from the model
        x0 = x0 if x0 is not None else self.get_default_x0()

        # Set initial state values
        for x_name in self.dae.x():
            self.ocp.subject_to(self.ocp.at_t0(self.dae.var(x_name)) == x0[x_name], scale=self.dae.nominal(x_name))

        # Set the initial guess based on the initial state values
        for x_name in self.dae.x():
            self.ocp.set_initial(self.dae.var(x_name), x0[x_name])

        # Set the initial guess for other provided variables
        if initial_guess is not None:
            for v_name in initial_guess:
                self.ocp.set_initial(self.dae.var(v_name), initial_guess[v_name])

        # Set initial time
        t0 = 0
        self.ocp.set_t0(0)

        # Formulate objective from provided list of objective terms
        for term in self.objective_terms:
            self.ocp.add_objective(self.ocp.integral(self.f_xu_xyu(x=self.x, u=self.u)['y'][self.dae.y().index(term)]))

        # Set constraints
        if constraints:
            for constrained_var in constraints.keys():
                self.ocp.subject_to(constraints[constrained_var][0] <= (self.dae.var(constrained_var) <= constraints[constrained_var][1]), scale=self.dae.nominal(constrained_var))

        # Set prescribed inputs
        if prescribed_inputs:
            for prescribed_input in prescribed_inputs.keys():
                self.ocp.set_value(self.dae.var(prescribed_input), prescribed_inputs[prescribed_input])

    def optimize(self, calculate_objective_terms_integrals: bool=True):

        # Solve the optimization problem
        try:
            sol = self.ocp.solve()
        except Exception as e:
            print(f"Optimization did not converge: {e}")
            sol = self.ocp.non_converged_solution

        # Extract the results
        t_ocp = sol.sample(self.dae.var(self.dae.u()[0]), grid="control")[0].T
        x_ocp = np.atleast_2d([sol.sample(self.dae.var(v), grid="control")[1].T for v in self.dae.x()])
        u_ocp = np.atleast_2d([sol.sample(self.dae.var(v), grid="control")[1].T for v in self.dae.u()])
        res_xyu_ocp = self.f_xu_xyu(x=x_ocp, u=u_ocp)
        y_ocp = res_xyu_ocp["y"].full()
        u_ocp = res_xyu_ocp["u"].full()

        t0 = 0

        res_df = get_dae_results(t_ocp, self.dae, x_ocp, y_ocp, u_ocp, t0)

        if calculate_objective_terms_integrals:
            # Calculate the cumulative integrated objective terms over the horizon
            for term in self.objective_terms:
                term_index = self.dae.y().index(term)
                term_values = y_ocp[term_index, :]
                dt_array = np.diff(t_ocp, prepend=t0)
                # Calculate cumulative integral (cumulative sum of term_values * dt_array)
                integral_array = np.cumsum(term_values * dt_array)
                # Add the integrated term to the results as a new variable
                res_df[f"{term}Int"] = integral_array

        return res_df