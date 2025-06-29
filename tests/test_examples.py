import os
import unittest
from tests import utilities
from examples.vdp import run_vdp

# Add the examples folder to the path to find the models.
root_path = utilities.get_root_path()
examples_path = os.path.join(root_path, 'examples')
os.environ['MODELICAPATH'] = os.environ.get('MODELICAPATH', '') + os.pathsep + examples_path

class Test_Optimo_Examples(unittest.TestCase, utilities.partialChecks):
    """Tests the examples of the Optimo package.
         
    """
    
    def test_vdp(self):
        """Test the Van der Pol oscillator example.  
        This example compiles, simulates and optimizes the Van der Pol oscillator model. 

        """
        file_ref_sim = os.path.join(root_path, 'tests', 'references', 'vdp_ref_sim.csv')
        file_ref_ocp = os.path.join(root_path, 'tests', 'references', 'vdp_ref_ocp.csv')
        res_sim_df, res_ocp_df = run_vdp(force_recompile=True, plot=False)
        self.compare_ref_timeseries_df(res_sim_df, file_ref_sim, tol=1e-3)
        self.compare_ref_timeseries_df(res_ocp_df, file_ref_ocp, tol=1e-3)
