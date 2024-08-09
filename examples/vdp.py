import numpy as np
from optimo.plots import plot_from_def
from optimo.model import OptimoModel

mo = OptimoModel()
mo.transfer_model(model='vdp', force_recompile=False)


plot_def = {}
plot_def['x'] = {}
plot_def['x']['values'] = mo.tgrid
plot_def['x']['title'] = 'Time (s)'
plot_def['Angle (rad)'] = {}
plot_def['Angle (rad)']['vars'] = ['x1', 'x2']
plot_def['Outputs'] = {}
plot_def['Outputs']['vars'] = ['objectiveIntegrand']
plot_def['Inputs'] = {}
plot_def['Inputs']['vars'] = ['u']

res_sim_df = mo.simulate()
plot_from_def(plot_def, res_sim_df, show=False, save_to_file=True, filename='plot_sim.html')

res_ocp_df = mo.optimize(objective_terms=['objectiveIntegrand'])
plot_from_def(plot_def, res_ocp_df, show=False, save_to_file=True, filename='plot_ocp.html')