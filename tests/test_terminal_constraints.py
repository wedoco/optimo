import math
import os
import shutil
import tempfile
import unittest

from tests import utilities
from wedoco_optimo.model import OptimoModel, TerminalConstraint

root_path = utilities.get_root_path()
vdp_file = os.path.join(root_path, "examples", "vdp.mo")


class Test_Terminal_Constraints(unittest.TestCase):
    """Terminal constraints on the Van der Pol oscillator, whose
    unconstrained optimum ends near x1 = x2 = 0 at t = 10 s."""

    @classmethod
    def setUpClass(cls):
        cls.original_dir = os.getcwd()
        cls.work_dir = tempfile.mkdtemp()
        os.chdir(cls.work_dir)
        _load_vdp(True)

    @classmethod
    def tearDownClass(cls):
        os.chdir(cls.original_dir)
        shutil.rmtree(cls.work_dir)

    def test_hard_lower_bound_on_a_state(self):
        res_df, _ = _optimize_vdp([TerminalConstraint({"x1": 1.0}, 0.5, math.inf, None)])

        self.assertGreaterEqual(res_df["x1"].iloc[-1], 0.5 - 1e-6)

    def test_hard_upper_bound_on_a_state(self):
        res_df, _ = _optimize_vdp([TerminalConstraint({"x2": 1.0}, -math.inf, -0.3, None)])

        self.assertLessEqual(res_df["x2"].iloc[-1], -0.3 + 1e-6)

    def test_bound_on_a_weighted_sum_of_states(self):
        res_df, _ = _optimize_vdp([TerminalConstraint({"x1": 0.5, "x2": 0.5}, 0.4, math.inf, None)])

        final_mean = 0.5 * res_df["x1"].iloc[-1] + 0.5 * res_df["x2"].iloc[-1]
        self.assertGreaterEqual(final_mean, 0.4 - 1e-6)

    def test_soft_bound_with_a_large_penalty_is_met(self):
        res_df, _ = _optimize_vdp([TerminalConstraint({"x1": 1.0}, 0.5, math.inf, 1e3)])

        self.assertGreaterEqual(res_df["x1"].iloc[-1], 0.5 - 1e-3)

    def test_soft_bound_with_a_small_penalty_is_violated_and_paid_for(self):
        _, unconstrained = _optimize_vdp([])
        res_df, soft = _optimize_vdp([TerminalConstraint({"x1": 1.0}, 0.5, math.inf, 1e-3)])

        violation = 0.5 - res_df["x1"].iloc[-1]
        self.assertGreater(violation, 0.4)
        self.assertAlmostEqual(soft.objective, unconstrained.objective + 1e-3 * violation, places=4)

    def test_solve_summary_reports_convergence_and_objective(self):
        _, summary = _optimize_vdp([])

        self.assertTrue(summary.converged)
        self.assertEqual(summary.return_status, "Solve_Succeeded")
        self.assertGreater(summary.iterations, 0)
        self.assertGreater(summary.objective, 0)

    def test_constraining_a_variable_that_is_not_a_state_is_rejected(self):
        mo = _load_vdp(False)

        with self.assertRaises(ValueError):
            mo.define_optimization(objective_terms=["objectiveIntegrand"],
                                   terminal_constraints=[TerminalConstraint({"u": 1.0}, 0.0, math.inf, None)])


def _load_vdp(force_recompile):
    mo = OptimoModel()
    mo.transfer_model(model="vdp", modelica_files=[vdp_file], force_recompile=force_recompile)
    return mo


def _optimize_vdp(terminal_constraints):
    mo = _load_vdp(False)
    mo.define_optimization(constraints={"u": (-1, 0.75)},
                           objective_terms=["objectiveIntegrand"],
                           terminal_constraints=terminal_constraints)
    res_df = mo.optimize(calculate_objective_terms_integrals=False)
    return res_df, mo.last_solve
