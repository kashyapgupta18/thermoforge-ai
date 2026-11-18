import unittest
import numpy as np
from backend.thermal_engine import (
    build_complex_heat_sink_geometry,
    run_solver_battle,
    verify_3d_analytical_benchmark,
    STATUS_CONVERGED,
    STATUS_MAXITER,
    STATUS_INVALID
)
from backend.experiment_manager import run_3x3_geometry_complexity_benchmark
from backend.mesh_visualizer import extract_surface_mesh_for_threejs


class TestThermoforgeEngine(unittest.TestCase):

    def test_3d_analytical_verification(self):
        """Validates 3D steady-state conduction against closed-form analytical solution and O(h^2) L2 order."""
        bench = verify_3d_analytical_benchmark()
        self.assertEqual(bench["status"], "PASS")
        self.assertAlmostEqual(bench["observed_order"], 2.0, delta=0.05)
        self.assertLess(bench["max_absolute_error_kelvin"], 1e-8)

    def test_complex_geometry_mesh(self):
        """Validates that complex heat sink assembly generates stepped base, fins, and heat source."""
        mesh = build_complex_heat_sink_geometry(preset="advanced_heat_sink", fin_count=7, resolution="coarse")
        self.assertGreater(mesh.num_nodes, 1000)
        self.assertTrue(np.any(mesh.heat_source_elements))
        self.assertFalse(np.all(mesh.active_elements))

    def test_all_three_solvers_converged_and_unsampled_history(self):
        """Validates that Direct LU, PCG, and SOR (omega=1.9, lumped T0) all converge on the identical system."""
        mesh = build_complex_heat_sink_geometry(preset="advanced_heat_sink", fin_count=7, resolution="coarse")
        battle = run_solver_battle(
            mesh, conductivity=167.0, heat_power_watts=65.0, convection_h=45.0,
            sor_omega=1.9, initial_guess="lumped"
        )

        lu_stats = battle["solvers"]["direct_lu"]
        cg_stats = battle["solvers"]["conjugate_gradient"]
        sor_stats = battle["solvers"]["sor"]

        self.assertTrue(lu_stats["converged"])
        self.assertTrue(cg_stats["converged"])
        self.assertTrue(sor_stats["converged"])
        self.assertEqual(lu_stats["status_label"], STATUS_CONVERGED)
        self.assertEqual(cg_stats["status_label"], STATUS_CONVERGED)
        self.assertEqual(sor_stats["status_label"], STATUS_CONVERGED)

        self.assertLess(lu_stats["final_residual"], 1e-10)
        self.assertLessEqual(cg_stats["final_residual"], 1e-6)
        self.assertLessEqual(sor_stats["final_residual"], 1e-6)

        # Unsampled history check: every iteration recorded (0..iterations)
        self.assertEqual(len(cg_stats["residuals_history"]), cg_stats["iterations"] + 1)
        self.assertEqual(len(sor_stats["residuals_history"]), sor_stats["iterations"] + 1)

        # Discrepancy against Direct LU must be < 0.001 C for both converged iterative solvers
        self.assertLess(cg_stats["max_discrepancy_vs_lu"], 1e-3)
        self.assertLess(sor_stats["max_discrepancy_vs_lu"], 1e-3)

        # Energy balance conservation check (< 0.01% error)
        self.assertLess(lu_stats["energy_balance"]["relative_error"], 1e-4)
        self.assertLess(cg_stats["energy_balance"]["relative_error"], 1e-4)
        self.assertLess(sor_stats["energy_balance"]["relative_error"], 1e-4)

    def test_sor_failure_and_invalid_detection(self):
        """Validates that unconverged SOR (omega=1.35) and invalid omega=2.05 are never reported as Converged."""
        mesh = build_complex_heat_sink_geometry(preset="advanced_heat_sink", fin_count=7, resolution="coarse")
        b_slow = run_solver_battle(mesh, sor_omega=1.35, max_iter=300, initial_guess="ambient")
        sor_slow = b_slow["solvers"]["sor"]
        self.assertFalse(sor_slow["converged"])
        self.assertEqual(sor_slow["status_label"], STATUS_MAXITER)

        b_inv = run_solver_battle(mesh, sor_omega=2.05, max_iter=100)
        sor_inv = b_inv["solvers"]["sor"]
        self.assertFalse(sor_inv["converged"])
        self.assertEqual(sor_inv["status_label"], STATUS_INVALID)

    def test_heat_flux_extraction_and_3x3_matrix(self):
        """Validates heat flux W/m^2 extraction and 9-experiment 3x3 benchmark."""
        mesh = build_complex_heat_sink_geometry(preset="advanced_heat_sink", fin_count=7, resolution="coarse")
        b = run_solver_battle(mesh, sor_omega=1.9)
        sm = extract_surface_mesh_for_threejs(mesh, b["temperature_fields"]["direct_lu"], conductivity=167.0)
        self.assertIn("heat_fluxes", sm)
        self.assertEqual(len(sm["heat_fluxes"]), len(sm["temperatures"]))
        self.assertGreater(sm["heat_flux_range"]["q_max_w_m2"], sm["heat_flux_range"]["q_min_w_m2"])

        mat = run_3x3_geometry_complexity_benchmark(resolution="coarse", sor_omega=1.9)
        self.assertEqual(len(mat["nine_experiments"]), 9)
        self.assertTrue(all(e["converged"] for e in mat["nine_experiments"]))


if __name__ == '__main__':
    unittest.main()
