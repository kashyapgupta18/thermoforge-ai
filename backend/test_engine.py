import unittest
import numpy as np
import scipy.sparse as sp
from backend.thermal_engine import (
    HexahedralMesh,
    build_complex_heat_sink_geometry,
    ThermalSystemAssembler,
    solve_direct_sparse_lu,
    solve_conjugate_gradient,
    solve_successive_over_relaxation,
    run_solver_battle,
    verify_1d_analytical_benchmark
)

class TestThermoforgeEngine(unittest.TestCase):

    def test_analytical_verification(self):
        """Validates 1D steady-state conduction against exact closed-form benchmark."""
        bench = verify_1d_analytical_benchmark()
        self.assertEqual(bench["status"], "PASS")
        self.assertLess(bench["max_absolute_error_kelvin"], 1e-10)

    def test_complex_geometry_mesh(self):
        """Validates that complex heat sink assembly generates stepped base, fins, and heat source."""
        mesh = build_complex_heat_sink_geometry(preset="advanced_heat_sink", fin_count=7, resolution="coarse")
        self.assertGreater(mesh.num_nodes, 1000)
        self.assertTrue(np.any(mesh.heat_source_elements))
        self.assertFalse(np.all(mesh.active_elements))  # Has cutouts / air gaps

    def test_all_three_solvers_consistency(self):
        """Validates that Direct LU, CG, and SOR solve the exact same system and match temperatures."""
        mesh = build_complex_heat_sink_geometry(preset="advanced_heat_sink", fin_count=5, resolution="coarse")
        battle = run_solver_battle(mesh, conductivity=167.0, heat_power_watts=65.0, convection_h=45.0)

        lu_stats = battle["solvers"]["direct_lu"]
        cg_stats = battle["solvers"]["conjugate_gradient"]
        sor_stats = battle["solvers"]["sor"]

        # Convergence and residuals
        self.assertTrue(lu_stats["converged"])
        self.assertTrue(cg_stats["converged"])
        self.assertLess(lu_stats["final_residual"], 1e-10)
        self.assertLess(cg_stats["final_residual"], 1e-5)

        # Discrepancy between CG and LU must be under 0.05 °C
        self.assertLess(cg_stats["max_discrepancy_vs_lu"], 0.05)
        # Peak temperature must be physically reasonable (between 50°C and 120°C for 65W dissipation)
        self.assertGreater(lu_stats["t_max"], 50.0)
        self.assertLess(lu_stats["t_max"], 120.0)

if __name__ == '__main__':
    unittest.main()
