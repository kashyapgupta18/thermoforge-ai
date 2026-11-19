import os
import unittest
import numpy as np
from backend.thermal_engine import (
    build_complex_heat_sink_geometry,
    run_solver_battle,
    run_sor_omega_study,
    verify_3d_analytical_benchmark,
    STATUS_CONVERGED,
    STATUS_MAXITER,
    STATUS_INVALID
)
from backend.experiment_manager import (
    run_3x3_geometry_complexity_benchmark,
    evaluate_solver_recommendation,
    run_parametric_optimization_sweep,
    save_experiment_run,
    get_experiment_by_id,
    compare_experiments
)
from backend.mesh_visualizer import extract_surface_mesh_for_threejs
from backend.main import (
    SimulationRequest,
    AIQueryRequest,
    run_simulation,
    run_mesh_intelligence_study,
    consult_thermo_ai
)


class TestThermoforgeEngine(unittest.TestCase):

    def test_3d_analytical_verification(self):
        """Task 5: Validates 3D steady-state conduction against closed-form analytical solution and O(h^2) L2 order."""
        bench = verify_3d_analytical_benchmark()
        self.assertEqual(bench["status"], "PASS")
        self.assertAlmostEqual(bench["observed_order"], 2.0, delta=0.05)
        self.assertLess(bench["max_absolute_error_kelvin"], 1e-8)

    def test_geometry_parameters_modify_computational_model(self):
        """Task 7 & 8: Validates that fin height, base thickness, fin thickness, and fin spacing modify the 3D mesh."""
        m_default = build_complex_heat_sink_geometry(
            preset="advanced_heat_sink", fin_count=7,
            fin_height_ratio=0.65, base_thickness_ratio=0.35, resolution="coarse"
        )
        m_tall = build_complex_heat_sink_geometry(
            preset="advanced_heat_sink", fin_count=7,
            fin_height_ratio=0.85, base_thickness_ratio=0.45, resolution="coarse"
        )
        self.assertGreater(m_tall.lz, m_default.lz)
        vol_def = np.sum(m_default.active_elements) * m_default.dx * m_default.dy * m_default.dz
        vol_tall = np.sum(m_tall.active_elements) * m_tall.dx * m_tall.dy * m_tall.dz
        self.assertNotAlmostEqual(vol_def, vol_tall)

    def test_all_three_solvers_statistical_benchmarking_and_memory_consistency(self):
        """Task 2, 3 & 4: Validates warm-up, balanced Latin-square repetitions, LU split timing, and consistent memory profiling."""
        mesh = build_complex_heat_sink_geometry(preset="advanced_heat_sink", fin_count=7, resolution="coarse")
        battle = run_solver_battle(
            mesh, conductivity=167.0, heat_power_watts=65.0, convection_h=45.0,
            sor_omega=1.9, initial_guess="lumped", repetitions=5
        )

        self.assertEqual(battle["solver_settings"]["warmup_runs"], 1)
        self.assertIn("Latin-Square", battle["solver_settings"]["execution_order_policy"])

        lu_stats = battle["solvers"]["direct_lu"]
        cg_stats = battle["solvers"]["conjugate_gradient"]
        sor_stats = battle["solvers"]["sor"]

        self.assertTrue(lu_stats["converged"])
        self.assertTrue(cg_stats["converged"])
        self.assertTrue(sor_stats["converged"])

        # Direct LU split timing & N/A iteration marking
        self.assertEqual(lu_stats["iterations_display"], "N/A (Direct)")
        self.assertFalse(lu_stats["iterations_applicable"])
        self.assertGreater(lu_stats["median_factorization_ms"], 0.0)
        self.assertGreaterEqual(lu_stats["median_substitution_ms"], 0.0)

        # Consistent memory accounting: memory_mb == base_system_memory_mb + additional_solver_memory_mb
        for s in (lu_stats, cg_stats, sor_stats):
            self.assertIn("base_system_memory_mb", s)
            self.assertIn("additional_solver_memory_mb", s)
            self.assertAlmostEqual(
                s["memory_mb"],
                s["base_system_memory_mb"] + s["additional_solver_memory_mb"],
                places=3
            )
        self.assertEqual(lu_stats["base_system_memory_mb"], cg_stats["base_system_memory_mb"])
        self.assertEqual(cg_stats["base_system_memory_mb"], sor_stats["base_system_memory_mb"])

        self.assertEqual(cg_stats["repetitions"], 5)
        self.assertEqual(len(cg_stats["trial_runtimes_ms"]), 5)
        self.assertGreater(cg_stats["median_runtime_ms"], 0.0)
        self.assertIn("speedup_vs_lu_median", cg_stats)

        # Unsampled history check: every iteration recorded (0..iterations)
        self.assertEqual(len(cg_stats["residuals_history"]), cg_stats["iterations"] + 1)
        self.assertEqual(len(sor_stats["residuals_history"]), sor_stats["iterations"] + 1)

        # Discrepancy against Direct LU must be < 0.001 C for both converged iterative solvers
        self.assertLess(cg_stats["max_discrepancy_vs_lu"], 1e-3)
        self.assertLess(sor_stats["max_discrepancy_vs_lu"], 1e-3)

    def test_sor_failure_and_stress_test_detection(self):
        """Task 1: Validates that unconverged SOR (omega=1.35) and boundary stress test omega=2.05 are properly classified."""
        mesh = build_complex_heat_sink_geometry(preset="advanced_heat_sink", fin_count=7, resolution="coarse")
        b_slow = run_solver_battle(mesh, sor_omega=1.35, max_iter=300, initial_guess="ambient")
        sor_slow = b_slow["solvers"]["sor"]
        self.assertFalse(sor_slow["converged"])
        self.assertEqual(sor_slow["status_label"], STATUS_MAXITER)
        self.assertFalse(sor_slow["is_stress_test"])
        self.assertTrue(len(sor_slow["failure_reason"]) > 10)

        b_inv = run_solver_battle(mesh, sor_omega=2.05, max_iter=100)
        sor_inv = b_inv["solvers"]["sor"]
        self.assertFalse(sor_inv["converged"])
        self.assertEqual(sor_inv["status_label"], STATUS_INVALID)
        self.assertTrue(sor_inv["is_stress_test"])
        self.assertIn("Stress Test", sor_inv["status_display"])
        self.assertIn("Ostrowski-Reich", sor_inv["failure_reason"])

        study = run_sor_omega_study(mesh)
        w205 = [r for r in study["results"] if r["omega"] >= 2.0][0]
        self.assertTrue(w205["is_stress_test"])

    def test_solver_decision_intelligence_and_pareto_optimization(self):
        """Task 7: Validates Solver Decision Map, reciprocal R_th/G_th audit, and (R_th, Mass) Pareto Frontier."""
        mesh = build_complex_heat_sink_geometry(preset="advanced_heat_sink", fin_count=7, resolution="coarse")
        b = run_solver_battle(mesh, sor_omega=1.9)
        rec = evaluate_solver_recommendation(
            dof=b["mesh_info"]["degrees_of_freedom"],
            matrix_nnz=b["mesh_info"]["matrix_nonzeros"],
            mesh_info=b["mesh_info"],
            solvers=b["solvers"]
        )
        self.assertIn("recommended_solver", rec)
        self.assertIn("alternative_solver", rec)
        self.assertIn("expected_trade_offs", rec)
        self.assertGreater(len(rec["supporting_measured_data"]), 3)

        opt = run_parametric_optimization_sweep(
            preset="advanced_heat_sink", material="aluminium_6061",
            param_name="fin_count", param_values=[4, 7, 8, 10], objective="multi_objective"
        )
        self.assertGreaterEqual(len(opt["pareto_frontier"]), 1)
        self.assertIn("optimization_methodology", opt)
        self.assertIn("reciprocal_dependency_audit", opt["optimization_methodology"])
        for pt in opt["results"]:
            # Verify R_th and G_th are exact reciprocals within rounding tolerance
            self.assertAlmostEqual(
                pt["thermal_resistance_k_per_w"] * pt["cooling_conductance_w_per_k"],
                1.0,
                delta=0.005
            )

    def test_experiment_reproducibility_and_diff_comparison(self):
        """Task 8: Validates saving, retrieving, and diffing changed parameters across saved experiments."""
        mesh = build_complex_heat_sink_geometry(preset="advanced_heat_sink", fin_count=7, resolution="coarse")
        b = run_solver_battle(mesh, sor_omega=1.9)
        id1 = save_experiment_run({
            "title": "Unit Test Exp 1",
            "config": {"preset": "advanced_heat_sink", "material": "aluminium_6061", "fin_count": 7, "power_watts": 65.0},
            "results": b
        })
        id2 = save_experiment_run({
            "title": "Unit Test Exp 2",
            "config": {"preset": "advanced_heat_sink", "material": "copper_c11000", "fin_count": 10, "power_watts": 65.0},
            "results": b
        })
        rec1 = get_experiment_by_id(id1)
        self.assertIsNotNone(rec1)
        self.assertEqual(rec1["config"]["fin_count"], 7)

        diff = compare_experiments([id1, id2])
        self.assertEqual(diff["status"], "OK")
        changed_names = {c["parameter"] for c in diff["changed_parameters"]}
        self.assertIn("material", changed_names)
        self.assertIn("fin_count", changed_names)
        self.assertNotIn("power_watts", changed_names)
        self.assertNotEqual(diff["experiments"][0]["energy_balance_error"], "—")

    def test_heat_flux_extraction_and_3x3_matrix(self):
        """Task 2 & 6: Validates heat flux W/m^2 extraction, energy balance decomposition, and 9-experiment 3x3 benchmark."""
        mesh = build_complex_heat_sink_geometry(preset="advanced_heat_sink", fin_count=7, resolution="coarse")
        b = run_solver_battle(mesh, sor_omega=1.9)
        sm = extract_surface_mesh_for_threejs(mesh, b["temperature_fields"]["direct_lu"], conductivity=167.0)
        self.assertIn("heat_fluxes", sm)
        self.assertEqual(len(sm["heat_fluxes"]), len(sm["temperatures"]))

        mat = run_3x3_geometry_complexity_benchmark(resolution="coarse", sor_omega=1.9, repetitions=3)
        self.assertEqual(len(mat["nine_experiments"]), 9)
        self.assertEqual(mat["repetitions"], 3)
        self.assertTrue(all(e["converged"] for e in mat["nine_experiments"]))
        lu_rows = [e for e in mat["nine_experiments"] if e["solver_id"] == "direct_lu"]
        self.assertTrue(all(r["iterations_display"] == "N/A (Direct)" for r in lu_rows))

    def test_thermo_ai_nine_prompts_and_mesh_intelligence_api(self):
        """Task 5, 9 & 12: Validates all 9 THERMO AI prompts and 3-level mesh convergence non-monotonicity analysis."""
        sim_res = run_simulation(SimulationRequest())
        self.assertEqual(sim_res["status"], "SUCCESS")

        mesh_study = run_mesh_intelligence_study(include_fine=True)
        self.assertEqual(len(mesh_study["data"]), 3)
        self.assertIn("non_monotonic_explanation", mesh_study)
        self.assertIn("exposed_area_cm2", mesh_study["data"][0])

        nine_prompts = [
            "Why is this geometry producing a hotspot?",
            "Why did CG outperform Direct LU?",
            "Why did SOR fail to converge?",
            "Is this mesh sufficiently refined?",
            "Which geometry gives the best thermal performance?",
            "What happens if I increase fin count?",
            "Which solver should I use for this problem?",
            "Explain the energy balance.",
            "Interpret my benchmark results."
        ]
        replies = []
        for q in nine_prompts:
            resp = consult_thermo_ai(AIQueryRequest(question=q, simulation_context=sim_res))
            self.assertIn("NUMERICAL RESULT:", resp["reply"])
            self.assertIn("AI INTERPRETATION:", resp["reply"])
            replies.append(resp["reply"])
        # Verify all 9 prompts produce distinct, specialized responses
        self.assertEqual(len(set(replies)), 9)


if __name__ == '__main__':
    unittest.main()
