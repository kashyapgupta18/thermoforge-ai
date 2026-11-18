# THERMOFORGE AI — Research-Grade 3D Thermal Simulation, Statistical Solver Benchmarking & Pareto Optimization Platform (`v3.1.0-RESEARCH`)

> **Tagline:** *Design. Simulate. Compare. Verify. Optimize.*  
> **Live Deployment:** `https://thermoforge-ai.onrender.com`  
> **GitHub Repository:** `https://github.com/kashyapgupta18/thermoforge-ai`  
> **Academic Submission Artifacts:**  
> - [`THERMOFORGE_AI_Technical_Report.pdf`](./THERMOFORGE_AI_Technical_Report.pdf) — 26-Section Comprehensive Academic Technical Report  
> - [`THERMOFORGE_AI_Website_Screenshots.pdf`](./THERMOFORGE_AI_Website_Screenshots.pdf) — Full-Resolution 12-Page Visual Platform Tour  

---

## 🎯 Executive Overview & Research-Grade Architecture

**THERMOFORGE AI (`v3.1.0-RESEARCH`)** is an end-to-end 3D Finite Element Method (FEM) thermal simulation, statistical numerical solver benchmarking, and multi-objective design optimization environment engineered for competitive academic evaluation:

1. **Parametric 3D Hexahedral FEM Geometries (3 Topologies, 8 Tunable Design Variables):**
   - **A. Standard Heat Sink (Uniform Fins)**
   - **B. Asymmetric Finned Cooling Plate**
   - **C. Complex Perforated Heat Sink** (stepped base plate, variable-thickness vertical fins, ventilation perforations, central silicon die heat source).
   - Every geometric parameter (`fin_count`, `fin_height_ratio`, `base_thickness_ratio`, `fin_thickness_ratio`, `fin_spacing_ratio`) directly modifies the 3D computational mesh ($L_x, L_y, L_z$), active element topology, solid volume ($\text{cm}^3$), mass ($\text{g}$), and global stiffness matrix $K$.

2. **Three Independently Implemented Sparse Linear Solvers on Identical $(K, F)$ Systems:**
   - **Solver A — Direct Sparse LU Factorization (`SuperLU` with COLAMD Ordering):** Exact $P_r K P_c = L U$ factorization and triangular solve ($\|r\|_2/\|F\|_2 \approx 5 \times 10^{-13}$).
   - **Solver B — Preconditioned Conjugate Gradient (`PCG` with Jacobi Preconditioner):** Krylov subspace projection ($M = \text{diag}(K)$) exploiting verified Symmetric Positive-Definite (SPD) structure ($\|K - K^T\|_F = 7.00 \times 10^{-16}$).
   - **Solver C — Successive Over-Relaxation (`SOR(ω)` with Lumped Energy-Balance Initialization & Spectral $\omega$-Sweep):** Matrix splitting $(D + \omega L)T^{(k+1)} = [\omega F - (\omega U + (\omega - 1)D)T^{(k)}]$ with automated failure classification (`Converged`, `Maximum iterations reached (UNCONVERGED)`, `Diverged`, `Invalid matrix`).

3. **Repeated-Run Statistical Benchmarking ($N = 10$ Repetitions):**
   - Executes each solver $N = 10$ independent times in both the **Solver Battle Arena** and the **3×3 Geometry Complexity Matrix** (90 solver runs total).
   - Records **Median Runtime (primary metric)**, **Mean**, **Standard Deviation**, **Min**, **Max**, and **Speedup vs. Direct LU Median**.

4. **Empirically Grounded Solver Decision Intelligence:**
   - Analyzes matrix dimension, sparsity ratio, SPD symmetry error, SuperLU factor fill-in ratio, memory ratio, median runtime speedup, and convergence status to recommend the optimal solver and backup alternative with confidence scores and measured trade-off tables.

5. **Numerical Verification, 3-Level Grid Convergence & Residual vs. Solution Error Separation:**
   - **3D Analytical Verification:** Validated against a closed-form 3D cosine eigenfunction + quadratic Z solution, achieving machine-verified second-order spatial convergence (**$p = 2.0000$**).
   - **3-Level Mesh Convergence Study:** Evaluates **Coarse ($16\times 12\times 10$, 2,080 DOFs)**, **Medium ($24\times 18\times 14$, 5,396 DOFs)**, and **Fine ($32\times 24\times 18$, 10,050 DOFs)** meshes.
   - **Global Energy Balance:** Verifies $|Q_{\text{in}} - \int_{\Gamma_c} h(T - T_\infty)\,dS| / Q_{\text{in}} < 10^{-10}\%$.
   - **Residual Error vs. Solution Error:** Explicitly separates algebraic relative residual $\|F - KT\|_2/\|F\|_2$ from true solution discrepancy $\|T - T_{\text{LU}}\|_2 / \|T_{\text{LU}} - T_\infty\|_2$ and $\max|T - T_{\text{LU}}|$.

6. **Multi-Objective Parametric Optimization & Non-Dominated Pareto Frontier:**
   - Sweeps **8 design parameters** across **5 optimization objectives** (`min_t_max`, `min_thermal_resistance`, `max_cooling_conductance`, `min_volume`, `multi_objective`) and extracts the non-dominated **Pareto Frontier** ($R_{\text{th}}$ in $\text{K/W}$ vs. solid volume $\text{cm}^3$ and mass $\text{g}$).

7. **Experiment Reproducibility, Live Replay & Multi-Experiment Parameter Diffing:**
   - Persists complete JSON records (`backend_version: "3.1.0-RESEARCH"`), supports one-click **Live Experiment Replay**, and provides **Multi-Experiment Comparison** isolating changed vs. unchanged input parameters alongside output deltas.

8. **Context-Aware AI Thermal Copilot (`THERMO AI`):**
   - Dynamically answers 9 suggested engineering queries from live simulation telemetry, strictly separating every response into `NUMERICAL RESULT:` and `AI INTERPRETATION:`.

---

## 📊 Actual Measured Benchmark Results

### 1. 3×3 Geometry–Solver Statistical Benchmark ($N = 10$ Repetitions, Coarse Grid $16\times 12\times 10$, $Q = 65\ \text{W}$, $k = 167\ \text{W/(m}\cdot\text{K)}$, $h = 45\ \text{W/(m}^2\cdot\text{K)}$, $T_\infty = 20^\circ\text{C}$)

| Geometry | Solver | DOFs | NNZ($K$) | Median Runtime (ms) | Mean ± Std (ms) | Speedup vs. LU Median | Iterations | Final Relative Residual | $T_{\max}$ ($^\circ\text{C}$) | Memory (MB) | Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **A. Standard Heat Sink** | Direct Sparse LU | 2,184 | 41,144 | **30.44 ms** | $32.32 \pm 10.96$ | $1.00\times$ | 1 | $4.77 \times 10^{-13}$ | $71.5824$ | 2.329 | Converged |
| **A. Standard Heat Sink** | Preconditioned CG | 2,184 | 41,144 | **13.43 ms** | $14.80 \pm 5.80$ | **$2.27\times$** | 95 | $5.78 \times 10^{-7}$ | $71.5824$ | **0.562** | Converged |
| **A. Standard Heat Sink** | $\text{SOR}(\omega=1.90)$ | 2,184 | 41,144 | **1223.24 ms** | $1175.13 \pm 310.29$ | $0.025\times$ | 446 | $9.81 \times 10^{-7}$ | $71.5823$ | 1.008 | Converged |
| **B. Asymmetric Finned** | Direct Sparse LU | 1,884 | 37,918 | **32.30 ms** | $41.23 \pm 19.45$ | $1.00\times$ | 1 | $4.93 \times 10^{-13}$ | $80.1266$ | 2.581 | Converged |
| **B. Asymmetric Finned** | Preconditioned CG | 1,884 | 37,918 | **14.54 ms** | $19.79 \pm 12.31$ | **$2.22\times$** | 96 | $9.97 \times 10^{-7}$ | $80.1266$ | **0.513** | Converged |
| **B. Asymmetric Finned** | $\text{SOR}(\omega=1.90)$ | 1,884 | 37,918 | **1141.97 ms** | $1241.23 \pm 490.30$ | $0.028\times$ | 426 | $9.90 \times 10^{-7}$ | $80.1265$ | 0.925 | Converged |
| **C. Complex Perforated** | Direct Sparse LU | 2,080 | 41,796 | **35.23 ms** | $35.68 \pm 3.07$ | $1.00\times$ | 1 | $5.44 \times 10^{-13}$ | $74.5586$ | 2.918 | Converged |
| **C. Complex Perforated** | Preconditioned CG | 2,080 | 41,796 | **10.70 ms** | $12.30 \pm 4.14$ | **$3.29\times$** | 88 | $9.50 \times 10^{-7}$ | $74.5586$ | **0.566** | Converged |
| **C. Complex Perforated** | $\text{SOR}(\omega=1.90)$ | 2,080 | 41,796 | **806.67 ms** | $866.46 \pm 165.77$ | $0.044\times$ | 393 | $9.88 \times 10^{-7}$ | $74.5585$ | 1.020 | Converged |

### 2. 3-Level Mesh Convergence Study (Complex Perforated Heat Sink, $Q = 65\ \text{W}$)

| Mesh Level | Structured Grid | Active Elements | Active Nodes (DOFs) | Matrix NNZ | $T_{\max}$ ($^\circ\text{C}$) | $T_{\text{avg}}$ ($^\circ\text{C}$) | Direct LU Runtime (ms) | Energy Balance Error |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Coarse** | $16 \times 12 \times 10$ | 1,302 | 2,080 | 41,796 | $74.5586^\circ\text{C}$ | $66.9692^\circ\text{C}$ | $17.13\ \text{ms}$ | $5.44 \times 10^{-11}\%$ |
| **Medium** | $24 \times 18 \times 14$ | 3,456 | 5,396 | 110,100 | $67.3810^\circ\text{C}$ | $60.2564^\circ\text{C}$ | $68.39\ \text{ms}$ | $6.31 \times 10^{-11}\%$ |
| **Fine** | $32 \times 24 \times 18$ | 6,846 | 10,050 | 206,826 | $69.5775^\circ\text{C}$ | $61.7721^\circ\text{C}$ | $207.55\ \text{ms}$ | $7.12 \times 10^{-11}\%$ |

### 3. Multi-Objective Parametric Sweep & Non-Dominated Pareto Frontier (`fin_count` Sweep, Aluminium 6061-T6)

| Fin Count | Solid Volume ($\text{cm}^3$) | Mass ($\text{g}$) | $T_{\max}$ ($^\circ\text{C}$) | Thermal Resistance $R_{\text{th}}$ ($\text{K/W}$) | Cooling Conductance $G_{\text{th}}$ ($\text{W/K}$) | Pareto Status |
| :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **4 fins** | $111.60$ | $301.32$ | $80.66^\circ\text{C}$ | $0.9333$ | $1.072$ | **★ Pareto-Optimal** (Min Mass) |
| **6 fins** | $123.20$ | $332.64$ | $76.19^\circ\text{C}$ | $0.8645$ | $1.157$ | **★ Pareto-Optimal** |
| **7 fins** | $130.20$ | $351.54$ | $74.56^\circ\text{C}$ | $0.8394$ | $1.191$ | **★ Pareto-Optimal** |
| **8 fins** | $137.20$ | $370.44$ | **$73.52^\circ\text{C}$** | **$0.8234$** | **$1.214$** | **★ Pareto-Optimal (Best Multi-Objective Balance)** |
| **10 fins** | $141.80$ | $382.86$ | $77.12^\circ\text{C}$ | $0.8788$ | $1.138$ | Dominated (Coarse-grid voxel fin merging) |
| **12 fins** | $155.80$ | $420.66$ | $85.33^\circ\text{C}$ | $1.0050$ | $0.995$ | Dominated (Coarse-grid voxel fin merging) |

---

## 🚀 Quick Start & Verification Commands

### 1. Run the Automated Unit Test Suite (7 Comprehensive Research Tests)
```bash
python -m unittest backend/test_engine.py -v
```

### 2. Start the FastAPI Numerical Backend
```bash
python -m uvicorn backend.main:app --reload --port 8000
```

### 3. Start or Build the React + Three.js Frontend
```bash
cd frontend
npm install
npm run dev      # Development server at http://localhost:5173
npm run build    # Production bundle verification
```

### 4. Regenerate the 26-Section Academic PDF Report & Website Screenshots PDF
```bash
python -X utf8 backend/report_generator.py
```
