# THERMOFORGE AI — Intelligent 3D Thermal Simulation & Solver Benchmarking Environment

> **Tagline:** *Design. Simulate. Compare. Optimize.*  
> **Course / Subject:** Advanced Computational Thermal Engineering & Numerical Methods  
> **Submission Format:** Single Comprehensive Academic Technical PDF Report (No source code in report)

---

## 🎯 Executive Overview & Requirements Fulfillment

This repository hosts **THERMOFORGE AI**, a production-quality engineering simulation software platform engineered to fulfill the assignment requirements:

1. **Complex 3D Geometry (Beyond Simple Cube):**  
   Simulates an **Advanced Finned Heat Sink Assembly** featuring a stepped base plate, variable-thickness vertical fins, unequal fin spacing, ventilation perforations, and a central 65W silicon die heat source.
2. **Three Genuine Numerical Solvers:**  
   - **Solver A — Direct Sparse LU Factorization (SuperLU):** COLAMD fill-in reduction, $P K Q = L U$, exact forward/backward triangular solves.
   - **Solver B — Conjugate Gradient (PCG):** Krylov subspace projection, Jacobi (diagonal) preconditioning, relative residual tracking for symmetric positive-definite systems.
   - **Solver C — Successive Over-Relaxation (SOR):** Stationary matrix splitting with relaxation factor $\omega = 1.35$.
3. **Exact Same Assembled System:**  
   All three solvers execute on the identical finite element stiffness matrix $K$ and load vector $F$ assembled from 3D trilinear hexahedral elements.
4. **Physical Thermal Engine:**  
   Steady-state 3D conduction governed by $\nabla \cdot (k \nabla T) + Q = 0$ with Newton's law of cooling convective Robin boundary conditions ($-k \frac{\partial T}{\partial n} = h(T - T_\infty)$) in physically consistent SI units.
5. **Interactive 3D WebGL Visualization:**  
   Three.js viewport supporting solid CAD, finite element wireframe, continuous temperature contours, hotspot isolation, and cross-section clipping.
6. **Solver Battle Arena & Mesh Intelligence:**  
   Quantitative performance benchmarks (runtime, memory, residual norms, iterations, discrepancy) and spatial grid convergence studies.
7. **THERMO AI Engineering Copilot:**  
   Context-aware AI thermal assistant providing physics-grounded explanations, solver selection guidance, and design optimization advice based on actual computed values.
8. **Single Academic Technical PDF Report:**  
   Publication-quality multi-page PDF technical report generated with ReportLab and Matplotlib, containing embedded convergence graphs, scaling curves, comparison tables, and thermal interpretations.

---

## 🚀 Quick Start Guide

### 1. Prerequisites
- **Python 3.10+**
- **Node.js 18+** & **npm**

### 2. Backend Setup & Startup
In a terminal:
```bash
cd backend
python -m pip install fastapi uvicorn scipy numpy matplotlib reportlab
python -m uvicorn main:app --reload --port 8000
```
Backend API will be live at: `http://localhost:8000`  
Interactive Swagger docs: `http://localhost:8000/docs`

### 3. Frontend Setup & Startup
In a separate terminal:
```bash
cd frontend
npm install
npm run dev
```
Open your browser at: `http://localhost:5173`

---

## 🧪 Running Automated Unit Tests
To verify the numerical engine against the 1D analytical benchmark and test solver convergence:
```bash
python -m unittest backend/test_engine.py
```

---

## 📄 Generating the Academic Submission PDF
To re-generate the comprehensive single PDF technical report:
```bash
python backend/report_generator.py
```
The output PDF is saved directly to:
- [`THERMOFORGE_AI_Technical_Report.pdf`](file:///c:/Users/kashy/Downloads/Pass%20Sccience/THERMOFORGE_AI_Technical_Report.pdf) (located in root directory for assignment submission)

---

## 📊 Summary of Audited Solver Benchmark Findings (Complex Perforated Heat Sink, 2,080 DOF)

| Metric | Solver A: Direct Sparse LU | Solver B: Conjugate Gradient (PCG) | Solver C1: SOR ($\omega = 1.90$, Lumped $T_0$) | Solver C2: SOR ($\omega = 1.35$, Ambient $T_0$) |
| :--- | :--- | :--- | :--- | :--- |
| **Algorithm Type** | Direct Sparse Factorization | Krylov Subspace Iteration | Over-Relaxed Stationary Sweep | Under-Damped Stationary Sweep |
| **Convergence Status** | **Converged** | **Converged** | **Converged** | **Max Iterations Reached (Unconverged)** |
| **Iterations** | 1 (Exact factorize) | 88 iterations | 393 iterations | 1,500 iterations (Limit) |
| **Wall-Time** | ~32.8 ms | **~15.3 ms (Fastest)** | ~731.4 ms | ~2,822 ms |
| **Relative Residual ($\|r\|_2/\|F\|_2$)** | $5.44 \times 10^{-13}$ | $9.50 \times 10^{-7}$ | $9.88 \times 10^{-7}$ | $4.96 \times 10^{-3}$ (Failed tol $10^{-6}$) |
| **Peak Temperature ($T_{\max}$)** | 74.56 °C | 74.56 °C | 74.56 °C | 72.48 °C |
| **Max Discrepancy ($\Delta T$ vs LU)** | 0.000000 °C (Ref) | $8.79 \times 10^{-6}$ °C | $1.02 \times 10^{-4}$ °C | 2.1048 °C |
| **Global Energy Balance Error** | $3.99 \times 10^{-10}\%$ | $6.09 \times 10^{-8}\%$ | $1.93 \times 10^{-4}\%$ | $19.50\%$ (Conservation violated) |
| **Working Memory Footprint** | 2.92 MB (Fill-in $6.1\times$) | **0.57 MB ($O(N)$ storage)** | 1.07 MB | 1.07 MB |

### 3D Analytical Benchmark Verification ($N_x \in \{8, 16, 32, 64\}$)
- **Exact Solution:** $T(x) = T_\infty + \frac{q''' L}{2h} + \frac{q'''}{2k}\left[\left(\frac{L}{2}\right)^2 - \left(x - \frac{L}{2}\right)^2\right]$
- **Observed $L_2$ Interior Convergence Order:** **$p = 2.0000$** (Theoretical $O(h^2)$)
- **Nodal Superconvergence Error:** **$3.20 \times 10^{-9}\text{ K}$** (`PASS`)
