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

## 📊 Summary of Solver Benchmark Findings

| Metric | Solver A: Direct Sparse LU | Solver B: Conjugate Gradient (PCG) | Solver C: SOR ($\omega = 1.35$) |
| :--- | :--- | :--- | :--- |
| **Algorithm Type** | Direct Sparse Factorization | Krylov Subspace Iteration | Stationary Relaxation |
| **Iterations** | 1 (Direct Factorization) | 94 iterations | 1500 (Max limit) |
| **Wall-Time** | ~24.1 ms | **~9.9 ms (Fastest)** | ~2822 ms |
| **Precision (Residual)** | $5.44 \times 10^{-13}$ (Exact) | $7.98 \times 10^{-7}$ | $4.96 \times 10^{-3}$ |
| **Peak Temperature ($T_{max}$)**| 74.56 °C | 74.56 °C ($\Delta T < 0.001$ °C) | 72.48 °C |
| **Peak Memory Footprint** | 3.9 MB (Matrix fill-in) | **0.4 MB ($O(N)$ storage)** | 0.8 MB |
| **Recommended Regimes** | Small/medium meshes, ill-conditioned matrices | **Production 3D grids, large DOF** | Simple educational models |
