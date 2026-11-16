import React, { useState, useEffect } from 'react';
import {
  Layers,
  Flame,
  Award,
  Zap,
  BarChart3,
  Bot,
  FileDown,
  RefreshCw,
  Eye,
  Sliders,
  Play,
  TrendingUp,
  Activity,
} from 'lucide-react';
import {
  LineChart, Line, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer
} from 'recharts';
import { Viewport3D } from './components/Viewport3D';

// Dynamically use relative API URL in production, or localhost:8000 in local dev
const API_BASE = import.meta.env.VITE_API_URL || '';

interface SolverStats {
  solver_id: string;
  solver_name: string;
  execution_time_ms: number;
  iterations: number;
  converged: boolean;
  final_residual: number;
  residuals_history: number[];
  memory_mb: number;
  t_min: number;
  t_max: number;
  t_avg: number;
  max_discrepancy_vs_lu: number;
  description: string;
}

interface SimulationData {
  mesh_info: {
    degrees_of_freedom: number;
    active_elements: number;
    matrix_nonzeros: number;
    assembly_time_ms: number;
  };
  solvers: {
    direct_lu: SolverStats;
    conjugate_gradient: SolverStats;
    sor: SolverStats;
  };
  winner_id: string;
  winner_explanation: string;
  surface_mesh: any;
}

export function App() {
  const [activeTab, setActiveTab] = useState<'simulation' | 'battle' | 'mesh' | 'optimization' | 'ai'>('simulation');

  // Simulation Parameters
  const [preset, setPreset] = useState<string>('advanced_heat_sink');
  const [material, setMaterial] = useState<string>('aluminium_6061');
  const [finCount, setFinCount] = useState<number>(7);
  const [powerWatts, setPowerWatts] = useState<number>(65.0);
  const [convectionH, setConvectionH] = useState<number>(45.0);
  const [resolution, setResolution] = useState<string>('medium');
  const [sorOmega, setSorOmega] = useState<number>(1.35);

  // 3D Viewport Mode
  const [vizMode, setVizMode] = useState<'solid' | 'wireframe' | 'temperature' | 'hotspot' | 'cross_section'>('temperature');

  // States
  const [loading, setLoading] = useState<boolean>(false);
  const [simData, setSimData] = useState<SimulationData | null>(null);

  // AI Chat States
  const [aiQuestion, setAiQuestion] = useState<string>('');
  const [aiMessages, setAiMessages] = useState<Array<{ sender: 'user' | 'thermo_ai'; text: string }>>([
    {
      sender: 'thermo_ai',
      text: 'Greetings. I am THERMO AI, your intelligent thermal engineering copilot. I am initialized with your finite element solver metrics and thermal conduction equations. Ask me regarding solver efficiency, temperature gradients, or heat sink design optimization.'
    }
  ]);
  const [aiLoading, setAiLoading] = useState<boolean>(false);

  // Optimization study state
  const [optData, setOptData] = useState<any>(null);
  const [optLoading, setOptLoading] = useState<boolean>(false);

  // Mesh Study State
  const [meshData, setMeshData] = useState<any>(null);
  const [meshLoading, setMeshLoading] = useState<boolean>(false);

  // Trigger simulation
  const executeSimulation = async () => {
    setLoading(true);
    try {
      const res = await fetch(`${API_BASE}/api/simulate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          preset,
          material,
          fin_count: finCount,
          power_watts: powerWatts,
          convection_h: convectionH,
          resolution,
          sor_omega: sorOmega,
          ambient_temp_celsius: 20.0
        })
      });
      const data = await res.json();
      setSimData(data);
    } catch (err) {
      console.error("Simulation failed:", err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    executeSimulation();
  }, []);

  const handleAskAI = async () => {
    if (!aiQuestion.trim()) return;
    const q = aiQuestion;
    setAiQuestion('');
    setAiMessages(prev => [...prev, { sender: 'user', text: q }]);
    setAiLoading(true);

    try {
      const res = await fetch(`${API_BASE}/api/ai-assistant`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          question: q,
          simulation_context: simData
        })
      });
      const data = await res.json();
      setAiMessages(prev => [...prev, { sender: 'thermo_ai', text: data.reply }]);
    } catch (err) {
      setAiMessages(prev => [...prev, { sender: 'thermo_ai', text: 'Error communicating with THERMO AI engine.' }]);
    } finally {
      setAiLoading(false);
    }
  };

  const handleRunMeshStudy = async () => {
    setMeshLoading(true);
    try {
      const res = await fetch(`${API_BASE}/api/mesh-intelligence?preset=${preset}&material=${material}&fin_count=${finCount}`);
      const data = await res.json();
      setMeshData(data);
    } catch (e) {
      console.error(e);
    } finally {
      setMeshLoading(false);
    }
  };

  const handleRunOptimization = async () => {
    setOptLoading(true);
    try {
      const res = await fetch(`${API_BASE}/api/design-optimization`);
      const data = await res.json();
      setOptData(data);
    } catch (e) {
      console.error(e);
    } finally {
      setOptLoading(false);
    }
  };

  // Prepare chart data for Convergence History
  const convergenceChartData = () => {
    if (!simData) return [];
    const cgHist = simData.solvers.conjugate_gradient.residuals_history || [];
    const sorHist = simData.solvers.sor.residuals_history || [];
    const maxLen = Math.max(cgHist.length, sorHist.length);
    const data = [];
    for (let i = 0; i < maxLen; i++) {
      data.push({
        step: i + 1,
        CG: cgHist[i] !== undefined ? Math.log10(Math.max(1e-12, cgHist[i])) : null,
        SOR: sorHist[i] !== undefined ? Math.log10(Math.max(1e-12, sorHist[i])) : null,
      });
    }
    return data;
  };

  // Solver battle bar chart
  const battleBarData = () => {
    if (!simData) return [];
    const s = simData.solvers;
    return [
      { name: 'Direct Sparse LU', time_ms: s.direct_lu.execution_time_ms, memory_mb: s.direct_lu.memory_mb, iters: 1 },
      { name: 'Conjugate Gradient (PCG)', time_ms: s.conjugate_gradient.execution_time_ms, memory_mb: s.conjugate_gradient.memory_mb, iters: s.conjugate_gradient.iterations },
      { name: 'SOR (ω=1.35)', time_ms: s.sor.execution_time_ms, memory_mb: s.sor.memory_mb, iters: s.sor.iterations },
    ];
  };

  const tMin = simData ? simData.solvers.direct_lu.t_min : 20.0;
  const tMax = simData ? simData.solvers.direct_lu.t_max : 75.0;

  return (
    <div className="min-h-screen bg-graphite-950 text-slate-200 flex flex-col font-sans">
      {/* TOP SCIENTIFIC HEADER */}
      <header className="h-16 border-b border-graphite-800 bg-graphite-900/80 backdrop-blur-md px-6 flex items-center justify-between sticky top-0 z-50">
        <div className="flex items-center space-x-3">
          <div className="w-9 h-9 rounded-lg bg-gradient-to-br from-cyan-500 to-blue-600 flex items-center justify-center shadow-lg shadow-cyan-500/20">
            <Flame className="w-5 h-5 text-white" />
          </div>
          <div>
            <div className="flex items-center space-x-2">
              <span className="font-bold text-lg tracking-wider text-white">THERMOFORGE AI</span>
              <span className="px-2 py-0.5 text-[10px] font-mono font-semibold bg-cyan-950 text-cyan-400 border border-cyan-800 rounded">v2.4 ACADEMIC</span>
            </div>
            <div className="text-[11px] text-gray-400">Intelligent 3D Thermal Simulation & Solver Benchmarking Arena</div>
          </div>
        </div>

        {/* Navigation tabs */}
        <nav className="flex items-center space-x-1 bg-graphite-850 p-1 rounded-lg border border-graphite-700">
          <button
            onClick={() => setActiveTab('simulation')}
            className={`px-3 py-1.5 rounded-md text-xs font-semibold transition-all flex items-center space-x-1.5 ${
              activeTab === 'simulation' ? 'bg-cyan-600 text-white shadow' : 'text-gray-400 hover:text-white'
            }`}
          >
            <Layers className="w-3.5 h-3.5" />
            <span>3D Simulation</span>
          </button>
          <button
            onClick={() => setActiveTab('battle')}
            className={`px-3 py-1.5 rounded-md text-xs font-semibold transition-all flex items-center space-x-1.5 ${
              activeTab === 'battle' ? 'bg-cyan-600 text-white shadow' : 'text-gray-400 hover:text-white'
            }`}
          >
            <Award className="w-3.5 h-3.5" />
            <span>Solver Battle Arena</span>
          </button>
          <button
            onClick={() => { setActiveTab('mesh'); if(!meshData) handleRunMeshStudy(); }}
            className={`px-3 py-1.5 rounded-md text-xs font-semibold transition-all flex items-center space-x-1.5 ${
              activeTab === 'mesh' ? 'bg-cyan-600 text-white shadow' : 'text-gray-400 hover:text-white'
            }`}
          >
            <BarChart3 className="w-3.5 h-3.5" />
            <span>Mesh Intelligence</span>
          </button>
          <button
            onClick={() => { setActiveTab('optimization'); if(!optData) handleRunOptimization(); }}
            className={`px-3 py-1.5 rounded-md text-xs font-semibold transition-all flex items-center space-x-1.5 ${
              activeTab === 'optimization' ? 'bg-cyan-600 text-white shadow' : 'text-gray-400 hover:text-white'
            }`}
          >
            <TrendingUp className="w-3.5 h-3.5" />
            <span>Design Studio</span>
          </button>
          <button
            onClick={() => setActiveTab('ai')}
            className={`px-3 py-1.5 rounded-md text-xs font-semibold transition-all flex items-center space-x-1.5 ${
              activeTab === 'ai' ? 'bg-cyan-600 text-white shadow' : 'text-gray-400 hover:text-white'
            }`}
          >
            <Bot className="w-3.5 h-3.5" />
            <span>THERMO AI Copilot</span>
          </button>
        </nav>

        {/* Download PDF button */}
        <a
          href={`${API_BASE}/api/download-report`}
          download="THERMOFORGE_AI_Technical_Report.pdf"
          className="px-4 py-2 bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-500 hover:to-indigo-500 text-white text-xs font-semibold rounded-lg shadow-md hover:shadow-indigo-500/25 transition-all flex items-center space-x-2 border border-indigo-400/30"
        >
          <FileDown className="w-4 h-4" />
          <span>Export Academic Report (PDF)</span>
        </a>
      </header>

      {/* MAIN WORKSPACE CONTENT */}
      <main className="flex-1 p-5 max-w-7xl mx-auto w-full">
        {/* TAB 1: 3D SIMULATION WORKSPACE */}
        {activeTab === 'simulation' && (
          <div className="grid grid-cols-12 gap-5 h-[calc(100vh-120px)]">
            {/* LEFT CONTROL PANEL */}
            <div className="col-span-4 bg-graphite-900 border border-graphite-800 rounded-xl p-5 flex flex-col justify-between overflow-y-auto">
              <div className="space-y-4">
                <div className="border-b border-graphite-800 pb-3">
                  <h2 className="text-sm font-bold tracking-wide text-white uppercase flex items-center space-x-2">
                    <Sliders className="w-4 h-4 text-cyan-400" />
                    <span>Thermal Model Setup</span>
                  </h2>
                  <p className="text-xs text-gray-400 mt-0.5">Parametric geometry and physics boundary setup</p>
                </div>

                {/* Preset Selector */}
                <div>
                  <label className="text-xs font-medium text-gray-300 block mb-1">Complex Engineering Geometry</label>
                  <select
                    value={preset}
                    onChange={(e) => setPreset(e.target.value)}
                    className="w-full bg-graphite-850 border border-graphite-700 text-xs rounded-lg px-3 py-2 text-white focus:outline-none focus:border-cyan-500"
                  >
                    <option value="advanced_heat_sink">Advanced Finned Heat Sink Assembly (Primary)</option>
                    <option value="standard_heat_sink">Standard Uniform Heat Sink</option>
                    <option value="asymmetric_cooling_plate">Asymmetric Industrial Cooling Plate</option>
                    <option value="cube">Cube (Baseline Verification Model)</option>
                  </select>
                </div>

                {/* Material Selection */}
                <div>
                  <label className="text-xs font-medium text-gray-300 block mb-1">Heat Sink Alloy Material</label>
                  <select
                    value={material}
                    onChange={(e) => setMaterial(e.target.value)}
                    className="w-full bg-graphite-850 border border-graphite-700 text-xs rounded-lg px-3 py-2 text-white focus:outline-none focus:border-cyan-500"
                  >
                    <option value="aluminium_6061">Aluminium 6061-T6 (k = 167 W/m·K)</option>
                    <option value="copper_c11000">Electrolytic Copper C11000 (k = 390 W/m·K)</option>
                    <option value="stainless_steel_304">Stainless Steel 304 (k = 16.2 W/m·K)</option>
                    <option value="synthetic_graphite">Synthetic Graphite Composite (k = 450 W/m·K)</option>
                  </select>
                </div>

                {/* Number of fins */}
                <div>
                  <div className="flex justify-between text-xs text-gray-300 mb-1">
                    <span>Vertical Fin Count:</span>
                    <span className="font-mono text-cyan-400 font-bold">{finCount} fins</span>
                  </div>
                  <input
                    type="range"
                    min="3"
                    max="12"
                    step="1"
                    value={finCount}
                    onChange={(e) => setFinCount(Number(e.target.value))}
                    className="w-full accent-cyan-500"
                  />
                </div>

                {/* Heat dissipation power */}
                <div>
                  <div className="flex justify-between text-xs text-gray-300 mb-1">
                    <span>Die Thermal Dissipation:</span>
                    <span className="font-mono text-rose-400 font-bold">{powerWatts} Watts</span>
                  </div>
                  <input
                    type="range"
                    min="15"
                    max="150"
                    step="5"
                    value={powerWatts}
                    onChange={(e) => setPowerWatts(Number(e.target.value))}
                    className="w-full accent-rose-500"
                  />
                </div>

                {/* Convective Heat Transfer Coefficient */}
                <div>
                  <div className="flex justify-between text-xs text-gray-300 mb-1">
                    <span>Convection Coefficient (h):</span>
                    <span className="font-mono text-blue-400 font-bold">{convectionH} W/(m²·K)</span>
                  </div>
                  <input
                    type="range"
                    min="10"
                    max="120"
                    step="5"
                    value={convectionH}
                    onChange={(e) => setConvectionH(Number(e.target.value))}
                    className="w-full accent-blue-500"
                  />
                </div>

                {/* Mesh Discretization */}
                <div>
                  <label className="text-xs font-medium text-gray-300 block mb-1">Grid Resolution</label>
                  <div className="grid grid-cols-2 gap-2">
                    {['coarse', 'medium'].map((res) => (
                      <button
                        key={res}
                        type="button"
                        onClick={() => setResolution(res)}
                        className={`py-1.5 rounded-lg text-xs font-mono font-bold capitalize transition-all border ${
                          resolution === res
                            ? 'bg-cyan-950 border-cyan-500 text-cyan-300'
                            : 'bg-graphite-850 border-graphite-700 text-gray-400'
                        }`}
                      >
                        {res}
                      </button>
                    ))}
                  </div>
                </div>

                {/* Governing Equation Info Card */}
                <div className="bg-graphite-950 p-3 rounded-lg border border-graphite-800 text-[11px] text-gray-400 font-mono">
                  <div className="text-cyan-400 font-bold mb-1">Governing Thermal PDE:</div>
                  <div>∇ · (k ∇ T) + Q = 0</div>
                  <div className="text-[10px] text-gray-500 mt-1">Convective Robin BC: -k(∂T/∂n) = h(T - T_∞)</div>
                </div>
              </div>

              {/* Run Simulation Action Button */}
              <button
                onClick={executeSimulation}
                disabled={loading}
                className="w-full mt-4 py-3 bg-gradient-to-r from-cyan-600 to-blue-600 hover:from-cyan-500 hover:to-blue-500 disabled:opacity-50 text-white font-bold text-xs rounded-xl shadow-lg shadow-cyan-600/30 transition-all flex items-center justify-center space-x-2"
              >
                {loading ? <RefreshCw className="w-4 h-4 animate-spin" /> : <Play className="w-4 h-4" />}
                <span>{loading ? 'Solving Assembled System...' : 'Run 3D Thermal Simulation'}</span>
              </button>
            </div>

            {/* 3D VIEWPORT & VISUALIZATION */}
            <div className="col-span-8 flex flex-col space-y-3">
              {/* Top View Mode Buttons */}
              <div className="bg-graphite-900 border border-graphite-800 p-2 rounded-xl flex items-center justify-between">
                <div className="flex items-center space-x-1">
                  <span className="text-xs text-gray-400 mr-2 flex items-center space-x-1">
                    <Eye className="w-3.5 h-3.5 text-cyan-400" />
                    <span>Mode:</span>
                  </span>
                  {[
                    { id: 'temperature', label: 'Temperature Contour' },
                    { id: 'hotspot', label: 'Hotspot Isolation' },
                    { id: 'wireframe', label: 'FEM Wireframe' },
                    { id: 'solid', label: 'Solid CAD' },
                    { id: 'cross_section', label: 'Cross-Section' },
                  ].map((mode) => (
                    <button
                      key={mode.id}
                      onClick={() => setVizMode(mode.id as any)}
                      className={`px-3 py-1 rounded-md text-xs font-medium transition-all ${
                        vizMode === mode.id
                          ? 'bg-cyan-600 text-white'
                          : 'bg-graphite-850 text-gray-400 hover:text-white'
                      }`}
                    >
                      {mode.label}
                    </button>
                  ))}
                </div>

                {simData && (
                  <div className="flex items-center space-x-4 text-xs font-mono">
                    <span className="text-gray-400">DOF: <strong className="text-cyan-400">{simData.mesh_info.degrees_of_freedom}</strong></span>
                    <span className="text-gray-400">Peak Temp: <strong className="text-rose-400">{simData.solvers.direct_lu.t_max}°C</strong></span>
                  </div>
                )}
              </div>

              {/* 3D Canvas */}
              <div className="flex-1 min-h-[420px]">
                <Viewport3D
                  surfaceMesh={simData?.surface_mesh || null}
                  vizMode={vizMode}
                  tMin={tMin}
                  tMax={tMax}
                />
              </div>

              {/* Quick Solver Snapshot Footer */}
              {simData && (
                <div className="grid grid-cols-3 gap-3">
                  <div className="bg-graphite-900 p-3 rounded-xl border border-graphite-800">
                    <div className="text-[10px] text-gray-400 font-mono">SOLVER A — DIRECT SPARSE LU</div>
                    <div className="text-lg font-bold font-mono text-cyan-400">{simData.solvers.direct_lu.execution_time_ms.toFixed(1)} ms</div>
                    <div className="text-[11px] text-gray-400">1 Factorization Pass • Exact</div>
                  </div>
                  <div className="bg-graphite-900 p-3 rounded-xl border border-graphite-800">
                    <div className="text-[10px] text-gray-400 font-mono">SOLVER B — CONJUGATE GRADIENT</div>
                    <div className="text-lg font-bold font-mono text-emerald-400">{simData.solvers.conjugate_gradient.execution_time_ms.toFixed(1)} ms</div>
                    <div className="text-[11px] text-gray-400">{simData.solvers.conjugate_gradient.iterations} Iterations • Jacobi Precond.</div>
                  </div>
                  <div className="bg-graphite-900 p-3 rounded-xl border border-graphite-800">
                    <div className="text-[10px] text-gray-400 font-mono">SOLVER C — SUCCESSIVE OVER-RELAXATION</div>
                    <div className="text-lg font-bold font-mono text-amber-400">{simData.solvers.sor.execution_time_ms.toFixed(1)} ms</div>
                    <div className="text-[11px] text-gray-400">{simData.solvers.sor.iterations} Iterations • ω={sorOmega}</div>
                  </div>
                </div>
              )}
            </div>
          </div>
        )}

        {/* TAB 2: SOLVER BATTLE ARENA */}
        {activeTab === 'battle' && simData && (
          <div className="space-y-6">
            {/* Winner Announcement Banner */}
            <div className="bg-gradient-to-r from-blue-950/80 via-indigo-950/80 to-graphite-900 border border-cyan-800/40 p-6 rounded-2xl shadow-xl flex items-start space-x-4">
              <div className="p-3 bg-cyan-500/20 rounded-xl text-cyan-400">
                <Award className="w-8 h-8" />
              </div>
              <div>
                <div className="text-xs font-mono font-bold text-cyan-400 uppercase tracking-widest">Benchmark Leaderboard Verdict</div>
                <h3 className="text-lg font-bold text-white mt-0.5">
                  Winner: {simData.winner_id === 'conjugate_gradient' ? 'Conjugate Gradient (PCG)' : 'Direct Sparse LU'}
                </h3>
                <p className="text-xs text-gray-300 mt-2 whitespace-pre-line leading-relaxed">
                  {simData.winner_explanation}
                </p>
              </div>
            </div>

            {/* Comparison Metrics Grid */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-5">
              {/* Card 1: Direct Sparse LU */}
              <div className="bg-graphite-900 border border-cyan-900/50 rounded-xl p-5 space-y-3">
                <div className="flex justify-between items-center">
                  <span className="text-xs font-bold text-cyan-400 font-mono">SOLVER A</span>
                  <span className="text-[10px] px-2 py-0.5 bg-cyan-950 text-cyan-300 rounded font-mono">DIRECT</span>
                </div>
                <h4 className="text-base font-bold text-white">{simData.solvers.direct_lu.solver_name}</h4>
                <div className="space-y-2 pt-2 border-t border-graphite-800 text-xs">
                  <div className="flex justify-between text-gray-400"><span>Wall-Clock Time:</span><strong className="text-white font-mono">{simData.solvers.direct_lu.execution_time_ms.toFixed(2)} ms</strong></div>
                  <div className="flex justify-between text-gray-400"><span>Iterations:</span><strong className="text-white font-mono">{simData.solvers.direct_lu.iterations}</strong></div>
                  <div className="flex justify-between text-gray-400"><span>Residual:</span><strong className="text-white font-mono">{simData.solvers.direct_lu.final_residual.toExponential(2)}</strong></div>
                  <div className="flex justify-between text-gray-400"><span>Memory Used:</span><strong className="text-white font-mono">{simData.solvers.direct_lu.memory_mb} MB</strong></div>
                  <div className="flex justify-between text-gray-400"><span>Peak Temperature:</span><strong className="text-rose-400 font-mono">{simData.solvers.direct_lu.t_max}°C</strong></div>
                </div>
              </div>

              {/* Card 2: Conjugate Gradient */}
              <div className="bg-graphite-900 border border-emerald-900/50 rounded-xl p-5 space-y-3">
                <div className="flex justify-between items-center">
                  <span className="text-xs font-bold text-emerald-400 font-mono">SOLVER B</span>
                  <span className="text-[10px] px-2 py-0.5 bg-emerald-950 text-emerald-300 rounded font-mono">KRYLOV</span>
                </div>
                <h4 className="text-base font-bold text-white">{simData.solvers.conjugate_gradient.solver_name}</h4>
                <div className="space-y-2 pt-2 border-t border-graphite-800 text-xs">
                  <div className="flex justify-between text-gray-400"><span>Wall-Clock Time:</span><strong className="text-emerald-400 font-mono">{simData.solvers.conjugate_gradient.execution_time_ms.toFixed(2)} ms</strong></div>
                  <div className="flex justify-between text-gray-400"><span>Iterations:</span><strong className="text-white font-mono">{simData.solvers.conjugate_gradient.iterations}</strong></div>
                  <div className="flex justify-between text-gray-400"><span>Residual:</span><strong className="text-white font-mono">{simData.solvers.conjugate_gradient.final_residual.toExponential(2)}</strong></div>
                  <div className="flex justify-between text-gray-400"><span>Memory Used:</span><strong className="text-emerald-400 font-mono">{simData.solvers.conjugate_gradient.memory_mb} MB</strong></div>
                  <div className="flex justify-between text-gray-400"><span>Error vs LU:</span><strong className="text-emerald-300 font-mono">{simData.solvers.conjugate_gradient.max_discrepancy_vs_lu.toFixed(4)}°C</strong></div>
                </div>
              </div>

              {/* Card 3: SOR */}
              <div className="bg-graphite-900 border border-amber-900/50 rounded-xl p-5 space-y-3">
                <div className="flex justify-between items-center">
                  <span className="text-xs font-bold text-amber-400 font-mono">SOLVER C</span>
                  <span className="text-[10px] px-2 py-0.5 bg-amber-950 text-amber-300 rounded font-mono">STATIONARY</span>
                </div>
                <h4 className="text-base font-bold text-white">{simData.solvers.sor.solver_name}</h4>
                <div className="space-y-2 pt-2 border-t border-graphite-800 text-xs">
                  <div className="flex justify-between text-gray-400"><span>Wall-Clock Time:</span><strong className="text-amber-400 font-mono">{simData.solvers.sor.execution_time_ms.toFixed(2)} ms</strong></div>
                  <div className="flex justify-between text-gray-400"><span>Iterations:</span><strong className="text-white font-mono">{simData.solvers.sor.iterations}</strong></div>
                  <div className="flex justify-between text-gray-400"><span>Residual:</span><strong className="text-white font-mono">{simData.solvers.sor.final_residual.toExponential(2)}</strong></div>
                  <div className="flex justify-between text-gray-400"><span>Memory Used:</span><strong className="text-amber-400 font-mono">{simData.solvers.sor.memory_mb} MB</strong></div>
                  <div className="flex justify-between text-gray-400"><span>Error vs LU:</span><strong className="text-amber-300 font-mono">{simData.solvers.sor.max_discrepancy_vs_lu.toFixed(4)}°C</strong></div>
                </div>
              </div>
            </div>

            {/* CHARTS SECTION */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
              {/* Convergence Trajectory Chart */}
              <div className="bg-graphite-900 border border-graphite-800 p-5 rounded-xl">
                <h4 className="text-sm font-bold text-white mb-1 flex items-center space-x-2">
                  <Activity className="w-4 h-4 text-cyan-400" />
                  <span>Iterative Residual Trajectory (log10 ||r|| / ||b||)</span>
                </h4>
                <p className="text-xs text-gray-400 mb-4">CG orthogonal projection vs SOR stationary relaxation damping</p>
                <div className="h-64">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={convergenceChartData()}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#1f293d" />
                      <XAxis dataKey="step" stroke="#64748b" tick={{ fontSize: 10 }} />
                      <YAxis stroke="#64748b" tick={{ fontSize: 10 }} />
                      <Tooltip contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155' }} />
                      <Legend />
                      <Line type="monotone" dataKey="CG" stroke="#10b981" strokeWidth={2} dot={false} name="Conjugate Gradient" />
                      <Line type="monotone" dataKey="SOR" stroke="#f59e0b" strokeWidth={2} dot={false} name="SOR (ω=1.35)" />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </div>

              {/* Execution Time Comparison Bar Chart */}
              <div className="bg-graphite-900 border border-graphite-800 p-5 rounded-xl">
                <h4 className="text-sm font-bold text-white mb-1 flex items-center space-x-2">
                  <Zap className="w-4 h-4 text-amber-400" />
                  <span>Solver Wall-Time Comparison (Milliseconds)</span>
                </h4>
                <p className="text-xs text-gray-400 mb-4">Actual profiling time executed on identical coefficient matrices</p>
                <div className="h-64">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={battleBarData()}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#1f293d" />
                      <XAxis dataKey="name" stroke="#64748b" tick={{ fontSize: 10 }} />
                      <YAxis stroke="#64748b" tick={{ fontSize: 10 }} />
                      <Tooltip contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155' }} />
                      <Bar dataKey="time_ms" fill="#3b82f6" name="Time (ms)" radius={[4, 4, 0, 0]} />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* TAB 3: MESH INTELLIGENCE */}
        {activeTab === 'mesh' && (
          <div className="space-y-6">
            <div className="bg-graphite-900 border border-graphite-800 p-5 rounded-xl">
              <h3 className="text-base font-bold text-white mb-1">Mesh Discretization & Grid Independence Laboratory</h3>
              <p className="text-xs text-gray-400 mb-4">
                Examines how spatial mesh refinement affects degrees of freedom, matrix bandwidth, and asymptotic temperature convergence.
              </p>
              {meshLoading ? (
                <div className="p-8 flex items-center justify-center space-x-2 text-cyan-400">
                  <RefreshCw className="w-5 h-5 animate-spin" />
                  <span className="text-xs font-mono">Running Multi-Grid FE Refinement Study...</span>
                </div>
              ) : meshData ? (
                <div className="space-y-4">
                  <div className="overflow-x-auto">
                    <table className="w-full text-left text-xs border border-graphite-800">
                      <thead className="bg-graphite-850 text-gray-300 font-mono">
                        <tr>
                          <th className="p-3">Resolution</th>
                          <th className="p-3">Active Elements</th>
                          <th className="p-3">DOF (Nodes)</th>
                          <th className="p-3">Peak Temp (T_max)</th>
                          <th className="p-3">Direct LU Time</th>
                          <th className="p-3">CG Time</th>
                          <th className="p-3">SOR Time</th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-graphite-800 font-mono">
                        {meshData.data.map((row: any, i: number) => (
                          <tr key={i} className="hover:bg-graphite-850/50">
                            <td className="p-3 font-bold text-cyan-400">{row.resolution}</td>
                            <td className="p-3">{row.elements}</td>
                            <td className="p-3 text-white font-bold">{row.dof}</td>
                            <td className="p-3 text-rose-400">{row.t_max.toFixed(2)} °C</td>
                            <td className="p-3 text-cyan-300">{row.direct_lu_ms.toFixed(1)} ms</td>
                            <td className="p-3 text-emerald-400">{row.cg_ms.toFixed(1)} ms</td>
                            <td className="p-3 text-amber-400">{row.sor_ms.toFixed(1)} ms</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                  <div className="bg-graphite-950 p-4 rounded-lg border border-graphite-800 text-xs text-gray-300 leading-relaxed font-mono">
                    <strong className="text-cyan-400">Analytical Convergence Remark:</strong> {meshData.convergence_analysis}
                  </div>
                </div>
              ) : null}
            </div>
          </div>
        )}

        {/* TAB 4: DESIGN OPTIMIZATION STUDIO */}
        {activeTab === 'optimization' && (
          <div className="space-y-6">
            <div className="bg-graphite-900 border border-graphite-800 p-5 rounded-xl">
              <h3 className="text-base font-bold text-white mb-1">Parametric Fin Optimization Studio</h3>
              <p className="text-xs text-gray-400 mb-4">
                Autonomous evaluation of candidate heat sink topologies to minimize thermal junction resistance.
              </p>
              {optLoading ? (
                <div className="p-8 flex items-center justify-center space-x-2 text-cyan-400">
                  <RefreshCw className="w-5 h-5 animate-spin" />
                  <span className="text-xs font-mono">Evaluating Candidate Fin Configurations...</span>
                </div>
              ) : optData ? (
                <div className="space-y-4">
                  <div className="grid grid-cols-3 gap-4">
                    {optData.results.map((cand: any, idx: number) => (
                      <div key={idx} className="bg-graphite-850 border border-graphite-700 p-4 rounded-xl space-y-2">
                        <div className="text-xs font-bold text-cyan-400 font-mono">{cand.design}</div>
                        <div className="text-2xl font-bold font-mono text-white">{cand.t_max.toFixed(1)}°C</div>
                        <div className="text-xs text-gray-400 space-y-1 pt-2 border-t border-graphite-700 font-mono">
                          <div>Fin Density: <strong>{cand.fin_count} Fins</strong></div>
                          <div>Average Temp: <strong>{cand.t_avg.toFixed(1)}°C</strong></div>
                          <div>Cooling Rate: <strong className="text-emerald-400">{cand.cooling_efficiency} W/K</strong></div>
                        </div>
                      </div>
                    ))}
                  </div>
                  <div className="bg-cyan-950/40 border border-cyan-800/60 p-4 rounded-xl text-xs text-cyan-200 leading-relaxed font-mono">
                    <strong>AI Optimization Recommendation:</strong> {optData.recommendation}
                  </div>
                </div>
              ) : null}
            </div>
          </div>
        )}

        {/* TAB 5: THERMO AI COPILOT */}
        {activeTab === 'ai' && (
          <div className="bg-graphite-900 border border-graphite-800 rounded-xl h-[calc(100vh-140px)] flex flex-col overflow-hidden">
            <div className="p-4 border-b border-graphite-800 bg-graphite-850 flex items-center justify-between">
              <div className="flex items-center space-x-2">
                <Bot className="w-5 h-5 text-cyan-400" />
                <span className="font-bold text-sm text-white">THERMO AI — Thermal Engineering Copilot</span>
              </div>
              <span className="text-[10px] font-mono text-emerald-400 bg-emerald-950 px-2 py-0.5 rounded border border-emerald-800">
                CONTEXT-AWARE REAL SIMULATION ENGINE
              </span>
            </div>

            {/* Messages feed */}
            <div className="flex-1 p-5 overflow-y-auto space-y-4">
              {aiMessages.map((msg, idx) => (
                <div
                  key={idx}
                  className={`flex ${msg.sender === 'user' ? 'justify-end' : 'justify-start'}`}
                >
                  <div
                    className={`max-w-2xl rounded-xl p-4 text-xs leading-relaxed ${
                      msg.sender === 'user'
                        ? 'bg-blue-600 text-white font-medium'
                        : 'bg-graphite-800 border border-graphite-700 text-gray-200 font-mono whitespace-pre-line'
                    }`}
                  >
                    {msg.text}
                  </div>
                </div>
              ))}
              {aiLoading && (
                <div className="flex justify-start">
                  <div className="bg-graphite-800 p-3 rounded-xl flex items-center space-x-2 text-xs text-cyan-400 font-mono">
                    <RefreshCw className="w-3.5 h-3.5 animate-spin" />
                    <span>Analyzing numerical residual and heat flux gradients...</span>
                  </div>
                </div>
              )}
            </div>

            {/* Quick suggested prompt chips */}
            <div className="px-5 py-2 bg-graphite-950 border-t border-graphite-800 flex items-center space-x-2 overflow-x-auto text-[11px]">
              <span className="text-gray-500 font-mono">Suggested:</span>
              {[
                "Why is the center of the heat sink hotter?",
                "Why did CG converge faster than SOR?",
                "Which solver should I use for a larger model?",
                "What happens if I change to copper alloy?"
              ].map((sug, i) => (
                <button
                  key={i}
                  onClick={() => { setAiQuestion(sug); }}
                  className="px-2.5 py-1 bg-graphite-850 hover:bg-graphite-800 text-gray-300 rounded border border-graphite-700 whitespace-nowrap"
                >
                  {sug}
                </button>
              ))}
            </div>

            {/* Input bar */}
            <div className="p-4 border-t border-graphite-800 bg-graphite-850 flex items-center space-x-2">
              <input
                type="text"
                value={aiQuestion}
                onChange={(e) => setAiQuestion(e.target.value)}
                onKeyDown={(e) => { if (e.key === 'Enter') handleAskAI(); }}
                placeholder="Ask THERMO AI about solver mechanics, thermal gradients, or geometry optimization..."
                className="flex-1 bg-graphite-900 border border-graphite-700 rounded-lg px-4 py-2.5 text-xs text-white focus:outline-none focus:border-cyan-500 font-mono"
              />
              <button
                onClick={handleAskAI}
                disabled={aiLoading}
                className="px-5 py-2.5 bg-cyan-600 hover:bg-cyan-500 disabled:opacity-50 text-white font-bold text-xs rounded-lg transition-all"
              >
                Query
              </button>
            </div>
          </div>
        )}
      </main>
    </div>
  );
}

export default App;
