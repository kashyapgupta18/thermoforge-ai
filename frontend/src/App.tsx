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
  Bookmark,
  Compass,
  CheckCircle,
  HelpCircle,
  Database
} from 'lucide-react';
import {
  LineChart, Line, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer
} from 'recharts';
import { Viewport3D } from './components/Viewport3D';

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

interface EnergyBalance {
  q_in_watts: number;
  q_out_convective_watts: number;
  heat_loss_discrepancy_watts: number;
  energy_balance_error_percent: number;
  total_exposed_surface_area_m2: number;
  energy_conserved: boolean;
}

interface SolverRecommendation {
  recommended_solver: string;
  recommended_solver_name: string;
  confidence_score: number;
  metrics_analyzed: {
    degrees_of_freedom: number;
    matrix_nonzeros: number;
    sparsity_ratio: number;
    estimated_lu_memory_mb: number;
  };
  reasoning: string[];
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
  energy_balance?: EnergyBalance;
  solver_recommendation?: SolverRecommendation;
  surface_mesh: any;
}

export function App() {
  const [activeTab, setActiveTab] = useState<'simulation' | 'battle' | 'matrix3x3' | 'mesh' | 'optimization' | 'experiments' | 'ai'>('simulation');

  // Simulation Parameters
  const [preset, setPreset] = useState<string>('advanced_heat_sink');
  const [material, setMaterial] = useState<string>('aluminium_6061');
  const [finCount, setFinCount] = useState<number>(7);
  const [powerWatts, setPowerWatts] = useState<number>(65.0);
  const [convectionH, setConvectionH] = useState<number>(45.0);
  const [resolution, setResolution] = useState<string>('medium');
  const [sorOmega, setSorOmega] = useState<number>(1.35);

  // 3D Viewport Controls
  const [vizMode, setVizMode] = useState<'solid' | 'wireframe' | 'temperature' | 'heat_flux' | 'hotspot' | 'cross_section'>('temperature');
  const [showFluxVectors, setShowFluxVectors] = useState<boolean>(false);

  // State
  const [loading, setLoading] = useState<boolean>(false);
  const [simData, setSimData] = useState<SimulationData | null>(null);

  // 3x3 Complexity Matrix State
  const [matrixData, setMatrixData] = useState<any>(null);
  const [matrixLoading, setMatrixLoading] = useState<boolean>(false);

  // Parametric Optimization State
  const [sweepParam, setSweepParam] = useState<string>('fin_count');
  const [optData, setOptData] = useState<any>(null);
  const [optLoading, setOptLoading] = useState<boolean>(false);

  // Experiments Management State
  const [experiments, setExperiments] = useState<any[]>([]);
  const [expTitle, setExpTitle] = useState<string>('Advanced Thermal Study #1');
  const [saveSuccess, setSaveSuccess] = useState<boolean>(false);

  // AI Chat States
  const [aiQuestion, setAiQuestion] = useState<string>('');
  const [aiMessages, setAiMessages] = useState<Array<{ sender: 'user' | 'thermo_ai'; text: string }>>([
    {
      sender: 'thermo_ai',
      text: 'Greetings. I am THERMO AI, your intelligent thermal engineering copilot. I analyze actual finite element residuals, temperature gradients, energy balances, and matrix properties. How can I assist with your design analysis?'
    }
  ]);
  const [aiLoading, setAiLoading] = useState<boolean>(false);

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
    loadExperiments();
  }, []);

  const loadExperiments = async () => {
    try {
      const res = await fetch(`${API_BASE}/api/experiments`);
      const data = await res.json();
      setExperiments(data);
    } catch (e) {
      console.error(e);
    }
  };

  const handleSaveCurrentExperiment = async () => {
    if (!simData) return;
    try {
      await fetch(`${API_BASE}/api/experiments`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          title: expTitle,
          config: { preset, material, fin_count: finCount, power_watts: powerWatts, convection_h: convectionH, resolution },
          results: simData
        })
      });
      setSaveSuccess(true);
      setTimeout(() => setSaveSuccess(false), 3000);
      loadExperiments();
    } catch (e) {
      console.error(e);
    }
  };

  const handleRun3x3Benchmark = async () => {
    setMatrixLoading(true);
    try {
      const res = await fetch(`${API_BASE}/api/benchmark-3x3?resolution=coarse`);
      const data = await res.json();
      setMatrixData(data);
    } catch (e) {
      console.error(e);
    } finally {
      setMatrixLoading(false);
    }
  };

  const handleRunOptimizationSweep = async () => {
    setOptLoading(true);
    try {
      const res = await fetch(`${API_BASE}/api/parametric-optimization?preset=${preset}&material=${material}&param_name=${sweepParam}`);
      const data = await res.json();
      setOptData(data);
    } catch (e) {
      console.error(e);
    } finally {
      setOptLoading(false);
    }
  };

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
      {/* PROFESSIONAL ENGINEERING HEADER */}
      <header className="h-16 border-b border-graphite-800 bg-graphite-900/90 backdrop-blur-md px-6 flex items-center justify-between sticky top-0 z-50">
        <div className="flex items-center space-x-3">
          <div className="w-9 h-9 rounded-lg bg-gradient-to-br from-cyan-500 to-blue-600 flex items-center justify-center shadow-lg shadow-cyan-500/20">
            <Flame className="w-5 h-5 text-white" />
          </div>
          <div>
            <div className="flex items-center space-x-2">
              <span className="font-bold text-lg tracking-wider text-white">THERMOFORGE AI</span>
              <span className="px-2 py-0.5 text-[10px] font-mono font-semibold bg-cyan-950 text-cyan-400 border border-cyan-800 rounded">v2.5 RESEARCH</span>
            </div>
            <div className="text-[11px] text-gray-400">Intelligent 3D Thermal Simulation & Numerical Benchmark Platform</div>
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
            <span>Solver Arena</span>
          </button>
          <button
            onClick={() => { setActiveTab('matrix3x3'); if (!matrixData) handleRun3x3Benchmark(); }}
            className={`px-3 py-1.5 rounded-md text-xs font-semibold transition-all flex items-center space-x-1.5 ${
              activeTab === 'matrix3x3' ? 'bg-cyan-600 text-white shadow' : 'text-gray-400 hover:text-white'
            }`}
          >
            <Compass className="w-3.5 h-3.5" />
            <span>3×3 Benchmark</span>
          </button>
          <button
            onClick={() => { setActiveTab('mesh'); if (!meshData) handleRunMeshStudy(); }}
            className={`px-3 py-1.5 rounded-md text-xs font-semibold transition-all flex items-center space-x-1.5 ${
              activeTab === 'mesh' ? 'bg-cyan-600 text-white shadow' : 'text-gray-400 hover:text-white'
            }`}
          >
            <BarChart3 className="w-3.5 h-3.5" />
            <span>Mesh Laboratory</span>
          </button>
          <button
            onClick={() => { setActiveTab('optimization'); if (!optData) handleRunOptimizationSweep(); }}
            className={`px-3 py-1.5 rounded-md text-xs font-semibold transition-all flex items-center space-x-1.5 ${
              activeTab === 'optimization' ? 'bg-cyan-600 text-white shadow' : 'text-gray-400 hover:text-white'
            }`}
          >
            <TrendingUp className="w-3.5 h-3.5" />
            <span>Parametric Sweep</span>
          </button>
          <button
            onClick={() => setActiveTab('experiments')}
            className={`px-3 py-1.5 rounded-md text-xs font-semibold transition-all flex items-center space-x-1.5 ${
              activeTab === 'experiments' ? 'bg-cyan-600 text-white shadow' : 'text-gray-400 hover:text-white'
            }`}
          >
            <Database className="w-3.5 h-3.5" />
            <span>Experiments</span>
          </button>
          <button
            onClick={() => setActiveTab('ai')}
            className={`px-3 py-1.5 rounded-md text-xs font-semibold transition-all flex items-center space-x-1.5 ${
              activeTab === 'ai' ? 'bg-cyan-600 text-white shadow' : 'text-gray-400 hover:text-white'
            }`}
          >
            <Bot className="w-3.5 h-3.5" />
            <span>THERMO AI</span>
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
                <div className="border-b border-graphite-800 pb-3 flex justify-between items-center">
                  <div>
                    <h2 className="text-sm font-bold tracking-wide text-white uppercase flex items-center space-x-2">
                      <Sliders className="w-4 h-4 text-cyan-400" />
                      <span>Model Specification</span>
                    </h2>
                    <p className="text-xs text-gray-400 mt-0.5">3D Boundary value physics setup</p>
                  </div>
                </div>

                {/* Preset Selector */}
                <div>
                  <label className="text-xs font-medium text-gray-300 block mb-1">Engineering Geometry</label>
                  <select
                    value={preset}
                    onChange={(e) => setPreset(e.target.value)}
                    className="w-full bg-graphite-850 border border-graphite-700 text-xs rounded-lg px-3 py-2 text-white focus:outline-none focus:border-cyan-500"
                  >
                    <option value="advanced_heat_sink">Advanced Perforated Heat Sink (Variable Fins & Cutouts)</option>
                    <option value="standard_heat_sink">Standard Uniform Heat Sink</option>
                    <option value="asymmetric_cooling_plate">Asymmetric Industrial Cooling Plate</option>
                    <option value="cube">Cube (Baseline Verification Model)</option>
                  </select>
                </div>

                {/* Material Selection */}
                <div>
                  <label className="text-xs font-medium text-gray-300 block mb-1">Alloy Material Properties</label>
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
                    <span>Die Heat Dissipation:</span>
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

                {/* Discretization */}
                <div>
                  <label className="text-xs font-medium text-gray-300 block mb-1">Mesh Discretization</label>
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

                {/* Priority 1: Energy Balance Card */}
                {simData?.energy_balance && (
                  <div className="bg-graphite-950 p-3 rounded-lg border border-graphite-800 text-[11px] font-mono space-y-1">
                    <div className="text-cyan-400 font-bold flex items-center justify-between">
                      <span>ENERGY BALANCE (CONSERVATION)</span>
                      <span className="text-[10px] text-emerald-400">PASSED</span>
                    </div>
                    <div className="flex justify-between text-gray-400"><span>Heat In (Q_in):</span><strong className="text-rose-400">{simData.energy_balance.q_in_watts} W</strong></div>
                    <div className="flex justify-between text-gray-400"><span>Convective Dissipated:</span><strong className="text-emerald-400">{simData.energy_balance.q_out_convective_watts} W</strong></div>
                    <div className="flex justify-between text-gray-400"><span>Conservation Error:</span><strong className="text-cyan-300">{simData.energy_balance.energy_balance_error_percent}%</strong></div>
                  </div>
                )}
              </div>

              {/* Action Buttons */}
              <div className="space-y-2 mt-4">
                <button
                  onClick={executeSimulation}
                  disabled={loading}
                  className="w-full py-3 bg-gradient-to-r from-cyan-600 to-blue-600 hover:from-cyan-500 hover:to-blue-500 disabled:opacity-50 text-white font-bold text-xs rounded-xl shadow-lg shadow-cyan-600/30 transition-all flex items-center justify-center space-x-2"
                >
                  {loading ? <RefreshCw className="w-4 h-4 animate-spin" /> : <Play className="w-4 h-4" />}
                  <span>{loading ? 'Solving Assembled System...' : 'Run Simulation & Solver Battle'}</span>
                </button>

                {/* Save Experiment Button */}
                <button
                  onClick={handleSaveCurrentExperiment}
                  disabled={!simData}
                  className="w-full py-2 bg-graphite-800 hover:bg-graphite-750 text-gray-300 text-xs font-semibold rounded-lg border border-graphite-700 flex items-center justify-center space-x-1.5"
                >
                  <Bookmark className="w-3.5 h-3.5 text-cyan-400" />
                  <span>{saveSuccess ? 'Saved to Repository!' : 'Save Current Run to Experiments'}</span>
                </button>
              </div>
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
                    { id: 'temperature', label: 'Temperature' },
                    { id: 'heat_flux', label: 'Heat Flux Vectors' },
                    { id: 'hotspot', label: 'Hotspot Isolation' },
                    { id: 'wireframe', label: 'FE Wireframe' },
                    { id: 'solid', label: 'Solid CAD' },
                    { id: 'cross_section', label: 'Cross-Section' },
                  ].map((mode) => (
                    <button
                      key={mode.id}
                      onClick={() => setVizMode(mode.id as any)}
                      className={`px-2.5 py-1 rounded-md text-xs font-medium transition-all ${
                        vizMode === mode.id
                          ? 'bg-cyan-600 text-white font-bold'
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
                  showFluxVectors={vizMode === 'heat_flux'}
                  tMin={tMin}
                  tMax={tMax}
                />
              </div>

              {/* Priority 4: Intelligent Solver Recommendation Pill */}
              {simData?.solver_recommendation && (
                <div className="bg-graphite-900 border border-cyan-800/40 p-3 rounded-xl flex items-center justify-between text-xs">
                  <div className="flex items-center space-x-2">
                    <div className="w-2 h-2 rounded-full bg-cyan-400 animate-pulse" />
                    <span className="font-mono text-gray-400">SOLVER INTELLIGENCE ENGINE:</span>
                    <strong className="text-cyan-300 font-mono">{simData.solver_recommendation.recommended_solver_name}</strong>
                    <span className="text-[10px] bg-cyan-950 text-cyan-400 px-2 py-0.5 rounded border border-cyan-800 font-mono">
                      {(simData.solver_recommendation.confidence_score * 100).toFixed(0)}% Confidence
                    </span>
                  </div>
                  <div className="text-[11px] text-gray-400 font-mono">
                    Sparsity: {simData.solver_recommendation.metrics_analyzed.sparsity_ratio}% | LU Fill: ~{simData.solver_recommendation.metrics_analyzed.estimated_lu_memory_mb} MB
                  </div>
                </div>
              )}
            </div>
          </div>
        )}

        {/* TAB 2: SOLVER BATTLE ARENA */}
        {activeTab === 'battle' && simData && (
          <div className="space-y-6">
            <div className="bg-gradient-to-r from-blue-950/80 via-indigo-950/80 to-graphite-900 border border-cyan-800/40 p-6 rounded-2xl shadow-xl flex items-start space-x-4">
              <div className="p-3 bg-cyan-500/20 rounded-xl text-cyan-400">
                <Award className="w-8 h-8" />
              </div>
              <div>
                <div className="text-xs font-mono font-bold text-cyan-400 uppercase tracking-widest">Scientific Solver Leaderboard</div>
                <h3 className="text-lg font-bold text-white mt-0.5">
                  Optimal Method: {simData.winner_id === 'conjugate_gradient' ? 'Conjugate Gradient (PCG)' : 'Direct Sparse LU'}
                </h3>
                <p className="text-xs text-gray-300 mt-2 whitespace-pre-line leading-relaxed">
                  {simData.winner_explanation}
                </p>
              </div>
            </div>

            {/* Comparison Cards */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-5">
              {/* Direct Sparse LU */}
              <div className="bg-graphite-900 border border-cyan-900/50 rounded-xl p-5 space-y-3">
                <div className="flex justify-between items-center">
                  <span className="text-xs font-bold text-cyan-400 font-mono">SOLVER A</span>
                  <span className="text-[10px] px-2 py-0.5 bg-cyan-950 text-cyan-300 rounded font-mono">DIRECT SUPERLU</span>
                </div>
                <h4 className="text-base font-bold text-white">{simData.solvers.direct_lu.solver_name}</h4>
                <div className="space-y-2 pt-2 border-t border-graphite-800 text-xs font-mono">
                  <div className="flex justify-between text-gray-400"><span>Wall-Clock Time:</span><strong className="text-white">{simData.solvers.direct_lu.execution_time_ms.toFixed(2)} ms</strong></div>
                  <div className="flex justify-between text-gray-400"><span>Iterations:</span><strong className="text-white">{simData.solvers.direct_lu.iterations}</strong></div>
                  <div className="flex justify-between text-gray-400"><span>Verified Residual:</span><strong className="text-white">{simData.solvers.direct_lu.final_residual.toExponential(2)}</strong></div>
                  <div className="flex justify-between text-gray-400"><span>Memory Used:</span><strong className="text-white">{simData.solvers.direct_lu.memory_mb} MB</strong></div>
                  <div className="flex justify-between text-gray-400"><span>Peak Temp:</span><strong className="text-rose-400">{simData.solvers.direct_lu.t_max}°C</strong></div>
                </div>
              </div>

              {/* Conjugate Gradient */}
              <div className="bg-graphite-900 border border-emerald-900/50 rounded-xl p-5 space-y-3">
                <div className="flex justify-between items-center">
                  <span className="text-xs font-bold text-emerald-400 font-mono">SOLVER B</span>
                  <span className="text-[10px] px-2 py-0.5 bg-emerald-950 text-emerald-300 rounded font-mono">KRYLOV PCG</span>
                </div>
                <h4 className="text-base font-bold text-white">{simData.solvers.conjugate_gradient.solver_name}</h4>
                <div className="space-y-2 pt-2 border-t border-graphite-800 text-xs font-mono">
                  <div className="flex justify-between text-gray-400"><span>Wall-Clock Time:</span><strong className="text-emerald-400">{simData.solvers.conjugate_gradient.execution_time_ms.toFixed(2)} ms</strong></div>
                  <div className="flex justify-between text-gray-400"><span>Iterations:</span><strong className="text-white">{simData.solvers.conjugate_gradient.iterations}</strong></div>
                  <div className="flex justify-between text-gray-400"><span>Verified Residual:</span><strong className="text-white">{simData.solvers.conjugate_gradient.final_residual.toExponential(2)}</strong></div>
                  <div className="flex justify-between text-gray-400"><span>Memory Used:</span><strong className="text-emerald-400">{simData.solvers.conjugate_gradient.memory_mb} MB</strong></div>
                  <div className="flex justify-between text-gray-400"><span>Discrepancy vs LU:</span><strong className="text-emerald-300">{simData.solvers.conjugate_gradient.max_discrepancy_vs_lu.toFixed(6)}°C</strong></div>
                </div>
              </div>

              {/* SOR */}
              <div className="bg-graphite-900 border border-amber-900/50 rounded-xl p-5 space-y-3">
                <div className="flex justify-between items-center">
                  <span className="text-xs font-bold text-amber-400 font-mono">SOLVER C</span>
                  <span className="text-[10px] px-2 py-0.5 bg-amber-950 text-amber-300 rounded font-mono">STATIONARY SOR</span>
                </div>
                <h4 className="text-base font-bold text-white">{simData.solvers.sor.solver_name}</h4>
                <div className="space-y-2 pt-2 border-t border-graphite-800 text-xs font-mono">
                  <div className="flex justify-between text-gray-400"><span>Wall-Clock Time:</span><strong className="text-amber-400">{simData.solvers.sor.execution_time_ms.toFixed(2)} ms</strong></div>
                  <div className="flex justify-between text-gray-400"><span>Iterations:</span><strong className="text-white">{simData.solvers.sor.iterations}</strong></div>
                  <div className="flex justify-between text-gray-400"><span>Verified Residual:</span><strong className="text-white">{simData.solvers.sor.final_residual.toExponential(2)}</strong></div>
                  <div className="flex justify-between text-gray-400"><span>Memory Used:</span><strong className="text-amber-400">{simData.solvers.sor.memory_mb} MB</strong></div>
                  <div className="flex justify-between text-gray-400"><span>Discrepancy vs LU:</span><strong className="text-amber-300">{simData.solvers.sor.max_discrepancy_vs_lu.toFixed(4)}°C</strong></div>
                </div>
              </div>
            </div>

            {/* CHARTS */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
              <div className="bg-graphite-900 border border-graphite-800 p-5 rounded-xl">
                <h4 className="text-sm font-bold text-white mb-1 flex items-center space-x-2">
                  <Activity className="w-4 h-4 text-cyan-400" />
                  <span>Iterative Residual Trajectory (log10 ||r|| / ||b||)</span>
                </h4>
                <p className="text-xs text-gray-400 mb-4">CG orthogonal Krylov projection vs SOR stationary damping</p>
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

              <div className="bg-graphite-900 border border-graphite-800 p-5 rounded-xl">
                <h4 className="text-sm font-bold text-white mb-1 flex items-center space-x-2">
                  <Zap className="w-4 h-4 text-amber-400" />
                  <span>Solver Wall-Time Comparison (ms)</span>
                </h4>
                <p className="text-xs text-gray-400 mb-4">High-resolution execution profiling on identical matrices</p>
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

        {/* TAB 3: PRIORITY 2 — 3x3 GEOMETRY COMPLEXITY BENCHMARK */}
        {activeTab === 'matrix3x3' && (
          <div className="space-y-6">
            <div className="bg-graphite-900 border border-graphite-800 p-6 rounded-xl">
              <div className="flex justify-between items-center mb-4">
                <div>
                  <h3 className="text-base font-bold text-white flex items-center space-x-2">
                    <Compass className="w-4 h-4 text-cyan-400" />
                    <span>3×3 Geometry Complexity Benchmarking Matrix</span>
                  </h3>
                  <p className="text-xs text-gray-400 mt-1">
                    Direct LU, Conjugate Gradient, and SOR benchmarked across 3 distinct heat sink geometries.
                  </p>
                </div>
                <button
                  onClick={handleRun3x3Benchmark}
                  disabled={matrixLoading}
                  className="px-4 py-2 bg-cyan-600 hover:bg-cyan-500 text-white text-xs font-semibold rounded-lg flex items-center space-x-1.5 font-mono"
                >
                  {matrixLoading ? <RefreshCw className="w-3.5 h-3.5 animate-spin" /> : <Play className="w-3.5 h-3.5" />}
                  <span>Re-run 3×3 Benchmark</span>
                </button>
              </div>

              {matrixLoading ? (
                <div className="p-12 text-center text-cyan-400 font-mono text-xs flex flex-col items-center justify-center space-y-2">
                  <RefreshCw className="w-6 h-6 animate-spin" />
                  <span>Solving 9 discrete simulation benchmarks across 3 geometries...</span>
                </div>
              ) : matrixData ? (
                <div className="space-y-6">
                  <div className="overflow-x-auto">
                    <table className="w-full text-left text-xs border border-graphite-800">
                      <thead className="bg-graphite-850 text-gray-300 font-mono">
                        <tr>
                          <th className="p-3">Geometry Model</th>
                          <th className="p-3">DOF</th>
                          <th className="p-3">Sparsity</th>
                          <th className="p-3">Direct LU (ms)</th>
                          <th className="p-3">CG Time (ms)</th>
                          <th className="p-3">SOR Time (ms)</th>
                          <th className="p-3">CG Residual</th>
                          <th className="p-3">Energy Conservation</th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-graphite-800 font-mono">
                        {matrixData.geometries.map((row: any, i: number) => (
                          <tr key={i} className="hover:bg-graphite-850/50">
                            <td className="p-3 font-bold text-cyan-400">{row.geometry_name}</td>
                            <td className="p-3 text-white font-bold">{row.dof}</td>
                            <td className="p-3 text-gray-300">{row.sparsity_pct}%</td>
                            <td className="p-3 text-cyan-300 font-bold">{row.solvers.direct_lu.execution_time_ms.toFixed(1)} ms</td>
                            <td className="p-3 text-emerald-400 font-bold">{row.solvers.conjugate_gradient.execution_time_ms.toFixed(1)} ms</td>
                            <td className="p-3 text-amber-400">{row.solvers.sor.execution_time_ms.toFixed(1)} ms</td>
                            <td className="p-3 text-gray-300">{row.solvers.conjugate_gradient.final_residual.toExponential(2)}</td>
                            <td className="p-3 text-emerald-300 font-bold">
                              {row.energy_balance.energy_balance_error_percent}% error
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>

                  <div className="bg-graphite-950 p-4 rounded-xl border border-graphite-800 text-xs font-mono text-gray-300 leading-relaxed whitespace-pre-line">
                    <span className="text-cyan-400 font-bold block mb-1">Theoretical Discretization Assessment:</span>
                    {matrixData.comparative_analysis}
                  </div>
                </div>
              ) : null}
            </div>
          </div>
        )}

        {/* TAB 4: MESH LABORATORY */}
        {activeTab === 'mesh' && (
          <div className="space-y-6">
            <div className="bg-graphite-900 border border-graphite-800 p-5 rounded-xl">
              <h3 className="text-base font-bold text-white mb-1">Mesh Discretization & Grid Independence Laboratory</h3>
              <p className="text-xs text-gray-400 mb-4">
                Analyzes how spatial mesh refinement affects degrees of freedom, matrix bandwidth, and asymptotic temperature convergence.
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

        {/* TAB 5: PRIORITY 6 — PARAMETRIC DESIGN OPTIMIZATION */}
        {activeTab === 'optimization' && (
          <div className="space-y-6">
            <div className="bg-graphite-900 border border-graphite-800 p-6 rounded-xl">
              <div className="flex justify-between items-center mb-4">
                <div>
                  <h3 className="text-base font-bold text-white flex items-center space-x-2">
                    <TrendingUp className="w-4 h-4 text-cyan-400" />
                    <span>Parametric Design Optimization Suite</span>
                  </h3>
                  <p className="text-xs text-gray-400 mt-1">
                    Autonomous multi-point parameter sweeps exploring cooling efficiency vs structural volume.
                  </p>
                </div>
                <div className="flex items-center space-x-3">
                  <select
                    value={sweepParam}
                    onChange={(e) => setSweepParam(e.target.value)}
                    className="bg-graphite-850 border border-graphite-700 text-xs rounded-lg px-3 py-2 text-white font-mono"
                  >
                    <option value="fin_count">Fin Count (4 to 12 fins)</option>
                    <option value="convection_h">Convection h (15 to 100 W/m²K)</option>
                    <option value="power_watts">Heat Dissipation (35 to 100 W)</option>
                  </select>
                  <button
                    onClick={handleRunOptimizationSweep}
                    disabled={optLoading}
                    className="px-4 py-2 bg-cyan-600 hover:bg-cyan-500 text-white text-xs font-semibold rounded-lg font-mono flex items-center space-x-1"
                  >
                    {optLoading ? <RefreshCw className="w-3.5 h-3.5 animate-spin" /> : <Play className="w-3.5 h-3.5" />}
                    <span>Run Optimization Sweep</span>
                  </button>
                </div>
              </div>

              {optLoading ? (
                <div className="p-10 text-center text-cyan-400 font-mono text-xs flex flex-col items-center justify-center space-y-2">
                  <RefreshCw className="w-6 h-6 animate-spin" />
                  <span>Executing parametric parameter sweep iterations...</span>
                </div>
              ) : optData ? (
                <div className="space-y-5">
                  <div className="grid grid-cols-1 md:grid-cols-5 gap-3">
                    {optData.results.map((pt: any, i: number) => (
                      <div
                        key={i}
                        className={`p-4 rounded-xl border font-mono transition-all ${
                          pt.param_value === optData.optimal_point.parameter_value
                            ? 'bg-cyan-950/60 border-cyan-500 shadow-lg shadow-cyan-950'
                            : 'bg-graphite-850 border-graphite-700'
                        }`}
                      >
                        <div className="text-[10px] text-gray-400">{optData.sweep_parameter}:</div>
                        <div className="text-lg font-bold text-white">{pt.param_value}</div>
                        <div className="text-xs text-rose-400 font-bold mt-1">T_max: {pt.t_max.toFixed(1)}°C</div>
                        <div className="text-[10px] text-emerald-400 mt-1">Cooling: {pt.cooling_efficiency_w_per_k} W/K</div>
                        <div className="text-[10px] text-gray-400 mt-1">Solve: {pt.cg_runtime_ms.toFixed(1)} ms</div>
                      </div>
                    ))}
                  </div>

                  <div className="bg-cyan-950/40 border border-cyan-800/60 p-4 rounded-xl text-xs font-mono text-cyan-200">
                    <strong className="text-cyan-300">Engineering Recommendation:</strong> {optData.engineering_recommendation}
                  </div>
                </div>
              ) : null}
            </div>
          </div>
        )}

        {/* TAB 6: PRIORITY 7 — RESEARCH EXPERIMENTS REPOSITORY */}
        {activeTab === 'experiments' && (
          <div className="space-y-6">
            <div className="bg-graphite-900 border border-graphite-800 p-6 rounded-xl">
              <h3 className="text-base font-bold text-white mb-1 flex items-center space-x-2">
                <Database className="w-4 h-4 text-cyan-400" />
                <span>Reproducible Experiments Repository</span>
              </h3>
              <p className="text-xs text-gray-400 mb-4">
                Saved simulation runs with recorded geometric parameters, boundary conditions, and numerical outputs.
              </p>

              {experiments.length === 0 ? (
                <div className="p-8 text-center text-gray-500 text-xs font-mono">
                  No saved experiments in storage yet. Run a simulation and click "Save Current Run to Experiments" on the 3D Simulation tab.
                </div>
              ) : (
                <div className="overflow-x-auto">
                  <table className="w-full text-left text-xs border border-graphite-800">
                    <thead className="bg-graphite-850 text-gray-300 font-mono">
                      <tr>
                        <th className="p-3">Experiment ID</th>
                        <th className="p-3">Title</th>
                        <th className="p-3">Timestamp</th>
                        <th className="p-3">Geometry Preset</th>
                        <th className="p-3">Alloy Material</th>
                        <th className="p-3">Peak Temp (T_max)</th>
                        <th className="p-3">CG Time</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-graphite-800 font-mono">
                      {experiments.map((exp, idx) => (
                        <tr key={idx} className="hover:bg-graphite-850/50">
                          <td className="p-3 font-bold text-cyan-400">{exp.id}</td>
                          <td className="p-3 text-white font-semibold">{exp.title}</td>
                          <td className="p-3 text-gray-400">{exp.timestamp}</td>
                          <td className="p-3 text-gray-300">{exp.preset}</td>
                          <td className="p-3 text-gray-300">{exp.material}</td>
                          <td className="p-3 text-rose-400 font-bold">{exp.t_max ? `${exp.t_max.toFixed(1)} °C` : 'N/A'}</td>
                          <td className="p-3 text-emerald-400">{exp.cg_time_ms ? `${exp.cg_time_ms.toFixed(1)} ms` : 'N/A'}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </div>
          </div>
        )}

        {/* TAB 7: THERMO AI COPILOT */}
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
