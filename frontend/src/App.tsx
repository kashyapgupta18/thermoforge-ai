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
  AlertTriangle,
  XCircle,
  Database
} from 'lucide-react';
import {
  LineChart,
  Line,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  ReferenceLine,
  ReferenceDot
} from 'recharts';
import { Viewport3D } from './components/Viewport3D';

const API_BASE = import.meta.env.VITE_API_URL || '';

interface ResidualPoint {
  iteration: number;
  residual: number;
}

interface EnergyBalance {
  q_in_watts: number;
  q_out_convective_watts: number;
  heat_loss_discrepancy_watts?: number;
  imbalance_watts?: number;
  relative_error?: number;
  energy_balance_error_percent: number;
  energy_balance_error_formatted?: string;
  total_exposed_surface_area_m2?: number;
  exposed_area_m2?: number;
  energy_conserved: boolean;
  calculation_explanation?: string;
}

interface SolverStats {
  solver_id: string;
  solver_name: string;
  execution_time_ms: number;
  iterations: number;
  converged: boolean;
  status_label?: string;
  final_residual: number;
  absolute_residual?: number;
  residuals_history: Array<ResidualPoint | number>;
  memory_mb: number;
  nnz_factors?: number;
  fill_in_ratio?: number;
  omega?: number;
  asymptotic_rate_estimate?: number | null;
  t_min: number;
  t_max: number;
  t_avg: number;
  max_discrepancy_vs_lu: number;
  relative_l2_error_vs_lu?: number;
  rms_discrepancy_vs_lu?: number;
  energy_balance?: EnergyBalance;
  description: string;
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
    sparsity_pct?: number;
    assembly_time_ms: number;
  };
  solver_settings?: {
    tolerance_rel: number;
    max_iterations: number;
    sor_omega: number;
    initial_guess: string;
    initial_guess_kelvin: number;
    residual_definition: string;
    history_sampling: string;
  };
  solvers: {
    direct_lu: SolverStats;
    conjugate_gradient: SolverStats;
    sor: SolverStats;
  };
  winner_id: string | null;
  winner_explanation: string;
  engineering_conclusions?: string[];
  energy_balance?: EnergyBalance;
  solver_recommendation?: SolverRecommendation;
  surface_mesh: any;
}

const StatusBadge: React.FC<{ converged: boolean; label?: string }> = ({ converged, label }) => {
  const text = label || (converged ? 'Converged' : 'Maximum iterations reached');
  if (converged && text === 'Converged') {
    return (
      <span className="inline-flex items-center space-x-1 px-2 py-0.5 text-[10px] font-mono font-bold bg-emerald-950 text-emerald-300 border border-emerald-700 rounded">
        <CheckCircle className="w-3 h-3 text-emerald-400" />
        <span>CONVERGED</span>
      </span>
    );
  }
  if (text.toLowerCase().includes('diverged') || text.toLowerCase().includes('invalid')) {
    return (
      <span className="inline-flex items-center space-x-1 px-2 py-0.5 text-[10px] font-mono font-bold bg-rose-950 text-rose-300 border border-rose-700 rounded">
        <XCircle className="w-3 h-3 text-rose-400" />
        <span>{text.toUpperCase()}</span>
      </span>
    );
  }
  return (
    <span className="inline-flex items-center space-x-1 px-2 py-0.5 text-[10px] font-mono font-bold bg-amber-950 text-amber-300 border border-amber-700 rounded">
      <AlertTriangle className="w-3 h-3 text-amber-400" />
      <span>{text.toUpperCase()} (UNCONVERGED)</span>
    </span>
  );
};

const formatPctError = (eb?: EnergyBalance) => {
  if (!eb) return '—';
  if (eb.energy_balance_error_formatted) return eb.energy_balance_error_formatted;
  const pct = eb.energy_balance_error_percent ?? (eb.relative_error ?? 0) * 100;
  return pct < 1e-2 ? `${pct.toExponential(2)}%` : `${pct.toFixed(3)}%`;
};

export function App() {
  const [activeTab, setActiveTab] = useState<'simulation' | 'battle' | 'matrix3x3' | 'mesh' | 'optimization' | 'experiments' | 'ai'>('simulation');

  // Simulation Parameters
  const [preset, setPreset] = useState<string>('advanced_heat_sink');
  const [material, setMaterial] = useState<string>('aluminium_6061');
  const [finCount, setFinCount] = useState<number>(7);
  const [powerWatts, setPowerWatts] = useState<number>(65.0);
  const [convectionH, setConvectionH] = useState<number>(45.0);
  const [resolution, setResolution] = useState<string>('coarse');
  const [sorOmega, setSorOmega] = useState<number>(1.9);
  const [initialGuess, setInitialGuess] = useState<string>('lumped');

  // 3D Viewport Controls
  const [vizMode, setVizMode] = useState<'solid' | 'wireframe' | 'temperature' | 'heat_flux' | 'hotspot' | 'cross_section'>('temperature');

  // State
  const [loading, setLoading] = useState<boolean>(false);
  const [simData, setSimData] = useState<SimulationData | null>(null);

  // SOR Omega Sweep Study State
  const [omegaStudy, setOmegaStudy] = useState<any>(null);
  const [omegaLoading, setOmegaLoading] = useState<boolean>(false);

  // 3x3 Complexity Matrix State
  const [matrixData, setMatrixData] = useState<any>(null);
  const [matrixLoading, setMatrixLoading] = useState<boolean>(false);
  const [matrixOmega, setMatrixOmega] = useState<number>(1.9);
  const [matrixInitGuess, setMatrixInitGuess] = useState<string>('lumped');

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
      text: 'Greetings. I am THERMO AI, your thermal engineering copilot. All interpretations are grounded in your actual computed FEM residuals, temperature fields, and global energy balances. Ask me about solver convergence, SOR ω sensitivity, or thermal spreading resistance.'
    }
  ]);
  const [aiLoading, setAiLoading] = useState<boolean>(false);

  // Mesh Study State
  const [meshData, setMeshData] = useState<any>(null);
  const [meshLoading, setMeshLoading] = useState<boolean>(false);

  const executeSimulation = async (overrideOmega?: number, overrideInit?: string) => {
    setLoading(true);
    const w = overrideOmega !== undefined ? overrideOmega : sorOmega;
    const ig = overrideInit !== undefined ? overrideInit : initialGuess;
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
          sor_omega: w,
          initial_guess: ig,
          ambient_temp_celsius: 20.0
        })
      });
      const data = await res.json();
      setSimData(data);
    } catch (err) {
      console.error('Simulation failed:', err);
    } finally {
      setLoading(false);
    }
  };

  const handleRunOmegaSweep = async (igMode?: string) => {
    setOmegaLoading(true);
    const ig = igMode || initialGuess;
    try {
      const res = await fetch(
        `${API_BASE}/api/sor-omega-sweep?preset=${preset}&material=${material}&fin_count=${finCount}&resolution=coarse&initial_guess=${ig}`
      );
      const data = await res.json();
      setOmegaStudy(data);
    } catch (e) {
      console.error(e);
    } finally {
      setOmegaLoading(false);
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
          config: { preset, material, fin_count: finCount, power_watts: powerWatts, convection_h: convectionH, resolution, sor_omega: sorOmega, initial_guess: initialGuess },
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

  const handleRun3x3Benchmark = async (wOverride?: number, igOverride?: string) => {
    setMatrixLoading(true);
    const w = wOverride !== undefined ? wOverride : matrixOmega;
    const ig = igOverride !== undefined ? igOverride : matrixInitGuess;
    try {
      const res = await fetch(`${API_BASE}/api/benchmark-3x3?resolution=coarse&sor_omega=${w}&initial_guess=${ig}`);
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
      const res = await fetch(`${API_BASE}/api/mesh-intelligence?preset=${preset}&material=${material}&fin_count=${finCount}&sor_omega=${sorOmega}`);
      const data = await res.json();
      setMeshData(data);
    } catch (e) {
      console.error(e);
    } finally {
      setMeshLoading(false);
    }
  };

  // Full unsampled convergence chart data keyed by true iteration index (0..maxIter)
  const convergenceChartData = () => {
    if (!simData) return { series: [], cgEnd: null, sorEnd: null };
    const cgRaw = simData.solvers.conjugate_gradient.residuals_history || [];
    const sorRaw = simData.solvers.sor.residuals_history || [];

    const byIter = new Map<number, { iteration: number; CG?: number; SOR?: number }>();

    cgRaw.forEach((item: any, idx: number) => {
      const it = typeof item === 'object' && item !== null ? Number(item.iteration) : idx;
      const val = typeof item === 'object' && item !== null ? Number(item.residual) : Number(item);
      if (Number.isFinite(val) && val > 0) {
        const cur = byIter.get(it) || { iteration: it };
        cur.CG = Number(Math.log10(Math.max(1e-15, val)).toFixed(4));
        byIter.set(it, cur);
      }
    });

    sorRaw.forEach((item: any, idx: number) => {
      const it = typeof item === 'object' && item !== null ? Number(item.iteration) : idx;
      const val = typeof item === 'object' && item !== null ? Number(item.residual) : Number(item);
      if (Number.isFinite(val) && val > 0) {
        const cur = byIter.get(it) || { iteration: it };
        cur.SOR = Number(Math.log10(Math.max(1e-15, val)).toFixed(4));
        byIter.set(it, cur);
      }
    });

    const series = Array.from(byIter.values()).sort((a, b) => a.iteration - b.iteration);
    const cgPts = series.filter(d => d.CG !== undefined);
    const sorPts = series.filter(d => d.SOR !== undefined);

    return {
      series,
      cgEnd: cgPts.length > 0 ? cgPts[cgPts.length - 1] : null,
      sorEnd: sorPts.length > 0 ? sorPts[sorPts.length - 1] : null
    };
  };

  const battleBarData = () => {
    if (!simData) return [];
    const s = simData.solvers;
    return [
      { name: 'Direct Sparse LU', time_ms: Number(s.direct_lu.execution_time_ms.toFixed(2)), memory_mb: s.direct_lu.memory_mb, iters: 1 },
      { name: 'Conjugate Gradient (PCG)', time_ms: Number(s.conjugate_gradient.execution_time_ms.toFixed(2)), memory_mb: s.conjugate_gradient.memory_mb, iters: s.conjugate_gradient.iterations },
      { name: `SOR (ω=${sorOmega.toFixed(2)})`, time_ms: Number(s.sor.execution_time_ms.toFixed(2)), memory_mb: s.sor.memory_mb, iters: s.sor.iterations },
    ];
  };

  const tMin = simData ? simData.solvers.direct_lu.t_min : 20.0;
  const tMax = simData ? simData.solvers.direct_lu.t_max : 75.0;
  const chartInfo = convergenceChartData();

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
              <span className="px-2 py-0.5 text-[10px] font-mono font-semibold bg-cyan-950 text-cyan-400 border border-cyan-800 rounded">v3.0 AUDIT</span>
            </div>
            <div className="text-[11px] text-gray-400">Intelligent 3D Thermal Simulation &amp; Numerical Solver Verification Platform</div>
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
            onClick={() => { setActiveTab('battle'); if (!omegaStudy) handleRunOmegaSweep(); }}
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
            <span>Verification &amp; Mesh</span>
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
          <div className="grid grid-cols-12 gap-5">
            {/* LEFT CONTROL PANEL */}
            <div className="col-span-4 bg-graphite-900 border border-graphite-800 rounded-xl p-5 flex flex-col justify-between">
              <div className="space-y-3.5">
                <div className="border-b border-graphite-800 pb-2.5">
                  <h2 className="text-sm font-bold tracking-wide text-white uppercase flex items-center space-x-2">
                    <Sliders className="w-4 h-4 text-cyan-400" />
                    <span>Physics &amp; Solver Setup</span>
                  </h2>
                  <p className="text-xs text-gray-400 mt-0.5">3D FEM Conduction &amp; Relaxation Controls</p>
                </div>

                {/* Preset Selector */}
                <div>
                  <label className="text-xs font-medium text-gray-300 block mb-1">Engineering Geometry</label>
                  <select
                    value={preset}
                    onChange={(e) => setPreset(e.target.value)}
                    className="w-full bg-graphite-850 border border-graphite-700 text-xs rounded-lg px-3 py-2 text-white focus:outline-none focus:border-cyan-500"
                  >
                    <option value="advanced_heat_sink">C. Complex Perforated Heat Sink (Stepped &amp; Cutouts)</option>
                    <option value="standard_heat_sink">A. Standard Heat Sink (Uniform Fins)</option>
                    <option value="asymmetric_cooling_plate">B. Asymmetric Finned Cooling Plate</option>
                    <option value="cube">Baseline Verification Block (Cube)</option>
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

                {/* Number of fins & power */}
                <div className="grid grid-cols-2 gap-3">
                  <div>
                    <div className="flex justify-between text-xs text-gray-300 mb-1">
                      <span>Fins:</span>
                      <span className="font-mono text-cyan-400 font-bold">{finCount}</span>
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
                  <div>
                    <div className="flex justify-between text-xs text-gray-300 mb-1">
                      <span>Die Power:</span>
                      <span className="font-mono text-rose-400 font-bold">{powerWatts} W</span>
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
                </div>

                {/* Convection & Mesh */}
                <div className="grid grid-cols-2 gap-3">
                  <div>
                    <div className="flex justify-between text-xs text-gray-300 mb-1">
                      <span>Conv. h:</span>
                      <span className="font-mono text-blue-400 font-bold">{convectionH} W/m²K</span>
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
                  <div>
                    <label className="text-xs font-medium text-gray-300 block mb-1">Mesh Grid</label>
                    <div className="grid grid-cols-2 gap-1.5">
                      {['coarse', 'medium'].map((res) => (
                        <button
                          key={res}
                          type="button"
                          onClick={() => setResolution(res)}
                          className={`py-1 rounded text-[11px] font-mono font-bold capitalize border ${
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
                </div>

                {/* SOR Relaxation Parameter ω & Initial Guess T0 Configurator */}
                <div className="bg-graphite-950 p-3 rounded-lg border border-graphite-800 space-y-2">
                  <div className="flex justify-between items-center text-xs">
                    <span className="font-bold text-amber-400 font-mono">SOR RELAXATION (ω):</span>
                    <span className="font-mono text-white font-bold">ω = {sorOmega.toFixed(2)}</span>
                  </div>
                  <div className="flex flex-wrap gap-1">
                    {[0.8, 1.0, 1.2, 1.35, 1.5, 1.7, 1.9, 2.05].map((w) => (
                      <button
                        key={w}
                        type="button"
                        onClick={() => { setSorOmega(w); executeSimulation(w, initialGuess); }}
                        className={`px-2 py-0.5 rounded text-[10px] font-mono border transition-all ${
                          Math.abs(sorOmega - w) < 0.01
                            ? 'bg-amber-600 text-white border-amber-400 font-bold'
                            : 'bg-graphite-850 text-gray-300 border-graphite-700 hover:border-amber-500'
                        }`}
                      >
                        {w.toFixed(2)}
                      </button>
                    ))}
                  </div>
                  <div className="flex items-center justify-between pt-1 border-t border-graphite-800 text-[11px]">
                    <span className="text-gray-400 font-mono">Initial Guess T₀:</span>
                    <div className="flex space-x-1">
                      {[
                        { id: 'lumped', label: 'Lumped T∞+Q/ΣhA' },
                        { id: 'ambient', label: 'Ambient T∞ (20°C)' }
                      ].map((g) => (
                        <button
                          key={g.id}
                          type="button"
                          onClick={() => { setInitialGuess(g.id); executeSimulation(sorOmega, g.id); }}
                          className={`px-2 py-0.5 rounded text-[10px] font-mono border ${
                            initialGuess === g.id
                              ? 'bg-cyan-950 border-cyan-500 text-cyan-300 font-bold'
                              : 'bg-graphite-850 border-graphite-700 text-gray-400'
                          }`}
                        >
                          {g.label}
                        </button>
                      ))}
                    </div>
                  </div>
                </div>

                {/* Global Energy Balance Verification Card */}
                {simData?.energy_balance && (
                  <div className="bg-graphite-950 p-3 rounded-lg border border-graphite-800 text-[11px] font-mono space-y-1">
                    <div className="text-cyan-400 font-bold flex items-center justify-between">
                      <span>GLOBAL ENERGY BALANCE (LU REF)</span>
                      <span className="text-[10px] text-emerald-400">
                        {simData.energy_balance.energy_conserved ? 'CONSERVED' : 'IMBALANCE'}
                      </span>
                    </div>
                    <div className="flex justify-between text-gray-400">
                      <span>Volumetric Input Q_in:</span>
                      <strong className="text-rose-400">{simData.energy_balance.q_in_watts.toFixed(4)} W</strong>
                    </div>
                    <div className="flex justify-between text-gray-400">
                      <span>Convective Integral Q_out:</span>
                      <strong className="text-emerald-400">{simData.energy_balance.q_out_convective_watts.toFixed(4)} W</strong>
                    </div>
                    <div className="flex justify-between text-gray-400">
                      <span>Conservation Error |ΔQ|/Q_in:</span>
                      <strong className="text-cyan-300">{formatPctError(simData.energy_balance)}</strong>
                    </div>
                    <div className="flex justify-between text-gray-400">
                      <span>SOR Energy Balance Error:</span>
                      <strong className={simData.solvers.sor.converged ? 'text-emerald-400' : 'text-amber-400'}>
                        {formatPctError(simData.solvers.sor.energy_balance)}
                      </strong>
                    </div>
                  </div>
                )}
              </div>

              {/* Action Buttons */}
              <div className="space-y-2 mt-4">
                <button
                  onClick={() => executeSimulation()}
                  disabled={loading}
                  className="w-full py-2.5 bg-gradient-to-r from-cyan-600 to-blue-600 hover:from-cyan-500 hover:to-blue-500 disabled:opacity-50 text-white font-bold text-xs rounded-xl shadow-lg shadow-cyan-600/30 transition-all flex items-center justify-center space-x-2"
                >
                  {loading ? <RefreshCw className="w-4 h-4 animate-spin" /> : <Play className="w-4 h-4" />}
                  <span>{loading ? 'Solving Assembled System...' : 'Run Simulation & Solver Audit'}</span>
                </button>

                <button
                  onClick={handleSaveCurrentExperiment}
                  disabled={!simData}
                  className="w-full py-1.5 bg-graphite-800 hover:bg-graphite-750 text-gray-300 text-xs font-semibold rounded-lg border border-graphite-700 flex items-center justify-center space-x-1.5"
                >
                  <Bookmark className="w-3.5 h-3.5 text-cyan-400" />
                  <span>{saveSuccess ? 'Saved to Repository!' : 'Save Current Run to Experiments'}</span>
                </button>
              </div>
            </div>

            {/* 3D VIEWPORT & SOLVER SUMMARY */}
            <div className="col-span-8 flex flex-col space-y-3">
              {/* Top View Mode Buttons */}
              <div className="bg-graphite-900 border border-graphite-800 p-2 rounded-xl flex items-center justify-between">
                <div className="flex items-center space-x-1">
                  <span className="text-xs text-gray-400 mr-2 flex items-center space-x-1">
                    <Eye className="w-3.5 h-3.5 text-cyan-400" />
                    <span>Mode:</span>
                  </span>
                  {[
                    { id: 'temperature', label: 'Temperature (°C)' },
                    { id: 'heat_flux', label: 'Heat Flux |q| (W/m²)' },
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
                    <span className="text-gray-400">Peak: <strong className="text-rose-400">{simData.solvers.direct_lu.t_max.toFixed(2)}°C</strong></span>
                  </div>
                )}
              </div>

              {/* 3D Canvas */}
              <div className="h-[430px]">
                <Viewport3D
                  surfaceMesh={simData?.surface_mesh || null}
                  vizMode={vizMode}
                  showFluxVectors={vizMode === 'heat_flux'}
                  tMin={tMin}
                  tMax={tMax}
                />
              </div>

              {/* Live 3-Solver Status Snapshot Bar */}
              {simData && (
                <div className="grid grid-cols-3 gap-3">
                  {(['direct_lu', 'conjugate_gradient', 'sor'] as const).map((sKey) => {
                    const s = simData.solvers[sKey];
                    return (
                      <div key={sKey} className="bg-graphite-900 border border-graphite-800 p-3 rounded-xl flex flex-col justify-between">
                        <div className="flex items-center justify-between mb-1">
                          <span className="text-[11px] font-bold text-white font-mono truncate">{s.solver_name}</span>
                          <StatusBadge converged={s.converged} label={s.status_label} />
                        </div>
                        <div className="grid grid-cols-3 gap-1 text-[11px] font-mono text-gray-400 mt-1">
                          <div>Time: <strong className="text-gray-200">{s.execution_time_ms.toFixed(1)}ms</strong></div>
                          <div>Iters: <strong className="text-gray-200">{s.iterations}</strong></div>
                          <div>Res: <strong className="text-cyan-300">{s.final_residual.toExponential(1)}</strong></div>
                        </div>
                        <div className="flex justify-between text-[10px] font-mono text-gray-400 mt-1 pt-1 border-t border-graphite-800">
                          <span>T_max: <strong className="text-rose-400">{s.t_max.toFixed(2)}°C</strong></span>
                          <span>ΔT vs LU: <strong className={s.max_discrepancy_vs_lu > 0.05 ? 'text-amber-400' : 'text-emerald-400'}>{s.max_discrepancy_vs_lu.toExponential(2)}°C</strong></span>
                        </div>
                      </div>
                    );
                  })}
                </div>
              )}
            </div>
          </div>
        )}

        {/* TAB 2: SOLVER BATTLE ARENA & SOR OMEGA AUDIT */}
        {activeTab === 'battle' && simData && (
          <div className="space-y-6">
            {/* Automatic Engineering Conclusions & Recommendation Panel (Requirement 6) */}
            <div className="bg-gradient-to-r from-blue-950/80 via-indigo-950/80 to-graphite-900 border border-cyan-800/40 p-5 rounded-2xl shadow-xl space-y-3">
              <div className="flex items-center justify-between">
                <div className="flex items-center space-x-3">
                  <div className="p-2.5 bg-cyan-500/20 rounded-xl text-cyan-400">
                    <Award className="w-6 h-6" />
                  </div>
                  <div>
                    <div className="text-[11px] font-mono font-bold text-cyan-400 uppercase tracking-widest">
                      Automated Engineering Conclusions &amp; Solver Eligibility Audit
                    </div>
                    <h3 className="text-base font-bold text-white">
                      Recommended Converged Solver:{' '}
                      <span className="text-emerald-400">
                        {simData.winner_id === 'conjugate_gradient'
                          ? 'Preconditioned Conjugate Gradient (Jacobi)'
                          : simData.winner_id === 'direct_lu'
                          ? 'Direct Sparse LU (SuperLU)'
                          : 'None (No Solver Converged)'}
                      </span>
                    </h3>
                  </div>
                </div>
                <div className="text-xs font-mono text-gray-300 bg-graphite-900/90 px-3 py-1.5 rounded-lg border border-graphite-700">
                  Residual Definition: <strong className="text-cyan-300">||F - K·T||₂ / ||F||₂</strong> (Tol = 1.0e-6)
                </div>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-3 pt-2">
                {(simData.engineering_conclusions || [simData.winner_explanation]).map((note, idx) => (
                  <div key={idx} className="bg-graphite-950/80 border border-graphite-800 p-3 rounded-lg text-xs text-gray-200 font-mono leading-relaxed">
                    {note}
                  </div>
                ))}
              </div>
            </div>

            {/* Detailed 3-Solver Audit Cards */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-5">
              {(['direct_lu', 'conjugate_gradient', 'sor'] as const).map((sKey, idx) => {
                const s = simData.solvers[sKey];
                const borderCol = s.converged ? 'border-emerald-800/60' : 'border-amber-700/80';
                return (
                  <div key={sKey} className={`bg-graphite-900 border ${borderCol} rounded-xl p-5 space-y-3`}>
                    <div className="flex justify-between items-center">
                      <span className="text-xs font-bold text-cyan-400 font-mono">SOLVER {String.fromCharCode(65 + idx)}</span>
                      <StatusBadge converged={s.converged} label={s.status_label} />
                    </div>
                    <h4 className="text-sm font-bold text-white">{s.solver_name}</h4>
                    <div className="space-y-1.5 pt-2 border-t border-graphite-800 text-xs font-mono">
                      <div className="flex justify-between text-gray-400">
                        <span>Execution Time:</span>
                        <strong className="text-white">{s.execution_time_ms.toFixed(2)} ms</strong>
                      </div>
                      <div className="flex justify-between text-gray-400">
                        <span>Iterations Executed:</span>
                        <strong className="text-white">{s.iterations}</strong>
                      </div>
                      <div className="flex justify-between text-gray-400">
                        <span>Relative Residual ||r||/||F||:</span>
                        <strong className={s.converged ? 'text-emerald-400' : 'text-amber-400'}>{s.final_residual.toExponential(3)}</strong>
                      </div>
                      <div className="flex justify-between text-gray-400">
                        <span>Absolute Residual ||F-KT||₂:</span>
                        <strong className="text-gray-200">{(s.absolute_residual ?? 0).toExponential(3)} W</strong>
                      </div>
                      <div className="flex justify-between text-gray-400">
                        <span>Working Memory:</span>
                        <strong className="text-white">{s.memory_mb.toFixed(3)} MB</strong>
                      </div>
                      <div className="flex justify-between text-gray-400">
                        <span>Temp Range [T_min, T_max]:</span>
                        <strong className="text-rose-400">[{s.t_min.toFixed(2)}°C, {s.t_max.toFixed(2)}°C]</strong>
                      </div>
                      <div className="flex justify-between text-gray-400">
                        <span>Mean Temp (T_avg):</span>
                        <strong className="text-gray-200">{s.t_avg.toFixed(2)} °C</strong>
                      </div>
                      <div className="flex justify-between text-gray-400">
                        <span>Max Discrepancy vs LU:</span>
                        <strong className={s.max_discrepancy_vs_lu > 0.05 ? 'text-amber-400' : 'text-emerald-300'}>
                          {s.max_discrepancy_vs_lu.toFixed(6)} °C
                        </strong>
                      </div>
                      <div className="flex justify-between text-gray-400">
                        <span>Energy Balance Error:</span>
                        <strong className="text-cyan-300">{formatPctError(s.energy_balance)}</strong>
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>

            {/* FULL-HISTORY CONVERGENCE CHART & RUNTIME CHART (Requirement 2) */}
            <div className="grid grid-cols-1 md:grid-cols-12 gap-5">
              <div className="md:col-span-7 bg-graphite-900 border border-graphite-800 p-5 rounded-xl">
                <div className="flex items-center justify-between mb-1">
                  <h4 className="text-sm font-bold text-white flex items-center space-x-2">
                    <Activity className="w-4 h-4 text-cyan-400" />
                    <span>Unsampled Iterative Residual Trajectory: log₁₀(||F - K·T||₂ / ||F||₂)</span>
                  </h4>
                  <span className="text-[10px] font-mono px-2 py-0.5 bg-cyan-950 text-cyan-300 border border-cyan-800 rounded">
                    EVERY ITERATION PLOTTED ({chartInfo.series.length} pts)
                  </span>
                </div>
                <p className="text-xs text-gray-400 mb-3">
                  True iteration count k on X-axis • Stopping tolerance line at 10⁻⁶ (-6.0) • Direct LU reference residual:{' '}
                  <strong className="text-cyan-300">{simData.solvers.direct_lu.final_residual.toExponential(2)}</strong>
                </p>
                <div className="h-72">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={chartInfo.series} margin={{ top: 10, right: 20, left: 0, bottom: 10 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#1f293d" />
                      <XAxis
                        dataKey="iteration"
                        type="number"
                        domain={[0, 'dataMax']}
                        stroke="#94a3b8"
                        tick={{ fontSize: 10 }}
                        label={{ value: 'True Iteration Count (k)', position: 'insideBottom', offset: -5, fill: '#94a3b8', fontSize: 10 }}
                      />
                      <YAxis
                        stroke="#94a3b8"
                        tick={{ fontSize: 10 }}
                        domain={[-8, 1]}
                        label={{ value: 'log10(||r|| / ||F||)', angle: -90, position: 'insideLeft', fill: '#94a3b8', fontSize: 10 }}
                      />
                      <Tooltip
                        contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', fontSize: '11px', fontFamily: 'monospace' }}
                        labelFormatter={(val) => `Iteration k = ${val}`}
                      />
                      <Legend wrapperStyle={{ fontSize: '11px' }} />
                      <ReferenceLine
                        y={-6}
                        stroke="#22c55e"
                        strokeDasharray="4 4"
                        label={{ value: 'Tolerance 1e-6 (log10 = -6)', fill: '#22c55e', fontSize: 10, position: 'insideBottomRight' }}
                      />
                      <Line
                        type="monotone"
                        dataKey="CG"
                        stroke="#10b981"
                        strokeWidth={2.2}
                        dot={false}
                        name={`PCG (${simData.solvers.conjugate_gradient.iterations} iters, ${simData.solvers.conjugate_gradient.status_label})`}
                      />
                      <Line
                        type="monotone"
                        dataKey="SOR"
                        stroke={simData.solvers.sor.converged ? '#3b82f6' : '#f59e0b'}
                        strokeWidth={2.2}
                        dot={false}
                        name={`SOR ω=${sorOmega.toFixed(2)} (${simData.solvers.sor.iterations} iters, ${simData.solvers.sor.status_label})`}
                      />
                      {chartInfo.cgEnd && chartInfo.cgEnd.CG !== undefined && (
                        <ReferenceDot
                          x={chartInfo.cgEnd.iteration}
                          y={chartInfo.cgEnd.CG}
                          r={5}
                          fill="#10b981"
                          stroke="#ffffff"
                        />
                      )}
                      {chartInfo.sorEnd && chartInfo.sorEnd.SOR !== undefined && (
                        <ReferenceDot
                          x={chartInfo.sorEnd.iteration}
                          y={chartInfo.sorEnd.SOR}
                          r={6}
                          fill={simData.solvers.sor.converged ? '#3b82f6' : '#ef4444'}
                          stroke="#ffffff"
                        />
                      )}
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </div>

              <div className="md:col-span-5 bg-graphite-900 border border-graphite-800 p-5 rounded-xl">
                <h4 className="text-sm font-bold text-white mb-1 flex items-center space-x-2">
                  <Zap className="w-4 h-4 text-amber-400" />
                  <span>Solver Wall-Time &amp; Memory Comparison</span>
                </h4>
                <p className="text-xs text-gray-400 mb-3">Measured execution time (ms) on the identical {simData.mesh_info.degrees_of_freedom}-DOF matrix</p>
                <div className="h-72">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={battleBarData()}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#1f293d" />
                      <XAxis dataKey="name" stroke="#94a3b8" tick={{ fontSize: 10 }} />
                      <YAxis stroke="#94a3b8" tick={{ fontSize: 10 }} />
                      <Tooltip contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', fontSize: '11px' }} />
                      <Bar dataKey="time_ms" fill="#3b82f6" name="Wall-Time (ms)" radius={[4, 4, 0, 0]} />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </div>
            </div>

            {/* SOR RELAXATION PARAMETER ω EMPIRICAL SWEEP AUDIT (Requirement 1) */}
            <div className="bg-graphite-900 border border-graphite-800 p-5 rounded-xl space-y-4">
              <div className="flex flex-wrap items-center justify-between gap-2">
                <div>
                  <h4 className="text-sm font-bold text-white">
                    SOR Relaxation Factor (ω) Empirical Convergence Audit (ω ∈ [0.80, 2.05])
                  </h4>
                  <p className="text-xs text-gray-400">
                    Measured convergence behavior across 8 relaxation parameters on the identical assembled matrix (never assumed).
                  </p>
                </div>
                <div className="flex items-center space-x-2">
                  <button
                    onClick={() => handleRunOmegaSweep('lumped')}
                    disabled={omegaLoading}
                    className="px-3 py-1.5 bg-cyan-700 hover:bg-cyan-600 text-white text-xs font-mono rounded-lg"
                  >
                    Test Lumped T₀
                  </button>
                  <button
                    onClick={() => handleRunOmegaSweep('ambient')}
                    disabled={omegaLoading}
                    className="px-3 py-1.5 bg-graphite-800 hover:bg-graphite-700 text-gray-200 text-xs font-mono rounded-lg border border-graphite-700"
                  >
                    Test Ambient T₀ (20°C)
                  </button>
                </div>
              </div>

              {omegaLoading ? (
                <div className="p-6 text-center text-xs font-mono text-cyan-400 flex items-center justify-center space-x-2">
                  <RefreshCw className="w-4 h-4 animate-spin" />
                  <span>Executing SOR parameter sweep across ω = 0.8, 1.0, 1.2, 1.35, 1.5, 1.7, 1.9, 2.05...</span>
                </div>
              ) : omegaStudy ? (
                <div className="space-y-3">
                  <div className="overflow-x-auto">
                    <table className="w-full text-left text-xs border border-graphite-800 font-mono">
                      <thead className="bg-graphite-850 text-gray-300">
                        <tr>
                          <th className="p-2.5">Relaxation ω</th>
                          <th className="p-2.5">Convergence Status</th>
                          <th className="p-2.5">Iterations</th>
                          <th className="p-2.5">Rel. Residual ||r||/||F||</th>
                          <th className="p-2.5">Abs. Residual (W)</th>
                          <th className="p-2.5">Runtime (ms)</th>
                          <th className="p-2.5">Peak T_max (°C)</th>
                          <th className="p-2.5">Max |T - T_LU| (°C)</th>
                          <th className="p-2.5">Action</th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-graphite-800">
                        {omegaStudy.results.map((r: any, idx: number) => (
                          <tr key={idx} className="hover:bg-graphite-850/50">
                            <td className="p-2.5 font-bold text-amber-400">ω = {r.omega.toFixed(2)}</td>
                            <td className="p-2.5">
                              <StatusBadge converged={r.converged} label={r.status} />
                            </td>
                            <td className="p-2.5 text-white">{r.iterations}</td>
                            <td className="p-2.5 text-cyan-300">{Number(r.final_residual).toExponential(2)}</td>
                            <td className="p-2.5 text-gray-300">{Number(r.absolute_residual ?? 0).toExponential(2)}</td>
                            <td className="p-2.5 text-gray-200">{r.execution_time_ms} ms</td>
                            <td className="p-2.5 text-rose-400">{r.t_max !== null ? `${r.t_max.toFixed(2)}°C` : '—'}</td>
                            <td className="p-2.5 text-emerald-300">
                              {r.max_discrepancy_vs_lu !== null ? `${Number(r.max_discrepancy_vs_lu).toFixed(5)}°C` : '—'}
                            </td>
                            <td className="p-2.5">
                              <button
                                onClick={() => {
                                  setSorOmega(r.omega);
                                  executeSimulation(r.omega, omegaStudy.initial_guess_mode);
                                }}
                                className="px-2 py-0.5 bg-graphite-800 hover:bg-cyan-700 text-cyan-300 hover:text-white rounded text-[10px]"
                              >
                                Load in Arena
                              </button>
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                  <div className="bg-graphite-950 p-3 rounded-lg border border-graphite-800 text-xs font-mono text-gray-300">
                    <strong className="text-cyan-400">Root-Cause Spectral Diagnosis:</strong> {omegaStudy.diagnosis}
                  </div>
                </div>
              ) : null}
            </div>
          </div>
        )}

        {/* TAB 3: 3x3 GEOMETRY-SOLVER RESEARCH BENCHMARK (9 EXPERIMENTS) */}
        {activeTab === 'matrix3x3' && (
          <div className="space-y-6">
            <div className="bg-graphite-900 border border-graphite-800 p-6 rounded-xl space-y-5">
              <div className="flex flex-wrap justify-between items-center gap-3">
                <div>
                  <h3 className="text-base font-bold text-white flex items-center space-x-2">
                    <Compass className="w-4 h-4 text-cyan-400" />
                    <span>3 × 3 Geometry-Solver Research Benchmark Dashboard (Nine Experiments)</span>
                  </h3>
                  <p className="text-xs text-gray-400 mt-1">
                    Each solver receives the exact same assembled matrix K and load vector F for (A) Standard, (B) Asymmetric, and (C) Perforated heat sinks.
                  </p>
                </div>
                <div className="flex items-center space-x-2 text-xs font-mono">
                  <span className="text-gray-400">SOR ω:</span>
                  {[1.35, 1.9].map((w) => (
                    <button
                      key={w}
                      onClick={() => { setMatrixOmega(w); handleRun3x3Benchmark(w, matrixInitGuess); }}
                      className={`px-2.5 py-1 rounded border ${
                        Math.abs(matrixOmega - w) < 0.01
                          ? 'bg-amber-600 text-white border-amber-400 font-bold'
                          : 'bg-graphite-850 text-gray-300 border-graphite-700'
                      }`}
                    >
                      ω={w.toFixed(2)}
                    </button>
                  ))}
                  <button
                    onClick={() => handleRun3x3Benchmark()}
                    disabled={matrixLoading}
                    className="px-4 py-1.5 bg-cyan-600 hover:bg-cyan-500 text-white font-semibold rounded-lg flex items-center space-x-1.5"
                  >
                    {matrixLoading ? <RefreshCw className="w-3.5 h-3.5 animate-spin" /> : <Play className="w-3.5 h-3.5" />}
                    <span>Execute 9 Experiments</span>
                  </button>
                </div>
              </div>

              {matrixLoading ? (
                <div className="p-12 text-center text-cyan-400 font-mono text-xs flex flex-col items-center justify-center space-y-2">
                  <RefreshCw className="w-6 h-6 animate-spin" />
                  <span>Assembling 3 geometries and running all 9 solver benchmarks...</span>
                </div>
              ) : matrixData ? (
                <div className="space-y-6">
                  {/* 9-Experiment Master Table */}
                  <div className="overflow-x-auto">
                    <table className="w-full text-left text-[11px] border border-graphite-800 font-mono">
                      <thead className="bg-graphite-850 text-gray-300">
                        <tr>
                          <th className="p-2.5">Exp</th>
                          <th className="p-2.5">Geometry</th>
                          <th className="p-2.5">DOF / NNZ (Sparsity)</th>
                          <th className="p-2.5">Solver</th>
                          <th className="p-2.5">Status</th>
                          <th className="p-2.5">Runtime</th>
                          <th className="p-2.5">Iters</th>
                          <th className="p-2.5">Rel. Residual</th>
                          <th className="p-2.5">Memory</th>
                          <th className="p-2.5">T_max / T_avg</th>
                          <th className="p-2.5">ΔT vs LU</th>
                          <th className="p-2.5">Energy Err</th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-graphite-800">
                        {(matrixData.nine_experiments || []).map((exp: any, i: number) => (
                          <tr key={i} className="hover:bg-graphite-850/50">
                            <td className="p-2.5 font-bold text-cyan-400">{exp.experiment_id}</td>
                            <td className="p-2.5 text-white font-semibold">{exp.geometry_name}</td>
                            <td className="p-2.5 text-gray-300">
                              {exp.dof} / {exp.matrix_nnz} ({exp.sparsity_pct}%)
                            </td>
                            <td className="p-2.5 text-cyan-200 font-bold">
                              {exp.solver_name}
                              {exp.is_recommended && (
                                <span className="ml-1.5 px-1.5 py-0.5 text-[9px] bg-emerald-950 text-emerald-300 border border-emerald-700 rounded">
                                  RECOMMENDED
                                </span>
                              )}
                            </td>
                            <td className="p-2.5">
                              <StatusBadge converged={exp.converged} label={exp.status_label} />
                            </td>
                            <td className="p-2.5 text-white font-bold">{exp.execution_time_ms.toFixed(1)} ms</td>
                            <td className="p-2.5 text-gray-200">{exp.iterations}</td>
                            <td className="p-2.5 text-cyan-300">{Number(exp.final_residual).toExponential(2)}</td>
                            <td className="p-2.5 text-gray-200">{exp.memory_mb.toFixed(2)} MB</td>
                            <td className="p-2.5 text-rose-400">
                              {exp.t_max.toFixed(2)}°C / {exp.t_avg.toFixed(2)}°C
                            </td>
                            <td className="p-2.5 text-emerald-300">{Number(exp.max_discrepancy_vs_lu).toExponential(2)}°C</td>
                            <td className="p-2.5 text-cyan-300">{exp.energy_balance_error_formatted}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>

                  <div className="bg-graphite-950 p-4 rounded-xl border border-graphite-800 text-xs font-mono text-gray-300 leading-relaxed whitespace-pre-line">
                    <span className="text-cyan-400 font-bold block mb-1">3×3 Research Benchmark Interpretation:</span>
                    {matrixData.comparative_analysis}
                  </div>
                </div>
              ) : null}
            </div>
          </div>
        )}

        {/* TAB 4: NUMERICAL VERIFICATION & MESH CONVERGENCE LABORATORY (Requirement 5) */}
        {activeTab === 'mesh' && (
          <div className="space-y-6">
            <div className="bg-graphite-900 border border-graphite-800 p-6 rounded-xl space-y-5">
              <div>
                <h3 className="text-base font-bold text-white mb-1">
                  Numerical Verification &amp; Spatial Grid Convergence Laboratory
                </h3>
                <p className="text-xs text-gray-400">
                  1) Closed-form 3D analytical boundary-value verification of the HEX-8 FEM assembler across 4 refinement levels. 2) 3D Heat sink mesh refinement study.
                </p>
              </div>

              {meshLoading ? (
                <div className="p-8 flex items-center justify-center space-x-2 text-cyan-400">
                  <RefreshCw className="w-5 h-5 animate-spin" />
                  <span className="text-xs font-mono">Running 3D Analytical Benchmark &amp; Multi-Grid FE Refinement Study...</span>
                </div>
              ) : meshData ? (
                <div className="space-y-6">
                  {/* Analytical Benchmark Section */}
                  {meshData.analytical_verification && (
                    <div className="bg-graphite-950 border border-graphite-800 p-4 rounded-xl space-y-3">
                      <div className="flex flex-wrap items-center justify-between gap-2">
                        <div>
                          <div className="text-xs font-mono font-bold text-cyan-400">
                            3D ANALYTICAL BENCHMARK: {meshData.analytical_verification.benchmark}
                          </div>
                          <div className="text-xs font-mono text-gray-300 mt-0.5">
                            Exact Solution: <strong className="text-white">{meshData.analytical_verification.exact_solution}</strong>
                          </div>
                        </div>
                        <div className="flex items-center space-x-3 font-mono text-xs">
                          <span className="px-2.5 py-1 bg-emerald-950 text-emerald-300 border border-emerald-700 rounded font-bold">
                            STATUS: {meshData.analytical_verification.status}
                          </span>
                          <span className="px-2.5 py-1 bg-cyan-950 text-cyan-300 border border-cyan-800 rounded font-bold">
                            Observed L2 Order p = {Number(meshData.analytical_verification.observed_order).toFixed(4)} (Expected 2.0000)
                          </span>
                        </div>
                      </div>

                      <div className="overflow-x-auto">
                        <table className="w-full text-left text-xs border border-graphite-800 font-mono">
                          <thead className="bg-graphite-850 text-gray-300">
                            <tr>
                              <th className="p-2.5">Grid (Nx×2×2)</th>
                              <th className="p-2.5">DOF</th>
                              <th className="p-2.5">Element Size h (m)</th>
                              <th className="p-2.5">Interior L2 Error (K)</th>
                              <th className="p-2.5">Observed Order p</th>
                              <th className="p-2.5">Nodal Max Error LU (K)</th>
                              <th className="p-2.5">CG Iters</th>
                              <th className="p-2.5">SOR Iters</th>
                              <th className="p-2.5">Energy Balance Err</th>
                            </tr>
                          </thead>
                          <tbody className="divide-y divide-graphite-800">
                            {meshData.analytical_verification.refinement.map((r: any, idx: number) => (
                              <tr key={idx} className="hover:bg-graphite-850/50">
                                <td className="p-2.5 font-bold text-cyan-400">{r.nx}×2×2</td>
                                <td className="p-2.5 text-white">{r.dof}</td>
                                <td className="p-2.5 text-gray-300">{r.h_m.toFixed(5)}</td>
                                <td className="p-2.5 text-amber-300">{Number(r.l2_error_K).toExponential(4)}</td>
                                <td className="p-2.5 text-emerald-400 font-bold">
                                  {r.observed_order ? Number(r.observed_order).toFixed(4) : '—'}
                                </td>
                                <td className="p-2.5 text-cyan-300">{Number(r.nodal_max_error_lu_K).toExponential(2)}</td>
                                <td className="p-2.5 text-gray-200">{r.cg_iterations} ({r.cg_status})</td>
                                <td className="p-2.5 text-gray-200">{r.sor_iterations} ({r.sor_status})</td>
                                <td className="p-2.5 text-emerald-300">{Number(r.energy_balance_rel_error * 100).toExponential(2)}%</td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                    </div>
                  )}

                  {/* 3D Finned Heat Sink Mesh Refinement Table */}
                  <div className="overflow-x-auto">
                    <table className="w-full text-left text-xs border border-graphite-800 font-mono">
                      <thead className="bg-graphite-850 text-gray-300">
                        <tr>
                          <th className="p-3">Resolution</th>
                          <th className="p-3">Active Elements</th>
                          <th className="p-3">DOF (Nodes)</th>
                          <th className="p-3">Peak Temp (T_max)</th>
                          <th className="p-3">Mean Temp (T_avg)</th>
                          <th className="p-3">Direct LU Time</th>
                          <th className="p-3">PCG Time (Iters)</th>
                          <th className="p-3">SOR Time (Status)</th>
                          <th className="p-3">Energy Balance Err</th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-graphite-800">
                        {meshData.data.map((row: any, i: number) => (
                          <tr key={i} className="hover:bg-graphite-850/50">
                            <td className="p-3 font-bold text-cyan-400">{row.resolution}</td>
                            <td className="p-3">{row.elements}</td>
                            <td className="p-3 text-white font-bold">{row.dof}</td>
                            <td className="p-3 text-rose-400">{row.t_max.toFixed(2)} °C</td>
                            <td className="p-3 text-gray-200">{row.t_avg.toFixed(2)} °C</td>
                            <td className="p-3 text-cyan-300">{row.direct_lu_ms.toFixed(1)} ms</td>
                            <td className="p-3 text-emerald-400">{row.cg_ms.toFixed(1)} ms ({row.cg_iterations})</td>
                            <td className="p-3 text-amber-400">{row.sor_ms.toFixed(1)} ms ({row.sor_status})</td>
                            <td className="p-3 text-cyan-300">{row.energy_balance_error_formatted}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                  <div className="bg-graphite-950 p-4 rounded-lg border border-graphite-800 text-xs text-gray-300 leading-relaxed font-mono">
                    <strong className="text-cyan-400">Grid Convergence Assessment:</strong> {meshData.convergence_analysis}
                  </div>
                </div>
              ) : null}
            </div>
          </div>
        )}

        {/* TAB 5: PARAMETRIC DESIGN OPTIMIZATION */}
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
                  <span>Executing parametric sweep iterations...</span>
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
                        <div className="text-xs text-rose-400 font-bold mt-1">T_max: {pt.t_max.toFixed(2)}°C</div>
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

        {/* TAB 6: RESEARCH EXPERIMENTS REPOSITORY */}
        {activeTab === 'experiments' && (
          <div className="space-y-6">
            <div className="bg-graphite-900 border border-graphite-800 p-6 rounded-xl">
              <div className="flex items-center justify-between mb-4">
                <div>
                  <h3 className="text-base font-bold text-white mb-1 flex items-center space-x-2">
                    <Database className="w-4 h-4 text-cyan-400" />
                    <span>Reproducible Experiments Repository</span>
                  </h3>
                  <p className="text-xs text-gray-400">
                    Saved simulation runs with recorded geometric parameters, boundary conditions, and numerical outputs.
                  </p>
                </div>
                <div className="flex items-center space-x-2">
                  <input
                    type="text"
                    value={expTitle}
                    onChange={(e) => setExpTitle(e.target.value)}
                    className="bg-graphite-850 border border-graphite-700 rounded-lg px-3 py-1.5 text-xs text-white font-mono"
                    placeholder="Experiment title..."
                  />
                  <button
                    onClick={handleSaveCurrentExperiment}
                    className="px-3 py-1.5 bg-cyan-600 hover:bg-cyan-500 text-white text-xs font-mono rounded-lg"
                  >
                    Save Current Run
                  </button>
                </div>
              </div>

              {experiments.length === 0 ? (
                <div className="p-8 text-center text-gray-500 text-xs font-mono">
                  No saved experiments in storage yet. Run a simulation and click "Save Current Run".
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
                          <td className="p-3 text-rose-400 font-bold">{exp.t_max ? `${exp.t_max.toFixed(2)} °C` : 'N/A'}</td>
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
                GROUNDED IN COMPUTED FEM DATA
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
                'Why is the center of the heat sink hotter?',
                'Why did CG converge faster than SOR?',
                'Which solver should I use for a larger model?',
                'What happens if I change to copper alloy?'
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
