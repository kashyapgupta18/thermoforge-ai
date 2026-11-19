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
  Database,
  RotateCcw,
  GitCompare
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
  single_run_time_ms?: number;
  median_runtime_ms?: number;
  mean_runtime_ms?: number;
  std_runtime_ms?: number;
  min_runtime_ms?: number;
  max_runtime_ms?: number;
  factorization_ms?: number;
  substitution_ms?: number;
  median_factorization_ms?: number;
  median_substitution_ms?: number;
  repetitions?: number;
  trial_runtimes_ms?: number[];
  speedup_vs_lu_median?: number;
  iterations: number;
  iterations_display?: string;
  iterations_applicable?: boolean;
  base_system_memory_mb?: number;
  additional_solver_memory_mb?: number;
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
  alternative_solver?: string;
  alternative_solver_name?: string;
  confidence_score: number;
  evidence_level?: string;
  primary_reason?: string;
  expected_trade_offs?: string[];
  supporting_measured_data?: Array<{
    metric: string;
    direct_lu: string;
    conjugate_gradient: string;
    sor: string;
  }>;
  metrics_analyzed: {
    degrees_of_freedom: number;
    matrix_dimension?: string;
    matrix_nonzeros: number;
    sparsity_ratio: number;
    is_symmetric_positive_definite?: boolean;
    measured_lu_fill_in_ratio?: number;
    estimated_lu_memory_mb: number;
    measured_cg_memory_mb?: number;
    memory_ratio_lu_to_cg?: number;
    speedup_cg_vs_lu_median?: number;
  };
  reasoning: string[];
}

interface SimulationData {
  mesh_info: {
    nx?: number;
    ny?: number;
    nz?: number;
    degrees_of_freedom: number;
    active_elements: number;
    matrix_nonzeros: number;
    sparsity_pct?: number;
    solid_volume_cm3?: number;
    mass_grams?: number;
    thermal_resistance_k_per_w?: number;
    cooling_conductance_w_per_k?: number;
    assembly_time_ms: number;
  };
  solver_settings?: {
    tolerance_rel: number;
    max_iterations: number;
    sor_omega: number;
    initial_guess: string;
    initial_guess_kelvin: number;
    repetitions?: number;
    primary_timing_metric?: string;
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
  const [finHeightRatio, setFinHeightRatio] = useState<number>(0.65);
  const [baseThicknessRatio, setBaseThicknessRatio] = useState<number>(0.35);
  const [finThicknessRatio, setFinThicknessRatio] = useState<number>(1.0);
  const [finSpacingRatio, setFinSpacingRatio] = useState<number>(1.0);
  const [powerWatts, setPowerWatts] = useState<number>(65.0);
  const [convectionH, setConvectionH] = useState<number>(45.0);
  const [resolution, setResolution] = useState<string>('coarse');
  const [sorOmega, setSorOmega] = useState<number>(1.9);
  const [initialGuess, setInitialGuess] = useState<string>('lumped');
  const [repetitions, setRepetitions] = useState<number>(1);

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
  const [matrixReps, setMatrixReps] = useState<number>(10);

  // Parametric Optimization State
  const [sweepParam, setSweepParam] = useState<string>('fin_count');
  const [optObjective, setOptObjective] = useState<string>('min_t_max');
  const [optData, setOptData] = useState<any>(null);
  const [optLoading, setOptLoading] = useState<boolean>(false);

  // Experiments Management State
  const [experiments, setExperiments] = useState<any[]>([]);
  const [expTitle, setExpTitle] = useState<string>('Advanced Thermal Study #1');
  const [saveSuccess, setSaveSuccess] = useState<boolean>(false);
  const [selectedExpIds, setSelectedExpIds] = useState<string[]>([]);
  const [comparisonData, setComparisonData] = useState<any>(null);
  const [replayBanner, setReplayBanner] = useState<string | null>(null);

  // AI Chat States
  const [aiQuestion, setAiQuestion] = useState<string>('');
  const [aiMessages, setAiMessages] = useState<Array<{ sender: 'user' | 'thermo_ai'; text: string }>>([
    {
      sender: 'thermo_ai',
      text: 'NUMERICAL RESULT:\nTHERMO AI is connected to the live 3D Hex-8 Finite Element solver engine.\n\nAI INTERPRETATION:\nEvery response explicitly separates computed FEM measurements from engineering interpretations. Select any suggested prompt below or ask about hotspot physics, PCG vs. Direct LU fill-in, SOR spectral radius, mesh convergence, or energy conservation.'
    }
  ]);
  const [aiLoading, setAiLoading] = useState<boolean>(false);

  // Mesh Study State
  const [meshData, setMeshData] = useState<any>(null);
  const [meshLoading, setMeshLoading] = useState<boolean>(false);

  const executeSimulation = async (
    overrideOmega?: number,
    overrideInit?: string,
    overrideReps?: number,
    overrideCfg?: any
  ) => {
    setLoading(true);
    const w = overrideOmega !== undefined ? overrideOmega : sorOmega;
    const ig = overrideInit !== undefined ? overrideInit : initialGuess;
    const reps = overrideReps !== undefined ? overrideReps : repetitions;
    const payload = overrideCfg || {
      preset,
      material,
      fin_count: finCount,
      fin_height_ratio: finHeightRatio,
      base_thickness_ratio: baseThicknessRatio,
      fin_thickness_ratio: finThicknessRatio,
      fin_spacing_ratio: finSpacingRatio,
      power_watts: powerWatts,
      convection_h: convectionH,
      resolution,
      sor_omega: w,
      initial_guess: ig,
      repetitions: reps,
      ambient_temp_celsius: 20.0
    };
    try {
      const res = await fetch(`${API_BASE}/api/simulate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
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
          config: {
            preset,
            material,
            fin_count: finCount,
            fin_height_ratio: finHeightRatio,
            base_thickness_ratio: baseThicknessRatio,
            fin_thickness_ratio: finThicknessRatio,
            fin_spacing_ratio: finSpacingRatio,
            power_watts: powerWatts,
            convection_h: convectionH,
            resolution,
            sor_omega: sorOmega,
            initial_guess: initialGuess,
            repetitions
          },
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

  const handleReplayExperiment = async (exp: any) => {
    const cfg = exp.config || exp;
    const p = cfg.preset || 'advanced_heat_sink';
    const m = cfg.material || 'aluminium_6061';
    const fc = Number(cfg.fin_count ?? 7);
    const fh = Number(cfg.fin_height_ratio ?? 0.65);
    const bt = Number(cfg.base_thickness_ratio ?? 0.35);
    const ft = Number(cfg.fin_thickness_ratio ?? 1.0);
    const fs = Number(cfg.fin_spacing_ratio ?? 1.0);
    const pw = Number(cfg.power_watts ?? 65.0);
    const ch = Number(cfg.convection_h ?? 45.0);
    const res = cfg.resolution || 'coarse';
    const w = Number(cfg.sor_omega ?? 1.9);
    const ig = cfg.initial_guess || 'lumped';
    const reps = Number(cfg.repetitions ?? 1);

    setPreset(p);
    setMaterial(m);
    setFinCount(fc);
    setFinHeightRatio(fh);
    setBaseThicknessRatio(bt);
    setFinThicknessRatio(ft);
    setFinSpacingRatio(fs);
    setPowerWatts(pw);
    setConvectionH(ch);
    setResolution(res);
    setSorOmega(w);
    setInitialGuess(ig);
    setRepetitions(reps);
    setReplayBanner(`Replaying Saved Experiment [${exp.id}] "${exp.title}" (${exp.timestamp})`);
    setActiveTab('simulation');

    await executeSimulation(w, ig, reps, {
      preset: p,
      material: m,
      fin_count: fc,
      fin_height_ratio: fh,
      base_thickness_ratio: bt,
      fin_thickness_ratio: ft,
      fin_spacing_ratio: fs,
      power_watts: pw,
      convection_h: ch,
      resolution: res,
      sor_omega: w,
      initial_guess: ig,
      repetitions: reps,
      ambient_temp_celsius: 20.0
    });
  };

  const handleToggleSelectExperiment = (id: string) => {
    setSelectedExpIds(prev => (prev.includes(id) ? prev.filter(x => x !== id) : [...prev, id]));
  };

  const handleCompareSelectedExperiments = async () => {
    if (selectedExpIds.length < 2) return;
    try {
      const res = await fetch(`${API_BASE}/api/experiments/compare`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ experiment_ids: selectedExpIds })
      });
      const data = await res.json();
      setComparisonData(data);
    } catch (e) {
      console.error(e);
    }
  };

  const handleRun3x3Benchmark = async (wOverride?: number, igOverride?: string, repsOverride?: number) => {
    setMatrixLoading(true);
    const w = wOverride !== undefined ? wOverride : matrixOmega;
    const ig = igOverride !== undefined ? igOverride : matrixInitGuess;
    const reps = repsOverride !== undefined ? repsOverride : matrixReps;
    try {
      const res = await fetch(`${API_BASE}/api/benchmark-3x3?resolution=coarse&sor_omega=${w}&initial_guess=${ig}&repetitions=${reps}`);
      const data = await res.json();
      setMatrixData(data);
    } catch (e) {
      console.error(e);
    } finally {
      setMatrixLoading(false);
    }
  };

  const handleRunOptimizationSweep = async (paramOverride?: string, objOverride?: string) => {
    setOptLoading(true);
    const p = paramOverride || sweepParam;
    const obj = objOverride || optObjective;
    try {
      const res = await fetch(`${API_BASE}/api/parametric-optimization?preset=${preset}&material=${material}&param_name=${p}&objective=${obj}`);
      const data = await res.json();
      setOptData(data);
    } catch (e) {
      console.error(e);
    } finally {
      setOptLoading(false);
    }
  };

  const handleAskAI = async (customQ?: string) => {
    const q = (customQ !== undefined ? customQ : aiQuestion).trim();
    if (!q) return;
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
      const res = await fetch(`${API_BASE}/api/mesh-intelligence?preset=${preset}&material=${material}&fin_count=${finCount}&sor_omega=${sorOmega}&include_fine=true`);
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
    const luMed = s.direct_lu.median_runtime_ms ?? s.direct_lu.execution_time_ms;
    const cgMed = s.conjugate_gradient.median_runtime_ms ?? s.conjugate_gradient.execution_time_ms;
    const sorMed = s.sor.median_runtime_ms ?? s.sor.execution_time_ms;
    return [
      {
        name: 'Direct Sparse LU',
        median_ms: Number(luMed.toFixed(2)),
        mean_ms: Number((s.direct_lu.mean_runtime_ms ?? luMed).toFixed(2)),
        min_ms: Number((s.direct_lu.min_runtime_ms ?? luMed).toFixed(2)),
        max_ms: Number((s.direct_lu.max_runtime_ms ?? luMed).toFixed(2)),
        memory_mb: s.direct_lu.memory_mb,
        speedup: 1.0,
        iters: 1
      },
      {
        name: 'Precond. CG (PCG)',
        median_ms: Number(cgMed.toFixed(2)),
        mean_ms: Number((s.conjugate_gradient.mean_runtime_ms ?? cgMed).toFixed(2)),
        min_ms: Number((s.conjugate_gradient.min_runtime_ms ?? cgMed).toFixed(2)),
        max_ms: Number((s.conjugate_gradient.max_runtime_ms ?? cgMed).toFixed(2)),
        memory_mb: s.conjugate_gradient.memory_mb,
        speedup: Number((luMed / Math.max(cgMed, 1e-6)).toFixed(2)),
        iters: s.conjugate_gradient.iterations
      },
      {
        name: `SOR (ω=${sorOmega.toFixed(2)})`,
        median_ms: Number(sorMed.toFixed(2)),
        mean_ms: Number((s.sor.mean_runtime_ms ?? sorMed).toFixed(2)),
        min_ms: Number((s.sor.min_runtime_ms ?? sorMed).toFixed(2)),
        max_ms: Number((s.sor.max_runtime_ms ?? sorMed).toFixed(2)),
        memory_mb: s.sor.memory_mb,
        speedup: Number((luMed / Math.max(sorMed, 1e-6)).toFixed(3)),
        iters: s.sor.iterations
      },
    ];
  };

  const tMin = simData ? simData.solvers.direct_lu.t_min : 20.0;
  const tMax = simData ? simData.solvers.direct_lu.t_max : 75.0;
  const chartInfo = convergenceChartData();
  const activeReps = simData?.solver_settings?.repetitions ?? repetitions;

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
              <span className="px-2 py-0.5 text-[10px] font-mono font-semibold bg-cyan-950 text-cyan-400 border border-cyan-800 rounded">v3.1 RESEARCH</span>
            </div>
            <div className="text-[11px] text-gray-400">3D Finite Element Thermal Conduction &amp; Statistical Solver Benchmarking Platform</div>
          </div>
        </div>

        {/* Navigation tabs */}
        <nav className="flex items-center space-x-1 bg-graphite-850 p-1 rounded-lg border border-graphite-700" role="tablist">
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
        {replayBanner && (
          <div className="mb-4 px-4 py-2.5 bg-cyan-950/80 border border-cyan-500/50 rounded-xl flex items-center justify-between text-xs font-mono text-cyan-200">
            <span>✓ {replayBanner} — Reconstructed &amp; Re-executed Live</span>
            <button onClick={() => setReplayBanner(null)} className="text-gray-400 hover:text-white">✕</button>
          </div>
        )}

        {/* TAB 1: 3D SIMULATION WORKSPACE */}
        {activeTab === 'simulation' && (
          <div className="grid grid-cols-12 gap-5">
            {/* LEFT CONTROL PANEL */}
            <div className="col-span-4 bg-graphite-900 border border-graphite-800 rounded-xl p-5 flex flex-col justify-between">
              <div className="space-y-3">
                <div className="border-b border-graphite-800 pb-2 flex items-center justify-between">
                  <div>
                    <h2 className="text-sm font-bold tracking-wide text-white uppercase flex items-center space-x-2">
                      <Sliders className="w-4 h-4 text-cyan-400" />
                      <span>Physics &amp; Solver Setup</span>
                    </h2>
                    <p className="text-[11px] text-gray-400">3D Parametric Geometry &amp; Boundary Conditions</p>
                  </div>
                  <div className="flex space-x-1">
                    {[1, 10].map((r) => (
                      <button
                        key={r}
                        type="button"
                        onClick={() => { setRepetitions(r); executeSimulation(sorOmega, initialGuess, r); }}
                        title={r === 1 ? 'Single-Run Execution' : '10-Repetition Statistical Benchmark (Median Timing)'}
                        className={`px-2 py-0.5 rounded text-[10px] font-mono border ${
                          repetitions === r
                            ? 'bg-cyan-950 border-cyan-500 text-cyan-300 font-bold'
                            : 'bg-graphite-850 border-graphite-700 text-gray-400'
                        }`}
                      >
                        N={r}
                      </button>
                    ))}
                  </div>
                </div>

                {/* Preset Selector */}
                <div>
                  <label className="text-xs font-medium text-gray-300 block mb-1">Engineering Geometry</label>
                  <select
                    value={preset}
                    onChange={(e) => setPreset(e.target.value)}
                    className="w-full bg-graphite-850 border border-graphite-700 text-xs rounded-lg px-3 py-1.5 text-white focus:outline-none focus:border-cyan-500"
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
                    className="w-full bg-graphite-850 border border-graphite-700 text-xs rounded-lg px-3 py-1.5 text-white focus:outline-none focus:border-cyan-500"
                  >
                    <option value="aluminium_6061">Aluminium 6061-T6 (k = 167 W/m·K, ρ = 2700 kg/m³)</option>
                    <option value="copper_c11000">Electrolytic Copper C11000 (k = 390 W/m·K, ρ = 8940 kg/m³)</option>
                    <option value="stainless_steel_304">Stainless Steel 304 (k = 16.2 W/m·K, ρ = 8000 kg/m³)</option>
                    <option value="synthetic_graphite">Synthetic Graphite (k = 450 W/m·K, ρ = 1800 kg/m³)</option>
                  </select>
                </div>

                {/* Number of fins & power */}
                <div className="grid grid-cols-2 gap-3">
                  <div>
                    <div className="flex justify-between text-xs text-gray-300 mb-0.5">
                      <span>Fin Count:</span>
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
                    <div className="flex justify-between text-xs text-gray-300 mb-0.5">
                      <span>Die Power Q:</span>
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

                {/* Fin Height & Base Thickness Ratios */}
                <div className="grid grid-cols-2 gap-3">
                  <div>
                    <div className="flex justify-between text-[11px] text-gray-300 mb-0.5">
                      <span>Fin Height Ratio:</span>
                      <span className="font-mono text-cyan-300">{finHeightRatio.toFixed(2)}</span>
                    </div>
                    <input
                      type="range"
                      min="0.40"
                      max="0.90"
                      step="0.05"
                      value={finHeightRatio}
                      onChange={(e) => setFinHeightRatio(Number(e.target.value))}
                      className="w-full accent-cyan-500"
                    />
                  </div>
                  <div>
                    <div className="flex justify-between text-[11px] text-gray-300 mb-0.5">
                      <span>Base Thickness:</span>
                      <span className="font-mono text-cyan-300">{baseThicknessRatio.toFixed(2)}</span>
                    </div>
                    <input
                      type="range"
                      min="0.20"
                      max="0.55"
                      step="0.05"
                      value={baseThicknessRatio}
                      onChange={(e) => setBaseThicknessRatio(Number(e.target.value))}
                      className="w-full accent-cyan-500"
                    />
                  </div>
                </div>

                {/* Convection & Mesh */}
                <div className="grid grid-cols-2 gap-3">
                  <div>
                    <div className="flex justify-between text-xs text-gray-300 mb-0.5">
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
                    <label className="text-xs font-medium text-gray-300 block mb-0.5">Mesh Grid</label>
                    <div className="grid grid-cols-3 gap-1">
                      {['coarse', 'medium', 'fine'].map((res) => (
                        <button
                          key={res}
                          type="button"
                          onClick={() => setResolution(res)}
                          className={`py-1 rounded text-[10px] font-mono font-bold capitalize border ${
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
                <div className="bg-graphite-950 p-2.5 rounded-lg border border-graphite-800 space-y-1.5">
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

                {/* Global Energy Balance & Physical Metrics Card */}
                {simData?.energy_balance && (
                  <div className="bg-graphite-950 p-2.5 rounded-lg border border-graphite-800 text-[11px] font-mono space-y-1">
                    <div className="text-cyan-400 font-bold flex items-center justify-between">
                      <span>GLOBAL ENERGY BALANCE &amp; MASS</span>
                      <span className="text-[10px] text-emerald-400">
                        {simData.energy_balance.energy_conserved ? 'CONSERVED' : 'IMBALANCE'}
                      </span>
                    </div>
                    <div className="flex justify-between text-gray-400">
                      <span>Input Q_in / Conv. Q_out:</span>
                      <strong className="text-gray-200">
                        {simData.energy_balance.q_in_watts.toFixed(2)} W / {simData.energy_balance.q_out_convective_watts.toFixed(2)} W
                      </strong>
                    </div>
                    <div className="flex justify-between text-gray-400">
                      <span>Conservation Error |ΔQ|/Q_in:</span>
                      <strong className="text-cyan-300">{formatPctError(simData.energy_balance)}</strong>
                    </div>
                    <div className="flex justify-between text-gray-400">
                      <span>Thermal Res R_th / Conductance:</span>
                      <strong className="text-emerald-400">
                        {simData.mesh_info.thermal_resistance_k_per_w ?? '0.839'} K/W ({simData.mesh_info.cooling_conductance_w_per_k ?? '1.19'} W/K)
                      </strong>
                    </div>
                    <div className="flex justify-between text-gray-400">
                      <span>Solid Volume / Mass:</span>
                      <strong className="text-gray-200">
                        {simData.mesh_info.solid_volume_cm3 ?? '130.2'} cm³ ({simData.mesh_info.mass_grams ?? '351.5'} g)
                      </strong>
                    </div>
                  </div>
                )}
              </div>

              {/* Action Buttons */}
              <div className="space-y-2 mt-3">
                <button
                  onClick={() => executeSimulation()}
                  disabled={loading}
                  className="w-full py-2.5 bg-gradient-to-r from-cyan-600 to-blue-600 hover:from-cyan-500 hover:to-blue-500 disabled:opacity-50 text-white font-bold text-xs rounded-xl shadow-lg shadow-cyan-600/30 transition-all flex items-center justify-center space-x-2"
                >
                  {loading ? <RefreshCw className="w-4 h-4 animate-spin" /> : <Play className="w-4 h-4" />}
                  <span>{loading ? 'Solving Assembled System...' : `Run Simulation & Audit (N=${repetitions})`}</span>
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
                    const medTime = s.median_runtime_ms ?? s.execution_time_ms;
                    return (
                      <div key={sKey} className="bg-graphite-900 border border-graphite-800 p-3 rounded-xl flex flex-col justify-between">
                        <div className="flex items-center justify-between mb-1">
                          <span className="text-[11px] font-bold text-white font-mono truncate">{s.solver_name}</span>
                          <StatusBadge converged={s.converged} label={s.status_label} />
                        </div>
                        <div className="grid grid-cols-3 gap-1 text-[11px] font-mono text-gray-400 mt-1">
                          <div>Median: <strong className="text-gray-200">{medTime.toFixed(1)}ms</strong></div>
                          <div>Iters: <strong className="text-gray-200">{sKey === 'direct_lu' ? 'N/A (Direct)' : s.iterations}</strong></div>
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

        {/* TAB 2: SOLVER BATTLE ARENA, STATISTICAL BENCHMARK & SOLVER DECISION MAP */}
        {activeTab === 'battle' && simData && (
          <div className="space-y-6">
            {/* Phase 3: Intelligent Solver Decision Map Panel */}
            <div className="bg-gradient-to-r from-blue-950/80 via-indigo-950/80 to-graphite-900 border border-cyan-800/40 p-5 rounded-2xl shadow-xl space-y-4">
              <div className="flex flex-wrap items-center justify-between gap-3">
                <div className="flex items-center space-x-3">
                  <div className="p-2.5 bg-cyan-500/20 rounded-xl text-cyan-400">
                    <Award className="w-6 h-6" />
                  </div>
                  <div>
                    <div className="text-[11px] font-mono font-bold text-cyan-400 uppercase tracking-widest">
                      Solver Decision Intelligence Engine ({activeReps > 1 ? `Repeated Statistical Benchmark: N=${activeReps} Trials` : 'Single-Run Mode'})
                    </div>
                    <h3 className="text-base font-bold text-white">
                      Recommended Solver:{' '}
                      <span className="text-emerald-400">
                        {simData.solver_recommendation?.recommended_solver_name || 'Preconditioned Conjugate Gradient (Jacobi)'}
                      </span>
                      {simData.solver_recommendation?.alternative_solver_name && (
                        <span className="ml-3 text-xs font-mono text-gray-300">
                          | Alternative: <strong className="text-cyan-300">{simData.solver_recommendation.alternative_solver_name}</strong>
                        </span>
                      )}
                    </h3>
                  </div>
                </div>
                <div className="flex items-center space-x-2">
                  <button
                    onClick={() => { setRepetitions(1); executeSimulation(sorOmega, initialGuess, 1); }}
                    disabled={loading}
                    className={`px-3 py-1.5 rounded-lg text-xs font-mono border ${
                      activeReps === 1 ? 'bg-cyan-700 text-white border-cyan-400 font-bold' : 'bg-graphite-900 text-gray-300 border-graphite-700'
                    }`}
                  >
                    Single Run (N=1)
                  </button>
                  <button
                    onClick={() => { setRepetitions(10); executeSimulation(sorOmega, initialGuess, 10); }}
                    disabled={loading}
                    className={`px-3 py-1.5 rounded-lg text-xs font-mono border flex items-center space-x-1 ${
                      activeReps >= 10 ? 'bg-emerald-700 text-white border-emerald-400 font-bold' : 'bg-graphite-900 text-cyan-300 border-cyan-700 hover:bg-cyan-950'
                    }`}
                  >
                    {loading ? <RefreshCw className="w-3.5 h-3.5 animate-spin" /> : <Play className="w-3.5 h-3.5" />}
                    <span>Run Statistical Benchmark (N=10 Trials)</span>
                  </button>
                </div>
              </div>

              {simData.solver_recommendation?.primary_reason && (
                <div className="bg-graphite-950/90 border border-cyan-900/60 p-3.5 rounded-xl text-xs font-mono text-gray-200 leading-relaxed">
                  <strong className="text-cyan-400">Empirical Decision Evidence ({simData.solver_recommendation.evidence_level || 'Verified'} — Confidence {(simData.solver_recommendation.confidence_score * 100).toFixed(0)}%):</strong>{' '}
                  {simData.solver_recommendation.primary_reason}
                </div>
              )}

              <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                {(simData.engineering_conclusions || [simData.winner_explanation]).map((note, idx) => (
                  <div key={idx} className="bg-graphite-950/80 border border-graphite-800 p-3 rounded-lg text-xs text-gray-200 font-mono leading-relaxed">
                    {note}
                  </div>
                ))}
              </div>
            </div>

            {/* Detailed 3-Solver Statistical Audit Cards */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-5">
              {(['direct_lu', 'conjugate_gradient', 'sor'] as const).map((sKey, idx) => {
                const s = simData.solvers[sKey];
                const borderCol = s.converged ? 'border-emerald-800/60' : 'border-amber-700/80';
                const medMs = s.median_runtime_ms ?? s.execution_time_ms;
                const meanMs = s.mean_runtime_ms ?? medMs;
                const stdMs = s.std_runtime_ms ?? 0.0;
                const minMs = s.min_runtime_ms ?? medMs;
                const maxMs = s.max_runtime_ms ?? medMs;
                const singleMs = s.single_run_time_ms ?? medMs;
                const speedup = s.speedup_vs_lu_median ?? 1.0;

                return (
                  <div key={sKey} className={`bg-graphite-900 border ${borderCol} rounded-xl p-5 space-y-3`}>
                    <div className="flex justify-between items-center">
                      <span className="text-xs font-bold text-cyan-400 font-mono">
                        SOLVER {String.fromCharCode(65 + idx)} ({s.repetitions ?? 1}x RUNS)
                      </span>
                      <StatusBadge converged={s.converged} label={s.status_label} />
                    </div>
                    <h4 className="text-sm font-bold text-white">{s.solver_name}</h4>
                    <div className="space-y-1.5 pt-2 border-t border-graphite-800 text-xs font-mono">
                      <div className="flex justify-between text-gray-400">
                        <span>Median Runtime (Primary):</span>
                        <strong className="text-emerald-400">{medMs.toFixed(2)} ms ({speedup.toFixed(2)}× vs LU)</strong>
                      </div>
                      <div className="flex justify-between text-gray-400">
                        <span>Mean ± Std Dev:</span>
                        <strong className="text-white">{meanMs.toFixed(2)} ± {stdMs.toFixed(2)} ms</strong>
                      </div>
                      <div className="flex justify-between text-gray-400">
                        <span>Min / Max / Single-Run:</span>
                        <strong className="text-gray-300">[{minMs.toFixed(1)}, {maxMs.toFixed(1)}] / {singleMs.toFixed(1)} ms</strong>
                      </div>
                      <div className="flex justify-between text-gray-400">
                        <span>Iterations Executed:</span>
                        <strong className="text-white">{sKey === 'direct_lu' ? 'N/A (Direct)' : s.iterations}</strong>
                      </div>
                      {sKey === 'direct_lu' && s.factorization_ms !== undefined && (
                        <div className="flex justify-between text-gray-400">
                          <span>Factorize / Solve Split:</span>
                          <strong className="text-cyan-300">
                            {(s.median_factorization_ms ?? s.factorization_ms).toFixed(2)} / {(s.median_substitution_ms ?? s.substitution_ms ?? 0).toFixed(2)} ms
                          </strong>
                        </div>
                      )}
                      <div className="flex justify-between text-gray-400">
                        <span>True Rel. Residual ||r||₂/||F||₂:</span>
                        <strong className={s.converged ? 'text-emerald-400' : 'text-amber-400'}>{s.final_residual.toExponential(3)}</strong>
                      </div>
                      <div className="flex justify-between text-gray-400">
                        <span>Absolute Residual ||F-KT||₂:</span>
                        <strong className="text-gray-200">{(s.absolute_residual ?? 0).toExponential(3)} W</strong>
                      </div>
                      <div className="flex justify-between text-gray-400">
                        <span>Total Memory (Base + Alloc):</span>
                        <strong className="text-white">
                          {s.memory_mb.toFixed(3)} MB
                          {s.additional_solver_memory_mb !== undefined ? ` (+${s.additional_solver_memory_mb.toFixed(3)})` : ''}
                        </strong>
                      </div>
                      <div className="flex justify-between text-gray-400">
                        <span>Temp Range [T_min, T_max]:</span>
                        <strong className="text-rose-400">[{s.t_min.toFixed(2)}°C, {s.t_max.toFixed(2)}°C]</strong>
                      </div>
                      <div className="flex justify-between text-gray-400">
                        <span>Max Discrepancy vs LU:</span>
                        <strong className={s.max_discrepancy_vs_lu > 0.05 ? 'text-amber-400' : 'text-emerald-300'}>
                          {s.max_discrepancy_vs_lu.toFixed(6)} °C
                        </strong>
                      </div>
                      <div className="flex justify-between text-gray-400">
                        <span>Rel. L2 Solution Error:</span>
                        <strong className="text-cyan-300">{(s.relative_l2_error_vs_lu ?? 0).toExponential(2)}</strong>
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

            {/* FULL-HISTORY CONVERGENCE CHART & STATISTICAL RUNTIME DISTRIBUTION CHART */}
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
                        <ReferenceDot x={chartInfo.cgEnd.iteration} y={chartInfo.cgEnd.CG} r={5} fill="#10b981" stroke="#ffffff" />
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
                  <span>Statistical Runtime Distribution (Min / Median / Mean / Max)</span>
                </h4>
                <p className="text-xs text-gray-400 mb-3">
                  Median vs. Mean wall-time (ms) over N={activeReps} independent repetitions on {simData.mesh_info.degrees_of_freedom} DOF
                </p>
                <div className="h-72">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={battleBarData()}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#1f293d" />
                      <XAxis dataKey="name" stroke="#94a3b8" tick={{ fontSize: 10 }} />
                      <YAxis stroke="#94a3b8" tick={{ fontSize: 10 }} />
                      <Tooltip contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', fontSize: '11px', fontFamily: 'monospace' }} />
                      <Legend wrapperStyle={{ fontSize: '11px' }} />
                      <Bar dataKey="median_ms" fill="#10b981" name="Median Runtime (ms)" radius={[4, 4, 0, 0]} />
                      <Bar dataKey="mean_ms" fill="#3b82f6" name="Mean Runtime (ms)" radius={[4, 4, 0, 0]} />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </div>
            </div>

            {/* SOR RELAXATION PARAMETER ω EMPIRICAL SWEEP AUDIT */}
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
                            <td className="p-2.5 font-bold text-amber-400">
                              ω = {r.omega.toFixed(2)}
                              {r.omega >= 2.0 && (
                                <span className="ml-1.5 px-1.5 py-0.5 text-[9px] bg-purple-950 text-purple-300 border border-purple-700 rounded">
                                  STRESS TEST
                                </span>
                              )}
                            </td>
                            <td className="p-2.5">
                              <StatusBadge
                                converged={r.converged}
                                label={r.omega >= 2.0 ? 'Stress Test: ω≥2 (Ostrowski-Reich Violation)' : r.status}
                              />
                            </td>
                            <td className="p-2.5 text-white">{r.omega >= 2.0 ? '0 (Rejected)' : r.iterations}</td>
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

        {/* TAB 3: 3x3 GEOMETRY-SOLVER STATISTICAL BENCHMARK (9 EXPERIMENTS) */}
        {activeTab === 'matrix3x3' && (
          <div className="space-y-6">
            <div className="bg-graphite-900 border border-graphite-800 p-6 rounded-xl space-y-5">
              <div className="flex flex-wrap justify-between items-center gap-3">
                <div>
                  <h3 className="text-base font-bold text-white flex items-center space-x-2">
                    <Compass className="w-4 h-4 text-cyan-400" />
                    <span>3 × 3 Geometry-Solver Statistical Research Benchmark (Nine Experiments, N={matrixData?.repetitions ?? matrixReps} Repetitions)</span>
                  </h3>
                  <p className="text-xs text-gray-400 mt-1">
                    Each solver runs N={matrixData?.repetitions ?? matrixReps} repeated trials on the identical (K, F) system for (A) Standard, (B) Asymmetric, and (C) Perforated heat sinks. Primary comparison metric: Median Runtime.
                  </p>
                </div>
                <div className="flex items-center space-x-2 text-xs font-mono">
                  <span className="text-gray-400">Trials:</span>
                  {[1, 10].map((n) => (
                    <button
                      key={n}
                      onClick={() => { setMatrixReps(n); handleRun3x3Benchmark(matrixOmega, matrixInitGuess, n); }}
                      className={`px-2 py-1 rounded border ${
                        matrixReps === n ? 'bg-emerald-700 text-white border-emerald-400 font-bold' : 'bg-graphite-850 text-gray-300 border-graphite-700'
                      }`}
                    >
                      N={n}
                    </button>
                  ))}
                  <span className="text-gray-400 ml-2">SOR ω:</span>
                  {[1.35, 1.9].map((w) => (
                    <button
                      key={w}
                      onClick={() => { setMatrixOmega(w); handleRun3x3Benchmark(w, matrixInitGuess, matrixReps); }}
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
                    <span>Execute 9×{matrixReps} Benchmark</span>
                  </button>
                </div>
              </div>

              {matrixLoading ? (
                <div className="p-12 text-center text-cyan-400 font-mono text-xs flex flex-col items-center justify-center space-y-2">
                  <RefreshCw className="w-6 h-6 animate-spin" />
                  <span>Assembling 3 geometries and running 9 × {matrixReps} = {9 * matrixReps} independent solver trials...</span>
                </div>
              ) : matrixData ? (
                <div className="space-y-6">
                  {/* 9-Experiment Master Statistical Table */}
                  <div className="overflow-x-auto">
                    <table className="w-full text-left text-[11px] border border-graphite-800 font-mono">
                      <thead className="bg-graphite-850 text-gray-300">
                        <tr>
                          <th className="p-2.5">Exp</th>
                          <th className="p-2.5">Geometry</th>
                          <th className="p-2.5">DOF / NNZ (Sparsity)</th>
                          <th className="p-2.5">Solver</th>
                          <th className="p-2.5">Status</th>
                          <th className="p-2.5">Median (ms)</th>
                          <th className="p-2.5">Mean ± SD (ms)</th>
                          <th className="p-2.5">Speedup vs LU</th>
                          <th className="p-2.5">Iters</th>
                          <th className="p-2.5">Rel. Residual</th>
                          <th className="p-2.5">Mem</th>
                          <th className="p-2.5">T_max / ΔT_LU</th>
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
                            <td className="p-2.5 text-emerald-400 font-bold">
                              {(exp.median_runtime_ms ?? exp.execution_time_ms).toFixed(2)} ms
                            </td>
                            <td className="p-2.5 text-gray-300">
                              {(exp.mean_runtime_ms ?? exp.execution_time_ms).toFixed(2)} ± {(exp.std_runtime_ms ?? 0).toFixed(2)}
                            </td>
                            <td className="p-2.5 text-cyan-300 font-bold">{(exp.speedup_vs_lu_median ?? 1.0).toFixed(2)}×</td>
                            <td className="p-2.5 text-gray-200">{exp.iterations_display ?? (exp.solver_id === 'direct_lu' ? 'N/A (Direct)' : exp.iterations)}</td>
                            <td className="p-2.5 text-cyan-300">{Number(exp.final_residual).toExponential(2)}</td>
                            <td className="p-2.5 text-gray-200">{exp.memory_mb.toFixed(2)} MB</td>
                            <td className="p-2.5 text-rose-400">
                              {exp.t_max.toFixed(2)}°C / <span className="text-emerald-300">{Number(exp.max_discrepancy_vs_lu).toExponential(1)}</span>
                            </td>
                            <td className="p-2.5 text-cyan-300">{exp.energy_balance_error_formatted}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>

                  {/* Comparative Charts: Median Runtime & Speedup vs Geometry Complexity */}
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
                    <div className="bg-graphite-950 border border-graphite-800 p-4 rounded-xl">
                      <h4 className="text-xs font-bold text-white mb-2 font-mono">
                        MEDIAN RUNTIME BY GEOMETRY COMPLEXITY (ms, N={matrixData.repetitions ?? 10})
                      </h4>
                      <div className="h-56">
                        <ResponsiveContainer width="100%" height="100%">
                          <BarChart
                            data={(matrixData.geometries || []).map((g: any) => ({
                              name: g.geometry_name.split('(')[0].trim(),
                              Direct_LU: Number((g.solvers.direct_lu.median_runtime_ms ?? g.solvers.direct_lu.execution_time_ms).toFixed(2)),
                              PCG: Number((g.solvers.conjugate_gradient.median_runtime_ms ?? g.solvers.conjugate_gradient.execution_time_ms).toFixed(2)),
                            }))}
                          >
                            <CartesianGrid strokeDasharray="3 3" stroke="#1f293d" />
                            <XAxis dataKey="name" stroke="#94a3b8" tick={{ fontSize: 10 }} />
                            <YAxis stroke="#94a3b8" tick={{ fontSize: 10 }} />
                            <Tooltip contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', fontSize: '11px' }} />
                            <Legend wrapperStyle={{ fontSize: '11px' }} />
                            <Bar dataKey="Direct_LU" fill="#3b82f6" name="Direct Sparse LU Median (ms)" />
                            <Bar dataKey="PCG" fill="#10b981" name="Preconditioned CG Median (ms)" />
                          </BarChart>
                        </ResponsiveContainer>
                      </div>
                    </div>

                    <div className="bg-graphite-950 border border-graphite-800 p-4 rounded-xl">
                      <h4 className="text-xs font-bold text-white mb-2 font-mono">
                        ITERATION COUNT &amp; PCG SPEEDUP VS. DIRECT LU ACROSS GEOMETRIES
                      </h4>
                      <div className="h-56">
                        <ResponsiveContainer width="100%" height="100%">
                          <BarChart
                            data={(matrixData.geometries || []).map((g: any) => ({
                              name: g.geometry_name.split('(')[0].trim(),
                              PCG_Iters: g.solvers.conjugate_gradient.iterations,
                              SOR_Iters: g.solvers.sor.iterations,
                            }))}
                          >
                            <CartesianGrid strokeDasharray="3 3" stroke="#1f293d" />
                            <XAxis dataKey="name" stroke="#94a3b8" tick={{ fontSize: 10 }} />
                            <YAxis stroke="#94a3b8" tick={{ fontSize: 10 }} />
                            <Tooltip contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', fontSize: '11px' }} />
                            <Legend wrapperStyle={{ fontSize: '11px' }} />
                            <Bar dataKey="PCG_Iters" fill="#10b981" name="PCG Iterations" />
                            <Bar dataKey="SOR_Iters" fill="#f59e0b" name="SOR Iterations" />
                          </BarChart>
                        </ResponsiveContainer>
                      </div>
                    </div>
                  </div>

                  <div className="bg-graphite-950 p-4 rounded-xl border border-graphite-800 text-xs font-mono text-gray-300 leading-relaxed whitespace-pre-line">
                    <span className="text-cyan-400 font-bold block mb-1">3×3 Statistical Benchmark Interpretation:</span>
                    {matrixData.comparative_analysis}
                  </div>
                </div>
              ) : null}
            </div>
          </div>
        )}

        {/* TAB 4: NUMERICAL VERIFICATION & MESH CONVERGENCE LABORATORY (Phase 6) */}
        {activeTab === 'mesh' && (
          <div className="space-y-6">
            <div className="bg-graphite-900 border border-graphite-800 p-6 rounded-xl space-y-5">
              <div className="flex items-center justify-between">
                <div>
                  <h3 className="text-base font-bold text-white mb-1">
                    Numerical Verification, 3-Level Grid Convergence (Coarse / Medium / Fine) &amp; Solver Error Analysis
                  </h3>
                  <p className="text-xs text-gray-400">
                    1) Closed-form 3D analytical boundary-value verification. 2) Coarse, Medium &amp; Fine mesh convergence. 3) Residual Error vs. Solution Error.
                  </p>
                </div>
                <button
                  onClick={handleRunMeshStudy}
                  disabled={meshLoading}
                  className="px-4 py-2 bg-cyan-600 hover:bg-cyan-500 text-white text-xs font-mono font-semibold rounded-lg flex items-center space-x-1.5"
                >
                  {meshLoading ? <RefreshCw className="w-3.5 h-3.5 animate-spin" /> : <Play className="w-3.5 h-3.5" />}
                  <span>Run Coarse / Medium / Fine Study</span>
                </button>
              </div>

              {meshLoading ? (
                <div className="p-10 flex items-center justify-center space-x-2 text-cyan-400">
                  <RefreshCw className="w-5 h-5 animate-spin" />
                  <span className="text-xs font-mono">Solving Coarse (16×12×10), Medium (24×18×14), and Fine (32×24×18) 3D FEM Grids...</span>
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
                            Measured L2 Order p = {Number(meshData.analytical_verification.observed_order).toFixed(4)} (Expected 2.0000)
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
                              <th className="p-2.5">Measured Order p</th>
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

                  {/* 3-Level Finned Heat Sink Mesh Refinement Table (Coarse, Medium, Fine) */}
                  <div className="overflow-x-auto">
                    <table className="w-full text-left text-xs border border-graphite-800 font-mono">
                      <thead className="bg-graphite-850 text-gray-300">
                        <tr>
                          <th className="p-2.5">Mesh Level</th>
                          <th className="p-2.5">Elements / DOF</th>
                          <th className="p-2.5">Char. h (mm)</th>
                          <th className="p-2.5">T_max / T_avg (°C)</th>
                          <th className="p-2.5">Step Change %</th>
                          <th className="p-2.5">Direct LU Time</th>
                          <th className="p-2.5">PCG Time (Iters)</th>
                          <th className="p-2.5">PCG Residual vs. Solution L2 Err</th>
                          <th className="p-2.5">Energy Balance Err</th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-graphite-800">
                        {meshData.data.map((row: any, i: number) => (
                          <tr key={i} className="hover:bg-graphite-850/50">
                            <td className="p-2.5 font-bold text-cyan-400">{row.resolution} ({row.grid_dims})</td>
                            <td className="p-2.5 text-white font-bold">{row.elements} / {row.dof}</td>
                            <td className="p-2.5 text-gray-300">{row.characteristic_h_mm} mm</td>
                            <td className="p-2.5 text-rose-400">{row.t_max.toFixed(2)}°C / {row.t_avg.toFixed(2)}°C</td>
                            <td className="p-2.5 text-amber-300">{row.step_relative_change_pct ?? 0}%</td>
                            <td className="p-2.5 text-cyan-300">{row.direct_lu_ms.toFixed(1)} ms</td>
                            <td className="p-2.5 text-emerald-400">{row.cg_ms.toFixed(1)} ms ({row.cg_iterations})</td>
                            <td className="p-2.5 text-gray-200">
                              Res: {Number(row.cg_residual).toExponential(2)} | L2: {Number(row.cg_rel_l2_error_vs_lu ?? 0).toExponential(2)}
                            </td>
                            <td className="p-2.5 text-cyan-300">{row.energy_balance_error_formatted}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>

                  {/* Mesh Convergence Curve & Residual vs Solution Error Explanation */}
                  <div className="grid grid-cols-1 md:grid-cols-12 gap-5">
                    <div className="md:col-span-6 bg-graphite-950 border border-graphite-800 p-4 rounded-xl">
                      <h4 className="text-xs font-bold text-white mb-2 font-mono">
                        SPATIAL GRID CONVERGENCE CURVE (T_max &amp; T_avg vs. Degrees of Freedom)
                      </h4>
                      <div className="h-52">
                        <ResponsiveContainer width="100%" height="100%">
                          <LineChart data={meshData.data}>
                            <CartesianGrid strokeDasharray="3 3" stroke="#1f293d" />
                            <XAxis dataKey="dof" stroke="#94a3b8" tick={{ fontSize: 10 }} />
                            <YAxis stroke="#94a3b8" tick={{ fontSize: 10 }} domain={['auto', 'auto']} />
                            <Tooltip contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', fontSize: '11px' }} />
                            <Legend wrapperStyle={{ fontSize: '11px' }} />
                            <Line type="monotone" dataKey="t_max" stroke="#f43f5e" strokeWidth={2.2} name="Peak Temp T_max (°C)" />
                            <Line type="monotone" dataKey="t_avg" stroke="#38bdf8" strokeWidth={2.2} name="Average Temp T_avg (°C)" />
                          </LineChart>
                        </ResponsiveContainer>
                      </div>
                    </div>

                    <div className="md:col-span-6 flex flex-col justify-between space-y-3">
                      <div className="bg-graphite-950 p-4 rounded-xl border border-graphite-800 text-xs text-gray-300 leading-relaxed font-mono whitespace-pre-line">
                        {meshData.residual_vs_solution_error_note}
                      </div>
                      <div className="bg-graphite-950 p-4 rounded-xl border border-cyan-900/50 text-xs text-gray-300 leading-relaxed font-mono">
                        <strong className="text-cyan-400">Grid Convergence Verdict:</strong> {meshData.convergence_analysis}
                      </div>
                    </div>
                  </div>
                </div>
              ) : null}
            </div>
          </div>
        )}

        {/* TAB 5: PARAMETRIC DESIGN OPTIMIZATION & PARETO FRONTIER (Phase 5) */}
        {activeTab === 'optimization' && (
          <div className="space-y-6">
            <div className="bg-graphite-900 border border-graphite-800 p-6 rounded-xl space-y-5">
              <div className="flex flex-wrap justify-between items-center gap-3">
                <div>
                  <h3 className="text-base font-bold text-white flex items-center space-x-2">
                    <TrendingUp className="w-4 h-4 text-cyan-400" />
                    <span>Parametric Design Optimization &amp; Multi-Objective Pareto Studio</span>
                  </h3>
                  <p className="text-xs text-gray-400 mt-1">
                    Evaluates peak temperature, thermal resistance R_th (K/W), cooling conductance G_th (W/K), solid volume (cm³), mass (g), and Pareto optimality.
                  </p>
                </div>
                <div className="flex flex-wrap items-center gap-2">
                  <select
                    value={sweepParam}
                    onChange={(e) => { setSweepParam(e.target.value); handleRunOptimizationSweep(e.target.value, optObjective); }}
                    className="bg-graphite-850 border border-graphite-700 text-xs rounded-lg px-3 py-2 text-white font-mono"
                  >
                    <option value="fin_count">1. Fin Count (4 to 12 fins)</option>
                    <option value="fin_spacing_ratio">2. Fin Spacing Ratio (0.70 to 1.30)</option>
                    <option value="fin_thickness_ratio">3. Fin Thickness Ratio (0.60 to 1.50)</option>
                    <option value="fin_height_ratio">4. Fin Height Ratio (0.40 to 0.90)</option>
                    <option value="base_thickness_ratio">5. Base Thickness Ratio (0.20 to 0.55)</option>
                    <option value="material_conductivity">6. Material Conductivity (Steel / Al / Cu / Graphite)</option>
                    <option value="power_watts">7. Heat Input Q (35 to 100 W)</option>
                    <option value="convection_h">8. Convection Coefficient h (15 to 100 W/m²K)</option>
                  </select>
                  <select
                    value={optObjective}
                    onChange={(e) => { setOptObjective(e.target.value); handleRunOptimizationSweep(sweepParam, e.target.value); }}
                    className="bg-graphite-850 border border-cyan-800 text-xs rounded-lg px-3 py-2 text-cyan-300 font-mono"
                  >
                    <option value="min_t_max">Objective 1: Minimize Peak Temp (T_max)</option>
                    <option value="min_thermal_resistance">Objective 2: Minimize Thermal Resistance (R_th)</option>
                    <option value="max_cooling_conductance">Objective 3: Maximize Cooling Conductance (G_th)</option>
                    <option value="min_volume">Objective 4: Minimize Material Volume / Mass</option>
                    <option value="multi_objective">Objective 5: Multi-Objective Pareto Balance (R_th vs Mass)</option>
                  </select>
                  <button
                    onClick={() => handleRunOptimizationSweep()}
                    disabled={optLoading}
                    className="px-4 py-2 bg-cyan-600 hover:bg-cyan-500 text-white text-xs font-semibold rounded-lg font-mono flex items-center space-x-1"
                  >
                    {optLoading ? <RefreshCw className="w-3.5 h-3.5 animate-spin" /> : <Play className="w-3.5 h-3.5" />}
                    <span>Execute Sweep</span>
                  </button>
                </div>
              </div>

              {optLoading ? (
                <div className="p-10 text-center text-cyan-400 font-mono text-xs flex flex-col items-center justify-center space-y-2">
                  <RefreshCw className="w-6 h-6 animate-spin" />
                  <span>Executing 3D FEM parametric sweep and computing Pareto frontier...</span>
                </div>
              ) : optData ? (
                <div className="space-y-5">
                  <div className="grid grid-cols-1 md:grid-cols-3 lg:grid-cols-6 gap-3">
                    {optData.results.map((pt: any, i: number) => {
                      const isBest = pt.param_value === optData.optimal_point.parameter_value;
                      return (
                        <div
                          key={i}
                          className={`p-3.5 rounded-xl border font-mono transition-all ${
                            isBest
                              ? 'bg-cyan-950/60 border-cyan-400 shadow-lg shadow-cyan-950'
                              : 'bg-graphite-850 border-graphite-700'
                          }`}
                        >
                          <div className="flex items-center justify-between">
                            <span className="text-[10px] text-gray-400">{optData.sweep_parameter}</span>
                            {pt.is_pareto_optimal && (
                              <span className="px-1.5 py-0.5 text-[9px] bg-emerald-950 text-emerald-300 border border-emerald-700 rounded">
                                PARETO
                              </span>
                            )}
                          </div>
                          <div className="text-sm font-bold text-white mt-0.5 truncate">{String(pt.param_value)}</div>
                          <div className="text-xs text-rose-400 font-bold mt-1">T_max: {pt.t_max.toFixed(2)}°C</div>
                          <div className="text-[10px] text-cyan-300">R_th: {pt.thermal_resistance_k_per_w} K/W</div>
                          <div className="text-[10px] text-emerald-400">G_th: {pt.cooling_conductance_w_per_k} W/K</div>
                          <div className="text-[10px] text-gray-300">
                            Vol: {pt.material_volume_cm3} cm³ ({pt.mass_grams} g)
                          </div>
                          <div className="text-[10px] text-gray-400">PCG: {pt.cg_runtime_ms.toFixed(1)} ms</div>
                        </div>
                      );
                    })}
                  </div>

                  {/* Pareto Frontier & Trade-off Charts */}
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
                    <div className="bg-graphite-950 border border-graphite-800 p-4 rounded-xl">
                      <h4 className="text-xs font-bold text-white mb-2 font-mono">
                        THERMAL RESISTANCE R_th (K/W) &amp; PEAK TEMPERATURE T_max (°C) ACROSS TESTED DESIGNS
                      </h4>
                      <div className="h-52">
                        <ResponsiveContainer width="100%" height="100%">
                          <LineChart data={optData.results}>
                            <CartesianGrid strokeDasharray="3 3" stroke="#1f293d" />
                            <XAxis dataKey="param_value" stroke="#94a3b8" tick={{ fontSize: 9 }} />
                            <YAxis yAxisId="left" stroke="#f43f5e" tick={{ fontSize: 10 }} />
                            <YAxis yAxisId="right" orientation="right" stroke="#06b6d4" tick={{ fontSize: 10 }} />
                            <Tooltip contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', fontSize: '11px' }} />
                            <Legend wrapperStyle={{ fontSize: '11px' }} />
                            <Line yAxisId="left" type="monotone" dataKey="t_max" stroke="#f43f5e" strokeWidth={2} name="T_max (°C)" />
                            <Line yAxisId="right" type="monotone" dataKey="thermal_resistance_k_per_w" stroke="#06b6d4" strokeWidth={2} name="R_th (K/W)" />
                          </LineChart>
                        </ResponsiveContainer>
                      </div>
                    </div>

                    <div className="bg-graphite-950 border border-graphite-800 p-4 rounded-xl">
                      <h4 className="text-xs font-bold text-white mb-2 font-mono">
                        MATERIAL MASS (g) VS. COOLING CONDUCTANCE G_th (W/K) — PARETO TRADE-OFF
                      </h4>
                      <div className="h-52">
                        <ResponsiveContainer width="100%" height="100%">
                          <BarChart data={optData.results}>
                            <CartesianGrid strokeDasharray="3 3" stroke="#1f293d" />
                            <XAxis dataKey="param_value" stroke="#94a3b8" tick={{ fontSize: 9 }} />
                            <YAxis yAxisId="left" stroke="#10b981" tick={{ fontSize: 10 }} />
                            <YAxis yAxisId="right" orientation="right" stroke="#94a3b8" tick={{ fontSize: 10 }} />
                            <Tooltip contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', fontSize: '11px' }} />
                            <Legend wrapperStyle={{ fontSize: '11px' }} />
                            <Bar yAxisId="left" dataKey="cooling_conductance_w_per_k" fill="#10b981" name="Conductance G_th (W/K)" />
                            <Bar yAxisId="right" dataKey="mass_grams" fill="#475569" name="Mass (grams)" />
                          </BarChart>
                        </ResponsiveContainer>
                      </div>
                    </div>
                  </div>

                  <div className="bg-cyan-950/40 border border-cyan-800/60 p-4 rounded-xl text-xs font-mono text-cyan-200 space-y-2">
                    <div>
                      <strong className="text-cyan-300">Best Design Found Within Tested Configurations:</strong> {optData.engineering_recommendation}
                    </div>
                    <div className="text-[11px] text-gray-300 leading-relaxed border-t border-cyan-900/50 pt-2">
                      <strong className="text-emerald-400">Objective Independence &amp; Pareto Dominance Methodology:</strong>{' '}
                      Under fixed power Q and ambient T_∞, Thermal Resistance R_th = (T_max - T_∞)/Q and Cooling Conductance G_th = 1/R_th are exact mathematical reciprocals (and T_max = T_∞ + Q·R_th is an affine transform), representing a single thermal objective axis f₁. The non-dominated Pareto Frontier strictly evaluates dominance across the two genuinely independent conflicting physical objectives: f₁ = Minimize Thermal Resistance R_th (K/W) vs. f₂ = Minimize Solid Mass m = ρ·V_solid (g).
                    </div>
                    <div className="text-[11px] text-gray-400">
                      Scope &amp; Search Limitation: {optData.optimal_point?.scope_note}
                    </div>
                  </div>
                </div>
              ) : null}
            </div>
          </div>
        )}

        {/* TAB 6: REPRODUCIBLE EXPERIMENTS REPOSITORY, REPLAY & DIFF COMPARISON (Phase 4) */}
        {activeTab === 'experiments' && (
          <div className="space-y-6">
            <div className="bg-graphite-900 border border-graphite-800 p-6 rounded-xl space-y-5">
              <div className="flex flex-wrap items-center justify-between gap-3">
                <div>
                  <h3 className="text-base font-bold text-white mb-1 flex items-center space-x-2">
                    <Database className="w-4 h-4 text-cyan-400" />
                    <span>Reproducible Experiments Repository, Replay &amp; Parameter Diff Comparison</span>
                  </h3>
                  <p className="text-xs text-gray-400">
                    Every saved experiment preserves full geometry, material, BCs, mesh, tolerances, backend version, and results. Select 2+ experiments to diff changed parameters.
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
                  <button
                    onClick={handleCompareSelectedExperiments}
                    disabled={selectedExpIds.length < 2}
                    className="px-3 py-1.5 bg-indigo-600 hover:bg-indigo-500 disabled:opacity-40 text-white text-xs font-mono rounded-lg flex items-center space-x-1"
                  >
                    <GitCompare className="w-3.5 h-3.5" />
                    <span>Compare Selected ({selectedExpIds.length})</span>
                  </button>
                </div>
              </div>

              {experiments.length === 0 ? (
                <div className="p-8 text-center text-gray-500 text-xs font-mono">
                  No saved experiments in storage yet. Run a simulation and click "Save Current Run".
                </div>
              ) : (
                <div className="overflow-x-auto">
                  <table className="w-full text-left text-xs border border-graphite-800 font-mono">
                    <thead className="bg-graphite-850 text-gray-300">
                      <tr>
                        <th className="p-2.5">Select</th>
                        <th className="p-2.5">ID / Version</th>
                        <th className="p-2.5">Title &amp; Timestamp</th>
                        <th className="p-2.5">Geometry / Fins</th>
                        <th className="p-2.5">Material / BCs</th>
                        <th className="p-2.5">Mesh / ω</th>
                        <th className="p-2.5">Peak Temp</th>
                        <th className="p-2.5">PCG Time</th>
                        <th className="p-2.5">Replay</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-graphite-800">
                      {experiments.map((exp, idx) => (
                        <tr key={idx} className="hover:bg-graphite-850/50">
                          <td className="p-2.5">
                            <input
                              type="checkbox"
                              checked={selectedExpIds.includes(exp.id)}
                              onChange={() => handleToggleSelectExperiment(exp.id)}
                              className="accent-cyan-500 cursor-pointer"
                            />
                          </td>
                          <td className="p-2.5">
                            <div className="font-bold text-cyan-400">{exp.id}</div>
                            <div className="text-[10px] text-gray-500">v{exp.backend_version || '3.1'}</div>
                          </td>
                          <td className="p-2.5">
                            <div className="text-white font-semibold">{exp.title}</div>
                            <div className="text-[10px] text-gray-400">{exp.timestamp}</div>
                          </td>
                          <td className="p-2.5 text-gray-300">
                            {exp.preset} ({exp.fin_count ?? 7} fins)
                          </td>
                          <td className="p-2.5 text-gray-300">
                            {exp.material} ({exp.power_watts ?? 65}W, h={exp.convection_h ?? 45})
                          </td>
                          <td className="p-2.5 text-gray-300">
                            {exp.resolution || 'coarse'} (ω={Number(exp.sor_omega ?? 1.9).toFixed(2)})
                          </td>
                          <td className="p-2.5 text-rose-400 font-bold">
                            {exp.t_max ? `${exp.t_max.toFixed(2)} °C` : 'N/A'}
                          </td>
                          <td className="p-2.5 text-emerald-400">
                            {exp.cg_time_ms ? `${exp.cg_time_ms.toFixed(1)} ms` : 'N/A'}
                          </td>
                          <td className="p-2.5">
                            <button
                              onClick={() => handleReplayExperiment(exp)}
                              className="px-2.5 py-1 bg-cyan-950 hover:bg-cyan-700 text-cyan-300 hover:text-white border border-cyan-700 rounded text-[10px] flex items-center space-x-1"
                            >
                              <RotateCcw className="w-3 h-3" />
                              <span>Replay Run</span>
                            </button>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}

              {/* Multi-Experiment Diff & Comparison Panel */}
              {comparisonData && comparisonData.status === 'OK' && (
                <div className="bg-graphite-950 border border-indigo-500/40 p-4 rounded-xl space-y-4 font-mono text-xs">
                  <div className="flex items-center justify-between">
                    <span className="font-bold text-indigo-300">
                      MULTI-EXPERIMENT PARAMETER DIFF &amp; OUTPUT COMPARISON ({comparisonData.experiment_ids.join(' vs. ')})
                    </span>
                    <button onClick={() => setComparisonData(null)} className="text-gray-400 hover:text-white">✕</button>
                  </div>

                  {comparisonData.changed_parameters.length > 0 ? (
                    <div>
                      <div className="text-[11px] text-amber-400 font-bold mb-1">
                        Changed Input Parameters Across Selected Experiments:
                      </div>
                      <div className="grid grid-cols-1 md:grid-cols-3 gap-2">
                        {comparisonData.changed_parameters.map((cp: any, i: number) => (
                          <div key={i} className="bg-graphite-900 border border-amber-800/50 p-2.5 rounded-lg">
                            <div className="text-amber-300 font-bold">{cp.parameter}</div>
                            {Object.entries(cp.values_by_experiment).map(([eid, val]: any) => (
                              <div key={eid} className="text-gray-300 text-[11px]">
                                [{eid}]: <strong className="text-white">{String(val)}</strong>
                              </div>
                            ))}
                          </div>
                        ))}
                      </div>
                    </div>
                  ) : (
                    <div className="text-emerald-400 text-[11px]">
                      All tracked input parameters are identical across the selected experiments (Reproducibility Check).
                    </div>
                  )}

                  <div className="overflow-x-auto">
                    <table className="w-full text-left text-[11px] border border-graphite-800">
                      <thead className="bg-graphite-850 text-gray-300">
                        <tr>
                          <th className="p-2">Exp ID</th>
                          <th className="p-2">Title</th>
                          <th className="p-2">DOF</th>
                          <th className="p-2">T_max (°C)</th>
                          <th className="p-2">T_avg (°C)</th>
                          <th className="p-2">R_th (K/W)</th>
                          <th className="p-2">LU Time</th>
                          <th className="p-2">PCG Time (Iters)</th>
                          <th className="p-2">SOR Status</th>
                          <th className="p-2">Energy Balance Err</th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-graphite-800">
                        {comparisonData.experiments.map((r: any, i: number) => (
                          <tr key={i}>
                            <td className="p-2 font-bold text-cyan-400">{r.id}</td>
                            <td className="p-2 text-white">{r.title}</td>
                            <td className="p-2">{r.dof}</td>
                            <td className="p-2 text-rose-400 font-bold">{r.t_max ? `${r.t_max.toFixed(2)}°C` : '—'}</td>
                            <td className="p-2">{r.t_avg ? `${r.t_avg.toFixed(2)}°C` : '—'}</td>
                            <td className="p-2 text-cyan-300">{r.thermal_resistance_k_per_w ?? '—'}</td>
                            <td className="p-2">{r.lu_time_ms ? `${r.lu_time_ms.toFixed(1)} ms` : '—'}</td>
                            <td className="p-2 text-emerald-400">{r.cg_time_ms ? `${r.cg_time_ms.toFixed(1)} ms (${r.cg_iterations})` : '—'}</td>
                            <td className="p-2 text-amber-300">{r.sor_status}</td>
                            <td className="p-2 text-cyan-300">{r.energy_balance_error}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              )}
            </div>
          </div>
        )}

        {/* TAB 7: ADVANCED CONTEXT-AWARE THERMO AI COPILOT (Phase 7) */}
        {activeTab === 'ai' && (
          <div className="bg-graphite-900 border border-graphite-800 rounded-xl h-[calc(100vh-140px)] flex flex-col overflow-hidden">
            <div className="p-4 border-b border-graphite-800 bg-graphite-850 flex items-center justify-between">
              <div className="flex items-center space-x-2">
                <Bot className="w-5 h-5 text-cyan-400" />
                <span className="font-bold text-sm text-white">THERMO AI — Context-Aware Thermal Engineering Copilot</span>
              </div>
              <span className="text-[10px] font-mono text-emerald-400 bg-emerald-950 px-2 py-0.5 rounded border border-emerald-800">
                STRICT SEPARATION: NUMERICAL RESULT vs. AI INTERPRETATION
              </span>
            </div>

            <div className="flex-1 p-5 overflow-y-auto space-y-4">
              {aiMessages.map((msg, idx) => (
                <div
                  key={idx}
                  className={`flex ${msg.sender === 'user' ? 'justify-end' : 'justify-start'}`}
                >
                  <div
                    className={`max-w-3xl rounded-xl p-4 text-xs leading-relaxed ${
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
                    <span>Inspecting live FEM matrices, residuals, and energy balances...</span>
                  </div>
                </div>
              )}
            </div>

            {/* All 9 Required Suggested Questions */}
            <div className="px-5 py-2.5 bg-graphite-950 border-t border-graphite-800 flex flex-wrap items-center gap-1.5 text-[11px]">
              <span className="text-gray-500 font-mono mr-1">Suggested Prompts:</span>
              {[
                'Why is this geometry producing a hotspot?',
                'Why did CG outperform Direct LU?',
                'Why did SOR fail to converge?',
                'Is this mesh sufficiently refined?',
                'Which geometry gives the best thermal performance?',
                'What happens if I increase fin count?',
                'Which solver should I use for this problem?',
                'Explain the energy balance.',
                'Interpret my benchmark results.'
              ].map((sug, i) => (
                <button
                  key={i}
                  onClick={() => handleAskAI(sug)}
                  className="px-2.5 py-1 bg-graphite-850 hover:bg-cyan-950 hover:border-cyan-600 text-gray-300 hover:text-cyan-200 rounded border border-graphite-700 transition-all"
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
                placeholder="Ask THERMO AI about hotspot physics, PCG vs Direct LU fill-in, SOR spectral radius, mesh sensitivity, or energy balance..."
                className="flex-1 bg-graphite-900 border border-graphite-700 rounded-lg px-4 py-2.5 text-xs text-white focus:outline-none focus:border-cyan-500 font-mono"
              />
              <button
                onClick={() => handleAskAI()}
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
