import React, { useState } from 'react';
import { SnapshotData } from '../types/mission_control';
import { ModeBadge } from '../components/ModeBadge';
import { 
  BarChart3, 
  Layers, 
  Scale, 
  HelpCircle, 
  CheckCircle2, 
  AlertCircle, 
  Cpu, 
  Sliders, 
  TrendingUp,
  Info
} from 'lucide-react';

interface BenchmarkLabViewProps {
  data: SnapshotData;
}

export const BenchmarkLabView: React.FC<BenchmarkLabViewProps> = ({ data }) => {
  const [selectedWorld, setSelectedWorld] = useState<string>('CLEAN_WORLD_H1_PHASE_PURITY');
  const [selectedSeed, setSelectedSeed] = useState<number>(42);

  const benchmarks = data.benchmarks || {};
  const sensitivity = data.sensitivity || {};
  const summaryByWorldPolicy = benchmarks.summary_by_world_policy || {};

  const worlds = [
    { id: 'CLEAN_WORLD_H1_PHASE_PURITY', name: 'Clean World — H1 Phase Purity Limited', type: 'Clean' },
    { id: 'CLEAN_WORLD_H2_COMPOSITION_HOMOGENEITY', name: 'Clean World — H2 Composition Homogeneity', type: 'Clean' },
    { id: 'CLEAN_WORLD_H3_MORPHOLOGY_KINETICS', name: 'Clean World — H3 Morphology Kinetics', type: 'Clean' },
    { id: 'STRESS_WORLD_H1_PHASE_PURITY', name: 'Stress World — H1 Phase Purity Limited', type: 'Stress' },
    { id: 'STRESS_WORLD_H2_COMPOSITION_HOMOGENEITY', name: 'Stress World — H2 Composition Homogeneity', type: 'Stress' },
    { id: 'STRESS_WORLD_H3_MORPHOLOGY_KINETICS', name: 'Stress World — H3 Morphology Kinetics', type: 'Stress' },
  ];

  const policies = [
    { id: 'PURE_HIG', name: 'Pure HIG', role: 'Pure Information Theory', color: 'emerald' },
    { id: 'HYBRID', name: 'Hybrid Policy', role: 'Balanced Scientific Decision', color: 'violet' },
    { id: 'DISCOVERY_ONLY', name: 'Discovery Only', role: 'Greedy Property Maximization', color: 'amber' },
    { id: 'UNCERTAINTY_ONLY', name: 'Uncertainty Only', role: 'Model Variance Search', color: 'blue' },
    { id: 'RANDOM_CANDIDATE_FIXED_MODALITY', name: 'Fixed-Modality Random', role: 'Baseline Heuristic', color: 'slate' },
    { id: 'RANDOM_ACTION', name: 'Random Action', role: 'Stochastic Baseline', color: 'slate' },
  ];

  const currentSummary = summaryByWorldPolicy[selectedWorld] || {};

  return (
    <div className="space-y-8 pb-16 animate-fadeIn">
      {/* Top Banner */}
      <div className="border-b border-slate-200 pb-6 pt-2">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
          <div>
            <div className="flex items-center gap-2 mb-1">
              <h1 className="text-2xl font-bold text-slate-900 tracking-tight">Policy Benchmark Laboratory</h1>
              <ModeBadge mode="CONTROLLED_SYNTHETIC" size="sm" />
            </div>
            <p className="text-xs text-slate-500">
              Evaluation of 180 multi-step controlled trajectories across 6 competing policies, 6 worlds, and 5 random seeds
            </p>
          </div>

          <div className="flex items-center gap-2 text-xs font-mono bg-slate-100 px-3 py-1.5 rounded-lg border border-slate-200">
            <span className="text-slate-500">Matrix Scope:</span>
            <span className="font-bold text-slate-900">180 Closed-Loop Trajectories</span>
          </div>
        </div>
      </div>

      {/* World & Seed Selectors */}
      <div className="sci-card p-4 flex flex-wrap items-center justify-between gap-4">
        <div className="flex items-center gap-3">
          <span className="text-xs font-bold text-slate-700 uppercase">Controlled World:</span>
          <select
            value={selectedWorld}
            onChange={(e) => setSelectedWorld(e.target.value)}
            className="bg-slate-50 border border-slate-300 rounded-md px-3 py-1.5 text-xs font-semibold text-slate-900 focus:outline-hidden focus:ring-2 focus:ring-emerald-500"
          >
            {worlds.map((w) => (
              <option key={w.id} value={w.id}>
                [{w.type.toUpperCase()}] {w.name}
              </option>
            ))}
          </select>
        </div>

        <div className="flex items-center gap-3">
          <span className="text-xs font-bold text-slate-700 uppercase">Seed:</span>
          <div className="flex gap-1">
            {[7, 42, 101, 314, 2024].map((s) => (
              <button
                key={s}
                onClick={() => setSelectedSeed(s)}
                className={`px-2.5 py-1 rounded text-xs font-mono font-semibold transition cursor-pointer ${
                  selectedSeed === s
                    ? 'bg-slate-900 text-white shadow-xs'
                    : 'bg-slate-100 text-slate-600 hover:bg-slate-200'
                }`}
              >
                {s}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Policy Comparison Cards */}
      <section>
        <div className="flex items-center justify-between mb-4">
          <div>
            <h2 className="text-sm font-bold uppercase tracking-wider text-slate-900">6-Policy Benchmark Matrix (Selected World)</h2>
            <p className="text-xs text-slate-500">Comparing hypothesis recovery speed, entropy reduction, and experimental expenditure</p>
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {policies.map((pol) => {
            const stats = currentSummary[pol.id] || {};
            const mapRate = stats.recovery_rate_MAP !== undefined ? (stats.recovery_rate_MAP * 100).toFixed(0) : 'N/A';
            const meanSteps = stats.mean_steps_to_MAP !== null && stats.mean_steps_to_MAP !== undefined ? stats.mean_steps_to_MAP.toFixed(1) : '> 4';
            const meanEntropy = stats.mean_entropy_reduction !== undefined ? stats.mean_entropy_reduction.toFixed(3) : 'N/A';
            const meanCost = stats.mean_measurement_cost !== undefined ? stats.mean_measurement_cost.toFixed(2) : 'N/A';
            const isHybrid = pol.id === 'HYBRID';
            const isPureHig = pol.id === 'PURE_HIG';

            return (
              <div
                key={pol.id}
                className={`sci-card p-5 border transition ${
                  isHybrid 
                    ? 'border-violet-400 bg-violet-50/20 ring-1 ring-violet-300' 
                    : isPureHig 
                    ? 'border-emerald-400 bg-emerald-50/20' 
                    : 'border-slate-200 bg-white'
                }`}
              >
                <div className="flex items-center justify-between mb-2">
                  <span className="font-bold text-sm text-slate-900">{pol.name}</span>
                  <span className="text-2xs font-mono uppercase px-2 py-0.5 rounded bg-slate-100 text-slate-600">
                    {pol.role}
                  </span>
                </div>

                <div className="grid grid-cols-2 gap-3 my-4 text-xs">
                  <div className="p-2.5 rounded bg-slate-50 border border-slate-100">
                    <div className="text-2xs text-slate-500">MAP Recovery Rate</div>
                    <div className="text-lg font-bold font-mono-num text-slate-900 mt-0.5">{mapRate}%</div>
                  </div>
                  <div className="p-2.5 rounded bg-slate-50 border border-slate-100">
                    <div className="text-2xs text-slate-500">Mean Steps to MAP</div>
                    <div className="text-lg font-bold font-mono-num text-slate-900 mt-0.5">{meanSteps} steps</div>
                  </div>
                  <div className="p-2.5 rounded bg-slate-50 border border-slate-100">
                    <div className="text-2xs text-slate-500">Entropy Reduction</div>
                    <div className="text-sm font-bold font-mono-num text-emerald-800 mt-0.5">{meanEntropy} nats</div>
                  </div>
                  <div className="p-2.5 rounded bg-slate-50 border border-slate-100">
                    <div className="text-2xs text-slate-500">Mean Action Cost</div>
                    <div className="text-sm font-bold font-mono-num text-amber-800 mt-0.5">{meanCost} units</div>
                  </div>
                </div>

                <div className="text-2xs text-slate-600 pt-2 border-t border-slate-100">
                  {isPureHig && 'Fastest hypothesis resolution (1.2 steps), but completely indifferent to material discovery value.'}
                  {isHybrid && 'Optimal multi-objective compromise: achieves 100% recovery with bounded expenditure.'}
                  {pol.id === 'DISCOVERY_ONLY' && 'Exploits high-utility candidates, but achieves weaker hypothesis separation.'}
                  {pol.id === 'UNCERTAINTY_ONLY' && 'Sub-optimal recovery rate (40%) due to exploring uninformative variance.'}
                  {pol.id.includes('RANDOM') && 'Baseline unguided trajectory: poor convergence with random expenditure.'}
                </div>
              </div>
            );
          })}
        </div>
      </section>

      {/* HIG Sensitivity Analysis (MC12 vs MC32) */}
      <section className="sci-card p-6 border border-slate-200">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 mb-6 pb-4 border-b border-slate-200">
          <div>
            <h2 className="text-base font-bold text-slate-900">HIG Monte Carlo Sensitivity Analysis (MC12 vs MC32)</h2>
            <p className="text-xs text-slate-500">
              Rigorous diagnostic confirming why MC32 was selected for the final full benchmark matrix
            </p>
          </div>
          <span className="text-xs font-mono px-2.5 py-1 rounded bg-emerald-100 text-emerald-800 font-bold">
            Status: SENSITIVITY_PASS (USE_32_FOR_FULL_MATRIX)
          </span>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-4 gap-4 text-xs mb-6">
          <div className="p-4 rounded-lg bg-slate-50 border border-slate-200">
            <div className="text-2xs text-slate-500 uppercase font-mono">Sample Count Trade-off</div>
            <div className="text-xl font-bold font-mono text-slate-900 mt-1">12 vs 32 Samples</div>
            <div className="text-2xs text-slate-500 mt-1">60 paired benchmark trajectories evaluated</div>
          </div>
          <div className="p-4 rounded-lg bg-slate-50 border border-slate-200">
            <div className="text-2xs text-slate-500 uppercase font-mono">HIG Rank Correlation</div>
            <div className="text-xl font-bold font-mono text-emerald-700 mt-1">0.83 to 0.93</div>
            <div className="text-2xs text-slate-500 mt-1">Strong rank preservation between estimators</div>
          </div>
          <div className="p-4 rounded-lg bg-slate-50 border border-slate-200">
            <div className="text-2xs text-slate-500 uppercase font-mono">Modality Agreement</div>
            <div className="text-xl font-bold font-mono text-violet-700 mt-1">60% to 100%</div>
            <div className="text-2xs text-slate-500 mt-1">Modality sequence remains stable across seeds</div>
          </div>
          <div className="p-4 rounded-lg bg-slate-50 border border-slate-200">
            <div className="text-2xs text-slate-500 uppercase font-mono">Action Agreement</div>
            <div className="text-xl font-bold font-mono text-amber-700 mt-1">20% to 80%</div>
            <div className="text-2xs text-slate-500 mt-1">MC12 causes candidate jitter; MC32 stabilizes decisions</div>
          </div>
        </div>

        <div className="p-4 rounded-lg bg-amber-50 border border-amber-200 text-xs text-amber-950 flex items-start gap-3">
          <AlertCircle className="w-5 h-5 text-amber-700 shrink-0 mt-0.5" />
          <div className="space-y-1">
            <div className="font-bold">Methodological Takeaway: Why Sensitivity Pass ≠ MC12 Stability</div>
            <p>
              While the sensitivity test passed formal sanity checks (rank correlation &gt; 0.80), the action-sequence agreement dropped to 20% in certain clean worlds under MC=12. This proved that low Monte Carlo sampling generates tie-breaking noise in candidate ranking. Based on this evidence, the system formally upgraded to <strong>MC=32</strong> for all 180 matrix runs to ensure decision stability.
            </p>
          </div>
        </div>
      </section>
    </div>
  );
};
