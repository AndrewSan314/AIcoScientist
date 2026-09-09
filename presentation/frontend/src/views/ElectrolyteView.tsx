import React from 'react';
import { SnapshotData } from '../types/mission_control';
import { ModeBadge } from '../components/ModeBadge';
import { 
  Zap, 
  Filter, 
  Database, 
  Cpu, 
  AlertTriangle, 
  TrendingUp, 
  Sliders, 
  CheckCircle2, 
  Clock,
  Info
} from 'lucide-react';

interface ElectrolyteViewProps {
  data: SnapshotData;
}

export const ElectrolyteView: React.FC<ElectrolyteViewProps> = ({ data }) => {
  const screen = data.electrolyte_screening || {};

  const funnelSteps = [
    {
      title: 'Virtual Formulation Universe',
      count: '333,333 Candidates',
      sub: 'Combinatorial space of salt, non-aqueous solvent blends, and functional additives',
      badge: 'Scientific Pool',
    },
    {
      title: 'LiFSI Scientific Focus Slice',
      count: '333,333 Candidates',
      sub: 'Concentrated lithium bis(fluorosulfonyl)imide formulation space',
      badge: 'Chemical Filter',
    },
    {
      title: 'Stage-1 Historically Informed Screen',
      count: 'Screened in 2.535s',
      sub: 'Ridge + RF ensemble (40% discovery) + distance (30% explore) + farthest-point diversity (20%) + random (10%)',
      badge: 'Multi-Objective Ensemble',
    },
    {
      title: 'Bounded Working Candidate Set',
      count: 'WS = 200 Candidates',
      sub: '100% latent maximum recovered (0.7886 latent cap preserved with 0.000 screening gap)',
      badge: 'Active Decision Space',
    },
    {
      title: 'Closed-Loop Scientific Policy',
      count: '15 Iterations',
      sub: 'Sequential candidate recommendation with firewalled surrogate oracle measurements',
      badge: 'Active Inference',
    },
  ];

  const simulationRows = [
    {
      policy: 'BOTORCH_EI_DIRECT',
      name: 'BoTorch Expected Improvement',
      latentCap: '0.7628 ± 0.036',
      latentRegret: '0.0257',
      cumHig: '1.520 nats',
      entropyRed: '0.564 nats',
      role: 'Greedy property maximizer. Lowest regret on single target.',
      highlight: false,
    },
    {
      policy: 'PURE_FALSIFICATION',
      name: 'Pure Falsification',
      latentCap: '0.7456 ± 0.056',
      latentRegret: '0.0430',
      cumHig: '1.843 nats',
      entropyRed: '0.842 nats',
      role: 'Maximizes information acquisition and hypothesis refutation.',
      highlight: false,
    },
    {
      policy: 'HYBRID_DEFAULT',
      name: 'Hybrid Scientific Policy',
      latentCap: '0.7097 ± 0.002',
      latentRegret: '0.0788',
      cumHig: '1.587 nats',
      entropyRed: '0.995 nats',
      role: 'Balanced policy: Highest overall scientific entropy reduction.',
      highlight: true,
    },
    {
      policy: 'BOTORCH_GPUCB_DIRECT',
      name: 'BoTorch GP-UCB',
      latentCap: '0.7157 ± 0.002',
      latentRegret: '0.0729',
      cumHig: '0.729 nats',
      entropyRed: '0.457 nats',
      role: 'Upper confidence bound baseline with lower information yield.',
      highlight: false,
    },
    {
      policy: 'RANDOM',
      name: 'Random Action Baseline',
      latentCap: '0.7223 ± 0.045',
      latentRegret: '0.0663',
      cumHig: '0.729 nats',
      entropyRed: '0.784 nats',
      role: 'Stochastic uniform control baseline.',
      highlight: false,
    },
  ];

  return (
    <div className="space-y-8 pb-16 animate-fadeIn">
      {/* Top Banner */}
      <div className="border-b border-slate-200 pb-6 pt-2">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
          <div>
            <div className="flex items-center gap-2 mb-1">
              <h1 className="text-2xl font-bold text-slate-900 tracking-tight">Electrolyte Discovery Scale</h1>
              <ModeBadge mode="CONTROLLED_SYNTHETIC" size="sm" />
            </div>
            <p className="text-xs text-slate-500">
              Screening 333,333 virtual LiFSI formulations with bounded Stage-1 down-selection & closed-loop simulation
            </p>
          </div>

          <div className="flex items-center gap-2 text-xs font-mono bg-slate-100 px-3 py-1.5 rounded-lg border border-slate-200">
            <span className="text-slate-500">Virtual Library:</span>
            <span className="font-bold text-slate-900">333,333 Formulations</span>
          </div>
        </div>
      </div>

      {/* 5-Stage Screening Funnel */}
      <section className="sci-card p-6">
        <div className="mb-6">
          <h2 className="text-sm font-bold uppercase tracking-wider text-slate-900">Large-Scale Candidate Screening Funnel</h2>
          <p className="text-xs text-slate-500">
            How AIcoScientist handles massive candidate universes without sacrificing Bayesian decision fidelity
          </p>
        </div>

        <div className="relative border-l-2 border-emerald-600/60 ml-4 pl-6 space-y-6">
          {funnelSteps.map((step, idx) => (
            <div key={idx} className="relative">
              {/* Dot */}
              <div className="absolute -left-[31px] top-1.5 w-4 h-4 rounded-full bg-white border-2 border-emerald-600 flex items-center justify-center">
                <div className="w-1.5 h-1.5 rounded-full bg-emerald-600" />
              </div>

              <div className="bg-slate-50/70 p-4 rounded-lg border border-slate-200 hover:bg-white hover:border-slate-300 transition">
                <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 mb-1">
                  <div className="font-bold text-sm text-slate-900">{step.title}</div>
                  <div className="flex items-center gap-2">
                    <span className="font-mono text-xs font-bold text-emerald-800 bg-emerald-100 px-2.5 py-0.5 rounded">
                      {step.count}
                    </span>
                    <span className="text-2xs font-mono uppercase bg-slate-200 text-slate-700 px-2 py-0.5 rounded">
                      {step.badge}
                    </span>
                  </div>
                </div>
                <div className="text-xs text-slate-600">{step.sub}</div>
              </div>
            </div>
          ))}
        </div>
      </section>

      {/* Surrogate Closed-Loop Benchmark */}
      <section className="sci-card p-6">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 mb-6 pb-4 border-b border-slate-200">
          <div>
            <h2 className="text-base font-bold text-slate-900">Closed-Loop Policy Benchmark (ExtraTrees Surrogate Oracle)</h2>
            <p className="text-xs text-slate-500">
              15-step closed-loop evaluation over Seeds 42, 101, and 2024 within the WS=200 candidate working set
            </p>
          </div>
          <span className="text-xs font-mono px-2.5 py-1 rounded bg-slate-100 text-slate-800 font-semibold">
            In-Silico Surrogate Benchmark
          </span>
        </div>

        <div className="overflow-x-auto">
          <table className="sci-table text-xs font-mono">
            <thead>
              <tr>
                <th>Policy</th>
                <th>Best Latent Max $f(x)$</th>
                <th>Latent Regret</th>
                <th>Cumulative HIG</th>
                <th>Entropy Reduction</th>
                <th>Scientific Trade-off Profile</th>
              </tr>
            </thead>
            <tbody>
              {simulationRows.map((row) => (
                <tr key={row.policy} className={row.highlight ? 'bg-violet-50/50 font-semibold' : ''}>
                  <td className="font-bold text-slate-900 font-sans">{row.name}</td>
                  <td>{row.latentCap}</td>
                  <td className={row.policy.includes('EI') ? 'text-emerald-700 font-bold' : ''}>{row.latentRegret}</td>
                  <td>{row.cumHig}</td>
                  <td className={row.highlight ? 'text-violet-800 font-bold' : ''}>{row.entropyRed}</td>
                  <td className="font-sans text-2xs text-slate-600">{row.role}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        {/* Honest Scientific Trade-off Callout */}
        <div className="mt-6 p-4 rounded-lg bg-amber-50 border border-amber-200 text-xs text-amber-950 space-y-2">
          <div className="flex items-center gap-2 font-bold text-amber-900">
            <AlertTriangle className="w-4 h-4 text-amber-700 shrink-0" />
            <span>Methodological Rigor & Negative Result Reporting</span>
          </div>
          <p className="leading-relaxed">
            <strong>Honest Comparison:</strong> We do not claim that Hybrid universally outperforms BoTorch EI on pure property optimization in this surrogate. When the single objective is scalar exploitation, BoTorch EI achieves the lowest latent regret (0.0257 vs 0.0788). However, Hybrid achieves the highest overall scientific entropy reduction (0.995 nats vs 0.564 nats), successfully balancing epistemic hypothesis discrimination with property utility.
          </p>
          <div className="text-2xs text-amber-800 font-mono pt-1">
            Disclaimer: The ExtraTrees surrogate is a frozen univariate target model; it does not represent independent physical wet-lab ground truth or causal process coupling.
          </div>
        </div>
      </section>
    </div>
  );
};
