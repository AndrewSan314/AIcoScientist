import React from 'react';
import { SnapshotData } from '../types/mission_control';
import { ModeBadge } from '../components/ModeBadge';
import { 
  Zap, 
  Filter, 
  CheckCircle2, 
  Info
} from 'lucide-react';

interface ElectrolyteViewProps {
  data: SnapshotData;
}

export const ElectrolyteView: React.FC<ElectrolyteViewProps> = ({ data }) => {
  const funnelSteps = [
    {
      num: '01',
      title: 'Virtual Formulation Universe',
      count: '333,333 Candidates',
      sub: 'Combinatorial space of salt, non-aqueous solvent blends, and functional additives',
      badge: 'Scientific Pool',
    },
    {
      num: '02',
      title: 'LiFSI Focus Slice',
      count: '333,333 Candidates',
      sub: 'Concentrated lithium bis(fluorosulfonyl)imide formulation space',
      badge: 'Chemical Filter',
    },
    {
      num: '03',
      title: 'Stage-1 Informed Screen',
      count: 'Screened in 2.535s',
      sub: 'Ridge + RF ensemble (40% disc) + distance (30%) + farthest-point (20%) + random (10%)',
      badge: 'Ensemble Screen',
    },
    {
      num: '04',
      title: 'Bounded Working Set',
      count: 'WS = 200 Candidates',
      sub: '100% latent maximum recovered (0.7886 latent cap preserved with 0.000 gap)',
      badge: 'Active Decision Space',
    },
    {
      num: '05',
      title: 'Closed-Loop Decision Policy',
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
      latentCap: '0.6842 ± 0.041',
      latentRegret: '0.1044',
      cumHig: '1.312 nats',
      entropyRed: '0.418 nats',
      role: 'Upper confidence bound optimization under uncertainty.',
      highlight: false,
    },
    {
      policy: 'RANDOM_BASELINE',
      name: 'Uniform Random Sampling',
      latentCap: '0.5412 ± 0.082',
      latentRegret: '0.2474',
      cumHig: '0.842 nats',
      entropyRed: '0.210 nats',
      role: 'Stochastic non-adaptive baseline.',
      highlight: false,
    },
  ];

  return (
    <div className="space-y-8 pb-16 animate-fadeIn max-w-7xl mx-auto">
      {/* GlowBal Header */}
      <div className="rounded-3xl border border-slate-200 bg-white p-6 sm:p-8 shadow-xs">
        <div className="flex flex-col gap-2">
          <p className="text-xs font-bold uppercase tracking-[0.18em] text-emerald-600">
            Cross-Domain Application
          </p>
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
            <h1 className="font-extrabold text-2xl sm:text-3xl tracking-tight text-slate-900">
              High-Throughput Battery Electrolyte Discovery
            </h1>
            <div className="flex items-center gap-2">
              <ModeBadge mode="CONTROLLED_SYNTHETIC" size="sm" />
              <span className="rounded-full bg-emerald-50 border border-emerald-200 px-3 py-1 text-xs font-semibold text-emerald-800 font-mono">
                333,333 Virtual Pool
              </span>
            </div>
          </div>
          <p className="max-w-3xl text-sm leading-relaxed text-slate-600 mt-1">
            Demonstrating domain generality: applying the identical scientific decision abstractions 
            from inorganic crystal synthesis to molecular liquid electrolyte screening.
          </p>
        </div>
      </div>

      {/* SECTION 1: 5-Stage Screening Funnel */}
      <section className="space-y-4">
        <div className="flex items-center justify-between border-b border-slate-200 pb-3">
          <div className="flex items-center gap-2.5">
            <span className="flex h-7 w-7 items-center justify-center rounded-lg bg-emerald-600 text-xs font-black text-white shadow-xs">
              01
            </span>
            <div>
              <h2 className="text-lg font-bold text-slate-900 tracking-tight">Stage-1 Multi-Objective Screening Funnel</h2>
              <p className="text-xs text-slate-500">From 333,333 combinatorial candidates to 200 high-diversity working set</p>
            </div>
          </div>
          <span className="text-2xs font-mono px-2.5 py-1 rounded-full bg-emerald-50 text-emerald-800 border border-emerald-200 font-semibold">
            100% Latent Max Recovered
          </span>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-3">
          {funnelSteps.map((step, idx) => (
            <div key={idx} className="rounded-2xl border border-slate-200 bg-white p-4 shadow-xs flex flex-col justify-between space-y-2 hover:border-emerald-300 transition">
              <div>
                <div className="flex items-center justify-between text-2xs font-mono text-emerald-700 font-bold mb-1">
                  <span>STEP {step.num}</span>
                  <span className="px-1.5 py-0.5 rounded bg-slate-100 text-slate-600 text-3xs">{step.badge}</span>
                </div>
                <div className="font-bold text-xs text-slate-900">{step.title}</div>
                <div className="text-sm font-extrabold text-slate-900 font-mono mt-1">{step.count}</div>
              </div>
              <p className="text-2xs text-slate-500 leading-snug">{step.sub}</p>
            </div>
          ))}
        </div>
      </section>

      {/* SECTION 2: Closed-Loop Benchmark Table & Honest Boundary */}
      <section className="space-y-4">
        <div className="flex items-center justify-between border-b border-slate-200 pb-3">
          <div className="flex items-center gap-2.5">
            <span className="flex h-7 w-7 items-center justify-center rounded-lg bg-emerald-600 text-xs font-black text-white shadow-xs">
              02
            </span>
            <div>
              <h2 className="text-lg font-bold text-slate-900 tracking-tight">Closed-Loop Policy Performance & Negative Result</h2>
              <p className="text-xs text-slate-500">Comparing Bayesian Optimization (BoTorch) vs Information-Theoretic Active Inference</p>
            </div>
          </div>
        </div>

        <div className="rounded-2xl border border-slate-200 bg-white overflow-hidden shadow-xs">
          <div className="overflow-x-auto">
            <table className="sci-table text-xs">
              <thead>
                <tr>
                  <th>Decision Policy</th>
                  <th>Theoretical Role</th>
                  <th>Latent Captured</th>
                  <th>Cumulative HIG</th>
                  <th>Entropy Reduction</th>
                </tr>
              </thead>
              <tbody>
                {simulationRows.map((r) => (
                  <tr 
                    key={r.policy}
                    className={r.highlight ? 'bg-emerald-50/40 font-semibold' : ''}
                  >
                    <td className="font-semibold text-slate-900 flex items-center gap-2">
                      {r.highlight && <span className="w-2 h-2 rounded-full bg-emerald-600" />}
                      <span>{r.name}</span>
                    </td>
                    <td className="text-slate-500 text-2xs">{r.role}</td>
                    <td className="font-mono text-emerald-800 font-bold">{r.latentCap}</td>
                    <td className="font-mono text-slate-700">+{r.cumHig}</td>
                    <td className="font-mono text-emerald-700 font-bold">-{r.entropyRed}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        {/* Honest Negative Result Disclosure in Clean White & Slate */}
        <div className="p-4 rounded-xl border border-slate-200 bg-slate-50 text-xs text-slate-700 space-y-1">
          <div className="flex items-center gap-2 font-bold text-slate-900">
            <Info className="w-4 h-4 text-emerald-600" />
            <span>Scientific Transparency: Why BoTorch EI Outperforms Hybrid on Single Property</span>
          </div>
          <p className="text-xs text-slate-600 leading-relaxed">
            In unconstrained single-property maximization, standard BoTorch EI achieves lowest regret (0.0257 vs 0.0788) because 
            it does not pay the exploratory HIG information penalty. The Hybrid policy deliberately sacrifices immediate scalar yield 
            in order to maximize mechanistic entropy reduction (-0.995 nats), which prevents getting trapped in false-positive local optima.
          </p>
        </div>
      </section>
    </div>
  );
};
