import React from 'react';
import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ReferenceLine,
} from 'recharts';
import { Zap, AlertCircle } from 'lucide-react';

interface Props {
  simulationData: any;
  selectedPolicy?: string;
  selectedSeed?: number;
}

const COLORS = ['#2563eb', '#B91C1C', '#0d9488', '#7c3aed', '#64748b', '#d97706'];

export const ElectrolyteOptimizationChart: React.FC<Props> = ({ simulationData, selectedPolicy, selectedSeed }) => {
  const detailedRuns = simulationData?.detailed_policy_seed_runs || {};
  const policies = Object.keys(detailedRuns);
  const series = policies.map((policy) => ({
    policy,
    run: (detailedRuns[policy] || []).find((candidate: any) => selectedSeed !== undefined && candidate.seed === selectedSeed),
  })).filter(({ run }) => run?.best_latent_curve?.length);
  const maxLen = Math.max(...series.map(({ run }) => run.best_latent_curve.length), 0);
  const chartData = Array.from({ length: maxLen }, (_, index) => Object.fromEntries([
    ['iteration', index + 1],
    ['query', `Q${index + 1}`],
    ...series.map(({ policy, run }) => [policy, run.best_latent_curve[index] ?? null]),
  ]));
  const diagnostics = simulationData?.screeningDiagnostics || simulationData;
  const trial = diagnostics?.working_set_trials?.[String(diagnostics?.chosen_default_working_set_size || 200)] || {};
  const latentMax = simulationData?.working_set_latent_max ?? diagnostics?.full_search_space_latent_max;
  const modelFamily = simulationData?.surrogate_model_family || 'Source model family unavailable';
  const physicalSynthesis = simulationData?.physical_synthesis;

  return (
    <div className="w-full flex flex-col">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 mb-3">
        <div>
          <div className="flex items-center gap-2">
            <Zap className="w-4 h-4 text-red-600" />
            <h4 className="text-sm font-bold text-slate-900 tracking-tight">Recorded Surrogate Trajectory</h4>
            <span className="text-2xs font-mono px-2 py-0.5 rounded-full bg-slate-100 text-slate-700 border border-slate-200">
              {selectedPolicy === 'HYBRID_DEFAULT' ? 'Hybrid Policy' : selectedPolicy === 'PURE_HIG' ? 'Pure HIG Policy' : selectedPolicy === 'DISCOVERY_ONLY' ? 'Discovery-Only Policy' : selectedPolicy || 'Acquisition Policy'}
            </span>
          </div>
          <p className="text-xs text-slate-500">Best selected latent value across the source query sequence</p>
        </div>
        <div className="text-2xs font-mono text-slate-500 bg-slate-100 px-2.5 py-1 rounded-md border border-slate-200">
          Latent maximum: <strong className="text-red-700 font-bold">{typeof latentMax === 'number' ? latentMax.toFixed(4) : 'Not recorded'}</strong>
        </div>
      </div>

      <div className="mb-3 px-3 py-2 bg-slate-50 border border-slate-200 rounded-lg text-xs text-slate-700 flex items-start gap-2">
        <AlertCircle className="w-4 h-4 text-slate-500 shrink-0 mt-0.5" />
        <div className="text-2xs">
          <strong>Evidence boundary:</strong>{' '}
          {physicalSynthesis === false ? 'All values shown are in-silico surrogate outputs; no physical synthesis or cycling is represented.' : 'Physical execution status is not recorded in this artifact.'}
        </div>
      </div>

      <div className="w-full h-72">
        {series.length ? (
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={chartData} margin={{ top: 16, right: 30, left: 10, bottom: 24 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#F1F5F9" />
              <XAxis dataKey="query" tick={{ fill: '#334155', fontSize: 12 }} tickLine={false} axisLine={{ stroke: '#D9DFDB' }} />
              <YAxis domain={['auto', 'auto']} tick={{ fill: '#475569', fontSize: 12 }} tickLine={false} axisLine={{ stroke: '#D9DFDB' }} />
              <Tooltip
                contentStyle={{ backgroundColor: '#ffffff', borderColor: '#D9DFDB', borderRadius: '8px', fontSize: '12px' }}
                formatter={(value: any, name: any) => [typeof value === 'number' ? value.toFixed(4) : 'Not recorded', name]}
              />
              <Legend formatter={(value) => <span className="text-2xs font-medium text-slate-700">{value}</span>} />
              {typeof latentMax === 'number' && <ReferenceLine y={latentMax} stroke="#B91C1C" strokeDasharray="4 4" label={{ value: `Latent max: ${latentMax.toFixed(4)}`, fill: '#B91C1C', fontSize: 10 }} />}
              {series.map(({ policy }, index) => (
                <Line key={policy} type="stepAfter" dataKey={policy} name={policy} stroke={COLORS[index % COLORS.length]} strokeWidth={policy === selectedPolicy ? 3 : 1.8} dot={policy === selectedPolicy ? { r: 3 } : false} />
              ))}
            </LineChart>
          </ResponsiveContainer>
        ) : <div className="h-full flex items-center justify-center text-sm text-slate-500">No trajectory is recorded for this policy.</div>}
      </div>

      <div className="mt-2 pt-2 border-t border-slate-100 flex flex-col sm:flex-row sm:items-center sm:justify-between gap-1 text-2xs font-mono text-slate-500">
        <div>
          Screening stage: {diagnostics?.search_space_size ?? 'Not recorded'} candidates → {trial?.working_set_size ?? diagnostics?.chosen_default_working_set_size ?? 'Not recorded'} working set ({trial?.screening_time_sec ?? 'Not recorded'} sec; latent gap {trial?.screening_latent_gap ?? 'Not recorded'})
        </div>
        <div className="text-slate-400">Surrogate model: {modelFamily}</div>
      </div>
    </div>
  );
};
