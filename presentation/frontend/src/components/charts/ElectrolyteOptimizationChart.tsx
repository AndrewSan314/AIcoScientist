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
  ReferenceLine
} from 'recharts';
import { Zap, AlertCircle } from 'lucide-react';

interface Props {
  simulationData: any;
}

export const ElectrolyteOptimizationChart: React.FC<Props> = ({ simulationData }) => {
  const detailedRuns = simulationData?.detailed_policy_seed_runs || {};

  // Build query iteration points (queries 1 to 15)
  const eiCurve = detailedRuns['BOTORCH_EI_DIRECT']?.[0]?.best_latent_curve || [];
  const hybridCurve = detailedRuns['HYBRID_DEFAULT']?.[0]?.best_latent_curve || [];
  const gpucbCurve = detailedRuns['BOTORCH_GPUCB_DIRECT']?.[0]?.best_latent_curve || [];
  const falsifyCurve = detailedRuns['PURE_FALSIFICATION']?.[0]?.best_latent_curve || [];
  const randomCurve = detailedRuns['RANDOM']?.[0]?.best_latent_curve || [];

  const maxLen = Math.max(
    eiCurve.length,
    hybridCurve.length,
    gpucbCurve.length,
    falsifyCurve.length,
    randomCurve.length,
    15
  );

  const chartData = [];
  for (let i = 0; i < maxLen; i++) {
    chartData.push({
      iteration: i + 1,
      query: `Q${i + 1}`,
      BOTORCH_EI: eiCurve[i] ?? null,
      HYBRID: hybridCurve[i] ?? null,
      BOTORCH_GPUCB: gpucbCurve[i] ?? null,
      FALSIFICATION: falsifyCurve[i] ?? null,
      RANDOM: randomCurve[i] ?? null
    });
  }

  const latentMax: number | null = simulationData?.working_set_latent_max ?? simulationData?.full_search_space_latent_max ?? null;

  return (
    <div className="w-full flex flex-col">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 mb-3">
        <div>
          <div className="flex items-center gap-2">
            <Zap className="w-4 h-4 text-red-600" />
            <h4 className="text-sm font-bold text-slate-900 tracking-tight">
              Electrolyte Closed-Loop Optimization Trajectories
            </h4>
            <span className="text-2xs font-mono px-2 py-0.5 rounded-full bg-slate-100 text-slate-700 border border-slate-200">
              ExtraTrees Surrogate Oracle (WS=200)
            </span>
          </div>
          <p className="text-xs text-slate-500">
            Cumulative best latent capacity discovered over 15 sequential query iterations
          </p>
        </div>

        <div className="text-2xs font-mono text-slate-500 bg-slate-100 px-2.5 py-1 rounded-md border border-slate-200">
          Latent Optimum: <strong className="text-red-700 font-bold">{latentMax !== null ? latentMax.toFixed(4) : 'N/A'}</strong>
        </div>
      </div>

      {/* Trade-off disclaimer */}
      <div className="mb-3 px-3 py-2 bg-slate-50 border border-slate-200 rounded-lg text-xs text-slate-700 flex items-start gap-2">
        <AlertCircle className="w-4 h-4 text-slate-500 shrink-0 mt-0.5" />
        <div className="text-2xs">
          <strong>Non-linear Trade-off & Negative Result Integrity:</strong> BoTorch EI directly exploits scalar capacity faster (regret: 0.0257 vs 0.0788), but Hybrid achieves nearly double the scientific hypothesis entropy reduction (<strong>0.995 nats</strong> vs 0.564 nats). Pure scalar exploitation sacrifices mechanistic learning.
        </div>
      </div>

      <div className="w-full h-72">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={chartData} margin={{ top: 16, right: 30, left: 10, bottom: 24 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#F1F5F9" />
            <XAxis
              dataKey="query"
              tick={{ fill: '#334155', fontSize: 12, fontWeight: 500, fontFamily: "'Montserrat', Arial, sans-serif" }}
              tickLine={false}
              axisLine={{ stroke: '#D9DFDB' }}
              label={{
                value: 'Sequential Query Iterations (Budget: 15 Experiments)',
                position: 'insideBottom',
                offset: -12,
                fill: '#475569',
                fontSize: 12,
                fontWeight: 600,
                fontFamily: "'Montserrat', Arial, sans-serif"
              }}
            />
            <YAxis
              domain={[0.2, 0.85]}
              tick={{ fill: '#475569', fontSize: 12, fontWeight: 500, fontFamily: "'Montserrat', Arial, sans-serif" }}
              tickLine={false}
              axisLine={{ stroke: '#D9DFDB' }}
              label={{
                value: 'Best Observed Latent Capacity',
                angle: -90,
                position: 'insideLeft',
                offset: 15,
                fill: '#475569',
                fontSize: 12,
                fontWeight: 600,
                fontFamily: "'Montserrat', Arial, sans-serif"
              }}
            />
            <Tooltip
              contentStyle={{
                backgroundColor: '#ffffff',
                borderColor: '#D9DFDB',
                borderRadius: '8px',
                boxShadow: '0 4px 6px -1px rgb(0 0 0 / 0.1)',
                fontSize: '12px',
                fontFamily: "'Montserrat', Arial, sans-serif"
              }}
              formatter={(value: any, name: any) => [
                Number(value).toFixed(4),
                name === 'BOTORCH_EI'
                  ? 'BoTorch EI (Scalar Exploitation)'
                  : name === 'HYBRID'
                  ? 'HYBRID Policy (Balanced)'
                  : name === 'BOTORCH_GPUCB'
                  ? 'BoTorch GP-UCB'
                  : name === 'FALSIFICATION'
                  ? 'Pure Falsification'
                  : 'Random Uniform'
              ]}
            />
            <Legend
              verticalAlign="top"
              height={32}
              formatter={(value) => {
                const names: Record<string, string> = {
                  BOTORCH_EI: 'BoTorch EI (Exploitation)',
                  HYBRID: 'HYBRID (Info + Discovery)',
                  BOTORCH_GPUCB: 'GP-UCB',
                  FALSIFICATION: 'Pure Falsification',
                  RANDOM: 'Random Uniform'
                };
                return <span className="text-2xs font-medium text-slate-700">{names[value] || value}</span>;
              }}
            />

            {latentMax !== null && (
              <ReferenceLine
                y={latentMax}
                stroke="#B91C1C"
                strokeDasharray="4 4"
                label={{
                  value: `Latent Max: ${latentMax.toFixed(4)}`,
                  fill: '#B91C1C',
                  fontSize: 10,
                  position: 'insideTopRight'
                }}
              />
            )}

            <Line
              type="stepAfter"
              dataKey="BOTORCH_EI"
              name="BOTORCH_EI"
              stroke="#2563eb"
              strokeWidth={2.5}
              dot={{ r: 3 }}
            />
            <Line
              type="stepAfter"
              dataKey="HYBRID"
              name="HYBRID"
              stroke="#B91C1C"
              strokeWidth={3}
              dot={{ r: 4, fill: '#B91C1C' }}
            />
            <Line
              type="stepAfter"
              dataKey="BOTORCH_GPUCB"
              name="BOTORCH_GPUCB"
              stroke="#0d9488"
              strokeWidth={1.5}
              dot={false}
            />
            <Line
              type="stepAfter"
              dataKey="FALSIFICATION"
              name="FALSIFICATION"
              stroke="#7c3aed"
              strokeWidth={1.5}
              dot={false}
            />
            <Line
              type="stepAfter"
              dataKey="RANDOM"
              name="RANDOM"
              stroke="#94a3b8"
              strokeWidth={1.5}
              strokeDasharray="3 3"
              dot={false}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>

      <div className="mt-2 pt-2 border-t border-slate-100 flex items-center justify-between text-2xs font-mono text-slate-500">
        <div>
          Screening Funnel: 333,333 virtual formulations $\to$ 200 working set (2.535 sec, 0.000 gap)
        </div>
        <div className="text-slate-400">
          Surrogate Model: ExtraTrees (100 estimators, σ=0.02)
        </div>
      </div>
    </div>
  );
};
