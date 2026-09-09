import React, { useState } from 'react';
import {
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  Cell
} from 'recharts';
import { BenchmarkSummaryData } from '../../types/mission_control';
import { Layers } from 'lucide-react';

interface Props {
  benchmarks: BenchmarkSummaryData;
}

type BenchmarkMetricKey =
  | 'recovery_rate_MAP'
  | 'mean_final_true_hypothesis_probability'
  | 'mean_entropy_reduction'
  | 'mean_measurement_cost'
  | 'steps_to_posterior_gt_0.8';

export const PolicyTrajectoryChart: React.FC<Props> = ({ benchmarks }) => {
  const worlds = benchmarks.worlds || Object.keys(benchmarks.summary_by_world_policy || {});
  const [selectedWorld, setSelectedWorld] = useState<string>(
    worlds.find((w) => w.includes('CLEAN')) || worlds[0] || 'CLEAN_WORLD_H1_PHASE_PURITY'
  );
  const [selectedMetric, setSelectedMetric] = useState<BenchmarkMetricKey>('recovery_rate_MAP');

  const metricConfig: Record<BenchmarkMetricKey, { label: string; unit: string; maxVal?: number }> = {
    recovery_rate_MAP: { label: 'MAP Recovery Rate', unit: '%', maxVal: 1.0 },
    mean_final_true_hypothesis_probability: { label: 'Final True Hypothesis Prob', unit: '%', maxVal: 1.0 },
    mean_entropy_reduction: { label: 'Mean Entropy Reduction', unit: ' nats' },
    mean_measurement_cost: { label: 'Mean Measurement Cost', unit: ' cost units' },
    'steps_to_posterior_gt_0.8': { label: 'Steps to Posterior > 0.8', unit: ' steps' }
  };

  const worldData = benchmarks.summary_by_world_policy?.[selectedWorld] || {};
  const policies = benchmarks.policies || Object.keys(worldData);

  const chartData = policies.map((p) => {
    const stats = worldData[p] || {};
    let val = 0;
    if (selectedMetric === 'steps_to_posterior_gt_0.8') {
      val = stats['steps_to_posterior_gt_0.8'] ?? stats['mean_steps_to_posterior_gt_0.8'] ?? 0;
    } else {
      val = stats[selectedMetric] ?? 0;
    }

    return {
      policy: p.replace('_ACTION', '').replace('_DEFAULT', ''),
      fullName: p,
      value: parseFloat(Number(val).toFixed(4)),
      isHybrid: p.includes('HYBRID'),
      isHig: p.includes('PURE_HIG')
    };
  });

  const isPercentage =
    selectedMetric === 'recovery_rate_MAP' ||
    selectedMetric === 'mean_final_true_hypothesis_probability';

  return (
    <div className="w-full flex flex-col">
      {/* Controls row */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 mb-4">
        <div className="flex items-center gap-2">
          <Layers className="w-4 h-4 text-emerald-600" />
          <span className="text-xs font-mono font-semibold text-slate-700">World:</span>
          <select
            value={selectedWorld}
            onChange={(e) => setSelectedWorld(e.target.value)}
            className="text-xs font-mono bg-white border border-slate-200 rounded-md px-3 py-1.5 text-slate-800 shadow-2xs cursor-pointer focus:outline-emerald-600 max-w-[280px]"
          >
            {worlds.map((w) => (
              <option key={w} value={w}>
                {w.replace(/_/g, ' ')}
              </option>
            ))}
          </select>
        </div>

        {/* Metric Switcher */}
        <div className="flex flex-wrap items-center gap-1 bg-slate-100 p-1 rounded-lg border border-slate-200 text-2xs font-mono">
          {(Object.keys(metricConfig) as BenchmarkMetricKey[]).map((k) => (
            <button
              key={k}
              onClick={() => setSelectedMetric(k)}
              className={`px-2 py-1 rounded-md transition-all cursor-pointer ${
                selectedMetric === k
                  ? 'bg-white text-emerald-800 font-bold shadow-2xs'
                  : 'text-slate-600 hover:text-slate-900'
              }`}
            >
              {metricConfig[k].label}
            </button>
          ))}
        </div>
      </div>

      {/* Chart */}
      <div className="w-full h-64">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={chartData} margin={{ top: 12, right: 24, left: 0, bottom: 24 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" vertical={false} />
            <XAxis
              dataKey="policy"
              tick={{ fill: '#475569', fontSize: 10, fontFamily: 'monospace' }}
              tickLine={false}
              axisLine={{ stroke: '#cbd5e1' }}
              interval={0}
            />
            <YAxis
              domain={isPercentage ? [0, 1] : ['auto', 'auto']}
              tickFormatter={(v) => (isPercentage ? `${(v * 100).toFixed(0)}%` : Number(v).toFixed(2))}
              tick={{ fill: '#64748b', fontSize: 10 }}
              tickLine={false}
              axisLine={{ stroke: '#cbd5e1' }}
              label={{
                value: metricConfig[selectedMetric].label,
                angle: -90,
                position: 'insideLeft',
                offset: 15,
                fill: '#64748b',
                fontSize: 10
              }}
            />
            <Tooltip
              contentStyle={{
                backgroundColor: '#ffffff',
                borderColor: '#e2e8f0',
                borderRadius: '8px',
                boxShadow: '0 4px 6px -1px rgb(0 0 0 / 0.1)',
                fontSize: '11px'
              }}
              formatter={(value: any, name: any, item: any) => [
                isPercentage
                  ? `${(Number(value) * 100).toFixed(1)}%`
                  : `${Number(value).toFixed(3)}${metricConfig[selectedMetric].unit}`,
                item.payload.fullName
              ]}
            />
            <Bar dataKey="value" radius={[4, 4, 0, 0]}>
              {chartData.map((entry, index) => {
                let fill = '#94a3b8'; // default slate
                if (entry.isHybrid) fill = '#059669'; // emerald for HYBRID
                else if (entry.isHig) fill = '#7c3aed'; // violet for PURE_HIG
                else if (entry.fullName.includes('DISCOVERY')) fill = '#d97706'; // amber for DISCOVERY
                return <Cell key={`cell-${index}`} fill={fill} />;
              })}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>

      <div className="mt-2 pt-2 border-t border-slate-100 flex items-center justify-between text-2xs font-mono text-slate-500">
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-1.5">
            <span className="w-2.5 h-2.5 rounded bg-emerald-600 inline-block" />
            <span className="font-bold text-emerald-800">HYBRID Policy (Recommended)</span>
          </div>
          <div className="flex items-center gap-1.5">
            <span className="w-2.5 h-2.5 rounded bg-violet-600 inline-block" />
            <span>PURE HIG Policy</span>
          </div>
          <div className="flex items-center gap-1.5">
            <span className="w-2.5 h-2.5 rounded bg-amber-600 inline-block" />
            <span>Discovery Only Policy</span>
          </div>
        </div>
        <div className="text-slate-400">
          Evaluated across 180 controlled closed-loop trajectories (5 seeds per policy)
        </div>
      </div>
    </div>
  );
};
