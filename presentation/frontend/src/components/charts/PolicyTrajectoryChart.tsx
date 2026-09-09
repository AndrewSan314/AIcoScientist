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
import { Layers, BarChart3 } from 'lucide-react';

interface Props {
  benchmarks: BenchmarkSummaryData;
}

type BenchmarkMetricKey =
  | 'recovery_rate_MAP'
  | 'mean_final_true_hypothesis_probability'
  | 'mean_entropy_reduction'
  | 'mean_measurement_cost'
  | 'steps_to_posterior_gt_0.8';

const POLICY_DISPLAY_NAMES: Record<string, string> = {
  RANDOM: 'Random',
  RANDOM_CANDIDATE_FIXED_MODALITY: 'Fixed Modality',
  UNCERTAINTY_ONLY: 'Uncertainty',
  DISCOVERY_ONLY: 'Discovery Only',
  PURE_HIG: 'Pure HIG',
  HYBRID: 'Hybrid'
};

const WORLD_DISPLAY_NAMES: Record<string, string> = {
  CLEAN_WORLD_H1_PHASE_PURITY: 'Clean H₁: Phase Purity',
  CLEAN_WORLD_H2_COMPOSITION_HOMOGENEITY: 'Clean H₂: Composition Homogeneity',
  CLEAN_WORLD_H3_MORPHOLOGY_KINETICS: 'Clean H₃: Morphology Kinetics',
  STRESS_WORLD_H1_PHASE_PURITY: 'Stress H₁: Phase Purity',
  STRESS_WORLD_H2_COMPOSITION_HOMOGENEITY: 'Stress H₂: Composition Homogeneity',
  STRESS_WORLD_H3_MORPHOLOGY_KINETICS: 'Stress H₃: Morphology Kinetics'
};

export const PolicyTrajectoryChart: React.FC<Props> = ({ benchmarks }) => {
  const worlds = benchmarks.worlds || [];
  const [selectedWorld, setSelectedWorld] = useState<string>(
    worlds.find((w) => w.includes('CLEAN')) || worlds[0] || ''
  );
  const [selectedMetric, setSelectedMetric] = useState<BenchmarkMetricKey>('recovery_rate_MAP');

  const metricConfig: Record<BenchmarkMetricKey, { label: string; unit: string; maxVal?: number }> = {
    recovery_rate_MAP: { label: 'MAP Recovery Rate', unit: '%', maxVal: 1.0 },
    mean_final_true_hypothesis_probability: { label: 'Final True H Probability', unit: '%', maxVal: 1.0 },
    mean_entropy_reduction: { label: 'Mean Entropy Reduction', unit: ' nats' },
    mean_measurement_cost: { label: 'Mean Measurement Cost', unit: ' cost units' },
    'steps_to_posterior_gt_0.8': { label: 'Steps to Posterior > 0.8', unit: ' steps' }
  };

  const worldData = benchmarks.summary_by_world_policy?.[selectedWorld] || {};
  const policies = benchmarks.policies || Object.keys(worldData);

  const chartData = policies.map((p) => {
    const stats = worldData[p] || {};
    let val: number | null = null;
    if (selectedMetric === 'steps_to_posterior_gt_0.8') {
      val = stats['steps_to_posterior_gt_0.8'] ?? stats['mean_steps_to_posterior_gt_0.8'] ?? null;
    } else {
      val = stats[selectedMetric] ?? null;
    }

    const shortKey = p.replace('_ACTION', '').replace('_DEFAULT', '');

    return {
      policy: POLICY_DISPLAY_NAMES[shortKey] || shortKey,
      fullName: p,
      value: typeof val === 'number' ? parseFloat(val.toFixed(4)) : null,
      isHybrid: p.includes('HYBRID'),
      isHig: p.includes('PURE_HIG')
    };
  });

  const isPercentage =
    selectedMetric === 'recovery_rate_MAP' ||
    selectedMetric === 'mean_final_true_hypothesis_probability';

  return (
    <div className="w-full flex flex-col">
      {/* Controls row with two clean dropdowns */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 mb-4">
        {/* World selector */}
        <div className="flex items-center gap-2">
          <Layers className="w-4 h-4 text-[#DC2626] shrink-0" />
          <span className="text-xs font-semibold text-[#17201F]">World:</span>
          <select
            value={selectedWorld}
            onChange={(e) => setSelectedWorld(e.target.value)}
            className="text-xs bg-white border border-[#D9DFDB] rounded-lg px-3 py-1.5 text-[#17201F] shadow-2xs cursor-pointer focus:outline-none focus:ring-1 focus:ring-[#DC2626] font-medium"
          >
            {worlds.map((w) => (
              <option key={w} value={w}>
                {WORLD_DISPLAY_NAMES[w] || w.replace(/_/g, ' ')}
              </option>
            ))}
          </select>
        </div>

        {/* Metric selector */}
        <div className="flex items-center gap-2">
          <BarChart3 className="w-4 h-4 text-[#DC2626] shrink-0" />
          <span className="text-xs font-semibold text-[#17201F]">Metric:</span>
          <select
            value={selectedMetric}
            onChange={(e) => setSelectedMetric(e.target.value as BenchmarkMetricKey)}
            className="text-xs bg-white border border-[#D9DFDB] rounded-lg px-3 py-1.5 text-[#17201F] shadow-2xs cursor-pointer focus:outline-none focus:ring-1 focus:ring-[#DC2626] font-medium"
          >
            {(Object.keys(metricConfig) as BenchmarkMetricKey[]).map((k) => (
              <option key={k} value={k}>
                {metricConfig[k].label}
              </option>
            ))}
          </select>
        </div>
      </div>

      {/* Chart */}
      <div className="w-full h-72">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={chartData} margin={{ top: 16, right: 30, left: 10, bottom: 24 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#F1F5F9" vertical={false} />
            <XAxis
              dataKey="policy"
              tick={{ fill: '#334155', fontSize: 12, fontWeight: 600, fontFamily: "'Montserrat', Arial, sans-serif" }}
              tickLine={false}
              axisLine={{ stroke: '#D9DFDB' }}
              interval={0}
            />
            <YAxis
              domain={isPercentage ? [0, 1] : ['auto', 'auto']}
              tickFormatter={(v) => (isPercentage ? `${(v * 100).toFixed(0)}%` : Number(v).toFixed(2))}
              tick={{ fill: '#475569', fontSize: 12, fontWeight: 500, fontFamily: "'Montserrat', Arial, sans-serif" }}
              tickLine={false}
              axisLine={{ stroke: '#D9DFDB' }}
              label={{
                value: metricConfig[selectedMetric].label,
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
              formatter={(value: any, name: any, item: any) => [
                isPercentage
                  ? `${(Number(value) * 100).toFixed(1)}%`
                  : `${Number(value).toFixed(3)}${metricConfig[selectedMetric].unit}`,
                item.payload.fullName
              ]}
            />
            <Bar dataKey="value" radius={[4, 4, 0, 0]} maxBarSize={48}>
              {chartData.map((entry, index) => {
                let fill = '#94a3b8'; // default slate
                if (entry.isHybrid) fill = '#DC2626'; // emerald for HYBRID
                else if (entry.isHig) fill = '#7C3AED'; // violet for PURE_HIG
                else if (entry.fullName.includes('DISCOVERY')) fill = '#D97706'; // amber for DISCOVERY
                return <Cell key={`cell-${index}`} fill={fill} />;
              })}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>

      <div className="mt-2 pt-2.5 border-t border-[#D9DFDB] flex flex-col sm:flex-row sm:items-center justify-between gap-2 text-xs text-[#66706C]">
          <div className="flex items-center gap-4">
          <div className="flex items-center gap-1.5">
            <span className="w-2.5 h-2.5 rounded bg-[#B91C1C] inline-block" />
            <span className="font-bold text-[#991B1B]">HYBRID Policy</span>
          </div>
          <div className="flex items-center gap-1.5">
            <span className="w-2.5 h-2.5 rounded bg-[#7C3AED] inline-block" />
            <span className="font-medium text-[#17201F]">PURE HIG Policy</span>
          </div>
          <div className="flex items-center gap-1.5">
            <span className="w-2.5 h-2.5 rounded bg-[#D97706] inline-block" />
            <span className="font-medium text-[#17201F]">Discovery Only</span>
          </div>
        </div>
        <div className="text-[#8F9995]">
          Evaluated across {benchmarks.trajectory_count ?? 'N/A'} source trajectories ({benchmarks.seeds?.length ?? 'N/A'} seeds per policy)
        </div>
      </div>
    </div>
  );
};
