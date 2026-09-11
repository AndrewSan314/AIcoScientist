import React from 'react';
import {
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
} from 'recharts';
import { Activity, ShieldCheck } from 'lucide-react';

interface Props {
  sensitivity: any;
}

const WORLD_LABELS: Record<string, string> = {
  CLEAN_WORLD_H1_PHASE_PURITY: 'Clean H1',
  CLEAN_WORLD_H2_COMPOSITION_HOMOGENEITY: 'Clean H2',
  CLEAN_WORLD_H3_MORPHOLOGY_KINETICS: 'Clean H3',
  STRESS_WORLD_H1_PHASE_PURITY: 'Stress H1',
  STRESS_WORLD_H2_COMPOSITION_HOMOGENEITY: 'Stress H2',
  STRESS_WORLD_H3_MORPHOLOGY_KINETICS: 'Stress H3'
};

export const SensitivityRankAgreementChart: React.FC<Props> = ({ sensitivity }) => {
  const aggregate = sensitivity?.aggregate_by_world_policy || {};
  const worlds = sensitivity?.design?.worlds || [];

  const chartData = worlds.map((worldKey: string) => {
    const pureHigKey = `${worldKey}:PURE_HIG`;
    const hybridKey = `${worldKey}:HYBRID`;

    const pureData = aggregate[pureHigKey] || {};
    const hybridData = aggregate[hybridKey] || {};

    const pureCorr = pureData.HIG_rank_correlation;
    const hybridCorr = hybridData.HIG_rank_correlation;

    return {
      worldKey,
      name: WORLD_LABELS[worldKey] || worldKey.replace(/_/g, ' '),
      pureHig: typeof pureCorr === 'number' ? parseFloat(pureCorr.toFixed(3)) : null,
      hybrid: typeof hybridCorr === 'number' ? parseFloat(hybridCorr.toFixed(3)) : null,
      pureModalityAgreement: typeof pureData.modality_sequence_agreement === 'number' ? pureData.modality_sequence_agreement * 100 : null,
      hybridModalityAgreement: typeof hybridData.modality_sequence_agreement === 'number' ? hybridData.modality_sequence_agreement * 100 : null
    };
  });
  const correlations = chartData.flatMap((row: { pureHig: number | null; hybrid: number | null }) => [row.pureHig, row.hybrid]).filter((value: number | null): value is number => typeof value === 'number');

  return (
    <div className="w-full flex flex-col">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 mb-3">
        <div>
          <div className="flex items-center gap-2">
            <Activity className="w-4 h-4 text-[#DC2626]" />
            <h4 className="text-sm font-bold text-[#17201F] tracking-tight">
              Monte Carlo Sample Sizing Sensitivity (MC12 vs MC32)
            </h4>
            <span className="text-xs font-semibold px-2 py-0.5 rounded-full bg-[#FEF2F2] text-[#991B1B] border border-[#FECACA]">
              Spearman Rank Correlation
            </span>
          </div>
          <p className="text-xs text-[#66706C] mt-0.5">
            Rank agreement of candidate acquisition scores between 12 and 32 particle draws across synthetic worlds
          </p>
        </div>

        <div className="text-xs text-[#991B1B] bg-[#FEF2F2] px-3 py-1 rounded-md border border-[#FECACA] font-medium shrink-0">
          Source range: <strong>{correlations.length ? `${Math.min(...correlations).toFixed(3)} – ${Math.max(...correlations).toFixed(3)}` : 'Not recorded'}</strong>
        </div>
      </div>

      <div className="w-full h-72">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={chartData} margin={{ top: 16, right: 110, left: 0, bottom: 24 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#F1F5F9" vertical={false} />
            <XAxis
              dataKey="name"
              tick={{ fill: '#334155', fontSize: 12, fontWeight: 500, fontFamily: "'Montserrat', Arial, sans-serif" }}
              tickLine={false}
              axisLine={{ stroke: '#D9DFDB' }}
            />
            <YAxis
              domain={[0.4, 1.0]}
              ticks={[0.5, 0.6, 0.7, 0.8, 0.9, 1.0]}
              tickFormatter={(v) => `ρ=${v.toFixed(1)}`}
              tick={{ fill: '#475569', fontSize: 12, fontWeight: 500, fontFamily: "'Montserrat', Arial, sans-serif" }}
              tickLine={false}
              axisLine={{ stroke: '#D9DFDB' }}
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
                `ρ = ${Number(value).toFixed(3)}`,
                name === 'pureHig' ? 'Pure HIG Policy' : 'Cost-Penalized Hybrid Policy'
              ]}
              labelFormatter={(label, payload) => payload?.[0]?.payload?.worldKey || label}
            />
            <Legend
              verticalAlign="top"
              height={32}
              formatter={(value) => (
                <span className="text-xs font-semibold text-[#17201F]">
                  {value === 'pureHig' ? 'Pure HIG Policy (MC12 vs MC32)' : 'Cost-Penalized Hybrid Policy (MC12 vs MC32)'}
                </span>
              )}
            />

            <Bar dataKey="pureHig" fill="#3B82F6" radius={[4, 4, 0, 0]} maxBarSize={36} />
            <Bar dataKey="hybrid" fill="#B91C1C" radius={[4, 4, 0, 0]} maxBarSize={36} />
          </BarChart>
        </ResponsiveContainer>
      </div>

      <div className="mt-2 pt-2.5 border-t border-[#D9DFDB] flex flex-col sm:flex-row sm:items-center justify-between gap-2 text-xs text-[#66706C]">
        <div className="flex items-center gap-2 text-[#991B1B]">
          <ShieldCheck className="w-4 h-4 text-[#DC2626] shrink-0" />
          <span>Source design: <strong>{sensitivity?.design?.low_samples ?? 'N/A'} vs {sensitivity?.design?.high_samples ?? 'N/A'} MC samples</strong></span>
        </div>
        <div className="text-[#8F9995]">
          Evaluation: {sensitivity?.trajectory_count ?? 'N/A'} paired source runs across {worlds.length || 'N/A'} physical worlds ({sensitivity?.design?.seeds?.length ?? 'N/A'} randomized trials)
        </div>
      </div>
    </div>
  );
};
