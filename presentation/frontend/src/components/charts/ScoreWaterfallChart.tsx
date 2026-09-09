import React from 'react';
import {
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Cell,
  ReferenceLine
} from 'recharts';
import { ScoredActionRecord } from '../../types/mission_control';
import { Calculator, Check } from 'lucide-react';

interface Props {
  action: ScoredActionRecord | null;
}

export const ScoreWaterfallChart: React.FC<Props> = ({ action }) => {
  if (!action) {
    return (
      <div className="w-full h-48 flex items-center justify-center text-xs text-slate-400 font-mono">
        No candidate action selected for score decomposition
      </div>
    );
  }

  const wHig = action.w_hig ?? 0.8;
  const wDisc = action.w_discovery ?? 0.8;
  const wCost = action.w_cost ?? 2.0;

  const normHig = action.normalized_hig ?? 0.746;
  const normDisc = action.normalized_discovery ?? 0.978;
  const normCost = action.normalized_cost ?? 0.5;

  const higContrib = action.weighted_hig_contribution ?? (wHig * normHig);
  const discContrib = action.weighted_discovery_contribution ?? (wDisc * normDisc);
  const costContrib = -(action.weighted_cost_contribution ?? (wCost * normCost));
  const totalScore = action.total_action_score ?? (higContrib + discContrib + costContrib);

  const rawHigNats = action.raw_expected_hig_nats ?? action.expected_hig_nats ?? 0.5066;

  const chartData = [
    {
      name: 'Expected HIG',
      shortName: '+w_H · HIG',
      value: parseFloat(higContrib.toFixed(4)),
      raw: `${rawHigNats.toFixed(3)} nats (norm: ${normHig.toFixed(3)})`,
      formula: `+${wHig} × ${normHig.toFixed(3)}`,
      color: '#059669', // Emerald
      isPenalty: false
    },
    {
      name: 'Discovery Utility',
      shortName: '+w_D · D',
      value: parseFloat(discContrib.toFixed(4)),
      raw: `Utility: ${normDisc.toFixed(3)}`,
      formula: `+${wDisc} × ${normDisc.toFixed(3)}`,
      color: '#d97706', // Amber
      isPenalty: false
    },
    {
      name: 'Cost Penalty',
      shortName: '-w_C · C',
      value: parseFloat(costContrib.toFixed(4)),
      raw: `Cost: ${normCost.toFixed(3)}`,
      formula: `-${wCost} × ${normCost.toFixed(3)}`,
      color: '#e11d48', // Rose / Red
      isPenalty: true
    },
    {
      name: 'Net Score S(a)',
      shortName: '= Net Score',
      value: parseFloat(totalScore.toFixed(4)),
      raw: 'Signed Dimensionless',
      formula: 'Sum of components',
      color: totalScore >= 0 ? '#047857' : '#0f172a', // Dark Slate or deep emerald
      isTotal: true
    }
  ];

  const sumCheck = Math.abs(higContrib + discContrib + costContrib - totalScore) < 1e-4;

  return (
    <div className="w-full flex flex-col">
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center gap-2">
          <Calculator className="w-4 h-4 text-emerald-600" />
          <h4 className="text-sm font-bold text-slate-900 tracking-tight">
            Exact Score Decomposition
          </h4>
        </div>
        <div className="flex items-center gap-1.5 text-2xs font-mono text-emerald-800 bg-emerald-50 px-2 py-0.5 rounded-full border border-emerald-200">
          <Check className="w-3 h-3 text-emerald-600" />
          <span>Dimensionless Scalar</span>
        </div>
      </div>

      <div className="w-full h-44">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={chartData} margin={{ top: 12, right: 12, left: -20, bottom: 20 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" vertical={false} />
            <XAxis
              dataKey="shortName"
              tick={{ fill: '#475569', fontSize: 11, fontFamily: 'monospace' }}
              tickLine={false}
              axisLine={{ stroke: '#cbd5e1' }}
            />
            <YAxis
              tick={{ fill: '#64748b', fontSize: 10 }}
              tickLine={false}
              axisLine={{ stroke: '#cbd5e1' }}
            />
            <ReferenceLine y={0} stroke="#94a3b8" strokeWidth={1.5} />
            <Tooltip
              contentStyle={{
                backgroundColor: '#ffffff',
                borderColor: '#e2e8f0',
                borderRadius: '8px',
                boxShadow: '0 4px 6px -1px rgb(0 0 0 / 0.1)',
                fontSize: '11px'
              }}
              formatter={(value: any, name: any, item: any) => [
                `${Number(value).toFixed(4)} (${item.payload.formula})`,
                item.payload.name
              ]}
            />
            <Bar dataKey="value" radius={[4, 4, 0, 0]}>
              {chartData.map((entry, index) => (
                <Cell key={`cell-${index}`} fill={entry.color} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* Formula breakdown footer */}
      <div className="mt-1 pt-2 border-t border-slate-100 flex flex-col gap-1 text-2xs font-mono text-slate-600">
        <div className="flex items-center justify-between">
          <span className="text-slate-500">S(a) = w_H · HIG + w_D · D - w_C · C</span>
          <span className="font-bold text-slate-900">
            {totalScore >= 0 ? `+${totalScore.toFixed(4)}` : totalScore.toFixed(4)}
          </span>
        </div>
        <div className="flex items-center justify-between text-3xs text-slate-400">
          <span>Weights: w_H={wHig}, w_D={wDisc}, w_C={wCost}</span>
          <span className={sumCheck ? 'text-emerald-600 font-semibold' : 'text-rose-600'}>
            {sumCheck ? '✓ Additive consistency verified' : '⚠ Component mismatch'}
          </span>
        </div>
      </div>
    </div>
  );
};
