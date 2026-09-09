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
      <div className="w-full h-48 flex items-center justify-center text-xs text-[#8F9995] font-mono">
        No candidate action selected for score decomposition
      </div>
    );
  }

  if (
    action.w_hig === undefined ||
    action.w_discovery === undefined ||
    action.w_cost === undefined ||
    action.normalized_hig === undefined ||
    action.normalized_discovery === undefined ||
    action.normalized_cost === undefined ||
    action.weighted_hig_contribution === undefined ||
    action.weighted_discovery_contribution === undefined ||
    action.weighted_cost_contribution === undefined ||
    action.total_action_score === undefined
  ) {
    return (
      <div className="w-full h-48 flex flex-col items-center justify-center text-xs text-[#8F9995] font-mono p-4 text-center">
        <span>Score decomposition unavailable for this action record</span>
      </div>
    );
  }

  const wHig = action.w_hig;
  const wDisc = action.w_discovery;
  const wCost = action.w_cost;

  const normHig = action.normalized_hig;
  const normDisc = action.normalized_discovery;
  const normCost = action.normalized_cost;

  const higContrib = action.weighted_hig_contribution;
  const discContrib = action.weighted_discovery_contribution;
  const costContrib = -action.weighted_cost_contribution;
  const totalScore = action.total_action_score;

  const rawHigNats = action.raw_expected_hig_nats ?? action.expected_hig_nats;
  const rawHigStr = rawHigNats !== undefined
    ? `${rawHigNats.toFixed(3)} nats (norm: ${normHig.toFixed(3)})`
    : `norm: ${normHig.toFixed(3)}`;

  const chartData = [
    {
      name: 'Expected HIG',
      shortName: '+w_H · HIG',
      value: parseFloat(higContrib.toFixed(4)),
      raw: rawHigStr,
      formula: 'Source-recorded component',
      color: '#DC2626', // Emerald
      isPenalty: false
    },
    {
      name: 'Discovery Utility',
      shortName: '+w_D · D',
      value: parseFloat(discContrib.toFixed(4)),
      raw: `Utility: ${normDisc.toFixed(3)}`,
      formula: 'Source-recorded component',
      color: '#d97706', // Amber
      isPenalty: false
    },
    {
      name: 'Cost Penalty',
      shortName: '-w_C · C',
      value: parseFloat(costContrib.toFixed(4)),
      raw: `Cost: ${normCost.toFixed(3)}`,
      formula: 'Source-recorded component',
      color: '#e11d48', // Rose / Red
      isPenalty: true
    },
    {
      name: 'Net Score S(a)',
      shortName: '= Net Score',
      value: parseFloat(totalScore.toFixed(4)),
      raw: 'Signed Dimensionless',
      formula: 'Sum of components',
      color: totalScore >= 0 ? '#991B1B' : '#0f172a', // Dark Slate or deep emerald
      isTotal: true
    }
  ];

  const sumCheck = Math.abs(higContrib + discContrib + costContrib - totalScore) < 1e-4;

  return (
    <div className="w-full flex flex-col">
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center gap-2">
          <Calculator className="w-4 h-4 text-red-600" />
          <h4 className="text-sm font-bold text-slate-900 tracking-tight">
            Exact Score Decomposition
          </h4>
        </div>
        <div className="flex items-center gap-1.5 text-2xs font-mono text-red-800 bg-red-50 px-2 py-0.5 rounded-full border border-red-200">
          <Check className="w-3 h-3 text-red-600" />
          <span>Dimensionless Scalar</span>
        </div>
      </div>

      <div className="w-full h-52">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={chartData} margin={{ top: 14, right: 16, left: -10, bottom: 20 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#F1F5F9" vertical={false} />
            <XAxis
              dataKey="shortName"
              tick={{ fill: '#334155', fontSize: 12, fontWeight: 500, fontFamily: "'Montserrat', Arial, sans-serif" }}
              tickLine={false}
              axisLine={{ stroke: '#D9DFDB' }}
            />
            <YAxis
              tick={{ fill: '#475569', fontSize: 12, fontWeight: 500, fontFamily: "'Montserrat', Arial, sans-serif" }}
              tickLine={false}
              axisLine={{ stroke: '#D9DFDB' }}
            />
            <ReferenceLine y={0} stroke="#94A3B8" strokeWidth={1.5} />
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
          <span className="text-slate-500">Source-recorded score components</span>
          <span className="font-bold text-slate-900">
            {totalScore >= 0 ? `+${totalScore.toFixed(4)}` : totalScore.toFixed(4)}
          </span>
        </div>
        <div className="flex items-center justify-between text-3xs text-slate-400">
          <span>Weights: w_H={wHig}, w_D={wDisc}, w_C={wCost}</span>
          <span className={sumCheck ? 'text-red-600 font-semibold' : 'text-rose-600'}>
            {sumCheck ? '✓ Additive consistency verified' : '⚠ Component mismatch'}
          </span>
        </div>
      </div>
    </div>
  );
};
