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
import { Calculator, AlertCircle } from 'lucide-react';

interface Props {
  action: ScoredActionRecord | null;
}

export const ScoreWaterfallChart: React.FC<Props> = ({ action }) => {
  if (!action) {
    return (
      <div className="w-full h-56 flex flex-col items-center justify-center text-xs text-[#8F9995] font-mono p-4 text-center border border-dashed border-[#D9DFDB] rounded-xl bg-[#FCFCFA]">
        <AlertCircle className="w-5 h-5 text-[#8F9995] mb-2" />
        <span className="font-semibold text-[#17201F]">Action not scored in this recorded step</span>
        <span className="text-2xs text-[#8F9995] mt-1">This candidate × modality combination was not evaluated by the decision policy.</span>
      </div>
    );
  }

  const rawHigNats = action.raw_expected_hig_nats ?? action.expected_hig_nats;
  const rawDisc = action.raw_discovery_utility ?? action.discovery_utility;
  const cost = action.normalized_cost ?? action.action?.estimated_cost;
  const totalScore = action.total_action_score;

  const chartData = [
    {
      name: 'Expected HIG',
      shortName: 'Expected HIG',
      value: rawHigNats !== undefined ? parseFloat(rawHigNats.toFixed(4)) : 0,
      displayVal: rawHigNats !== undefined ? `${rawHigNats.toFixed(3)} nats` : 'Not recorded',
      description: 'Mutual information / entropy reduction',
      color: '#DC2626',
    },
    {
      name: 'Discovery Utility',
      shortName: 'Discovery Utility',
      value: rawDisc !== undefined ? parseFloat(rawDisc.toFixed(4)) : 0,
      displayVal: rawDisc !== undefined ? rawDisc.toFixed(4) : 'Not recorded',
      description: 'Exploitation utility value',
      color: '#D97706',
    },
    {
      name: 'Recorded Cost',
      shortName: 'Cost Metric',
      value: cost !== undefined ? parseFloat(cost.toFixed(4)) : 0,
      displayVal: cost !== undefined ? cost.toFixed(3) : 'Not recorded',
      description: 'Normalized measurement cost',
      color: '#64748B',
    },
    {
      name: 'Total Score',
      shortName: 'Net Score S(a)',
      value: totalScore !== undefined ? parseFloat(totalScore.toFixed(4)) : 0,
      displayVal: totalScore !== undefined ? `${totalScore >= 0 ? '+' : ''}${totalScore.toFixed(4)}` : 'Not recorded',
      description: 'Recorded decision score',
      color: (totalScore ?? 0) >= 0 ? '#991B1B' : '#0F172A',
      isTotal: true
    }
  ];

  return (
    <div className="w-full flex flex-col">
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center gap-2">
          <Calculator className="w-4 h-4 text-[#DC2626]" />
          <h4 className="text-sm font-bold text-[#17201F] tracking-tight">
            Recorded Decision Score
          </h4>
        </div>
        <div className="flex items-center gap-1.5 text-2xs font-mono text-[#991B1B] bg-[#FEF2F2] px-2 py-0.5 rounded-full border border-[#FECACA]">
          <span>{action.policy_name ? `Policy: ${action.policy_name}` : 'Recorded Action'}</span>
        </div>
      </div>

      <div className="w-full h-52">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={chartData} margin={{ top: 14, right: 16, left: -10, bottom: 20 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#F1F5F9" vertical={false} />
            <XAxis
              dataKey="shortName"
              tick={{ fill: '#334155', fontSize: 11, fontWeight: 500, fontFamily: "'Montserrat', Arial, sans-serif" }}
              tickLine={false}
              axisLine={{ stroke: '#D9DFDB' }}
            />
            <YAxis
              tick={{ fill: '#475569', fontSize: 11, fontWeight: 500, fontFamily: "'Montserrat', Arial, sans-serif" }}
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
                `${item.payload.displayVal} (${item.payload.description})`,
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

      {/* Honest disclosure footer */}
      <div className="mt-1 pt-2 border-t border-[#D9DFDB] flex flex-col gap-1 text-2xs font-mono text-[#66706C]">
        <div className="flex items-center justify-between">
          <span className="text-[#8F9995]">Total Action Score S(a)</span>
          <span className="font-bold text-[#17201F]">
            {totalScore !== undefined ? `${totalScore >= 0 ? '+' : ''}${totalScore.toFixed(4)}` : 'Not recorded'}
          </span>
        </div>
        <div className="text-3xs text-[#8F9995] mt-0.5">
          Full weighted decomposition is not persisted in the source record.
        </div>
      </div>
    </div>
  );
};
