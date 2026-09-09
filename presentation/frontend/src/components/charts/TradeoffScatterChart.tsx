import React from 'react';
import {
  ResponsiveContainer,
  ScatterChart,
  Scatter,
  XAxis,
  YAxis,
  ZAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  Cell
} from 'recharts';
import { ScoredActionRecord } from '../../types/mission_control';
import { Compass } from 'lucide-react';

interface Props {
  actions: ScoredActionRecord[];
  winnerActionId?: string;
  selectedCandidateId: string;
  selectedModality: string;
  onSelectAction: (candidateId: string, modality: string) => void;
}

export const TradeoffScatterChart: React.FC<Props> = ({
  actions,
  winnerActionId,
  selectedCandidateId,
  selectedModality,
  onSelectAction
}) => {
  // Format actions into scatter points
  const points = actions.map((a) => {
    const rawHig = a.raw_expected_hig_nats ?? a.expected_hig_nats ?? 0;
    const disc = a.raw_discovery_utility ?? a.discovery_utility ?? 0;
    const cost = a.raw_estimated_cost ?? a.action?.estimated_cost ?? 0;
    const cId = a.action?.candidate_id || '';
    const mod = a.action?.action_type || '';
    const isWinner = a.action?.action_id === winnerActionId;
    const isSelected = cId === selectedCandidateId && mod === selectedModality;

    return {
      x: parseFloat(rawHig.toFixed(4)),
      y: parseFloat(disc.toFixed(4)),
      z: cost,
      candidateId: cId,
      modality: mod,
      score: a.total_action_score ?? 0,
      cost,
      rawHig,
      isWinner,
      isSelected
    };
  });

  return (
    <div className="w-full flex flex-col">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2.5 mb-3">
        <div>
          <h4 className="text-xs sm:text-sm font-bold text-[#17201F] tracking-tight flex items-center gap-2">
            <Compass className="w-4 h-4 text-[#DC2626]" />
            <span>Information–Discovery–Cost Action Trade-off</span>
          </h4>
          <p className="text-2xs text-[#66706C] mt-0.5">
            Expected HIG (nats) vs Discovery Utility with measurement cost weighting
          </p>
        </div>
        <div className="text-2xs font-mono text-[#8F9995] whitespace-nowrap">
          Click point to inspect candidate
        </div>
      </div>

      <div className="w-full h-[300px]">
        <ResponsiveContainer width="100%" height={300}>
          <ScatterChart margin={{ top: 16, right: 30, left: 0, bottom: 24 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#F1F5F9" />
            <XAxis
              type="number"
              dataKey="x"
              name="Expected HIG"
              unit=" nats"
              tick={{ fill: '#334155', fontSize: 12, fontWeight: 500, fontFamily: "'Montserrat', Arial, sans-serif" }}
              tickLine={false}
              axisLine={{ stroke: '#D9DFDB' }}
              label={{
                value: 'Expected Hypothesis Information Gain (nats)',
                position: 'insideBottom',
                offset: -12,
                fill: '#17201F',
                fontSize: 12,
                fontWeight: 600,
                fontFamily: "'Montserrat', Arial, sans-serif"
              }}
            />
            <YAxis
              type="number"
              dataKey="y"
              name="Discovery Utility"
              domain={[0, 1]}
              tick={{ fill: '#475569', fontSize: 12, fontWeight: 500, fontFamily: "'Montserrat', Arial, sans-serif" }}
              tickLine={false}
              axisLine={{ stroke: '#D9DFDB' }}
              label={{
                value: 'Discovery Utility [0, 1]',
                angle: -90,
                position: 'insideLeft',
                offset: 15,
                fill: '#17201F',
                fontSize: 12,
                fontWeight: 600,
                fontFamily: "'Montserrat', Arial, sans-serif"
              }}
            />
            <ZAxis type="number" dataKey="z" range={[60, 200]} name="Cost" />
            <Tooltip
              cursor={{ strokeDasharray: '3 3' }}
              contentStyle={{
                backgroundColor: '#ffffff',
                borderColor: '#D9DFDB',
                borderRadius: '8px',
                boxShadow: '0 4px 6px -1px rgb(0 0 0 / 0.1)',
                fontSize: '12px',
                fontFamily: "'Montserrat', Arial, sans-serif"
              }}
              formatter={(value: any, name: any, item: any) => {
                const p = item.payload;
                if (name === 'Expected HIG') return [`${Number(value).toFixed(3)} nats`, name];
                if (name === 'Discovery Utility') return [Number(value).toFixed(3), name];
                return [value, name];
              }}
              labelFormatter={() => ''}
            />
            <Scatter
              name="Actions"
              data={points}
              onClick={(entry: any) => {
                if (entry && entry.candidateId && entry.modality) {
                  onSelectAction(entry.candidateId, entry.modality);
                }
              }}
              className="cursor-pointer"
            >
              {points.map((p, idx) => {
                let fill = '#94a3b8';
                if (p.isWinner) fill = '#DC2626';
                else if (p.isSelected) fill = '#991B1B';
                else if (p.modality === 'XRD') fill = '#EF4444';
                else if (p.modality === 'TEM' || p.modality === 'REFINEMENT') fill = '#8B5CF6';

                return (
                  <Cell
                    key={idx}
                    fill={fill}
                    stroke={p.isSelected ? '#17201F' : '#ffffff'}
                    strokeWidth={p.isSelected ? 2.5 : 1}
                  />
                );
              })}
            </Scatter>
          </ScatterChart>
        </ResponsiveContainer>
      </div>

      <div className="mt-3 pt-2.5 border-t border-[#D9DFDB] flex flex-col sm:flex-row sm:items-center justify-between gap-1.5 text-2xs font-mono text-[#66706C]">
        <div className="flex items-center gap-3">
          <span className="flex items-center gap-1">
            <span className="w-2.5 h-2.5 rounded-full bg-[#B91C1C]" />
            <span>Winner (#1)</span>
          </span>
          <span className="flex items-center gap-1">
            <span className="w-2.5 h-2.5 rounded-full bg-[#EF4444]" />
            <span>XRD Diagnostic</span>
          </span>
          <span className="flex items-center gap-1">
            <span className="w-2.5 h-2.5 rounded-full bg-[#8B5CF6]" />
            <span>Refinement / TEM</span>
          </span>
        </div>
        <div className="text-[#8F9995] whitespace-nowrap">
          Circle size indicates characterization cost
        </div>
      </div>
    </div>
  );
};
