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
    const cost = a.raw_estimated_cost ?? 1.0;
    const cId = a.action?.candidate_id || '';
    const mod = a.action?.action_type || '';
    const isWinner = a.action?.action_id === winnerActionId;
    const isSelected = cId === selectedCandidateId && mod === selectedModality;

    return {
      x: parseFloat(rawHig.toFixed(4)),
      y: parseFloat(disc.toFixed(4)),
      z: cost * 10,
      candidateId: cId,
      modality: mod,
      cost,
      score: a.total_action_score ?? 0,
      isWinner,
      isSelected
    };
  });

  return (
    <div className="w-full h-full flex flex-col">
      <div className="flex items-center justify-between mb-3">
        <div>
          <h4 className="text-sm font-bold text-slate-900 tracking-tight flex items-center gap-2">
            <Compass className="w-4 h-4 text-emerald-600" />
            <span>Information–Discovery–Cost Action Trade-off</span>
          </h4>
          <p className="text-xs text-slate-500">
            Expected HIG (nats) vs Discovery Utility with measurement cost weighting
          </p>
        </div>
        <div className="text-2xs font-mono text-slate-400">
          Click point to inspect candidate
        </div>
      </div>

      <div className="flex-1 w-full min-h-[250px]">
        <ResponsiveContainer width="100%" height="100%">
          <ScatterChart margin={{ top: 12, right: 24, left: -10, bottom: 20 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
            <XAxis
              type="number"
              dataKey="x"
              name="Expected HIG"
              unit=" nats"
              tick={{ fill: '#64748b', fontSize: 11 }}
              tickLine={{ stroke: '#cbd5e1' }}
              axisLine={{ stroke: '#cbd5e1' }}
              label={{
                value: 'Expected Hypothesis Information Gain (nats)',
                position: 'insideBottom',
                offset: -12,
                fill: '#475569',
                fontSize: 11
              }}
            />
            <YAxis
              type="number"
              dataKey="y"
              name="Discovery Utility"
              domain={[0, 1]}
              tick={{ fill: '#64748b', fontSize: 11 }}
              tickLine={{ stroke: '#cbd5e1' }}
              axisLine={{ stroke: '#cbd5e1' }}
              label={{
                value: 'Discovery Utility [0, 1]',
                angle: -90,
                position: 'insideLeft',
                offset: 20,
                fill: '#475569',
                fontSize: 11
              }}
            />
            <ZAxis type="number" dataKey="z" range={[60, 200]} name="Cost" />
            <Tooltip
              cursor={{ strokeDasharray: '3 3' }}
              contentStyle={{
                backgroundColor: '#ffffff',
                borderColor: '#e2e8f0',
                borderRadius: '8px',
                boxShadow: '0 4px 6px -1px rgb(0 0 0 / 0.1)',
                fontSize: '12px'
              }}
              formatter={(value: any, name: any, item: any) => {
                const p = item.payload;
                return [
                  `HIG: ${p.x} nats, Discovery: ${p.y}, Cost: ${p.cost}, Net S(a): ${p.score.toFixed(3)}`,
                  `${p.candidateId} — ${p.modality}${p.isWinner ? ' (Recommendation)' : ''}`
                ];
              }}
            />
            <Legend
              verticalAlign="top"
              height={32}
              formatter={() => (
                <span className="text-xs font-medium text-slate-700">
                  Green: XRD (cost: 1.0) | Violet: Refinement (cost: 0.5) | Ring: Selected/Winner
                </span>
              )}
            />
            <Scatter
              name="Actions"
              data={points}
              onClick={(e: any) => {
                if (e?.candidateId && e?.modality) {
                  onSelectAction(e.candidateId, e.modality);
                }
              }}
              className="cursor-pointer"
            >
              {points.map((entry, index) => {
                let fill = entry.modality === 'XRD' ? '#059669' : '#7c3aed';
                if (entry.isWinner) fill = '#10b981';
                if (entry.isSelected) fill = '#2563eb';

                const stroke = entry.isSelected
                  ? '#1d4ed8'
                  : entry.isWinner
                  ? '#047857'
                  : '#ffffff';

                return (
                  <Cell
                    key={`cell-${index}`}
                    fill={fill}
                    stroke={stroke}
                    strokeWidth={entry.isSelected || entry.isWinner ? 3 : 1}
                  />
                );
              })}
            </Scatter>
          </ScatterChart>
        </ResponsiveContainer>
      </div>

      <div className="mt-2 pt-2 border-t border-slate-100 flex items-center justify-between text-2xs font-mono text-slate-500">
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-1.5">
            <span className="w-2.5 h-2.5 rounded-full bg-emerald-600 inline-block" />
            <span>XRD Modality</span>
          </div>
          <div className="flex items-center gap-1.5">
            <span className="w-2.5 h-2.5 rounded-full bg-violet-600 inline-block" />
            <span>Refinement Modality</span>
          </div>
          <div className="flex items-center gap-1.5">
            <span className="w-2.5 h-2.5 rounded-full bg-blue-600 inline-block" />
            <span>Selected Action</span>
          </div>
        </div>
        <div className="text-slate-400">
          Joint Candidate × Modality Optimization
        </div>
      </div>
    </div>
  );
};
