import React from 'react';
import {
  ScoredActionRecord,
  HeatmapMetricMode
} from '../../types/mission_control';
import { Award, CheckCircle2 } from 'lucide-react';

interface Props {
  actions: ScoredActionRecord[];
  winnerActionId?: string;
  selectedCandidateId: string;
  selectedModality: string;
  activeMetric: HeatmapMetricMode;
  onChangeMetric: (metric: HeatmapMetricMode) => void;
  onSelectAction: (candidateId: string, modality: string) => void;
}

export const CandidateModalityHeatmap: React.FC<Props> = ({
  actions,
  winnerActionId,
  selectedCandidateId,
  selectedModality,
  activeMetric,
  onChangeMetric,
  onSelectAction
}) => {
  // Dynamically derive modalities present in the step
  const modalities = Array.from(
    new Set(actions.map((a) => a.action?.action_type).filter(Boolean) as string[])
  );
  if (modalities.length === 0) {
    modalities.push('XRD', 'REFINEMENT');
  }

  // Derive genuine costs per modality
  const modalityCosts: Record<string, number | undefined> = {};
  modalities.forEach((mod) => {
    const act = actions.find((a) => a.action?.action_type === mod);
    modalityCosts[mod] = act?.action?.estimated_cost;
  });

  // Unique candidate IDs sorted
  const candidateIds = Array.from(
    new Set(actions.map((a) => a.action?.candidate_id).filter(Boolean) as string[])
  ).sort((a, b) => {
    const numA = parseInt(a.replace(/\D/g, ''), 10) || 0;
    const numB = parseInt(b.replace(/\D/g, ''), 10) || 0;
    return numA - numB;
  });

  // Create lookup map [candidateId][modality] -> ScoredActionRecord
  const matrix: Record<string, Record<string, ScoredActionRecord>> = {};
  candidateIds.forEach((cId) => {
    matrix[cId] = {};
  });

  let minVal = Infinity;
  let maxVal = -Infinity;

  const getMetricValue = (rec: ScoredActionRecord | undefined, mode: HeatmapMetricMode): number | null => {
    if (!rec) return null;
    switch (mode) {
      case 'composite':
        return rec.total_action_score !== undefined ? rec.total_action_score : null;
      case 'raw_hig':
        return rec.raw_expected_hig_nats ?? rec.expected_hig_nats ?? null;
      case 'norm_hig':
        return rec.normalized_hig !== undefined ? rec.normalized_hig : null;
      case 'discovery':
        return rec.raw_discovery_utility ?? rec.discovery_utility ?? null;
      case 'cost':
        return rec.normalized_cost !== undefined ? rec.normalized_cost : null;
      default:
        return rec.total_action_score !== undefined ? rec.total_action_score : null;
    }
  };

  actions.forEach((a) => {
    const cId = a.action?.candidate_id;
    const mod = a.action?.action_type;
    if (cId && mod) {
      if (!matrix[cId]) matrix[cId] = {};
      matrix[cId][mod] = a;
      const v = getMetricValue(a, activeMetric);
      if (v !== null) {
        if (v < minVal) minVal = v;
        if (v > maxVal) maxVal = v;
      }
    }
  });

  if (minVal === Infinity) minVal = 0;
  if (maxVal === -Infinity) maxVal = 1;
  const valRange = maxVal - minVal || 1;

  // Background color intensity helper with fail-closed non-feasible handling
  const getCellBg = (val: number | null, isWinner: boolean, isSelected: boolean, isFeasible: boolean) => {
    if (!isFeasible || val === null) {
      return 'bg-[#F4F3EE]/60 border-dashed border-[#D9DFDB] text-[#8F9995] cursor-not-allowed';
    }
    if (isSelected) return 'bg-[#FEF2F2] border-[#B91C1C] ring-2 ring-[#DC2626] text-[#991B1B] font-bold';
    if (isWinner) return 'bg-[#FEF2F2]/80 border-[#B91C1C] ring-2 ring-[#FECACA] text-[#991B1B] font-bold';

    const normalized = (val - minVal) / valRange; // 0 to 1
    if (activeMetric === 'cost') {
      if (normalized > 0.7) return 'bg-amber-50 border-amber-200 text-amber-900';
      return 'bg-[#FCFCFA] border-[#D9DFDB] text-[#66706C]';
    }

    // Emerald gradient matching White & Emerald theme
    if (normalized > 0.8) return 'bg-[#B91C1C] text-white font-bold border-[#991B1B]';
    if (normalized > 0.6) return 'bg-[#FECACA] text-[#991B1B] font-semibold border-[#B91C1C]';
    if (normalized > 0.4) return 'bg-[#FEF2F2] text-[#991B1B] border-[#FECACA]';
    if (normalized > 0.2) return 'bg-[#FCFCFA] text-[#17201F] border-[#D9DFDB]';
    return 'bg-[#F4F3EE] text-[#66706C] border-[#D9DFDB]';
  };

  const metricLabels: Record<HeatmapMetricMode, { title: string; unit: string }> = {
    composite: { title: 'Net Composite Score S(a)', unit: 'signed dimensionless' },
    raw_hig: { title: 'Raw Expected HIG', unit: 'nats' },
    norm_hig: { title: 'Normalized HIG', unit: '[0, 1]' },
    discovery: { title: 'Discovery Utility', unit: '[0, 1]' },
    cost: { title: 'Normalized Cost', unit: '[0, 1]' }
  };

  return (
    <div className="w-full flex flex-col">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 mb-3">
        <div>
          <div className="flex items-center gap-2">
            <h4 className="text-sm font-bold text-slate-900 tracking-tight">
              Candidate × Modality Action Matrix
            </h4>
            <span className="text-2xs font-mono px-2 py-0.5 rounded-full bg-slate-100 text-slate-700 border border-slate-200">
              {candidateIds.length} Candidates × {modalities.length} Modalities
            </span>
          </div>
          <p className="text-xs text-slate-500">
            Click any cell to inspect expected information gain, costs, and counterfactuals
          </p>
        </div>

        {/* Metric Switcher Pills */}
        <div className="flex items-center gap-1 bg-slate-100 p-0.5 rounded-lg border border-slate-200 text-xs font-mono">
          {(['composite', 'raw_hig', 'discovery', 'cost'] as HeatmapMetricMode[]).map((m) => (
            <button
              key={m}
              onClick={() => onChangeMetric(m)}
              className={`px-2 py-1 rounded-md transition-all cursor-pointer ${
                activeMetric === m
                  ? 'bg-white text-red-800 font-bold shadow-2xs'
                  : 'text-slate-600 hover:text-slate-900'
              }`}
            >
              {m === 'composite' ? 'S(a)' : m === 'raw_hig' ? 'HIG' : m === 'discovery' ? 'Discovery' : 'Cost'}
            </button>
          ))}
        </div>
      </div>

      {/* Matrix Table */}
      <div className="overflow-x-auto border border-slate-200 rounded-xl bg-white shadow-2xs">
        <table className="w-full text-center border-collapse">
          <thead>
            <tr className="bg-slate-50/80 border-b border-slate-200 text-2xs font-mono text-slate-500 uppercase tracking-wider">
              <th className="py-2.5 px-3 text-left font-semibold sticky left-0 bg-slate-50 z-10">
                Modality
              </th>
              {candidateIds.map((cId) => {
                const isWinnerCol = actions.some(
                  (a) => a.action?.candidate_id === cId && a.action?.action_id === winnerActionId
                );
                const isSelectedCol = cId === selectedCandidateId;
                return (
                  <th
                    key={cId}
                    className={`py-2 px-1.5 font-semibold text-center whitespace-nowrap ${
                      isSelectedCol
                        ? 'text-red-800 bg-red-50/50'
                        : isWinnerCol
                        ? 'text-red-700 font-bold'
                        : ''
                    }`}
                  >
                    <span className="block text-2xs">{cId.replace('controlled-', 'C-')}</span>
                    {isWinnerCol && (
                      <span className="inline-block w-1.5 h-1.5 rounded-full bg-red-500 mt-0.5" />
                    )}
                  </th>
                );
              })}
            </tr>
          </thead>
          <tbody>
            {modalities.map((mod) => (
              <tr key={mod} className="border-b border-slate-100 last:border-b-0 hover:bg-slate-50/40">
                <td className="py-2 px-3 text-left font-mono font-bold text-xs text-slate-800 sticky left-0 bg-white z-10 border-r border-slate-100">
                  <div className="flex items-center gap-1.5">
                    <span className={`w-2 h-2 rounded-full ${mod === 'XRD' ? 'bg-[#B91C1C]' : 'bg-[#7C3AED]'}`} />
                    <span>{mod}</span>
                    <span className="text-2xs text-[#8F9995] font-normal">
                      (cost: {modalityCosts[mod] !== undefined ? modalityCosts[mod]?.toFixed(1) : 'N/A'})
                    </span>
                  </div>
                </td>
                {candidateIds.map((cId) => {
                  const rec = matrix[cId]?.[mod];
                  const val = getMetricValue(rec, activeMetric);
                  const isWinner = rec?.action?.action_id === winnerActionId;
                  const isSelected = cId === selectedCandidateId && mod === selectedModality;
                  const isFeasible = Boolean(rec);

                  return (
                    <td key={cId} className="p-1">
                      <button
                        disabled={!isFeasible}
                        onClick={() => isFeasible && onSelectAction(cId, mod)}
                        title={isFeasible ? `${cId} — ${mod}: ${val !== null ? val.toFixed(4) : 'N/A'}` : `${cId} — ${mod}: Infeasible`}
                        className={`w-full py-2 px-1 rounded-lg border text-2xs font-mono transition-all relative flex flex-col items-center justify-center ${
                          isFeasible ? 'cursor-pointer' : 'cursor-not-allowed opacity-50'
                        } ${getCellBg(
                          val,
                          isWinner,
                          isSelected,
                          isFeasible
                        )}`}
                      >
                        {isWinner && (
                          <div className="absolute -top-1 -right-1 bg-[#B91C1C] text-white rounded-full p-0.5 shadow-2xs">
                            <Award className="w-2.5 h-2.5" />
                          </div>
                        )}
                        <span>{val !== null ? val.toFixed(2) : '—'}</span>
                        {isSelected && (
                          <span className="text-3xs tracking-tight text-[#991B1B] font-bold uppercase mt-0.5">
                            Active
                          </span>
                        )}
                      </button>
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div className="mt-2.5 flex flex-col sm:flex-row items-center justify-between text-2xs font-mono text-slate-500 gap-2">
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-1">
            <span className="w-2.5 h-2.5 rounded bg-red-500 inline-block" />
            <span>High {metricLabels[activeMetric].title}</span>
          </div>
          <div className="flex items-center gap-1">
            <div className="w-3 h-3 rounded-full bg-red-600 text-white flex items-center justify-center text-3xs">
              <Award className="w-2 h-2" />
            </div>
            <span className="font-semibold text-red-800">Recorded Recommendation</span>
          </div>
          <div className="flex items-center gap-1">
            <CheckCircle2 className="w-3 h-3 text-red-600" />
            <span>Currently Selected</span>
          </div>
        </div>

        <div className="text-slate-400">
          Showing Metric: <strong className="text-slate-700">{metricLabels[activeMetric].title}</strong> ({metricLabels[activeMetric].unit})
        </div>
      </div>
    </div>
  );
};
