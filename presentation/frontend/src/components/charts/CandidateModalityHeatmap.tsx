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
    return <div className="h-48 flex items-center justify-center text-sm text-[#66706C]">Action matrix unavailable for this source step.</div>;
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

  if (minVal === Infinity || maxVal === -Infinity) {
    return <div className="h-48 flex items-center justify-center text-sm text-[#66706C]">Selected metric unavailable for this source step.</div>;
  }
  const valRange = maxVal - minVal || 1;

  // Refined, intuitive color mapping:
  // Positive/high scores light up in warm ruby/coral; negative scores recede into soft slate.
  const getCellBg = (val: number | null, isWinner: boolean, isSelected: boolean, isFeasible: boolean) => {
    if (!isFeasible || val === null) {
      return 'bg-[#F4F3EE]/50 border-dashed border-[#D9DFDB] text-[#8F9995] cursor-not-allowed';
    }

    let baseBg = 'bg-[#FCFCFA] text-[#17201F] border-[#D9DFDB]';

    if (activeMetric === 'composite') {
      if (val >= 0.25) {
        baseBg = 'bg-[#DC2626] text-white font-bold border-[#B91C1C] shadow-2xs';
      } else if (val >= 0.10) {
        baseBg = 'bg-[#F87171] text-white font-semibold border-[#EF4444]';
      } else if (val >= 0.00) {
        baseBg = 'bg-[#FEE2E2] text-[#991B1B] font-semibold border-[#FECACA]';
      } else if (val >= -0.20) {
        baseBg = 'bg-[#F8FAFC] text-[#334155] border-[#E2E8F0]';
      } else {
        baseBg = 'bg-[#F1F5F9]/80 text-[#64748B] border-[#E2E8F0]/70';
      }
    } else if (activeMetric === 'cost') {
      const norm = (val - minVal) / valRange;
      if (norm > 0.7) {
        baseBg = 'bg-amber-100 text-amber-900 border-amber-300/80 font-medium';
      } else {
        baseBg = 'bg-[#F8FAFC] text-[#334155] border-[#E2E8F0]';
      }
    } else if (activeMetric === 'discovery') {
      const norm = (val - minVal) / valRange;
      if (norm >= 0.75) {
        baseBg = 'bg-[#D97706] text-white font-bold border-[#B45309]';
      } else if (norm >= 0.50) {
        baseBg = 'bg-[#FBBF24] text-amber-950 font-semibold border-[#F59E0B]';
      } else if (norm >= 0.25) {
        baseBg = 'bg-[#FEF3C7] text-[#92400E] font-medium border-[#FDE68A]';
      } else {
        baseBg = 'bg-[#F8FAFC] text-[#64748B] border-[#E2E8F0]';
      }
    } else {
      // HIG / Norm HIG
      const norm = (val - minVal) / valRange;
      if (norm >= 0.75) {
        baseBg = 'bg-[#DC2626] text-white font-bold border-[#B91C1C]';
      } else if (norm >= 0.50) {
        baseBg = 'bg-[#F87171] text-white font-semibold border-[#EF4444]';
      } else if (norm >= 0.25) {
        baseBg = 'bg-[#FEE2E2] text-[#991B1B] font-medium border-[#FECACA]';
      } else {
        baseBg = 'bg-[#F8FAFC] text-[#64748B] border-[#E2E8F0]';
      }
    }

    if (isSelected) {
      return `${baseBg} ring-2 ring-[#17201F] ring-offset-1 z-20 scale-[1.03] shadow-xs`;
    }
    if (isWinner) {
      return `${baseBg} ring-2 ring-[#DC2626] ring-offset-1`;
    }
    return baseBg;
  };

  const metricLabels: Record<HeatmapMetricMode, { title: string; unit: string }> = {
    composite: { title: 'Net Composite Score S(a)', unit: 'signed dimensionless' },
    raw_hig: { title: 'Raw Expected HIG', unit: 'nats' },
    norm_hig: { title: 'Normalized HIG', unit: '[0, 1]' },
    discovery: { title: 'Discovery Utility', unit: '[0, 1]' },
    cost: { title: 'Normalized Cost', unit: '[0, 1]' }
  };

  return (
    <div className="w-full flex flex-col space-y-3">
      {/* Header with Title and Metric Switcher */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2.5 pb-1">
        <div>
          <div className="flex items-center gap-2">
            <h4 className="text-sm font-bold text-[#17201F] tracking-tight">
              Candidate × Measurement Decision Matrix
            </h4>
            <span className="text-2xs font-semibold px-2 py-0.5 rounded-full bg-[#F4F3EE] text-[#66706C] border border-[#D9DFDB]">
              {candidateIds.length} Candidates × {modalities.length} Modalities
            </span>
          </div>
          <p className="text-xs text-[#66706C] mt-0.5 font-normal">
            Each cell is one possible experiment. Select a cell to inspect its source-recorded values.
          </p>
        </div>

        {/* Metric Switcher Pills */}
        <div className="flex items-center gap-1 bg-[#F4F3EE] p-1 rounded-lg border border-[#D9DFDB] self-start sm:self-auto">
          {(['composite', 'raw_hig', 'discovery', 'cost'] as HeatmapMetricMode[]).map((m) => (
            <button
              key={m}
              onClick={() => onChangeMetric(m)}
              className={`px-2.5 py-1 rounded-md text-xs font-semibold transition-all cursor-pointer ${
                activeMetric === m
                  ? 'bg-white text-[#DC2626] font-bold shadow-2xs'
                  : 'text-[#66706C] hover:text-[#17201F]'
              }`}
            >
              {m === 'composite' ? 'Composite score' : m === 'raw_hig' ? 'Information gain' : m === 'discovery' ? 'Discovery value' : 'Cost'}
            </button>
          ))}
        </div>
      </div>

      {/* Matrix Table with guaranteed width so C-11 is NEVER cut off */}
      <div className="overflow-x-auto border border-[#D9DFDB] rounded-xl bg-white shadow-2xs">
        <table className="w-full text-center border-collapse min-w-[720px]">
          <thead>
            <tr className="bg-[#F8FAFC] border-b border-[#D9DFDB] text-xs font-semibold text-slate-500 uppercase tracking-wider">
              <th className="py-2.5 px-3.5 text-left sticky left-0 bg-[#F8FAFC] z-10 w-[140px] min-w-[130px] border-r border-[#E2E8F0]">
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
                    className={`py-2 px-1 font-semibold text-center whitespace-nowrap min-w-[44px] ${
                      isSelectedCol
                        ? 'text-[#DC2626] bg-red-50/60 font-bold'
                        : isWinnerCol
                        ? 'text-[#991B1B] font-bold'
                        : 'text-slate-600'
                    }`}
                  >
                    <span className="block text-2xs font-bold">{cId.replace('controlled-', 'C-')}</span>
                    {isWinnerCol && (
                      <span className="inline-block w-1.5 h-1.5 rounded-full bg-[#DC2626] mt-0.5" />
                    )}
                  </th>
                );
              })}
            </tr>
          </thead>
          <tbody>
            {modalities.map((mod) => (
              <tr key={mod} className="border-b border-slate-100 last:border-b-0 hover:bg-slate-50/50">
                <td className="py-2 px-3.5 text-left font-bold text-xs text-slate-800 sticky left-0 bg-white z-10 border-r border-[#E2E8F0] shadow-2xs whitespace-nowrap">
                  <div className="flex items-center gap-2">
                    <span className={`w-2.5 h-2.5 rounded-full shrink-0 ${mod === 'XRD' ? 'bg-[#DC2626]' : 'bg-[#7C3AED]'}`} />
                    <span className="text-slate-900">{mod}</span>
                    <span className="text-3xs font-medium text-slate-500 bg-slate-100 px-1.5 py-0.5 rounded border border-slate-200">
                      cost {modalityCosts[mod] !== undefined ? modalityCosts[mod]?.toFixed(1) : 'N/A'}
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
                    <td key={cId} className="p-1 min-w-[44px]">
                      <button
                        disabled={!isFeasible}
                        onClick={() => isFeasible && onSelectAction(cId, mod)}
                        title={isFeasible ? `${cId} — ${mod}: ${val !== null ? val.toFixed(4) : 'N/A'}` : `${cId} — ${mod}: Infeasible`}
                        className={`w-full h-11 py-1 px-1 rounded-lg border text-xs font-semibold tabular-nums transition-all relative flex flex-col items-center justify-center ${
                          isFeasible ? 'cursor-pointer hover:shadow-sm' : 'cursor-not-allowed opacity-40'
                        } ${getCellBg(
                          val,
                          isWinner,
                          isSelected,
                          isFeasible
                        )}`}
                      >
                        {isWinner && (
                          <div className="absolute -top-1 -right-1 bg-[#DC2626] text-white rounded-full p-0.5 shadow-2xs z-10">
                            <Award className="w-2.5 h-2.5" />
                          </div>
                        )}
                        <span>{val !== null ? (val >= 0 && activeMetric === 'composite' ? `+${val.toFixed(2)}` : val.toFixed(2)) : '—'}</span>
                        {isSelected && (
                          <span className={`text-3xs tracking-wider uppercase font-extrabold leading-none mt-0.5 ${
                            (activeMetric === 'composite' && (val ?? 0) >= 0.10) || (activeMetric !== 'composite' && (val ?? 0) >= 0.5)
                              ? 'text-white'
                              : 'text-[#DC2626]'
                          }`}>
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

      {/* Clean, spacious legend bar */}
      <div className="pt-2 flex flex-col sm:flex-row items-center justify-between text-xs text-slate-600 gap-2 font-medium">
        <div className="flex items-center gap-3 flex-wrap">
          <div className="flex items-center gap-1.5">
            <span className="w-3 h-3 rounded bg-[#DC2626] border border-[#B91C1C] inline-block shadow-2xs" />
            <span>More preferred under this metric</span>
          </div>
          <div className="flex items-center gap-1.5">
            <span className="w-3 h-3 rounded bg-slate-100 border border-slate-300 inline-block" />
            <span>Less preferred / penalized</span>
          </div>
          <div className="flex items-center gap-1.5 text-[#991B1B] font-semibold">
            <div className="w-3.5 h-3.5 rounded-full bg-[#DC2626] text-white flex items-center justify-center text-3xs">
              <Award className="w-2.5 h-2.5" />
            </div>
            <span>Recorded selected action</span>
          </div>
          <div className="flex items-center gap-1.5 text-slate-900 font-semibold">
            <CheckCircle2 className="w-3.5 h-3.5 text-slate-900" />
            <span>Currently Selected</span>
          </div>
        </div>

        <div className="text-xs text-slate-500">
          Showing: <strong className="text-slate-800">{metricLabels[activeMetric].title}</strong> ({metricLabels[activeMetric].unit})
        </div>
      </div>
    </div>
  );
};
