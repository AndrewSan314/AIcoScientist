import React from 'react';
import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ReferenceLine
} from 'recharts';
import { FlagshipCampaign, ReplayCampaign, RevealPhase } from '../../types/mission_control';

interface Props {
  campaign?: FlagshipCampaign | ReplayCampaign | null;
  currentStepIndex: number;
  revealPhase: RevealPhase;
  onSelectStep?: (step: number) => void;
}

// Calculate Shannon entropy in nats: H(p) = -sum(p * ln(p))
function shannonEntropyNats(probs: number[]): number {
  return probs.reduce((acc, p) => (p > 0 ? acc - p * Math.log(p) : acc), 0);
}

export const HypothesisBeliefTrajectoryChart: React.FC<Props> = ({
  campaign,
  currentStepIndex,
  revealPhase,
  onSelectStep
}) => {
  const steps = campaign?.steps || [];
  const initBeliefs = campaign?.initial_beliefs || {};

  const initH1 = initBeliefs['H1_PHASE_PURITY_LIMITED'];
  const initH2 = initBeliefs['H2_COMPOSITION_HOMOGENEITY_LIMITED'];
  const initH3 = initBeliefs['H3_MORPHOLOGY_KINETICS_LIMITED'];
  if (![initH1, initH2, initH3].every((value) => typeof value === 'number' && Number.isFinite(value))) {
    return <div className="h-72 flex items-center justify-center text-sm text-[#66706C]">Belief trajectory unavailable for this source run.</div>;
  }
  const initEntropy = shannonEntropyNats([initH1, initH2, initH3]);

  const chartPoints = [
    {
      step: 0,
      label: 'Prior',
      action: 'Prior State',
      H1: parseFloat(initH1.toFixed(4)),
      H2: parseFloat(initH2.toFixed(4)),
      H3: parseFloat(initH3.toFixed(4)),
      entropy: parseFloat(initEntropy.toFixed(4))
    }
  ];

  steps.forEach((s) => {
    const stepNum = s.step;
    const actionType = s.preregistration?.action?.action_type || 'Not recorded';
    const cand = s.preregistration?.action?.candidate_id || 'Not recorded';
    const bu = s.belief_update?.beliefs_after;
    const isPast = stepNum < currentStepIndex;
    const isCurrent = stepNum === currentStepIndex;

    let h1Val: number | null = null;
    let h2Val: number | null = null;
    let h3Val: number | null = null;

    if (isPast) {
      h1Val = bu?.['H1_PHASE_PURITY_LIMITED'] ?? null;
      h2Val = bu?.['H2_COMPOSITION_HOMOGENEITY_LIMITED'] ?? null;
      h3Val = bu?.['H3_MORPHOLOGY_KINETICS_LIMITED'] ?? null;
    } else if (isCurrent) {
      if (revealPhase === 'D_UPDATED') {
        h1Val = bu?.['H1_PHASE_PURITY_LIMITED'] ?? null;
        h2Val = bu?.['H2_COMPOSITION_HOMOGENEITY_LIMITED'] ?? null;
        h3Val = bu?.['H3_MORPHOLOGY_KINETICS_LIMITED'] ?? null;
      } else {
        const prevPoint = chartPoints[chartPoints.length - 1];
        h1Val = prevPoint.H1;
        h2Val = prevPoint.H2;
        h3Val = prevPoint.H3;
      }
    }

    if (h1Val !== null && h2Val !== null && h3Val !== null) {
      chartPoints.push({
        step: stepNum,
        label: `Step ${stepNum}`,
        action: `${actionType} (${cand})`,
        H1: parseFloat(h1Val.toFixed(4)),
        H2: parseFloat(h2Val.toFixed(4)),
        H3: parseFloat(h3Val.toFixed(4)),
        entropy: parseFloat(shannonEntropyNats([h1Val, h2Val, h3Val]).toFixed(4))
      });
    }
  });

  return (
    <div className="w-full flex flex-col">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2.5 mb-3">
        <div>
          <div className="flex items-center gap-2 flex-wrap">
            <h4 className="text-xs sm:text-sm font-bold text-[#17201F] tracking-tight whitespace-nowrap">
              Hypothesis Belief Trajectory
            </h4>
            <span className="text-3xs font-mono px-2 py-0.5 rounded-full bg-[#FEF2F2] text-[#991B1B] border border-[#FECACA] font-semibold whitespace-nowrap">
              P(H | e₁:t)
            </span>
          </div>
          <p className="text-2xs text-[#66706C] mt-0.5">
            Sequential Bayesian posterior probability shifts over characterization measurements
          </p>
        </div>

        <div className="flex items-center gap-1.5 text-2xs font-mono shrink-0">
          <span className="text-[#8F9995] whitespace-nowrap">Step:</span>
          {steps.map((s) => (
            <button
              key={s.step}
              onClick={() => onSelectStep?.(s.step)}
              className={`px-2 py-0.5 rounded font-bold transition-colors cursor-pointer text-2xs ${
                currentStepIndex === s.step
                  ? 'bg-[#B91C1C] text-white shadow-xs'
                  : 'bg-[#F4F3EE] text-[#66706C] hover:bg-[#D9DFDB]'
              }`}
            >
              Step {s.step}
            </button>
          ))}
        </div>
      </div>

      <div className="w-full h-[300px]">
        <ResponsiveContainer width="100%" height={300}>
          <LineChart data={chartPoints} margin={{ top: 12, right: 30, left: 0, bottom: 24 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#F1F5F9" />
            <XAxis
              dataKey="label"
              tick={{ fill: '#334155', fontSize: 12, fontWeight: 500, fontFamily: "'Montserrat', Arial, sans-serif" }}
              tickLine={false}
              axisLine={{ stroke: '#D9DFDB' }}
            />
            <YAxis
              domain={[0, 1]}
              ticks={[0, 0.25, 0.5, 0.75, 1.0]}
              tickFormatter={(v) => `${(v * 100).toFixed(0)}%`}
              tick={{ fill: '#475569', fontSize: 12, fontWeight: 500, fontFamily: "'Montserrat', Arial, sans-serif" }}
              tickLine={false}
              axisLine={{ stroke: '#D9DFDB' }}
              label={{
                value: 'Posterior Belief P(H)',
                angle: -90,
                position: 'insideLeft',
                offset: 15,
                fill: '#17201F',
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
              formatter={(value: any, name: any) => {
                const labelMap: Record<string, string> = {
                  H1: 'H₁ Phase Purity',
                  H2: 'H₂ Homogeneity',
                  H3: 'H₃ Kinetics'
                };
                return [`${(Number(value) * 100).toFixed(1)}%`, labelMap[String(name)] || name];
              }}
              labelFormatter={(label, payload) => {
                if (payload?.[0]?.payload?.action) {
                  return `${label} — ${payload[0].payload.action}`;
                }
                return label;
              }}
            />
            <Legend
              verticalAlign="top"
              height={36}
              formatter={(value) => {
                const names: Record<string, string> = {
                  H1: 'H₁ Phase Purity (Crimson)',
                  H2: 'H₂ Homogeneity (Amber)',
                  H3: 'H₃ Kinetics (Violet)'
                };
                return <span className="text-xs font-semibold text-[#17201F]">{names[value] || value}</span>;
              }}
            />
            <ReferenceLine
              x={`Step ${currentStepIndex}`}
              stroke="#DC2626"
              strokeDasharray="4 4"
              label={{
                value: 'Current Step',
                position: 'insideTopRight',
                fill: '#DC2626',
                fontSize: 10,
                fontWeight: 700
              }}
            />
            <Line
              type="monotone"
              dataKey="H1"
              name="H1"
              stroke="#DC2626"
              strokeWidth={3}
              dot={{ r: 5, fill: '#DC2626', stroke: '#ffffff', strokeWidth: 2 }}
              activeDot={{ r: 7 }}
            />
            <Line
              type="monotone"
              dataKey="H2"
              name="H2"
              stroke="#D97706"
              strokeWidth={2.5}
              dot={{ r: 5, fill: '#D97706', stroke: '#ffffff', strokeWidth: 2 }}
              activeDot={{ r: 7 }}
            />
            <Line
              type="monotone"
              dataKey="H3"
              name="H3"
              stroke="#7C3AED"
              strokeWidth={2.5}
              dot={{ r: 5, fill: '#7C3AED', stroke: '#ffffff', strokeWidth: 2 }}
              activeDot={{ r: 7 }}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>

      <div className="mt-3 pt-2.5 border-t border-[#D9DFDB] flex flex-col sm:flex-row sm:items-center justify-between gap-1.5 text-2xs font-mono text-[#66706C]">
        <div>
          Current Model Weights:{' '}
          <span className="font-bold text-[#DC2626]">
            H₁: {(chartPoints[chartPoints.length - 1].H1 * 100).toFixed(1)}%
          </span>{' '}
          |{' '}
          <span className="font-bold text-[#D97706]">
            H₂: {(chartPoints[chartPoints.length - 1].H2 * 100).toFixed(1)}%
          </span>{' '}
          |{' '}
          <span className="font-bold text-[#7C3AED]">
            H₃: {(chartPoints[chartPoints.length - 1].H3 * 100).toFixed(1)}%
          </span>
        </div>
        <div className="text-[#8F9995] whitespace-nowrap">
          Hypothesis Space: Source-registered competing models
        </div>
      </div>
    </div>
  );
};
