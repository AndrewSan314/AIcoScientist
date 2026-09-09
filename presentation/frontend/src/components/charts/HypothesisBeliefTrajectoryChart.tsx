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
import { SnapshotData, RevealPhase } from '../../types/mission_control';

interface Props {
  data: SnapshotData;
  currentStepIndex: number;
  revealPhase: RevealPhase;
  onSelectStep: (step: number) => void;
}

export const HypothesisBeliefTrajectoryChart: React.FC<Props> = ({
  data,
  currentStepIndex,
  revealPhase,
  onSelectStep
}) => {
  const steps = data.flagship_campaign?.steps || [];

  // Build sequential trajectory data points
  // Point 0: Step 0 (Prior beliefs 33.3% each)
  // Point 1: Step 1 posterior
  // Point 2: Step 2 posterior
  // Point 3: Step 3 posterior
  // Point 4: Step 4 posterior
  const chartPoints = [
    {
      step: 0,
      label: 'Prior',
      action: 'Prior State',
      H1: 0.3333,
      H2: 0.3333,
      H3: 0.3333,
      entropy: 1.0986
    }
  ];

  steps.forEach((s) => {
    const stepNum = s.step;
    const actionType = s.preregistration?.action?.action_type || 'XRD';
    const cand = s.preregistration?.action?.candidate_id || '';
    const bu = s.belief_update?.beliefs_after;
    const isPast = stepNum < currentStepIndex;
    const isCurrent = stepNum === currentStepIndex;

    // Determine what beliefs to show based on revealPhase
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
        // Before State D, show prior of this step
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
        entropy: parseFloat((s.observation?.realized_entropy_reduction_nats ?? 0).toFixed(4))
      });
    }
  });

  return (
    <div className="w-full h-full flex flex-col">
      <div className="flex items-center justify-between mb-3">
        <div>
          <h4 className="text-sm font-bold text-slate-900 tracking-tight flex items-center gap-2">
            <span>Hypothesis Belief Trajectory</span>
            <span className="text-2xs font-mono px-2 py-0.5 rounded-full bg-emerald-50 text-emerald-800 border border-emerald-200">
              P(H | e₁:t)
            </span>
          </h4>
          <p className="text-xs text-slate-500">
            Sequential Bayesian posterior probability shifts over characterization measurements
          </p>
        </div>
        <div className="flex items-center gap-1.5 text-xs font-mono">
          <span className="text-slate-400">Step:</span>
          {steps.map((s) => (
            <button
              key={s.step}
              onClick={() => onSelectStep(s.step)}
              className={`px-2.5 py-1 rounded-md font-bold transition-colors cursor-pointer text-xs ${
                currentStepIndex === s.step
                  ? 'bg-emerald-600 text-white shadow-xs'
                  : 'bg-slate-100 text-slate-600 hover:bg-slate-200'
              }`}
            >
              Step {s.step}
            </button>
          ))}
        </div>
      </div>

      <div className="flex-1 w-full min-h-[260px]">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={chartPoints} margin={{ top: 12, right: 24, left: -10, bottom: 20 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
            <XAxis
              dataKey="label"
              tick={{ fill: '#64748b', fontSize: 11 }}
              tickLine={{ stroke: '#cbd5e1' }}
              axisLine={{ stroke: '#cbd5e1' }}
            />
            <YAxis
              domain={[0, 1]}
              ticks={[0, 0.25, 0.5, 0.75, 1.0]}
              tickFormatter={(v) => `${(v * 100).toFixed(0)}%`}
              tick={{ fill: '#64748b', fontSize: 11 }}
              tickLine={{ stroke: '#cbd5e1' }}
              axisLine={{ stroke: '#cbd5e1' }}
            />
            <Tooltip
              contentStyle={{
                backgroundColor: '#ffffff',
                borderColor: '#e2e8f0',
                borderRadius: '8px',
                boxShadow: '0 4px 6px -1px rgb(0 0 0 / 0.1)',
                fontSize: '12px'
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
                  H1: 'H₁ Phase Purity Limited (Emerald)',
                  H2: 'H₂ Homogeneity Limited (Amber)',
                  H3: 'H₃ Morphology Kinetics (Violet)'
                };
                return <span className="text-xs font-medium text-slate-700">{names[value] || value}</span>;
              }}
            />
            <ReferenceLine
              x={`Step ${currentStepIndex}`}
              stroke="#059669"
              strokeDasharray="4 4"
              label={{
                value: 'Current Step',
                position: 'insideTopRight',
                fill: '#059669',
                fontSize: 10,
                fontWeight: 700
              }}
            />
            <Line
              type="monotone"
              dataKey="H1"
              name="H1"
              stroke="#059669"
              strokeWidth={3}
              dot={{ r: 5, fill: '#059669', stroke: '#ffffff', strokeWidth: 2 }}
              activeDot={{ r: 7 }}
            />
            <Line
              type="monotone"
              dataKey="H2"
              name="H2"
              stroke="#d97706"
              strokeWidth={2.5}
              dot={{ r: 5, fill: '#d97706', stroke: '#ffffff', strokeWidth: 2 }}
              activeDot={{ r: 7 }}
            />
            <Line
              type="monotone"
              dataKey="H3"
              name="H3"
              stroke="#7c3aed"
              strokeWidth={2.5}
              dot={{ r: 5, fill: '#7c3aed', stroke: '#ffffff', strokeWidth: 2 }}
              activeDot={{ r: 7 }}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>

      <div className="mt-2 pt-2 border-t border-slate-100 flex items-center justify-between text-2xs font-mono text-slate-500">
        <div>
          Current Model Weights:{' '}
          <span className="font-bold text-emerald-700">
            H₁: {(chartPoints[chartPoints.length - 1].H1 * 100).toFixed(1)}%
          </span>{' '}
          |{' '}
          <span className="font-bold text-amber-700">
            H₂: {(chartPoints[chartPoints.length - 1].H2 * 100).toFixed(1)}%
          </span>{' '}
          |{' '}
          <span className="font-bold text-violet-700">
            H₃: {(chartPoints[chartPoints.length - 1].H3 * 100).toFixed(1)}%
          </span>
        </div>
        <div className="text-slate-400">
          Hypothesis Space: 3 Exhaustive Mutually Exclusive Models
        </div>
      </div>
    </div>
  );
};
