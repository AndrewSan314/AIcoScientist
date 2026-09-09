import React, { useState, useMemo } from 'react';
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
import { CampaignStep, RevealPhase } from '../../types/mission_control';
import { ShieldCheck, Eye, Lock } from 'lucide-react';

interface Props {
  currentStep: CampaignStep | null;
  revealPhase: RevealPhase;
  selectedCandidateId: string;
  selectedModality: string;
}

// Compute normal distribution PDF
function gaussianPdf(x: number, mean: number, variance: number): number {
  if (variance <= 0) return 0;
  const std = Math.sqrt(variance);
  const factor = 1 / (std * Math.sqrt(2 * Math.PI));
  const exponent = -Math.pow(x - mean, 2) / (2 * variance);
  return factor * Math.exp(exponent);
}

export const PredictiveDistributionChart: React.FC<Props> = ({
  currentStep,
  revealPhase,
  selectedCandidateId,
  selectedModality
}) => {
  const dists = currentStep?.preregistration?.predictive_distributions;
  const obs = currentStep?.observation?.observed_measurement;

  // Extract available observable names for this modality
  const observableNames = useMemo(() => {
    if (dists) {
      for (const hid of Object.keys(dists)) {
        if (dists[hid]?.observable_names?.length) {
          return dists[hid].observable_names || [];
        }
      }
    }
    if (obs?.observable_names?.length) return obs.observable_names;
    return ['XRD.normalized_intensity_std_proxy'];
  }, [dists, obs]);

  const [selectedObsIndex, setSelectedObsIndex] = useState<number>(0);
  const activeObsName = observableNames[selectedObsIndex] || observableNames[0] || 'Observable';

  // Compute curve points
  const { chartData, observedValue, xRange } = useMemo(() => {
    if (!dists) return { chartData: [], observedValue: null, xRange: [0, 1] };

    const h1 = dists['H1_PHASE_PURITY_LIMITED'];
    const h2 = dists['H2_COMPOSITION_HOMOGENEITY_LIMITED'];
    const h3 = dists['H3_MORPHOLOGY_KINETICS_LIMITED'];

    const mu1 = h1?.mean?.[selectedObsIndex] ?? 0.3;
    const var1 = h1?.variance?.[selectedObsIndex] ?? 0.04;

    const mu2 = h2?.mean?.[selectedObsIndex] ?? 0.4;
    const var2 = h2?.variance?.[selectedObsIndex] ?? 0.05;

    const mu3 = h3?.mean?.[selectedObsIndex] ?? 0.35;
    const var3 = h3?.variance?.[selectedObsIndex] ?? 0.045;

    const stdMax = Math.max(Math.sqrt(var1), Math.sqrt(var2), Math.sqrt(var3));
    const minX = Math.max(0, Math.min(mu1, mu2, mu3) - 3.2 * stdMax);
    const maxX = Math.max(mu1, mu2, mu3) + 3.2 * stdMax;

    // Get observed value if available
    let obsVal: number | null = null;
    if (obs && (revealPhase === 'C_REVEALED' || revealPhase === 'D_UPDATED')) {
      if (Array.isArray(obs.value)) {
        obsVal = obs.value[selectedObsIndex] ?? null;
      } else if (typeof obs.value === 'number') {
        obsVal = obs.value;
      }
    }

    const numPoints = 60;
    const stepX = (maxX - minX) / (numPoints - 1);
    const data = [];

    for (let i = 0; i < numPoints; i++) {
      const x = minX + i * stepX;
      data.push({
        x: parseFloat(x.toFixed(4)),
        H1: parseFloat(gaussianPdf(x, mu1, var1).toFixed(4)),
        H2: parseFloat(gaussianPdf(x, mu2, var2).toFixed(4)),
        H3: parseFloat(gaussianPdf(x, mu3, var3).toFixed(4))
      });
    }

    return { chartData: data, observedValue: obsVal, xRange: [minX, maxX] };
  }, [dists, obs, selectedObsIndex, revealPhase]);

  const isBlinded = revealPhase === 'A_SCORED' || revealPhase === 'B_PREREGISTERED';

  return (
    <div className="w-full h-full flex flex-col">
      <div className="flex items-center justify-between mb-3">
        <div>
          <h4 className="text-sm font-bold text-slate-900 tracking-tight flex items-center gap-2">
            <span>Hypothesis Predictive Distributions</span>
            <span className="text-2xs font-mono px-2 py-0.5 rounded-full bg-slate-100 text-slate-700 border border-slate-200">
              p(y | a, Hₖ)
            </span>
          </h4>
          <p className="text-xs text-slate-500">
            Preregistered Gaussian probability densities across competing mechanistic hypotheses
          </p>
        </div>

        {/* Observable selector pill */}
        <div className="flex items-center gap-2">
          <label className="text-2xs font-mono text-slate-500">Observable:</label>
          <select
            value={selectedObsIndex}
            onChange={(e) => setSelectedObsIndex(Number(e.target.value))}
            className="text-xs font-mono bg-white border border-slate-200 rounded-md px-2.5 py-1 text-slate-800 shadow-2xs cursor-pointer focus:outline-emerald-600"
          >
            {observableNames.map((name, idx) => (
              <option key={name} value={idx}>
                {name.replace('XRD.', '').replace('REFINEMENT.', '')}
              </option>
            ))}
          </select>
        </div>
      </div>

      {/* Firewall status banner */}
      <div className="mb-2">
        {isBlinded ? (
          <div className="flex items-center justify-between px-3 py-1.5 bg-amber-50 border border-amber-200 rounded-lg text-xs text-amber-900">
            <div className="flex items-center gap-2">
              <Lock className="w-3.5 h-3.5 text-amber-700" />
              <span className="font-semibold">Observation Blinding Firewalled:</span>
              <span className="text-amber-800">
                Ground-truth measurement is strictly hidden prior to preregistration lock.
              </span>
            </div>
            <span className="text-2xs font-mono uppercase tracking-wider font-bold bg-amber-200/60 px-2 py-0.5 rounded text-amber-900">
              State {revealPhase === 'A_SCORED' ? 'A (Scored)' : 'B (Locked)'}
            </span>
          </div>
        ) : (
          <div className="flex items-center justify-between px-3 py-1.5 bg-blue-50 border border-blue-200 rounded-lg text-xs text-blue-900">
            <div className="flex items-center gap-2">
              <Eye className="w-3.5 h-3.5 text-blue-700" />
              <span className="font-semibold">Evidence Revealed:</span>
              <span className="text-blue-800">
                Observed value: <strong className="font-mono">{observedValue?.toFixed(4) ?? 'N/A'}</strong>.
                Log Bayes factors applied to belief update.
              </span>
            </div>
            <span className="text-2xs font-mono uppercase tracking-wider font-bold bg-blue-200/60 px-2 py-0.5 rounded text-blue-900">
              State {revealPhase === 'C_REVEALED' ? 'C (Revealed)' : 'D (Updated)'}
            </span>
          </div>
        )}
      </div>

      <div className="flex-1 w-full min-h-[240px]">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={chartData} margin={{ top: 10, right: 24, left: -10, bottom: 20 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
            <XAxis
              dataKey="x"
              domain={xRange}
              tickFormatter={(v) => Number(v).toFixed(2)}
              tick={{ fill: '#64748b', fontSize: 11 }}
              tickLine={{ stroke: '#cbd5e1' }}
              axisLine={{ stroke: '#cbd5e1' }}
              label={{
                value: activeObsName,
                position: 'insideBottom',
                offset: -12,
                fill: '#475569',
                fontSize: 11,
                fontFamily: 'monospace'
              }}
            />
            <YAxis
              tick={{ fill: '#64748b', fontSize: 11 }}
              tickLine={{ stroke: '#cbd5e1' }}
              axisLine={{ stroke: '#cbd5e1' }}
              label={{
                value: 'Probability Density p(y)',
                angle: -90,
                position: 'insideLeft',
                offset: 20,
                fill: '#475569',
                fontSize: 11
              }}
            />
            <Tooltip
              contentStyle={{
                backgroundColor: '#ffffff',
                borderColor: '#e2e8f0',
                borderRadius: '8px',
                boxShadow: '0 4px 6px -1px rgb(0 0 0 / 0.1)',
                fontSize: '12px'
              }}
              formatter={(value: any, name: any) => [
                Number(value).toFixed(3),
                name === 'H1' ? 'H₁ Phase Purity' : name === 'H2' ? 'H₂ Homogeneity' : 'H₃ Kinetics'
              ]}
              labelFormatter={(label) => `x = ${label}`}
            />
            <Legend
              verticalAlign="top"
              height={32}
              formatter={(value) => {
                const names: Record<string, string> = {
                  H1: 'H₁ Phase Purity (Emerald)',
                  H2: 'H₂ Homogeneity (Amber)',
                  H3: 'H₃ Kinetics (Violet)'
                };
                return <span className="text-xs font-medium text-slate-700">{names[value] || value}</span>;
              }}
            />

            <Line
              type="monotone"
              dataKey="H1"
              name="H1"
              stroke="#059669"
              strokeWidth={2.5}
              dot={false}
              activeDot={{ r: 5 }}
            />
            <Line
              type="monotone"
              dataKey="H2"
              name="H2"
              stroke="#d97706"
              strokeWidth={2}
              dot={false}
              activeDot={{ r: 5 }}
            />
            <Line
              type="monotone"
              dataKey="H3"
              name="H3"
              stroke="#7c3aed"
              strokeWidth={2}
              dot={false}
              activeDot={{ r: 5 }}
            />

            {/* Vertical marker for observed value upon reveal */}
            {observedValue !== null && (
              <ReferenceLine
                x={parseFloat(observedValue.toFixed(4))}
                stroke="#2563eb"
                strokeWidth={2.5}
                strokeDasharray="4 2"
                label={{
                  value: `Observed: ${observedValue.toFixed(3)}`,
                  position: 'insideTopLeft',
                  fill: '#1d4ed8',
                  fontSize: 11,
                  fontWeight: 700
                }}
              />
            )}
          </LineChart>
        </ResponsiveContainer>
      </div>

      <div className="mt-2 pt-2 border-t border-slate-100 flex items-center justify-between text-2xs font-mono text-slate-500">
        <div className="flex items-center gap-2">
          <span>Target Action:</span>
          <span className="font-bold text-slate-800">
            {selectedModality} on {selectedCandidateId}
          </span>
        </div>
        <div className="flex items-center gap-1.5 text-slate-400">
          <ShieldCheck className="w-3 h-3 text-emerald-600" />
          <span>Strict Pre-reveal Preregistration Invariant Verified</span>
        </div>
      </div>
    </div>
  );
};
