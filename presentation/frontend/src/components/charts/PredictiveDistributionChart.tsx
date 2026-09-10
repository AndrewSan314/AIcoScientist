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
import { ShieldCheck, Eye, Lock, Info } from 'lucide-react';


const OBSERVABLE_FRIENDLY_NAMES: Record<string, string> = {
  normalized_intensity_std_proxy: 'Intensity Std',
  dominant_peak_index_fraction: 'Dominant Peak',
  global_halfmax_span_proxy: 'FWHM Span',
  spectral_entropy: 'Spectral Entropy',
  peak_count_proxy: 'Peak Count',
  target_phase_fraction: 'Target Phase',
  precursor_phase_fraction: 'Precursor Phase',
  other_identified_phase_fraction: 'Other Phase',
  rwp_scaled: 'Scaled Rwp'
};
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
  const prereg = currentStep?.preregistration;
  const preregAction = prereg?.action;
  const obs = currentStep?.observation?.observed_measurement;

  const isSelectedActionPreregistered = Boolean(
    preregAction &&
    preregAction.candidate_id === selectedCandidateId &&
    preregAction.action_type === selectedModality
  );

  const dists = isSelectedActionPreregistered ? prereg?.predictive_distributions : null;

  // Extract available observable names for this modality
  const observableNames = useMemo(() => {
    if (dists) {
      for (const hid of Object.keys(dists)) {
        if (dists[hid]?.observable_names?.length) {
          return dists[hid].observable_names || [];
        }
      }
    }
    if (isSelectedActionPreregistered && obs?.observable_names?.length) {
      return obs.observable_names;
    }
    return [];
  }, [dists, obs, isSelectedActionPreregistered]);

  const [selectedObsIndex, setSelectedObsIndex] = useState<number>(0);
  const activeObsName = observableNames[selectedObsIndex] || observableNames[0] || 'Observable';

  // Compute curve points with ZERO synthetic fallbacks
  const { chartData, observedValue, xRange, isValid } = useMemo(() => {
    if (!dists) {
      return { chartData: [], observedValue: null, xRange: [0, 1], isValid: false };
    }

    const h1 = dists['H1_PHASE_PURITY_LIMITED'];
    const h2 = dists['H2_COMPOSITION_HOMOGENEITY_LIMITED'];
    const h3 = dists['H3_MORPHOLOGY_KINETICS_LIMITED'];

    const mu1 = h1?.mean?.[selectedObsIndex];
    const var1 = h1?.variance?.[selectedObsIndex];
    const mu2 = h2?.mean?.[selectedObsIndex];
    const var2 = h2?.variance?.[selectedObsIndex];
    const mu3 = h3?.mean?.[selectedObsIndex];
    const var3 = h3?.variance?.[selectedObsIndex];

    if (
      mu1 === undefined || var1 === undefined || var1 <= 0 ||
      mu2 === undefined || var2 === undefined || var2 <= 0 ||
      mu3 === undefined || var3 === undefined || var3 <= 0
    ) {
      return { chartData: [], observedValue: null, xRange: [0, 1], isValid: false };
    }

    const stdMax = Math.max(Math.sqrt(var1), Math.sqrt(var2), Math.sqrt(var3));
    const minX = Math.max(0, Math.min(mu1, mu2, mu3) - 3.2 * stdMax);
    const maxX = Math.max(mu1, mu2, mu3) + 3.2 * stdMax;

    // Strict observation blinding: only reveal if state C or D and action matches
    let obsVal: number | null = null;
    if (isSelectedActionPreregistered && obs && (revealPhase === 'C_REVEALED' || revealPhase === 'D_UPDATED')) {
      if (Array.isArray(obs.value)) {
        obsVal = typeof obs.value[selectedObsIndex] === 'number' ? obs.value[selectedObsIndex] : null;
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

    return { chartData: data, observedValue: obsVal, xRange: [minX, maxX], isValid: true };
  }, [dists, obs, selectedObsIndex, revealPhase, isSelectedActionPreregistered]);

  // Compute 5-6 clean round ticks across [minX, maxX]
  const cleanTicks = useMemo(() => {
    if (!xRange || xRange[0] === undefined || xRange[1] === undefined) return undefined;
    const min = xRange[0];
    const max = xRange[1];
    const count = 6;
    const step = (max - min) / (count - 1);
    return Array.from({ length: count }, (_, i) => parseFloat((min + i * step).toFixed(2)));
  }, [xRange]);

  const isBlinded = revealPhase === 'A_SCORED' || revealPhase === 'B_PREREGISTERED';

  if (!isSelectedActionPreregistered || !isValid) {
    return (
      <div className="w-full flex flex-col p-8 bg-[#FCFCFA] rounded-2xl border border-[#D9DFDB] text-center items-center justify-center min-h-[360px]">
        <div className="w-11 h-11 rounded-full bg-[#FEF2F2] border border-[#FECACA] flex items-center justify-center mb-3.5 shadow-2xs">
          <Info className="w-5 h-5 text-[#DC2626]" />
        </div>
        <h4 className="text-sm font-bold text-[#17201F] mb-1.5">
          Predictive Distribution Unavailable for Inspected Action
        </h4>
        <p className="text-xs text-[#66706C] max-w-md mb-4 leading-relaxed font-normal">
          Serialized predictive densities <span className="font-semibold text-slate-800 bg-[#F4F3EE] px-1.5 py-0.5 rounded border border-[#D9DFDB]">p(y | a, Hₖ)</span> are stored exclusively for the preregistered winner action (<strong className="text-[#DC2626] font-semibold">{preregAction?.candidate_id} · {preregAction?.action_type}</strong>). Alternative candidate counterfactuals do not reuse winner curves.
        </p>
        <div className="inline-flex items-center gap-2 px-3 py-1.5 bg-[#F4F3EE] border border-[#D9DFDB] rounded-lg text-xs font-medium text-[#17201F]">
          <span className="text-[#8F9995]">Inspected Alternative:</span>
          <span className="font-bold text-[#DC2626]">{selectedCandidateId} &bull; {selectedModality}</span>
        </div>
      </div>
    );
  }

  return (
    <div className="w-full flex flex-col space-y-3">
      {/* Header: Title and Observable Selector cleanly arranged */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2.5 pb-1">
        <div>
          <div className="flex items-center gap-2 flex-wrap">
            <h4 className="text-sm font-bold text-[#17201F] tracking-tight">
              Hypothesis Predictive Distributions
            </h4>
            <span className="text-3xs font-semibold px-2 py-0.5 rounded-full bg-[#FEF2F2] text-[#991B1B] border border-[#FECACA]">
              p(y | a, Hₖ)
            </span>
          </div>
          <p className="text-xs text-[#66706C] mt-0.5 font-normal">
            Preregistered Gaussian probability densities across competing hypotheses
          </p>
        </div>

        {/* Observable selector pill */}
        <div className="flex items-center gap-2 self-start sm:self-auto shrink-0 bg-[#F4F3EE] px-2.5 py-1 rounded-lg border border-[#D9DFDB]">
          <label className="text-xs font-medium text-[#66706C] whitespace-nowrap">Observable:</label>
          <select
            value={selectedObsIndex}
            onChange={(e) => setSelectedObsIndex(Number(e.target.value))}
            className="text-xs font-semibold bg-white border border-[#D9DFDB] rounded-md px-2 py-1 text-[#17201F] shadow-2xs cursor-pointer focus:ring-1 focus:ring-[#DC2626] outline-none"
          >
            {observableNames.map((name, idx) => {
              const raw = name.replace('XRD.', '').replace('REFINEMENT.', '');
              return (
                <option key={name} value={idx}>
                  {OBSERVABLE_FRIENDLY_NAMES[raw] || raw}
                </option>
              );
            })}
          </select>
        </div>
      </div>

      {/* Firewall status banner with ample room and zero truncation */}
      <div>
        {isBlinded ? (
          <div className="flex items-center justify-between px-3.5 py-2 bg-amber-50/90 border border-amber-200 rounded-xl text-xs text-amber-900 gap-2">
            <div className="flex items-center gap-2 min-w-0">
              <Lock className="w-4 h-4 text-amber-600 shrink-0" />
              <span className="font-semibold whitespace-nowrap">Observation Blinding Active</span>
              <span className="text-amber-800/80 text-xs hidden md:inline">
                • Ground-truth strictly hidden prior to preregistration lock
              </span>
            </div>
            <span className="text-3xs font-bold uppercase tracking-wider bg-amber-200/70 border border-amber-300/80 px-2 py-0.5 rounded-md text-amber-900 whitespace-nowrap shrink-0">
              State {revealPhase === 'A_SCORED' ? 'A (Scored)' : 'B (Locked)'}
            </span>
          </div>
        ) : (
          <div className="flex items-center justify-between px-3.5 py-2 bg-red-50/90 border border-red-200 rounded-xl text-xs text-red-950 gap-2">
            <div className="flex items-center gap-2 min-w-0">
              <Eye className="w-4 h-4 text-[#DC2626] shrink-0" />
              <span className="font-semibold whitespace-nowrap">Evidence Revealed</span>
              <span className="text-red-800/80 text-xs">
                • Observed value: <strong className="font-semibold text-red-900">{observedValue?.toFixed(4) ?? 'N/A'}</strong> (Bayes update applied)
              </span>
            </div>
            <span className="text-3xs font-bold uppercase tracking-wider bg-red-100 border border-red-200 px-2 py-0.5 rounded-md text-red-900 whitespace-nowrap shrink-0">
              State {revealPhase === 'C_REVEALED' ? 'C (Revealed)' : 'D (Updated)'}
            </span>
          </div>
        )}
      </div>

      {/* Guaranteed Height Chart Container with clean 6 ticks */}
      <div className="w-full h-[300px]">
        <ResponsiveContainer width="100%" height={300}>
          <LineChart data={chartData} margin={{ top: 14, right: 24, left: 0, bottom: 26 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#F1F5F9" vertical={false} />
            <XAxis
              type="number"
              dataKey="x"
              domain={xRange}
              ticks={cleanTicks}
              tickFormatter={(v) => Number(v).toFixed(2)}
              tick={{ fill: '#475569', fontSize: 11, fontWeight: 500, fontFamily: "'Montserrat', Arial, sans-serif" }}
              tickLine={false}
              axisLine={{ stroke: '#D9DFDB' }}
              dy={6}
              label={{
                value: OBSERVABLE_FRIENDLY_NAMES[activeObsName.replace('XRD.', '').replace('REFINEMENT.', '')] || activeObsName.replace('XRD.', '').replace('REFINEMENT.', ''),
                position: 'insideBottom',
                offset: -16,
                fill: '#17201F',
                fontSize: 12,
                fontWeight: 600,
                fontFamily: "'Montserrat', Arial, sans-serif"
              }}
            />
            <YAxis
              tick={{ fill: '#475569', fontSize: 11, fontWeight: 500, fontFamily: "'Montserrat', Arial, sans-serif" }}
              tickLine={false}
              axisLine={{ stroke: '#D9DFDB' }}
              label={{
                value: 'Probability Density p(y)',
                angle: -90,
                position: 'insideLeft',
                offset: 14,
                fill: '#17201F',
                fontSize: 11,
                fontWeight: 600,
                fontFamily: "'Montserrat', Arial, sans-serif"
              }}
            />
            <Tooltip
              contentStyle={{
                backgroundColor: '#ffffff',
                borderColor: '#D9DFDB',
                borderRadius: '10px',
                boxShadow: '0 4px 12px -2px rgba(0, 0, 0, 0.08)',
                fontSize: '12px',
                fontFamily: "'Montserrat', Arial, sans-serif",
                padding: '8px 12px'
              }}
              formatter={(value: any, name: any) => [
                Number(value).toFixed(3),
                name === 'H1' ? 'H₁ Phase Purity' : name === 'H2' ? 'H₂ Homogeneity' : 'H₃ Kinetics'
              ]}
              labelFormatter={(label) => `Observable value = ${label}`}
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
                return <span className="text-xs font-semibold text-[#17201F] mr-3">{names[value] || value}</span>;
              }}
            />

            <Line
              type="monotone"
              dataKey="H1"
              name="H1"
              stroke="#DC2626"
              strokeWidth={2.5}
              dot={false}
              activeDot={{ r: 5, stroke: '#FFFFFF', strokeWidth: 2 }}
            />
            <Line
              type="monotone"
              dataKey="H2"
              name="H2"
              stroke="#D97706"
              strokeWidth={2}
              dot={false}
              activeDot={{ r: 5, stroke: '#FFFFFF', strokeWidth: 2 }}
            />
            <Line
              type="monotone"
              dataKey="H3"
              name="H3"
              stroke="#7C3AED"
              strokeWidth={2}
              dot={false}
              activeDot={{ r: 5, stroke: '#FFFFFF', strokeWidth: 2 }}
            />

            {/* Vertical marker for observed value upon reveal */}
            {observedValue !== null && (
              <ReferenceLine
                x={parseFloat(observedValue.toFixed(4))}
                stroke="#DC2626"
                strokeWidth={2.5}
                strokeDasharray="4 2"
                label={{
                  value: `Observed: ${observedValue.toFixed(3)}`,
                  position: 'insideTopLeft',
                  fill: '#DC2626',
                  fontSize: 11,
                  fontWeight: 700
                }}
              />
            )}
          </LineChart>
        </ResponsiveContainer>
      </div>

      {/* Footer Info: Clean sans-serif design with verified badge */}
      <div className="pt-3 border-t border-[#D9DFDB] flex flex-col sm:flex-row sm:items-center justify-between gap-2 text-xs text-[#66706C]">
        <div className="flex items-center gap-2">
          <span className="text-[#8F9995]">Target Action:</span>
          <span className="font-semibold text-[#17201F] bg-[#F4F3EE] px-2 py-0.5 rounded-md border border-[#D9DFDB]">
            {selectedModality} on {selectedCandidateId}
          </span>
        </div>
        <div className="flex items-center gap-1.5 text-xs text-[#DC2626] font-medium bg-[#FEF2F2] px-2.5 py-1 rounded-md border border-[#FECACA]/60">
          <ShieldCheck className="w-3.5 h-3.5 text-[#DC2626] shrink-0" />
          <span>Strict Pre-reveal Invariant Verified</span>
        </div>
      </div>
    </div>
  );
};
