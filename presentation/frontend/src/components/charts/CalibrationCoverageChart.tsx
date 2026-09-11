import React, { useState } from 'react';
import {
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ReferenceLine,
  Cell
} from 'recharts';
import { CalibrationData } from '../../types/mission_control';
import { Target, AlertTriangle } from 'lucide-react';

interface Props {
  calibration: CalibrationData;
}

const FRIENDLY_NAMES: Record<string, string> = {
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

export const CalibrationCoverageChart: React.FC<Props> = ({ calibration }) => {
  const [activeModality, setActiveModality] = useState<'XRD' | 'REFINEMENT'>('XRD');

  const modData = calibration[activeModality] || {};
  const obsKeys = Object.keys(modData);

  const chartData = obsKeys.map((k) => {
    const item = modData[k];
    const cov50 = item.coverage50;
    const cov90 = item.coverage90;
    const rawShortName = k.replace('XRD.', '').replace('REFINEMENT.', '');
    const cleanName = FRIENDLY_NAMES[rawShortName] || rawShortName;
    const isOverdispersed = activeModality === 'REFINEMENT' && k.includes('target_phase_fraction');

    return {
      fullName: k,
      rawKey: rawShortName,
      name: cleanName,
      coverage50: typeof cov50 === 'number' ? parseFloat((cov50 * 100).toFixed(1)) : null,
      coverage90: typeof cov90 === 'number' ? parseFloat((cov90 * 100).toFixed(1)) : null,
      mae: item.MAE,
      isOverdispersed
    };
  });

  return (
    <div className="w-full flex flex-col">
      {/* Chart Header Controls */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 mb-3">
        <div>
          <div className="flex items-center gap-2">
            <Target className="w-4 h-4 text-[#DC2626]" />
            <h4 className="text-sm font-bold text-[#17201F] tracking-tight">
              A-Lab Predictive Calibration Coverage
            </h4>
            <span className="text-xs font-semibold px-2.5 py-0.5 rounded-full bg-[#FEF2F2] text-[#991B1B] border border-[#FECACA]">
              50% & 90% Empirical Intervals
            </span>
          </div>
          <p className="text-xs text-[#66706C] mt-0.5">
            Empirical proportion of ground-truth observations falling inside model predictive credible intervals
          </p>
        </div>

        {/* Modality toggle */}
        <div className="flex items-center gap-1.5 bg-[#F4F3EE] p-1 rounded-lg border border-[#D9DFDB] text-xs shrink-0">
          <button
            onClick={() => setActiveModality('XRD')}
            className={`px-3 py-1 rounded-md transition-all cursor-pointer font-semibold ${
              activeModality === 'XRD'
                ? 'bg-white text-[#DC2626] shadow-2xs border border-[#FECACA]'
                : 'text-[#66706C] hover:text-[#17201F]'
            }`}
          >
            XRD Descriptors
          </button>
          <button
            onClick={() => setActiveModality('REFINEMENT')}
            className={`px-3 py-1 rounded-md transition-all cursor-pointer font-semibold ${
              activeModality === 'REFINEMENT'
                ? 'bg-white text-[#7C3AED] shadow-2xs border border-[#DDD6FE]'
                : 'text-[#66706C] hover:text-[#17201F]'
            }`}
          >
            Rietveld Refinement
          </button>
        </div>
      </div>

      {activeModality === 'REFINEMENT' && (
        <div className="mb-3 flex items-center justify-between px-3.5 py-2.5 bg-amber-50 border border-amber-200 rounded-lg text-xs text-amber-900">
          <div className="flex items-center gap-2">
            <AlertTriangle className="w-4 h-4 text-amber-600 shrink-0" />
            <span>
              <strong>A_LAB_CALIBRATION_PARTIAL:</strong> Target phase fraction 50% interval covers{' '}
              <strong>{chartData.find((item) => item.rawKey === 'target_phase_fraction')?.coverage50 ?? 'Not recorded'}%</strong> of linked samples (source calibration behavior; interpretation remains bounded by the artifact).
            </span>
          </div>
          <span className="text-2xs font-mono bg-amber-200/60 px-2 py-0.5 rounded font-bold">FAIL GATE</span>
        </div>
      )}

      <div className="w-full h-72">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={chartData} margin={{ top: 16, right: 80, left: 0, bottom: 24 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#F1F5F9" vertical={false} />
            <XAxis
              dataKey="name"
              tick={{ fill: '#334155', fontSize: 12, fontWeight: 500, fontFamily: "'Montserrat', Arial, sans-serif" }}
              tickLine={false}
              axisLine={{ stroke: '#D9DFDB' }}
            />
            <YAxis
              domain={[0, 100]}
              ticks={[0, 25, 50, 75, 90, 100]}
              tickFormatter={(v) => `${v}%`}
              tick={{ fill: '#475569', fontSize: 12, fontWeight: 500, fontFamily: "'Montserrat', Arial, sans-serif" }}
              tickLine={false}
              axisLine={{ stroke: '#D9DFDB' }}
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
              formatter={(value: any, name: any) => [
                `${Number(value).toFixed(1)}%`,
                name === 'coverage50' ? '50% Interval Coverage' : '90% Interval Coverage'
              ]}
              labelFormatter={(label, payload) => {
                const item = payload?.[0]?.payload;
                return item ? `${item.name} (${item.rawKey})` : label;
              }}
            />
            <Legend
              verticalAlign="top"
              height={32}
              formatter={(value) => (
                <span className="text-xs font-semibold text-[#17201F]">
                  {value === 'coverage50' ? '50% Interval Coverage (Target: 50%)' : '90% Interval Coverage (Target: 90%)'}
                </span>
              )}
            />

            {/* Target reference lines placed insideTopRight to prevent overlapping bars */}
            <ReferenceLine
              y={50}
              stroke="#B91C1C"
              strokeDasharray="4 4"
              strokeWidth={1.5}
              label={{
                value: 'Target 50%',
                position: 'right',
                fill: '#B91C1C',
                fontSize: 12,
                fontWeight: 600,
                
              }}
            />
            <ReferenceLine
              y={90}
              stroke="#2563EB"
              strokeDasharray="4 4"
              strokeWidth={1.5}
              label={{
                value: 'Target 90%',
                position: 'right',
                fill: '#2563EB',
                fontSize: 12,
                fontWeight: 600,
                
              }}
            />

            <Bar dataKey="coverage50" fill="#EF4444" radius={[4, 4, 0, 0]} maxBarSize={36}>
              {chartData.map((entry, index) => (
                <Cell
                  key={`cov50-${index}`}
                  fill={entry.isOverdispersed ? '#D97706' : '#B91C1C'}
                />
              ))}
            </Bar>
            <Bar dataKey="coverage90" fill="#3B82F6" radius={[4, 4, 0, 0]} maxBarSize={36} />
          </BarChart>
        </ResponsiveContainer>
      </div>

      <div className="mt-2 pt-2.5 border-t border-[#D9DFDB] flex flex-col sm:flex-row sm:items-center justify-between gap-2 text-xs text-[#66706C]">
        <div className="flex items-center gap-4">
          <span>Nominal 50% target: <strong>±0.674σ</strong></span>
          <span>Nominal 90% target: <strong>±1.645σ</strong></span>
        </div>
        <div className="text-[#8F9995]">
          Source: Zenodo DOI 10.5281/zenodo.21285546 (1,030 linked samples)
        </div>
      </div>
    </div>
  );
};
