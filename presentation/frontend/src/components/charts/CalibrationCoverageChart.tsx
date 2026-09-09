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

export const CalibrationCoverageChart: React.FC<Props> = ({ calibration }) => {
  const [activeModality, setActiveModality] = useState<'XRD' | 'REFINEMENT'>('XRD');

  const modData = calibration[activeModality] || {};
  const obsKeys = Object.keys(modData);

  const chartData = obsKeys.map((k) => {
    const item = modData[k];
    const cov50 = item.coverage50 ?? 0;
    const cov90 = item.coverage90 ?? 0;
    const shortName = k.replace('XRD.', '').replace('REFINEMENT.', '');
    const isOverdispersed = activeModality === 'REFINEMENT' && k.includes('target_phase_fraction');

    return {
      fullName: k,
      name: shortName,
      coverage50: parseFloat((cov50 * 100).toFixed(1)),
      coverage90: parseFloat((cov90 * 100).toFixed(1)),
      mae: item.MAE ?? 0,
      isOverdispersed
    };
  });

  return (
    <div className="w-full flex flex-col">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 mb-3">
        <div>
          <div className="flex items-center gap-2">
            <Target className="w-4 h-4 text-emerald-600" />
            <h4 className="text-sm font-bold text-slate-900 tracking-tight">
              A-Lab Predictive Calibration Coverage
            </h4>
            <span className="text-2xs font-mono px-2 py-0.5 rounded-full bg-slate-100 text-slate-700 border border-slate-200">
              50% & 90% Empirical Credible Intervals
            </span>
          </div>
          <p className="text-xs text-slate-500">
            Empirical proportion of true observations falling inside model predictive intervals
          </p>
        </div>

        {/* Modality toggle */}
        <div className="flex items-center gap-1 bg-slate-100 p-0.5 rounded-lg border border-slate-200 text-xs font-mono">
          <button
            onClick={() => setActiveModality('XRD')}
            className={`px-3 py-1 rounded-md transition-all cursor-pointer ${
              activeModality === 'XRD'
                ? 'bg-white text-emerald-800 font-bold shadow-2xs'
                : 'text-slate-600 hover:text-slate-900'
            }`}
          >
            XRD Descriptors
          </button>
          <button
            onClick={() => setActiveModality('REFINEMENT')}
            className={`px-3 py-1 rounded-md transition-all cursor-pointer ${
              activeModality === 'REFINEMENT'
                ? 'bg-white text-violet-800 font-bold shadow-2xs'
                : 'text-slate-600 hover:text-slate-900'
            }`}
          >
            Rietveld Refinement
          </button>
        </div>
      </div>

      {activeModality === 'REFINEMENT' && (
        <div className="mb-3 flex items-center justify-between px-3 py-2 bg-amber-50 border border-amber-200 rounded-lg text-xs text-amber-900">
          <div className="flex items-center gap-2">
            <AlertTriangle className="w-4 h-4 text-amber-600 shrink-0" />
            <span>
              <strong>A_LAB_CALIBRATION_PARTIAL:</strong> Target phase fraction 50% interval covers{' '}
              <strong>95.2%</strong> of samples (over-dispersed / conservative variance rather than overconfident).
            </span>
          </div>
          <span className="text-2xs font-mono bg-amber-200/60 px-2 py-0.5 rounded font-bold">FAIL GATE</span>
        </div>
      )}

      <div className="w-full h-60">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={chartData} margin={{ top: 12, right: 24, left: 0, bottom: 24 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" vertical={false} />
            <XAxis
              dataKey="name"
              tick={{ fill: '#475569', fontSize: 10, fontFamily: 'monospace' }}
              tickLine={false}
              axisLine={{ stroke: '#cbd5e1' }}
            />
            <YAxis
              domain={[0, 100]}
              ticks={[0, 25, 50, 75, 90, 100]}
              tickFormatter={(v) => `${v}%`}
              tick={{ fill: '#64748b', fontSize: 10 }}
              tickLine={false}
              axisLine={{ stroke: '#cbd5e1' }}
            />
            <Tooltip
              contentStyle={{
                backgroundColor: '#ffffff',
                borderColor: '#e2e8f0',
                borderRadius: '8px',
                boxShadow: '0 4px 6px -1px rgb(0 0 0 / 0.1)',
                fontSize: '11px'
              }}
              formatter={(value: any, name: any) => [
                `${Number(value).toFixed(1)}%`,
                name === 'coverage50' ? '50% Interval Coverage' : '90% Interval Coverage'
              ]}
              labelFormatter={(label, payload) => payload?.[0]?.payload?.fullName || label}
            />
            <Legend
              verticalAlign="top"
              height={32}
              formatter={(value) => (
                <span className="text-xs font-medium text-slate-700">
                  {value === 'coverage50' ? '50% Interval Coverage (Target: 50%)' : '90% Interval Coverage (Target: 90%)'}
                </span>
              )}
            />

            {/* Target reference lines */}
            <ReferenceLine y={50} stroke="#059669" strokeDasharray="3 3" label={{ value: 'Target 50%', fill: '#059669', fontSize: 10 }} />
            <ReferenceLine y={90} stroke="#2563eb" strokeDasharray="3 3" label={{ value: 'Target 90%', fill: '#2563eb', fontSize: 10 }} />

            <Bar dataKey="coverage50" fill="#10b981" radius={[3, 3, 0, 0]}>
              {chartData.map((entry, index) => (
                <Cell
                  key={`cov50-${index}`}
                  fill={entry.isOverdispersed ? '#d97706' : '#059669'}
                />
              ))}
            </Bar>
            <Bar dataKey="coverage90" fill="#3b82f6" radius={[3, 3, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </div>

      <div className="mt-2 pt-2 border-t border-slate-100 flex items-center justify-between text-2xs font-mono text-slate-500">
        <div className="flex items-center gap-3">
          <span>Nominal 50% target: ±0.674σ</span>
          <span>Nominal 90% target: ±1.645σ</span>
        </div>
        <div className="text-slate-400">
          Source: Zenodo DOI 10.5281/zenodo.21285546 (1,030 linked samples)
        </div>
      </div>
    </div>
  );
};
