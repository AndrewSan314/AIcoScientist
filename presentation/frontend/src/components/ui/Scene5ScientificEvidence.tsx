import React from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from 'recharts';
import { Award, CheckCircle2, ShieldAlert, ArrowRight, RotateCcw, ExternalLink, FileText, Zap } from 'lucide-react';
import { ExhibitionScenario, ScenarioId } from '../../data/types';

interface Scene5ScientificEvidenceProps {
  scenario: ExhibitionScenario;
  onToggleLimitationsDrawer: () => void;
  onSwitchScenario: (id: ScenarioId) => void;
  onRestartJourney: () => void;
}

export const Scene5ScientificEvidence: React.FC<Scene5ScientificEvidenceProps> = ({
  scenario,
  onToggleLimitationsDrawer,
  onSwitchScenario,
  onRestartJourney
}) => {
  const bm = scenario.benchmark;
  const isWarwick = scenario.id === 'warwick_nmc622_calendering';
  const otherScenarioId: ScenarioId = isWarwick ? 'drakopoulos_graphite' : 'warwick_nmc622_calendering';
  const otherScenarioLabel = isWarwick ? 'Graphite Anode (Drakopoulos)' : 'Warwick NMC622 Cathode';

  // Benchmark comparison chart data
  const chartData = [
    {
      metric: 'Hit@1',
      aicoscientist: bm.hitAt1Pct,
      randomBaseline: isWarwick ? 5.5 : 11.1
    },
    {
      metric: 'Hit@3',
      aicoscientist: bm.hitAt3Pct,
      randomBaseline: isWarwick ? 16.7 : 33.3
    },
    {
      metric: 'Hit@5 (Full Budget)',
      aicoscientist: bm.hitAt5Pct,
      randomBaseline: bm.randomBaselineHitAt5Pct
    }
  ];

  return (
    <div className="absolute top-20 left-8 right-8 bottom-24 z-20 pointer-events-none flex justify-between gap-6">
      {/* Left Column: Primary Scientific Evaluation & Telemetry */}
      <div className="w-full max-w-2xl pointer-events-auto bg-white/94 backdrop-blur-xl p-7 rounded-3xl shadow-xl border border-[#DCE8EC] flex flex-col justify-between overflow-y-auto animate-in fade-in slide-in-from-left-4 duration-500">
        <div className="space-y-5">
          {/* Header & Badges */}
          <div className="flex items-center justify-between">
            <div className="space-y-1">
              <div className="inline-flex items-center space-x-1.5 px-3 py-1 rounded-full bg-[#E8F4F2] text-[#087F8C] text-xs font-semibold border border-[#087F8C]/20">
                <CheckCircle2 className="w-3.5 h-3.5" />
                <span>Audited Benchmark Verification</span>
              </div>
              <h2 className="text-2xl font-extrabold text-[#142A35]">
                {bm.policyName}
              </h2>
            </div>

            <div className="text-right">
              <div className="text-2xl font-black font-mono text-[#087F8C]">
                {bm.hitAt5Pct.toFixed(0)}%
              </div>
              <div className="text-[11px] font-semibold text-slate-500 uppercase tracking-wider">
                Hit@5 Success Rate
              </div>
            </div>
          </div>

          {/* Core Benchmark Comparison Chart */}
          <div className="p-4 rounded-2xl bg-slate-50 border border-slate-100">
            <div className="flex items-center justify-between mb-3 text-xs">
              <span className="font-semibold text-slate-700">Rediscovery Probability vs Random Search</span>
              <div className="flex items-center space-x-3 text-[11px]">
                <span className="flex items-center space-x-1">
                  <span className="w-2.5 h-2.5 rounded-sm bg-[#087F8C]" />
                  <span className="text-slate-600 font-medium">AIcoScientist</span>
                </span>
                <span className="flex items-center space-x-1">
                  <span className="w-2.5 h-2.5 rounded-sm bg-slate-300" />
                  <span className="text-slate-400 font-medium">Random Analytical</span>
                </span>
              </div>
            </div>

            <div className="h-56 w-full">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={chartData} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#E2E8F0" />
                  <XAxis dataKey="metric" tick={{ fontSize: 11, fill: '#475569' }} axisLine={false} tickLine={false} />
                  <YAxis unit="%" tick={{ fontSize: 11, fill: '#475569' }} axisLine={false} tickLine={false} domain={[0, 100]} />
                  <Tooltip
                    formatter={(val: any) => [`${Number(val).toFixed(1)}%`, '']}
                    contentStyle={{ backgroundColor: '#142A35', borderRadius: '12px', border: 'none', color: '#fff', fontSize: '12px' }}
                  />
                  <Bar dataKey="aicoscientist" name="AIcoScientist" fill="#087F8C" radius={[6, 6, 0, 0]} />
                  <Bar dataKey="randomBaseline" name="Random Baseline" fill="#CBD5E1" radius={[6, 6, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Supported Scientific Claim Card */}
          <div className="p-4 rounded-2xl bg-[#E8F4F2]/70 border border-[#087F8C]/20 space-y-2">
            <div className="flex items-center space-x-2 text-xs font-bold text-[#087F8C]">
              <Award className="w-4 h-4 text-[#087F8C]" />
              <span>Formally Verified Claim</span>
            </div>
            <p className="text-xs text-slate-700 leading-relaxed">
              {bm.supportedClaim}
            </p>
          </div>

          {/* Scientific Provenance Summary */}
          <div className="grid grid-cols-2 gap-3 text-xs">
            <div className="p-3 rounded-xl bg-slate-50 border border-slate-100">
              <div className="text-slate-400 text-[10px] uppercase font-bold">Source Reference</div>
              <div className="font-semibold text-slate-800 truncate mt-0.5">{scenario.provenance.authors} ({scenario.provenance.year})</div>
              <div className="text-[10px] text-slate-500 truncate">{scenario.provenance.journal}</div>
            </div>

            <div className="p-3 rounded-xl bg-slate-50 border border-slate-100">
              <div className="text-slate-400 text-[10px] uppercase font-bold">Verification Scope</div>
              <div className="font-semibold text-slate-800 mt-0.5">{bm.seedsEvaluated} Independent Seeds</div>
              <div className="text-[10px] font-mono text-emerald-600 font-bold">Simple Regret: {bm.simpleRegretAt5.toFixed(4)}</div>
            </div>
          </div>
        </div>

        {/* Footer Actions */}
        <div className="pt-4 border-t border-slate-100 flex items-center justify-between gap-3">
          <button
            onClick={onRestartJourney}
            className="flex items-center space-x-1.5 px-4 py-2.5 bg-slate-100 hover:bg-slate-200 text-slate-700 rounded-xl text-xs font-semibold transition-all cursor-pointer"
          >
            <RotateCcw className="w-3.5 h-3.5" />
            <span>Restart Tour</span>
          </button>

          <button
            onClick={onToggleLimitationsDrawer}
            className="flex items-center space-x-1.5 px-4 py-2.5 bg-amber-50 hover:bg-amber-100 text-amber-900 border border-amber-200/80 rounded-xl text-xs font-semibold transition-all cursor-pointer"
          >
            <ShieldAlert className="w-3.5 h-3.5 text-amber-600" />
            <span>Known Limitations Drawer</span>
          </button>

          <button
            onClick={() => onSwitchScenario(otherScenarioId)}
            className="flex items-center space-x-1.5 px-4 py-2.5 bg-[#087F8C] hover:bg-[#076a75] text-white rounded-xl text-xs font-semibold shadow-xs transition-all cursor-pointer"
          >
            <span>Switch to {otherScenarioLabel}</span>
            <ArrowRight className="w-3.5 h-3.5" />
          </button>
        </div>
      </div>

      {/* Right Column: Research Rigor & Methodology Card */}
      <div className="w-96 pointer-events-auto bg-white/94 backdrop-blur-xl p-6 rounded-3xl shadow-xl border border-[#DCE8EC] flex flex-col justify-between overflow-y-auto animate-in fade-in slide-in-from-right-4 duration-500">
        <div className="space-y-4">
          <div className="flex items-center space-x-2 pb-3 border-b border-slate-100">
            <FileText className="w-4 h-4 text-[#087F8C]" />
            <span className="text-xs font-bold text-[#142A35]">Rigorous Methodology</span>
          </div>

          <div className="space-y-3 text-xs">
            <div className="p-3 rounded-xl bg-slate-50 border border-slate-100 space-y-1">
              <div className="font-bold text-[#142A35]">Pre-Decision Information Horizon</div>
              <p className="text-slate-600 text-[11px] leading-relaxed">
                The surrogate Gaussian Process only receives features available prior to the decision point. Electrochemical cycling results are held out until official candidate acquisition.
              </p>
            </div>

            <div className="p-3 rounded-xl bg-slate-50 border border-slate-100 space-y-1">
              <div className="font-bold text-[#142A35]">Finite Candidate Pool Acquisition</div>
              <p className="text-slate-600 text-[11px] leading-relaxed">
                Rather than proposing synthetic unmanufacturable parameter combinations, the optimizer selects exclusively from physical pilot-line candidates with verified chemistry bounds.
              </p>
            </div>

            <div className="p-3 rounded-xl bg-slate-50 border border-slate-100 space-y-1">
              <div className="font-bold text-[#142A35]">Analytical Combinatorial Baseline</div>
              <p className="text-slate-600 text-[11px] leading-relaxed">
                Baselines are calculated using exact hypergeometric combinatorial probabilities conditioned on the specific initial designs, guaranteeing unbiased evaluation.
              </p>
            </div>
          </div>
        </div>

        <div className="pt-4 border-t border-slate-100 text-center">
          <button
            onClick={onToggleLimitationsDrawer}
            className="text-xs font-semibold text-[#087F8C] hover:underline cursor-pointer"
          >
            Read Complete Verification Audit →
          </button>
        </div>
      </div>
    </div>
  );
};
