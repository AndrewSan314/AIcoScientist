import React from 'react';
import { ArrowRight, Sparkles, CheckCircle2, Shield, Activity, Microscope } from 'lucide-react';
import { ExhibitionScenario } from '../../data/types';

interface Scene1HeroProps {
  scenario: ExhibitionScenario;
  onExploreLine: () => void;
  onExploreMicrostructure: () => void;
}

export const Scene1Hero: React.FC<Scene1HeroProps> = ({
  scenario,
  onExploreLine,
  onExploreMicrostructure
}) => {
  return (
    <div className="absolute top-28 left-8 max-w-xl z-20 pointer-events-auto">
      <div className="bg-white/90 backdrop-blur-xl p-8 rounded-3xl shadow-xl border border-[#DCE8EC] space-y-6 animate-in fade-in slide-in-from-left-4 duration-700">
        {/* Badge */}
        <div className="inline-flex items-center space-x-2 px-3 py-1 rounded-full bg-[#E8F4F2] border border-[#087F8C]/20 text-[#087F8C] text-xs font-semibold">
          <Sparkles className="w-3.5 h-3.5 text-[#087F8C]" />
          <span>Battery Manufacturing · Scientific Exhibition</span>
        </div>

        {/* Headlines */}
        <div className="space-y-2">
          <h1 className="text-3xl sm:text-4xl font-extrabold tracking-tight text-[#142A35] leading-tight">
            Optimize Battery Manufacturing with Process-Aware AI
          </h1>
          <p className="text-sm sm:text-base text-slate-600 leading-relaxed">
            Explore electrode manufacturing, an illustrative material cutaway, and recorded AI decisions across two independent historical datasets.
          </p>
        </div>

        {/* Key Scientific Evidence Pillars */}
        <div className="grid grid-cols-2 gap-3 pt-1">
          <div className="p-3 rounded-2xl bg-slate-50 border border-slate-100 flex items-start space-x-2.5">
            <CheckCircle2 className="w-4 h-4 text-[#087F8C] shrink-0 mt-0.5" />
            <div>
              <div className="text-xs font-bold text-[#142A35]">100% Hit@5 Rediscovery</div>
              <div className="text-[11px] text-slate-500 font-medium">{scenario.benchmark.seedsEvaluated} evaluated seeds · historical replay</div>
            </div>
          </div>

          <div className="p-3 rounded-2xl bg-slate-50 border border-slate-100 flex items-start space-x-2.5">
            <Shield className="w-4 h-4 text-[#087F8C] shrink-0 mt-0.5" />
            <div>
              <div className="text-xs font-bold text-[#142A35]">Information Horizon</div>
              <div className="text-[11px] text-slate-500 font-medium">Outcomes appear after reveal</div>
            </div>
          </div>

          <div className="p-3 rounded-2xl bg-slate-50 border border-slate-100 flex items-start space-x-2.5">
            <Activity className="w-4 h-4 text-[#F59E42] shrink-0 mt-0.5" />
            <div>
              <div className="text-xs font-bold text-[#142A35]">{scenario.evidenceKind === 'PILOT_LINE_HISTORICAL' ? 'Pilot-Line Historical' : 'Physical Historical'}</div>
              <div className="text-[11px] text-slate-500 font-medium">Audited physical experiments only</div>
            </div>
          </div>

          <div className="p-3 rounded-2xl bg-slate-50 border border-slate-100 flex items-start space-x-2.5">
            <Microscope className="w-4 h-4 text-[#087F8C] shrink-0 mt-0.5" />
            <div>
              <div className="text-xs font-bold text-[#142A35]">3D Microstructure Morph</div>
              <div className="text-[11px] text-slate-500 font-medium">Illustrative geometry, not simulation</div>
            </div>
          </div>
        </div>

        {/* Primary Call to Action */}
        <div className="flex flex-col sm:flex-row items-stretch sm:items-center gap-3 pt-2">
          <button
            onClick={onExploreLine}
            className="flex-1 flex items-center justify-center space-x-2 px-6 py-3.5 bg-[#087F8C] hover:bg-[#076a75] text-white font-semibold rounded-2xl shadow-md hover:shadow-lg transition-all cursor-pointer text-sm"
          >
            <span>Explore Manufacturing Line</span>
            <ArrowRight className="w-4 h-4" />
          </button>

          <button
            onClick={onExploreMicrostructure}
            className="flex items-center justify-center space-x-2 px-5 py-3.5 bg-slate-100 hover:bg-slate-200 text-[#142A35] font-semibold rounded-2xl transition-all cursor-pointer text-sm border border-slate-200/80"
          >
            <Microscope className="w-4 h-4 text-[#087F8C]" />
            <span>Inside Electrode</span>
          </button>
        </div>

        {/* Active Scenario Notice */}
        <div className="text-xs text-slate-500 flex items-center justify-between border-t border-slate-100 pt-3">
          <span>Active Demonstration:</span>
          <span className="font-semibold text-[#142A35]">{scenario.title}</span>
        </div>
      </div>
    </div>
  );
};
