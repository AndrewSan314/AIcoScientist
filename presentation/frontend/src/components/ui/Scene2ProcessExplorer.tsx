import React from 'react';
import { ArrowRight, Sliders, Activity, Sparkles, Microscope, Cpu, Layers } from 'lucide-react';
import { ExhibitionScenario } from '../../data/types';

interface Scene2ProcessExplorerProps {
  scenario: ExhibitionScenario;
  selectedStageId: string;
  onSelectStage: (stageId: string) => void;
  onNavigateMicrostructure: () => void;
  onNavigateOptimization: () => void;
}

export const Scene2ProcessExplorer: React.FC<Scene2ProcessExplorerProps> = ({
  scenario,
  selectedStageId,
  onSelectStage,
  onNavigateMicrostructure,
  onNavigateOptimization
}) => {
  const isOverview = selectedStageId === 'overview';
  const currentStage = !isOverview
    ? scenario.stages.find((s) => s.id === selectedStageId) || scenario.stages[0]
    : undefined;

  return (
    <div className="absolute top-24 left-8 max-w-lg z-20 pointer-events-auto">
      <div className="bg-white/92 backdrop-blur-xl p-6 rounded-3xl shadow-xl border border-[#DCE8EC] space-y-5 animate-in fade-in slide-in-from-left-4 duration-500">
        {/* Stage Timeline Navigation Chips */}
        <div>
          <div className="text-[11px] font-bold uppercase tracking-wider text-slate-400 mb-2">
            Manufacturing Line Stages
          </div>
          <div className="flex flex-wrap gap-1.5">
            {/* Overview Button */}
            <button
              onClick={() => onSelectStage('overview')}
              className={`px-3 py-1.5 rounded-xl text-xs font-semibold transition-all cursor-pointer flex items-center space-x-1.5 ${
                isOverview
                  ? 'bg-[#142A35] text-white shadow-xs'
                  : 'bg-slate-100 text-slate-600 hover:bg-slate-200'
              }`}
            >
              <Layers className="w-3.5 h-3.5" />
              <span>Full Line Overview</span>
            </button>

            {scenario.stages.map((stage) => {
              const isSelected = !isOverview && stage.id === currentStage?.id;
              return (
                <button
                  key={stage.id}
                  onClick={() => onSelectStage(stage.id)}
                  className={`px-3 py-1.5 rounded-xl text-xs font-semibold transition-all cursor-pointer flex items-center space-x-1.5 ${
                    isSelected
                      ? 'bg-[#142A35] text-white shadow-xs'
                      : stage.isOptimizationTarget
                      ? 'bg-[#E8F4F2] text-[#087F8C] border border-[#087F8C]/30 hover:bg-[#D8EDE9]'
                      : 'bg-slate-100 text-slate-600 hover:bg-slate-200'
                  }`}
                >
                  <span>{stage.order}. {stage.label}</span>
                  {stage.isOptimizationTarget && (
                    <span className="w-1.5 h-1.5 rounded-full bg-[#F59E42]" />
                  )}
                </button>
              );
            })}
          </div>
        </div>

        {/* Overview Mode Card */}
        {isOverview && (
          <div className="space-y-3 border-t border-slate-100 pt-4">
            <div className="flex items-center justify-between">
              <span className="text-xs font-bold text-[#087F8C] tracking-wide uppercase">
                Continuous Cleanroom Line
              </span>
              <span className="text-[10px] font-medium text-slate-400">
                Click Equipment in 3D to Focus
              </span>
            </div>

            <h2 className="text-xl font-bold text-[#142A35]">
              Connected Multi-Stage Manufacturing Digital Twin
            </h2>
            <p className="text-xs text-slate-600 leading-relaxed">
              Electrode manufacturing spans powder formulation, vacuum planetary mixing, slot-die web coating, convection/IR drying, precision roll calendering, and electrochemical testing. Click any machine along the moving web to explore its physics.
            </p>

            <div className="pt-2 flex items-center space-x-2.5">
              <button
                onClick={() => onSelectStage(scenario.stages.find(s => s.isOptimizationTarget)?.id || scenario.stages[0].id)}
                className="flex-1 flex items-center justify-center space-x-2 px-4 py-3 bg-[#087F8C] hover:bg-[#076a75] text-white rounded-2xl text-xs font-semibold shadow-xs transition-all cursor-pointer"
              >
                <Cpu className="w-4 h-4" />
                <span>Explore Optimization Stage</span>
                <ArrowRight className="w-3.5 h-3.5" />
              </button>
            </div>
          </div>
        )}

        {/* Specific Stage Card */}
        {!isOverview && currentStage && (
          <>
            <div className="space-y-2.5 border-t border-slate-100 pt-4">
              <div className="flex items-center justify-between">
                <span className="text-xs font-bold text-[#087F8C] tracking-wide uppercase">
                  Stage 0{currentStage.order} • {currentStage.label}
                </span>
                {currentStage.isOptimizationTarget ? (
                  <span className="inline-flex items-center space-x-1 px-2.5 py-0.5 rounded-full bg-[#F59E42]/15 text-[#D97706] text-[10px] font-bold border border-[#F59E42]/30">
                    <Sparkles className="w-3 h-3 text-[#F59E42]" />
                    <span>AI Optimization Target</span>
                  </span>
                ) : (
                  <span className="text-[10px] font-medium text-slate-400">
                    Process Explorer Mode
                  </span>
                )}
              </div>

              <h2 className="text-xl font-bold text-[#142A35]">
                {currentStage.shortDescription}
              </h2>
              <p className="text-xs text-slate-600 leading-relaxed">
                {currentStage.detailedDescription}
              </p>
            </div>

            {/* Controllable Parameters & Observations */}
            <div className="space-y-3 border-t border-slate-100 pt-3">
              {currentStage.controllableParameters.length > 0 && (
                <div>
                  <div className="text-[11px] font-semibold text-slate-500 mb-1.5 flex items-center space-x-1">
                    <Sliders className="w-3.5 h-3.5 text-[#087F8C]" />
                    <span>Controllable Process Parameters</span>
                  </div>
                  <div className="grid grid-cols-2 gap-2">
                    {currentStage.controllableParameters.map((p) => (
                      <div key={p.name} className="p-2.5 rounded-xl bg-slate-50 border border-slate-100">
                        <div className="text-[11px] font-semibold text-[#142A35] truncate">{p.label}</div>
                        <div className="text-[10px] font-mono text-slate-500 mt-0.5">{p.typicalRange}</div>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {currentStage.measuredProperties.length > 0 && (
                <div>
                  <div className="text-[11px] font-semibold text-slate-500 mb-1.5 flex items-center space-x-1">
                    <Activity className="w-3.5 h-3.5 text-[#F59E42]" />
                    <span>Observed Physical Metrology</span>
                  </div>
                  <div className="grid grid-cols-2 gap-2">
                    {currentStage.measuredProperties.map((m) => (
                      <div key={m.name} className="p-2.5 rounded-xl bg-amber-50/50 border border-amber-100/80">
                        <div className="text-[11px] font-semibold text-slate-800 truncate">{m.label}</div>
                        <div className="text-[10px] font-mono text-amber-700 mt-0.5">[{m.unit}]</div>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>

            {/* Next Action CTAs */}
            <div className="flex items-center space-x-2.5 pt-2 border-t border-slate-100">
              {currentStage.id === 'calendering' && (
                <button
                  onClick={onNavigateMicrostructure}
                  className="flex-1 flex items-center justify-center space-x-2 px-4 py-3 bg-[#142A35] hover:bg-[#1f3b49] text-white rounded-2xl text-xs font-semibold shadow-xs transition-all cursor-pointer"
                >
                  <Microscope className="w-4 h-4 text-[#087F8C]" />
                  <span>Inspect Microstructure</span>
                </button>
              )}

              {currentStage.isOptimizationTarget && (
                <button
                  onClick={onNavigateOptimization}
                  className="flex-1 flex items-center justify-center space-x-2 px-4 py-3 bg-[#087F8C] hover:bg-[#076a75] text-white rounded-2xl text-xs font-semibold shadow-xs transition-all cursor-pointer"
                >
                  <Cpu className="w-4 h-4" />
                  <span>Launch AI Studio</span>
                  <ArrowRight className="w-3.5 h-3.5" />
                </button>
              )}

              {!currentStage.isOptimizationTarget && currentStage.id !== 'calendering' && (
                <button
                  onClick={onNavigateOptimization}
                  className="flex-1 flex items-center justify-center space-x-2 px-4 py-3 bg-[#087F8C] hover:bg-[#076a75] text-white rounded-2xl text-xs font-semibold shadow-xs transition-all cursor-pointer"
                >
                  <span>Explore AI Studio</span>
                  <ArrowRight className="w-3.5 h-3.5" />
                </button>
              )}
            </div>
          </>
        )}
      </div>
    </div>
  );
};
