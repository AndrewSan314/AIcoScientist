import React, { useState } from 'react';
import { ArrowLeft, ChevronDown, Cpu, Eye, Layers, Microscope } from 'lucide-react';
import type { ExhibitionScenario } from '../../data/types';
import { isDecisionStage, type ProcessExperiencePhase } from '../../data/processExperience';
import { ProcessDecisionExperience } from './ProcessDecisionExperience';

interface Scene2ProcessExplorerProps {
  scenario: ExhibitionScenario;
  selectedStageId: string;
  processPhase: ProcessExperiencePhase;
  replayStep: number;
  onSelectStage: (stageId: string) => void;
  onBeginDecision: () => void;
  onRunProcess: () => void;
  onNextDecision: () => void;
  onRepeatDecision: () => void;
  onReturnOverview: () => void;
  onNavigateMicrostructure: () => void;
  onNavigateOptimization: () => void;
}

const MECHANISM: Record<string, string> = {
  formulation: 'Select a material vessel in the 3D exploded view to locate active material, conductive additive and binder delivery.',
  slurry_prep: 'Inspect the restrained material-dosing exploded view. Composition is qualitative where the replay has no legal composition control.',
  mixing: 'The vessel opens into a cutaway while the actual agitator assembly rotates. Motion illustrates dispersion and is not a validated flow simulation.',
  coating: 'Enter the slot-die bead, backing roll and moving web. The recorded decision combines coating, drying and calendering status as one recipe.',
  pilot_coating: 'Inspect the backing roll, die lips and aluminum web. Warwick optimization remains attached to calendering, not this stage.',
  drying: 'Follow the coated web through exposed oven zones. Thermal color is qualitative; no evaporation curve is inferred.',
  calendering: 'Enter the roller nip and observe counter-rotating rolls, web motion and an illustrative compression bridge.',
  cell_assembly: 'Follow the electrode toward half-cell assembly. The station remains contextual because no assembly control is present in the replay.',
  characterization: 'A measurement trace emerges at the cycler only after the corresponding historical recipe has been revealed.',
  rate_characterization: 'The rate-performance measurement belongs to the selected historical Warwick condition and appears after reveal.',
};

const MATERIAL_ROLES = [
  ['Active material', 'Primary electrochemically active powder delivered into the formulation.'],
  ['Conductive additive', 'Builds electronic pathways through the composite electrode.'],
  ['Binder', 'Maintains particle-to-particle and particle-to-foil adhesion.'],
] as const;

export const Scene2ProcessExplorer: React.FC<Scene2ProcessExplorerProps> = ({
  scenario,
  selectedStageId,
  processPhase,
  replayStep,
  onSelectStage,
  onBeginDecision,
  onRunProcess,
  onNextDecision,
  onRepeatDecision,
  onReturnOverview,
  onNavigateMicrostructure,
  onNavigateOptimization,
}) => {
  const [learnMore, setLearnMore] = useState(false);
  const [materialRole, setMaterialRole] = useState(0);
  const isOverview = selectedStageId === 'overview';
  const stage = scenario.stages.find((item) => item.id === selectedStageId);
  const decisionEnabled = stage ? isDecisionStage(scenario, stage.id) : false;
  const revealedResult = replayStep > 0 ? scenario.replaySteps[Math.min(replayStep, scenario.replaySteps.length) - 1] : undefined;

  const selectStage = (stageId: string) => {
    setLearnMore(false);
    onSelectStage(stageId);
  };

  return (
    <>
      <div className="pointer-events-auto absolute left-1/2 top-32 z-20 flex w-[calc(100vw-48px)] max-w-[calc(100vw-48px)] -translate-x-1/2 gap-1.5 overflow-x-auto rounded-2xl border border-white/70 bg-white/88 p-1.5 shadow-lg backdrop-blur-xl xl:w-auto">
        <button onClick={onReturnOverview} aria-label="Full Line Overview" className={`rounded-xl px-3 py-2 text-xs font-semibold ${isOverview ? 'bg-[#142A35] text-white' : 'text-slate-500 hover:bg-slate-100'}`}><Layers className="h-4 w-4" /></button>
        {scenario.stages.map((item) => (
          <button
            key={item.id}
            onClick={() => selectStage(item.id)}
            className={`whitespace-nowrap rounded-xl px-3 py-2 text-[11px] font-semibold transition ${item.id === selectedStageId ? 'bg-[#087F8C] text-white shadow' : 'text-slate-600 hover:bg-slate-100'}`}
          >
            {item.order}. {item.label}
          </button>
        ))}
      </div>

      {isOverview ? (
        <section className="pointer-events-auto absolute bottom-28 left-8 z-20 w-[390px] rounded-3xl border border-[#DCE8EC] bg-white/92 p-5 shadow-xl backdrop-blur-xl">
          <div className="text-[10px] font-bold uppercase tracking-[0.18em] text-[#087F8C]">MANUFACTURING LINE STAGES · Process Journey</div>
          <h2 className="mt-2 text-xl font-bold text-[#142A35]">Enter a working mechanism</h2>
          <p className="mt-2 text-xs leading-relaxed text-slate-600">Select a machine in 3D or above. Flagship stages keep the recorded AI decision, illustrated process response and historical result inside the machine experience.</p>
          <div className="mt-4 grid grid-cols-2 gap-2">
            <button onClick={() => selectStage(scenario.id === 'drakopoulos_graphite' ? 'coating' : 'calendering')} className="rounded-xl bg-[#087F8C] px-4 py-3 text-xs font-semibold text-white"><span className="flex items-center justify-center gap-2"><Cpu className="h-4 w-4" />Enter flagship</span></button>
            <button onClick={() => selectStage(scenario.stages[0].id)} className="rounded-xl border border-slate-200 px-4 py-3 text-xs font-semibold text-slate-700"><span className="flex items-center justify-center gap-2"><Eye className="h-4 w-4" />Start at stage 1</span></button>
          </div>
        </section>
      ) : stage && (
        <section className="pointer-events-auto absolute bottom-28 left-8 top-44 z-20 flex w-[410px] flex-col justify-end gap-3">
          <div className="rounded-3xl border border-[#DCE8EC] bg-white/92 p-5 shadow-xl backdrop-blur-xl">
            <div className="flex items-center justify-between gap-3">
              <div className="text-[10px] font-bold uppercase tracking-[0.16em] text-[#087F8C]">Stage {String(stage.order).padStart(2, '0')} · {processPhase.replaceAll('_', ' ')}</div>
              <button onClick={onReturnOverview} className="rounded-full border border-slate-200 p-2 text-slate-500 hover:bg-slate-50" aria-label="Return to manufacturing overview"><ArrowLeft className="h-4 w-4" /></button>
            </div>
            <h2 className="mt-2 text-xl font-bold text-[#142A35]">{stage.label}</h2>
            {stage.shortDescription && <p className="mt-1 text-xs font-semibold text-slate-700">{stage.shortDescription}</p>}
            <p className="mt-1 text-xs leading-relaxed text-slate-600">{MECHANISM[stage.id] || stage.detailedDescription}</p>

            {processPhase === 'PROCESS_ENTERING' && <div className="mt-4 flex items-center gap-2 text-xs font-semibold text-[#087F8C]"><span className="h-2 w-2 animate-pulse rounded-full bg-[#F59E42]" />Entering mechanism…</div>}

            {processPhase === 'PROCESS_EXPLORE' && !decisionEnabled && (
              <div className="mt-4 rounded-2xl border border-teal-100 bg-teal-50/70 px-4 py-3 text-xs text-teal-950">
                <div className="font-semibold">Mechanism inspection active</div>
                <div className="mt-1 text-[11px] text-teal-800">Drag the scene where supported and inspect the revealed assembly. No new quantitative process outcome is inferred.</div>
              </div>
            )}

            {processPhase === 'PROCESS_EXPLORE' && (stage.id === 'formulation' || stage.id === 'slurry_prep') && (
              <div className="mt-3 rounded-2xl border border-slate-200 bg-white/80 p-3">
                <div className="flex flex-wrap gap-1.5">
                  {MATERIAL_ROLES.map(([label], index) => <button key={label} onClick={() => setMaterialRole(index)} className={`rounded-lg px-2 py-1 text-[10px] font-semibold ${materialRole === index ? 'bg-[#142A35] text-white' : 'bg-slate-100 text-slate-600'}`}>{label}</button>)}
                </div>
                <p className="mt-2 text-[11px] leading-relaxed text-slate-600">{MATERIAL_ROLES[materialRole][1]}</p>
              </div>
            )}

            {processPhase === 'PROCESS_EXPLORE' && revealedResult && (stage.id === 'characterization' || stage.id === 'rate_characterization') && (
              <div className="mt-3 rounded-2xl border border-[#CFE7EA] bg-[#F0FAFA] px-4 py-3">
                <div className="text-[10px] font-bold uppercase tracking-[0.15em] text-[#087F8C]">Revealed historical measurement</div>
                <div className="mt-1 text-lg font-bold text-[#142A35]">{revealedResult.revealedTarget} {scenario.targetUnit}</div>
                <div className="text-[11px] text-slate-600">{revealedResult.selectedCandidateDisplay} · source-backed replay record</div>
              </div>
            )}

            <button onClick={() => setLearnMore((open) => !open)} className="mt-4 flex items-center gap-1.5 text-[11px] font-semibold text-slate-600 hover:text-[#087F8C]">Learn more <ChevronDown className={`h-3.5 w-3.5 transition ${learnMore ? 'rotate-180' : ''}`} /></button>
            {learnMore && (
              <div className="mt-2 space-y-2 border-t border-slate-100 pt-3 text-[11px] leading-relaxed text-slate-600">
                <p>{stage.detailedDescription}</p>
                <p className="font-medium text-slate-500">Scientific parameters appear on the 3D mechanism only when they belong to the active recorded decision.</p>
                {stage.id === 'calendering' && <button onClick={onNavigateMicrostructure} className="flex items-center gap-1.5 font-semibold text-[#087F8C]"><Microscope className="h-3.5 w-3.5" />Open illustrative microstructure</button>}
              </div>
            )}
          </div>

          {decisionEnabled && processPhase !== 'PROCESS_ENTERING' && (
            <ProcessDecisionExperience
              scenario={scenario}
              phase={processPhase}
              revealedSteps={replayStep}
              onBeginDecision={onBeginDecision}
              onRunProcess={onRunProcess}
              onNextDecision={onNextDecision}
              onRepeatDecision={onRepeatDecision}
              onOpenAdvanced={onNavigateOptimization}
            />
          )}
        </section>
      )}
    </>
  );
};
