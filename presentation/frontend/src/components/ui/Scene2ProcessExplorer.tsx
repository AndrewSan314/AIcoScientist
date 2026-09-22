import React, { useState } from 'react';
import {
  ArrowLeft,
  Camera,
  ChevronDown,
  Cpu,
  Eye,
  Layers,
  Microscope,
  Pause,
  Play,
  SkipBack,
  SkipForward,
  Square,
  Video,
} from 'lucide-react';
import type { ExhibitionScenario } from '../../data/types';
import { isDecisionStage, type ProcessExperiencePhase } from '../../data/processExperience';
import { ProcessDecisionExperience } from './ProcessDecisionExperience';
import type { TourUpdateInfo } from '../3d/CameraRig';

interface Scene2ProcessExplorerProps {
  scenario: ExhibitionScenario;
  selectedStageId: string;
  processPhase: ProcessExperiencePhase;
  replayStep: number;
  isCinematicTourActive?: boolean;
  tourInfo?: TourUpdateInfo | null;
  onSelectStage: (stageId: string) => void;
  onBeginDecision: () => void;
  onRunProcess: () => void;
  onNextDecision: () => void;
  onRepeatDecision: () => void;
  onReturnOverview: () => void;
  onNavigateMicrostructure: () => void;
  onNavigateOptimization: () => void;
  onStartVideoTour?: () => void;
  onExitVideoTour?: () => void;
  onToggleTourPause?: () => void;
  onStepTour?: (direction: 1 | -1) => void;
  onToggleCleanScreen?: () => void;
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
  isCinematicTourActive,
  tourInfo,
  onSelectStage,
  onBeginDecision,
  onRunProcess,
  onNextDecision,
  onRepeatDecision,
  onReturnOverview,
  onNavigateMicrostructure,
  onNavigateOptimization,
  onStartVideoTour,
  onExitVideoTour,
  onToggleTourPause,
  onStepTour,
  onToggleCleanScreen,
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
      <div className="pointer-events-auto absolute left-1/2 top-32 z-20 flex w-[calc(100vw-48px)] max-w-[calc(100vw-48px)] -translate-x-1/2 items-center gap-1.5 overflow-x-auto rounded-2xl border border-white/70 bg-white/88 p-1.5 shadow-lg backdrop-blur-xl xl:w-auto">
        <button onClick={onReturnOverview} aria-label="Full Line Overview" className={`rounded-xl px-3 py-2 text-xs font-semibold ${isOverview && !isCinematicTourActive ? 'bg-[#142A35] text-white' : 'text-slate-500 hover:bg-slate-100'}`}><Layers className="h-4 w-4" /></button>
        {scenario.stages.map((item) => (
          <button
            key={item.id}
            onClick={() => selectStage(item.id)}
            className={`whitespace-nowrap rounded-xl px-3 py-2 text-[11px] font-semibold transition ${item.id === selectedStageId && !isCinematicTourActive ? 'bg-[#087F8C] text-white shadow' : 'text-slate-600 hover:bg-slate-100'}`}
          >
            {item.order}. {item.label}
          </button>
        ))}

        {/* Video Tour Trigger Button */}
        <button
          onClick={isCinematicTourActive ? onExitVideoTour : onStartVideoTour}
          className={`flex items-center gap-1.5 whitespace-nowrap rounded-xl px-3.5 py-2 text-[11px] font-bold transition shadow-sm ${
            isCinematicTourActive
              ? 'bg-gradient-to-r from-[#F59E42] to-[#E53935] text-white animate-pulse'
              : 'bg-teal-50 text-[#087F8C] border border-teal-200/80 hover:bg-teal-100'
          }`}
          title="Start continuous cinematic camera tour across the manufacturing pipeline (ideal for video capture)"
        >
          <Video className="h-3.5 w-3.5" />
          <span>{isCinematicTourActive ? 'Stop Tour' : '🎬 Video Tour'}</span>
        </button>

        {/* Clean Screen Trigger Button */}
        <button
          onClick={onToggleCleanScreen}
          className="flex items-center gap-1.5 whitespace-nowrap rounded-xl border border-slate-200 bg-white px-3 py-2 text-[11px] font-semibold text-slate-700 hover:bg-slate-50"
          title="Toggle Clean Screen mode (Hotkey: H) for clean screen recording"
        >
          <Camera className="h-3.5 w-3.5 text-[#087F8C]" />
          <span className="hidden sm:inline">Clean (H)</span>
        </button>
      </div>

      {/* 1. CINEMATIC VIDEO TOUR LOWER-THIRD TELEMETRY OVERLAY */}
      {isCinematicTourActive && tourInfo ? (
        <div className="pointer-events-auto absolute bottom-24 left-1/2 -translate-x-1/2 z-30 w-[94vw] max-w-4xl rounded-3xl border border-white/25 bg-[#142A35]/94 p-5 text-white shadow-2xl backdrop-blur-2xl transition-all duration-300">
          <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-3 border-b border-white/10 pb-3">
            <div className="flex items-center gap-2.5">
              <span className="flex h-3 w-3 relative">
                <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-[#F59E42] opacity-75" />
                <span className="relative inline-flex rounded-full h-3 w-3 bg-[#F59E42]" />
              </span>
              <span className="text-[10px] font-extrabold uppercase tracking-[0.2em] text-[#38BDF8]">
                {tourInfo.stage.category} · STATION {String(tourInfo.stageIndex + 1).padStart(2, '0')}/{String(tourInfo.totalStages).padStart(2, '0')}
              </span>
            </div>

            <div className="flex items-center gap-2">
              <button
                onClick={() => onStepTour?.(-1)}
                className="rounded-xl border border-white/15 bg-white/10 p-2 text-white/80 hover:bg-white/20 transition"
                title="Previous Station"
              >
                <SkipBack className="h-4 w-4" />
              </button>
              <button
                onClick={onToggleTourPause}
                className="flex items-center gap-1.5 rounded-xl bg-[#087F8C] px-3.5 py-2 text-xs font-bold text-white hover:bg-[#0aa2b3] transition shadow-md"
              >
                {tourInfo.isPaused ? <Play className="h-3.5 w-3.5 fill-current" /> : <Pause className="h-3.5 w-3.5 fill-current" />}
                <span>{tourInfo.isPaused ? 'Resume' : 'Pause'}</span>
              </button>
              <button
                onClick={() => onStepTour?.(1)}
                className="rounded-xl border border-white/15 bg-white/10 p-2 text-white/80 hover:bg-white/20 transition"
                title="Next Station"
              >
                <SkipForward className="h-4 w-4" />
              </button>
              <button
                onClick={onExitVideoTour}
                className="rounded-xl border border-red-500/40 bg-red-500/20 px-3 py-2 text-xs font-semibold text-red-200 hover:bg-red-500/30 transition"
              >
                <Square className="h-3.5 w-3.5 inline mr-1 fill-current" /> Exit
              </button>
            </div>
          </div>

          <div className="mt-3.5 grid grid-cols-1 md:grid-cols-3 gap-3.5">
            <div className="md:col-span-2">
              <h3 className="text-xl font-extrabold tracking-tight text-white">{tourInfo.stage.label}</h3>
              <p className="mt-1.5 text-xs leading-relaxed text-slate-300">{tourInfo.stage.action}</p>
            </div>
            <div className="rounded-2xl border border-white/10 bg-black/35 p-3 flex flex-col justify-center">
              <div className="text-[9px] font-bold uppercase tracking-widest text-[#38BDF8]">Process Context & Verified Controls</div>
              <div className="mt-1 font-mono text-[11px] leading-relaxed text-emerald-300">{tourInfo.stage.contextDescription || tourInfo.stage.telemetry}</div>
            </div>
          </div>

          {/* Explicit Scientific Provenance Disclosure */}
          <div className="mt-2.5 flex items-center justify-between text-[10px] text-slate-400 border-t border-white/10 pt-2">
            <span>Illustrative Process Journey · Independent Historical Datasets (Not a single continuous batch or live plant)</span>
            <span className="font-mono text-[#38BDF8]">STATION {tourInfo.stageIndex + 1}/{tourInfo.totalStages}</span>
          </div>

          {/* Station Progress Bar */}
          <div className="mt-2.5 h-1.5 w-full overflow-hidden rounded-full bg-white/10">
            <div
              className="h-full bg-gradient-to-r from-[#087F8C] via-[#38BDF8] to-[#F59E42] transition-all duration-150"
              style={{ width: `${Math.round(tourInfo.progress * 100)}%` }}
            />
          </div>
        </div>
      ) : isOverview ? (
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
              onInspectMeasurement={() => {
                selectStage(scenario.id === 'warwick_nmc622_calendering' ? 'rate_characterization' : 'characterization');
              }}
            />
          )}
        </section>
      )}
    </>
  );
};
