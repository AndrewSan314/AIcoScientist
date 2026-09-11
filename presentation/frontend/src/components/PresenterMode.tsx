import React, { useEffect } from 'react';
import { 
  ChevronLeft, 
  ChevronRight, 
  X, 
  BookOpen, 
  RotateCcw
} from 'lucide-react';
import { WorkspaceTab, DiscoveryFlowState, DatasetOption, RevealPhase } from '../types/mission_control';

interface PresenterModeProps {
  currentScene: number;
  totalScenes: number;
  onPrevScene: () => void;
  onNextScene: () => void;
  onSelectScene: (index: number) => void;
  onExit: () => void;
  onToggleNotes: () => void;
  onJumpToWorkspace: (workspace: WorkspaceTab, questionId?: number) => void;
}

export interface SceneMeta {
  index: number;
  title: string;
  workspace: WorkspaceTab;
  flowState?: DiscoveryFlowState;
  datasetOption?: DatasetOption;
  questionId?: number;
  stepIndex?: number;
  revealPhase?: RevealPhase;
  subtab?: 'architecture' | 'audit' | 'verification';
  tagline: string;
  keyMetric: string;
}

export const SCENES: SceneMeta[] = [
  {
    index: 0,
    title: '1. What problem does AIcoScientist solve?',
    workspace: 'discovery',
    flowState: 'setup',
    datasetOption: 'controlled_synthesis',
    tagline: '“It chooses the next candidate × measurement experiment—not simply the material predicted to be best.”',
    keyMetric: 'Next Experiment ≠ Best Material',
  },
  {
    index: 1,
    title: '2. What should we test next?',
    workspace: 'discovery',
    flowState: 'results',
    stepIndex: 1,
    revealPhase: 'A_SCORED',
    tagline: '“The Next Experiment card names the recorded candidate × measurement action and its score.”',
    keyMetric: 'Recorded Selected Action',
  },
  {
    index: 2,
    title: '3. Why this experiment?',
    workspace: 'discovery',
    flowState: 'results',
    stepIndex: 1,
    revealPhase: 'A_SCORED',
    tagline: '“Every candidate × measurement cell is a feasible experiment; the highlighted cell is the recorded choice.”',
    keyMetric: 'Decision Matrix',
  },
  {
    index: 3,
    title: '4. Lock prediction → reveal evidence → update model support',
    workspace: 'discovery',
    flowState: 'results',
    stepIndex: 2,
    revealPhase: 'D_UPDATED',
    tagline: '“The expected outcome is recorded before the measurement, then evidence changes support for competing models.”',
    keyMetric: 'Recorded Evidence Loop',
  },
  {
    index: 4,
    title: '5. Best candidate found in an optimization task',
    workspace: 'discovery',
    flowState: 'results',
    datasetOption: 'electrolyte_search',
    tagline: '“Best Found is separate from Next Experiment and is limited to a frozen surrogate benchmark.”',
    keyMetric: 'Best Found So Far',
  },
  {
    index: 5,
    title: '6. Does the decision strategy work in controlled worlds?',
    workspace: 'benchmarks',
    questionId: 1,
    tagline: '“Controlled ground truth lets the benchmark test whether model-support updates and decisions behave as intended.”',
    keyMetric: 'Controlled Validation',
  },
  {
    index: 6,
    title: '7. Can the framework replay real historical records?',
    workspace: 'benchmarks',
    questionId: 4,
    tagline: '“A-Lab is a historical physical-data replay, not a new autonomous laboratory execution.”',
    keyMetric: 'A-Lab Historical Replay',
  },
  {
    index: 7,
    title: '8. Can it scale candidate search?',
    workspace: 'benchmarks',
    questionId: 5,
    tagline: '“333,333 virtual formulations are screened to a recorded working set and sequential surrogate queries.”',
    keyMetric: 'Large-Space Surrogate Optimization',
  },
  {
    index: 8,
    title: '9. Evidence boundaries and next research step',
    workspace: 'system',
    subtab: 'verification',
    tagline: '“The presentation distinguishes controlled validation, historical replay, and simulation from prospective autonomous experimentation.”',
    keyMetric: 'Evidence & Limitations',
  },
];

export const PresenterMode: React.FC<PresenterModeProps> = ({
  currentScene,
  totalScenes,
  onPrevScene,
  onNextScene,
  onSelectScene,
  onExit,
  onToggleNotes,
}) => {
  const scene = SCENES[currentScene] || SCENES[0];

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (['INPUT', 'SELECT', 'TEXTAREA'].includes((e.target as HTMLElement)?.tagName)) {
        return;
      }
      if (e.key === 'ArrowRight' || e.key === ' ' || e.key === 'PageDown') {
        e.preventDefault();
        onNextScene();
      } else if (e.key === 'ArrowLeft' || e.key === 'PageUp') {
        e.preventDefault();
        onPrevScene();
      } else if (e.key === 'Escape') {
        e.preventDefault();
        onExit();
      } else if (e.key === 'n' || e.key === 'N') {
        e.preventDefault();
        onToggleNotes();
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [onNextScene, onPrevScene, onExit, onToggleNotes]);

  return (
    <div className="fixed top-0 left-0 right-0 z-50 bg-[#17201F] text-[#FCFCFA] border-b border-[#B91C1C]/40 shadow-md">
      <div className="max-w-[1720px] mx-auto px-4 sm:px-8 lg:px-12 h-14 flex items-center justify-between gap-4">
        {/* Left: Brand & Scene Title */}
        <div className="flex items-center gap-3 min-w-0">
          <div className="flex items-center gap-1.5 px-2.5 py-1 rounded bg-[#B91C1C] text-white text-2xs font-mono font-bold shrink-0">
            <span>SCENE {currentScene + 1}/{totalScenes}</span>
          </div>

          <div className="min-w-0">
            <div className="text-xs sm:text-sm font-bold text-white truncate">
              {scene.title}
            </div>
            <div className="text-2xs text-[#FECACA] hidden md:block truncate max-w-xl">
              {scene.tagline}
            </div>
          </div>
        </div>

        {/* Center: Key Metric Badge */}
        <div className="hidden lg:flex items-center">
          <span className="px-2.5 py-1 rounded-full bg-[#B91C1C]/30 text-[#FECACA] border border-[#B91C1C]/50 text-2xs font-mono font-semibold">
            {scene.keyMetric}
          </span>
        </div>

        {/* Right: Controls & Actions */}
        <div className="flex items-center gap-2 shrink-0">
          {/* Speaker Notes Toggle */}
          <button
            onClick={onToggleNotes}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-[#243331] hover:bg-[#2F4240] text-xs font-semibold text-[#FCFCFA] border border-[#3E5653] transition cursor-pointer"
            title="Speaker notes (N)"
          >
            <BookOpen className="w-3.5 h-3.5 text-[#FECACA]" />
            <span className="hidden sm:inline">Notes</span>
          </button>

          {/* Prev / Next Navigation */}
          <div className="flex items-center bg-[#243331] rounded-lg border border-[#3E5653] p-0.5">
            <button
              onClick={onPrevScene}
              disabled={currentScene === 0}
              className="p-1.5 rounded hover:bg-[#2F4240] text-white disabled:opacity-30 disabled:cursor-not-allowed transition cursor-pointer"
              title="Previous scene (Left Arrow)"
            >
              <ChevronLeft className="w-4 h-4" />
            </button>
            <button
              onClick={onNextScene}
              disabled={currentScene === totalScenes - 1}
              className="p-1.5 rounded hover:bg-[#2F4240] text-white disabled:opacity-30 disabled:cursor-not-allowed transition cursor-pointer"
              title="Next scene (Right Arrow / Space)"
            >
              <ChevronRight className="w-4 h-4" />
            </button>
          </div>

          {/* Exit Presenter Mode */}
          <button
            onClick={onExit}
            className="p-1.5 rounded-lg hover:bg-[#243331] text-[#FECACA] hover:text-white transition cursor-pointer"
            title="Exit presenter mode (Esc)"
          >
            <X className="w-4 h-4" />
          </button>
        </div>
      </div>
    </div>
  );
};
