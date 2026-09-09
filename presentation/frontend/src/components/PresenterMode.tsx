import React, { useEffect } from 'react';
import { 
  ChevronLeft, 
  ChevronRight, 
  X, 
  BookOpen, 
  RotateCcw
} from 'lucide-react';
import { WorkspaceTab } from '../types/mission_control';

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
  questionId?: number;
  tagline: string;
  keyMetric: string;
  stepIndex?: number;
}

export const SCENES: SceneMeta[] = [
  {
    index: 0,
    title: '1. The Research Question & Contribution',
    workspace: 'discovery',
    tagline: '“Traditional optimization asks which material is best. We ask which experiment should be performed next, and why.”',
    keyMetric: 'Hypothesis-Driven Decisions',
    stepIndex: 1
  },
  {
    index: 1,
    title: '2. Discovery Lab: Competing Hypotheses & Predictions',
    workspace: 'discovery',
    tagline: '“Maintaining three formal competing hypotheses with preregistered Gaussian predictive distributions.”',
    keyMetric: 'H₁ vs H₂ vs H₃ Observatory',
    stepIndex: 1
  },
  {
    index: 2,
    title: '3. Next Experiment: Candidate × Modality Trade-off',
    workspace: 'discovery',
    tagline: '“Jointly selecting candidate and modality using exact score decomposition S(a) = w_H·HIG + w_D·D - w_C·C.”',
    keyMetric: 'Dimensionless Scalar S(a)',
    stepIndex: 1
  },
  {
    index: 3,
    title: '4. The Scientific Wow Moment (Preregister → Reveal → Update)',
    workspace: 'discovery',
    tagline: '“Preregistration before reveal strictly firewalls observations, preventing hindsight bias with immutable audit logging.”',
    keyMetric: 'State A → B → C → D',
    stepIndex: 2
  },
  {
    index: 4,
    title: '5. Policy Benchmarks: Clean vs Stress Worlds',
    workspace: 'benchmarks',
    questionId: 1,
    tagline: '“180 controlled trajectories prove HYBRID achieves 100% MAP hypothesis recovery with bounded experimental expenditure.”',
    keyMetric: '180 Full Trajectories',
    stepIndex: 1
  },
  {
    index: 5,
    title: '6. Real A-Lab Replay, Electrolyte Scaling & Governance',
    workspace: 'benchmarks',
    questionId: 4,
    tagline: '“1,035 real physical samples and 333k electrolyte screening with honest disclosure of empirical boundaries.”',
    keyMetric: '1,035 Real Samples • 48/50 Gates',
    stepIndex: 1
  }
];

export const PresenterMode: React.FC<PresenterModeProps> = ({
  currentScene,
  totalScenes,
  onPrevScene,
  onNextScene,
  onSelectScene,
  onExit,
  onToggleNotes,
  onJumpToWorkspace
}) => {
  const scene = SCENES[currentScene] || SCENES[0];

  // Global keyboard shortcuts in Presenter Mode
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (['INPUT', 'SELECT', 'TEXTAREA'].includes((e.target as HTMLElement)?.tagName)) {
        return;
      }

      if (e.key === 'ArrowRight' || e.key === ' ') {
        e.preventDefault();
        onNextScene();
      } else if (e.key === 'ArrowLeft') {
        e.preventDefault();
        onPrevScene();
      } else if (e.key === 'Escape') {
        e.preventDefault();
        onExit();
      } else if (e.key === 'n' || e.key === 'N') {
        e.preventDefault();
        onToggleNotes();
      } else if (e.key >= '1' && e.key <= '6') {
        const idx = parseInt(e.key, 10) - 1;
        if (idx >= 0 && idx < SCENES.length) {
          onSelectScene(idx);
        }
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [currentScene, onNextScene, onPrevScene, onExit, onToggleNotes, onSelectScene]);

  return (
    <div className="fixed top-0 left-0 right-0 z-50 bg-slate-900 text-white shadow-xl border-b border-emerald-500/30">
      {/* Top Progress Line */}
      <div className="w-full bg-slate-800 h-1">
        <div
          className="bg-emerald-400 h-1 transition-all duration-300"
          style={{ width: `${((currentScene + 1) / totalScenes) * 100}%` }}
        />
      </div>

      <div className="max-w-7xl mx-auto px-4 py-2 flex items-center justify-between gap-3">
        {/* Left: Current Scene Info */}
        <div className="flex items-center gap-3 min-w-0">
          <div className="flex items-center gap-2">
            <span className="px-2 py-0.5 rounded bg-emerald-700 text-emerald-100 font-mono text-xs font-bold shrink-0">
              Scene {currentScene + 1}/{totalScenes}
            </span>
            <span className="font-bold text-sm tracking-tight truncate hidden sm:inline">
              {scene.title}
            </span>
          </div>

          <span className="text-slate-600 hidden md:inline">|</span>

          <span className="text-xs text-slate-300 italic truncate hidden lg:inline max-w-md">
            {scene.tagline}
          </span>
        </div>

        {/* Center: Key Metric Badge */}
        <div className="hidden xl:flex items-center gap-2 px-3 py-1 bg-slate-800/80 rounded-lg border border-slate-700/60 font-mono text-xs text-emerald-300">
          <span>Focus:</span>
          <strong className="text-white">{scene.keyMetric}</strong>
        </div>

        {/* Right: Scene Navigation Buttons & Controls */}
        <div className="flex items-center gap-2 shrink-0">
          <button
            onClick={() => onSelectScene(0)}
            className="p-1.5 text-slate-400 hover:text-white rounded transition hover:bg-slate-800 cursor-pointer"
            title="Reset to Scene 1"
          >
            <RotateCcw className="w-4 h-4" />
          </button>

          <button
            onClick={onToggleNotes}
            className="inline-flex items-center gap-1 px-2.5 py-1 text-xs font-medium text-slate-200 bg-slate-800 hover:bg-slate-700 border border-slate-700 rounded-md transition cursor-pointer"
            title="Toggle Speaker Notes Drawer (Key: N)"
          >
            <BookOpen className="w-3.5 h-3.5 text-emerald-400" />
            <span className="hidden sm:inline">Notes</span>
            <kbd className="text-3xs font-mono bg-slate-900 px-1 py-0.5 rounded text-slate-400">N</kbd>
          </button>

          <div className="flex items-center bg-slate-800 rounded-md border border-slate-700">
            <button
              onClick={onPrevScene}
              disabled={currentScene === 0}
              className={`p-1.5 transition ${
                currentScene === 0
                  ? 'text-slate-600 cursor-not-allowed'
                  : 'text-slate-200 hover:text-white hover:bg-slate-700 cursor-pointer'
              }`}
              title="Previous Scene (Left Arrow)"
            >
              <ChevronLeft className="w-4 h-4" />
            </button>
            <button
              onClick={onNextScene}
              disabled={currentScene === totalScenes - 1}
              className={`p-1.5 transition ${
                currentScene === totalScenes - 1
                  ? 'text-slate-600 cursor-not-allowed'
                  : 'text-slate-200 hover:text-white hover:bg-slate-700 cursor-pointer'
              }`}
              title="Next Scene (Right Arrow or Space)"
            >
              <ChevronRight className="w-4 h-4" />
            </button>
          </div>

          <button
            onClick={onExit}
            className="p-1.5 text-slate-400 hover:text-white rounded transition hover:bg-slate-800 cursor-pointer"
            title="Exit Presenter Mode (Esc)"
          >
            <X className="w-4 h-4" />
          </button>
        </div>
      </div>
    </div>
  );
};
