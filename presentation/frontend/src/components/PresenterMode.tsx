import React, { useEffect } from 'react';
import { 
  ChevronLeft, 
  ChevronRight, 
  X, 
  BookOpen, 
  FlaskConical, 
  Layers, 
  BarChart3, 
  Zap, 
  GitBranch, 
  ShieldCheck,
  Compass
} from 'lucide-react';
import { NavTab } from './Header';

interface PresenterModeProps {
  currentScene: number;
  totalScenes: number;
  onPrevScene: () => void;
  onNextScene: () => void;
  onSelectScene: (index: number) => void;
  onExit: () => void;
  onToggleNotes: () => void;
  onJumpToTab: (tab: NavTab) => void;
}

export interface SceneMeta {
  index: number;
  title: string;
  tab: NavTab;
  tagline: string;
  keyMetric: string;
}

export const SCENES: SceneMeta[] = [
  {
    index: 0,
    title: 'The Research Question',
    tab: 'overview',
    tagline: '“Traditional optimization asks which material is best. We ask which experiment should be performed next, and why.”',
    keyMetric: 'Hypothesis-Driven Decisions',
  },
  {
    index: 1,
    title: 'Universal Architecture',
    tab: 'architecture',
    tagline: '“One reusable decision engine decoupling candidate schemas from multi-modal physical characterization.”',
    keyMetric: '4 Scientific Domains',
  },
  {
    index: 2,
    title: 'Decision Cockpit',
    tab: 'cockpit',
    tagline: '“Maintaining competing mechanistic hypotheses while jointly scoring candidate materials and measurement modalities.”',
    keyMetric: 'HIG + Discovery - Cost',
  },
  {
    index: 3,
    title: 'The Scientific Wow Moment',
    tab: 'cockpit',
    tagline: '“Preregistration before reveal prevents retrospective bias and provides an immutable evidence trail.”',
    keyMetric: 'Preregister → Reveal → Update',
  },
  {
    index: 4,
    title: 'Evaluation Breadth',
    tab: 'benchmarks',
    tagline: '“180 controlled trajectories across clean and stress worlds demonstrate rigorous policy trade-offs.”',
    keyMetric: '180 Full Trajectories',
  },
  {
    index: 5,
    title: 'A-Lab Evidence Atlas',
    tab: 'alab',
    tagline: '“Validating on 1,035 real inorganic synthesis samples while honestly reporting partial calibration and holdout boundaries.”',
    keyMetric: '1,035 Real Samples',
  },
  {
    index: 6,
    title: 'Electrolyte Screening Scale',
    tab: 'electrolyte',
    tagline: '“Screening a 333,333 virtual formulation space with bounded working-set execution and honest surrogate results.”',
    keyMetric: '333,333 Candidates',
  },
  {
    index: 7,
    title: 'Contributions & Readiness',
    tab: 'readiness',
    tagline: '“48 out of 50 boolean validation gates passed with complete evidence ledger provenance.”',
    keyMetric: '48 / 50 Gates Pass',
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
      if (e.key === 'ArrowRight' || e.key === 'Space') {
        e.preventDefault();
        onNextScene();
      } else if (e.key === 'ArrowLeft') {
        e.preventDefault();
        onPrevScene();
      } else if (e.key === 'Escape') {
        e.preventDefault();
        onExit();
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [onNextScene, onPrevScene, onExit]);

  const progressPercent = ((currentScene + 1) / totalScenes) * 100;

  return (
    <div className="fixed top-0 left-0 right-0 z-50 bg-slate-950 text-white border-b border-slate-800 shadow-xl">
      {/* Progress Bar */}
      <div className="h-1 w-full bg-slate-800">
        <div 
          className="h-full bg-emerald-500 transition-all duration-300 ease-out"
          style={{ width: `${progressPercent}%` }}
        />
      </div>

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-3 flex items-center justify-between">
        {/* Left: Scene Status */}
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-2">
            <span className="px-2 py-0.5 rounded bg-emerald-950 border border-emerald-700 text-emerald-400 font-mono text-xs font-bold">
              SCENE {currentScene + 1}/{totalScenes}
            </span>
            <span className="font-bold text-sm tracking-tight text-slate-100 hidden sm:inline">
              {scene.title}
            </span>
          </div>

          <div className="hidden md:flex items-center gap-1">
            {SCENES.map((s, idx) => (
              <button
                key={s.index}
                onClick={() => onSelectScene(idx)}
                className={`w-7 h-7 rounded text-xs font-mono font-medium transition cursor-pointer flex items-center justify-center ${
                  idx === currentScene
                    ? 'bg-emerald-600 text-white font-bold'
                    : 'bg-slate-800 text-slate-400 hover:bg-slate-700 hover:text-slate-200'
                }`}
                title={`Jump to Scene ${idx + 1}: ${s.title}`}
              >
                {idx + 1}
              </button>
            ))}
          </div>
        </div>

        {/* Center: Quote / Tagline */}
        <div className="hidden lg:block max-w-xl text-center">
          <p className="text-xs text-slate-300 italic truncate font-sans">
            {scene.tagline}
          </p>
        </div>

        {/* Right: Navigation Controls */}
        <div className="flex items-center gap-2">
          <button
            onClick={onToggleNotes}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-md bg-slate-800 hover:bg-slate-700 border border-slate-700 text-xs font-medium text-slate-200 transition cursor-pointer"
            title="Open Speaker Script"
          >
            <BookOpen className="w-3.5 h-3.5 text-emerald-400" />
            <span className="hidden sm:inline">Notes</span>
          </button>

          <div className="flex items-center border border-slate-700 rounded-md overflow-hidden">
            <button
              onClick={onPrevScene}
              disabled={currentScene === 0}
              className="px-2.5 py-1.5 bg-slate-800 hover:bg-slate-700 disabled:opacity-30 disabled:cursor-not-allowed text-xs transition cursor-pointer border-r border-slate-700 flex items-center gap-1"
              title="Previous Scene (Left Arrow)"
            >
              <ChevronLeft className="w-4 h-4" />
              <span className="hidden sm:inline">Prev</span>
            </button>
            <button
              onClick={onNextScene}
              disabled={currentScene === totalScenes - 1}
              className="px-2.5 py-1.5 bg-emerald-700 hover:bg-emerald-600 disabled:opacity-30 disabled:cursor-not-allowed text-xs font-semibold transition cursor-pointer flex items-center gap-1"
              title="Next Scene (Right Arrow / Space)"
            >
              <span className="hidden sm:inline">Next</span>
              <ChevronRight className="w-4 h-4" />
            </button>
          </div>

          <button
            onClick={onExit}
            className="p-1.5 rounded-md text-slate-400 hover:text-white hover:bg-slate-800 transition cursor-pointer ml-1"
            title="Exit Presenter Mode (Esc)"
          >
            <X className="w-5 h-5" />
          </button>
        </div>
      </div>
    </div>
  );
};
