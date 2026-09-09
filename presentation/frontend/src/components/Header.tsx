import { DataMode, WorkspaceTab, LegacyNavTab } from '../types/mission_control';

export type NavTab = LegacyNavTab;
import { ModeBadge } from './ModeBadge';
import { 
  Compass, 
  Layers, 
  ShieldCheck, 
  Presentation, 
  BookOpen,
  Play
} from 'lucide-react';

interface HeaderProps {
  currentWorkspace: WorkspaceTab;
  onSelectWorkspace: (tab: WorkspaceTab) => void;
  currentMode: DataMode;
  onLaunchPresenter: () => void;
  onToggleNotes: () => void;
  headCommit?: string;
  branch?: string;
}

export const Header: React.FC<HeaderProps> = ({
  currentWorkspace,
  onSelectWorkspace,
  currentMode,
  onLaunchPresenter,
  onToggleNotes,
  headCommit = 'bf08542',
  branch = 'integration/multimodal-scientific-engine',
}) => {
  const workspaces: { id: WorkspaceTab; label: string; sub: string; hotkey: string; icon: React.ReactNode }[] = [
    {
      id: 'discovery',
      label: 'Discovery Lab',
      sub: 'Scientific Decision Engine & Active Loop',
      hotkey: '1',
      icon: <Compass className="w-4 h-4" />
    },
    {
      id: 'benchmarks',
      label: 'Evidence & Benchmarks',
      sub: '5 Research Questions & Retrospective Replay',
      hotkey: '2',
      icon: <Layers className="w-4 h-4" />
    },
    {
      id: 'system',
      label: 'Research System',
      sub: 'Architecture, Ledger & 50 Validation Gates',
      hotkey: '3',
      icon: <ShieldCheck className="w-4 h-4" />
    },
  ];

  return (
    <header className="sticky top-0 z-40 bg-white border-b border-slate-200 shadow-xs">
      {/* Top Banner */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="w-9 h-9 rounded-lg bg-emerald-700 flex items-center justify-center text-white font-bold text-lg shadow-xs">
            Ψ
          </div>
          <div>
            <div className="flex items-center gap-2">
              <span className="font-bold text-slate-900 tracking-tight text-base">AIcoScientist</span>
              <span className="text-2xs bg-emerald-50 text-emerald-800 font-mono px-2 py-0.5 rounded border border-emerald-200 font-semibold">
                Mission Control v2.5
              </span>
              <span className="text-2xs bg-slate-100 text-slate-600 px-2 py-0.5 rounded border border-slate-200 font-mono hidden md:inline-block">
                {branch.replace('integration/', '')}
              </span>
            </div>
            <p className="text-xs text-slate-500 hidden sm:block">
              Multimodal Scientific Decision Engine & Autonomous Discovery Control
            </p>
          </div>
        </div>

        {/* Status & Actions */}
        <div className="flex items-center gap-2.5">
          <ModeBadge mode={currentMode} />

          <button
            onClick={onToggleNotes}
            className="hidden md:inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium text-slate-700 bg-slate-50 hover:bg-slate-100 border border-slate-200 rounded-md transition cursor-pointer"
            title="Open Advisor Demonstration Script & Speaker Notes (Key: N)"
          >
            <BookOpen className="w-3.5 h-3.5 text-slate-500" />
            <span>Speaker Notes</span>
            <kbd className="hidden lg:inline text-3xs font-mono bg-white px-1 py-0.2 rounded border border-slate-300 text-slate-500">N</kbd>
          </button>

          <button
            onClick={onLaunchPresenter}
            className="inline-flex items-center gap-2 px-3.5 py-1.5 text-xs font-semibold text-white bg-emerald-700 hover:bg-emerald-600 rounded-md shadow-xs transition cursor-pointer"
            title="Start Guided Advisor Presentation (Key: P)"
          >
            <Play className="w-3.5 h-3.5 text-emerald-200 fill-emerald-200" />
            <span>Start Demo</span>
            <kbd className="hidden sm:inline text-3xs font-mono bg-emerald-800/80 px-1 py-0.2 rounded text-emerald-200">P</kbd>
          </button>
        </div>
      </div>

      {/* Navigation Workspaces Bar */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 border-t border-slate-100">
        <nav className="flex space-x-2 py-1" aria-label="Workspaces">
          {workspaces.map((tab) => {
            const isActive = currentWorkspace === tab.id;
            return (
              <button
                key={tab.id}
                onClick={() => onSelectWorkspace(tab.id)}
                className={`flex items-center gap-2.5 px-4 py-2 rounded-lg text-xs transition cursor-pointer text-left ${
                  isActive
                    ? 'bg-emerald-50 text-emerald-900 font-bold border border-emerald-200 shadow-2xs'
                    : 'text-slate-600 hover:text-slate-900 hover:bg-slate-50 border border-transparent'
                }`}
              >
                <span className={isActive ? 'text-emerald-700' : 'text-slate-400'}>{tab.icon}</span>
                <div className="flex flex-col">
                  <div className="flex items-center gap-1.5">
                    <span>{tab.label}</span>
                    <kbd className={`text-3xs font-mono px-1 py-0.2 rounded ${
                      isActive ? 'bg-emerald-200/60 text-emerald-900' : 'bg-slate-100 text-slate-400'
                    }`}>
                      {tab.hotkey}
                    </kbd>
                  </div>
                  <span className="text-3xs text-slate-400 font-normal hidden lg:block">
                    {tab.sub}
                  </span>
                </div>
              </button>
            );
          })}
        </nav>
      </div>
    </header>
  );
};
