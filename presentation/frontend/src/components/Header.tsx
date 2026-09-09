import { DataMode, WorkspaceTab, LegacyNavTab } from '../types/mission_control';

export type NavTab = LegacyNavTab;
import { Compass, Layers, ShieldCheck } from 'lucide-react';

interface HeaderProps {
  currentWorkspace: WorkspaceTab;
  onSelectWorkspace: (tab: WorkspaceTab) => void;
  currentMode?: DataMode;
  onLaunchPresenter?: () => void;
  onToggleNotes?: () => void;
  headCommit?: string;
  branch?: string;
}

export const Header: React.FC<HeaderProps> = ({
  currentWorkspace,
  onSelectWorkspace,
}) => {
  const workspaces: { id: WorkspaceTab; label: string; hotkey: string; icon: React.ReactNode }[] = [
    {
      id: 'discovery',
      label: 'Discovery Lab',
      hotkey: '1',
      icon: <Compass className="w-4 h-4" />
    },
    {
      id: 'benchmarks',
      label: 'Evidence & Benchmarks',
      hotkey: '2',
      icon: <Layers className="w-4 h-4" />
    },
    {
      id: 'system',
      label: 'Research System',
      hotkey: '3',
      icon: <ShieldCheck className="w-4 h-4" />
    },
  ];

  return (
    <header className="sticky top-0 z-40 bg-white border-b border-slate-200 shadow-xs">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-14 flex items-center justify-between">
        {/* Left: Project Name Only */}
        <div className="flex items-center gap-2.5">
          <div className="w-8 h-8 rounded-lg bg-emerald-700 flex items-center justify-center text-white font-bold text-base shadow-xs">
            Ψ
          </div>
          <span className="font-bold text-slate-900 tracking-tight text-lg">
            AIcoScientist
          </span>
        </div>

        {/* Same Line: 3 Main Tabs (No Sublines) */}
        <nav className="flex items-center gap-1 sm:gap-2 overflow-x-auto" aria-label="Workspaces">
          {workspaces.map((tab) => {
            const isActive = currentWorkspace === tab.id;
            return (
              <button
                key={tab.id}
                onClick={() => onSelectWorkspace(tab.id)}
                className={`flex items-center gap-2 px-3 sm:px-3.5 py-1.5 rounded-lg text-xs sm:text-sm font-semibold transition cursor-pointer ${
                  isActive
                    ? 'bg-emerald-50 text-emerald-900 border border-emerald-200 shadow-2xs font-bold'
                    : 'text-slate-600 hover:text-slate-900 hover:bg-slate-50 border border-transparent'
                }`}
              >
                <span className={isActive ? 'text-emerald-700' : 'text-slate-400'}>{tab.icon}</span>
                <span>{tab.label}</span>
                <kbd className={`text-3xs font-mono px-1 py-0.5 rounded ${
                  isActive ? 'bg-emerald-200/60 text-emerald-900' : 'bg-slate-100 text-slate-400'
                }`}>
                  {tab.hotkey}
                </kbd>
              </button>
            );
          })}
        </nav>
      </div>
    </header>
  );
};
