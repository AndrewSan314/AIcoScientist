import { DataMode, WorkspaceTab, LegacyNavTab } from '../types/mission_control';

export type NavTab = LegacyNavTab;
import { Compass, Layers, ShieldCheck, Play } from 'lucide-react';

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
  onLaunchPresenter,
}) => {
  const workspaces: { id: WorkspaceTab; label: string; icon: React.ReactNode }[] = [
    {
      id: 'discovery',
      label: 'Run discovery',
      icon: <Compass className="w-4 h-4" />
    },
    {
      id: 'benchmarks',
      label: 'Evidence',
      icon: <Layers className="w-4 h-4" />
    },
    {
      id: 'system',
      label: 'How it works',
      icon: <ShieldCheck className="w-4 h-4" />
    },
  ];

  return (
    <header className="sticky top-0 z-40 bg-[#FCFCFA] border-b border-[#D9DFDB] shadow-2xs">
      <div className="max-w-[1720px] mx-auto px-4 sm:px-8 lg:px-12 h-14 flex items-center justify-between">
        {/* Left: Project Wordmark */}
        <div className="flex items-center gap-2.5">
          <div className="w-7 h-7 rounded-lg bg-[#B91C1C] flex items-center justify-center text-white font-bold text-sm shadow-xs">
            Ψ
          </div>
          <span className="font-bold text-[#17201F] tracking-tight text-base sm:text-lg">
            AIcoScientist
          </span>
        </div>

        {/* Center: 3 Clean Navigation Items */}
        <nav className="flex items-center gap-1 sm:gap-1.5 overflow-x-auto" aria-label="Workspaces">
          {workspaces.map((tab) => {
            const isActive = currentWorkspace === tab.id;
            return (
              <button
                key={tab.id}
                onClick={() => onSelectWorkspace(tab.id)}
                className={`flex items-center gap-2 px-3 sm:px-3.5 py-1.5 rounded-lg text-xs sm:text-sm font-medium transition cursor-pointer ${
                  isActive
                    ? 'bg-[#FEF2F2] text-[#991B1B] font-semibold border border-[#FECACA]'
                    : 'text-[#66706C] hover:text-[#17201F] hover:bg-[#F4F3EE] border border-transparent'
                }`}
              >
                <span className={isActive ? 'text-[#DC2626]' : 'text-[#8F9995]'}>{tab.icon}</span>
                <span>{tab.label}</span>
              </button>
            );
          })}
        </nav>

        {/* Right: Start Guided Demo Action */}
        <div className="flex items-center">
          {onLaunchPresenter && (
            <button
              onClick={onLaunchPresenter}
              className="inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-semibold text-white bg-[#B91C1C] hover:bg-[#991B1B] rounded-lg shadow-xs transition cursor-pointer"
              title="Launch guided presentation (Hotkey: P)"
            >
              <Play className="w-3.5 h-3.5 fill-white" />
              <span>Start guided demo</span>
            </button>
          )}
        </div>
      </div>
    </header>
  );
};
