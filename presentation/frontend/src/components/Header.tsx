import React from 'react';
import { DataMode } from '../types/mission_control';
import { ModeBadge } from './ModeBadge';
import { 
  Compass, 
  Layers, 
  FlaskConical, 
  BarChart3, 
  Zap, 
  GitBranch, 
  ShieldCheck, 
  Presentation, 
  BookOpen
} from 'lucide-react';

export type NavTab = 
  | 'overview' 
  | 'cockpit' 
  | 'alab' 
  | 'benchmarks' 
  | 'electrolyte' 
  | 'architecture' 
  | 'readiness';

interface HeaderProps {
  currentTab: NavTab;
  onSelectTab: (tab: NavTab) => void;
  currentMode: DataMode;
  onLaunchPresenter: () => void;
  onToggleNotes: () => void;
  headCommit?: string;
  branch?: string;
}

export const Header: React.FC<HeaderProps> = ({
  currentTab,
  onSelectTab,
  currentMode,
  onLaunchPresenter,
  onToggleNotes,
  headCommit = 'dc1f5fda',
  branch = 'integration/multimodal-scientific-engine',
}) => {
  const tabs: { id: NavTab; label: string; icon: React.ReactNode }[] = [
    { id: 'overview', label: 'Research Overview', icon: <Compass className="w-4 h-4" /> },
    { id: 'cockpit', label: 'Decision Cockpit', icon: <FlaskConical className="w-4 h-4" /> },
    { id: 'alab', label: 'A-Lab Evidence Atlas', icon: <Layers className="w-4 h-4" /> },
    { id: 'benchmarks', label: 'Policy Benchmark Lab', icon: <BarChart3 className="w-4 h-4" /> },
    { id: 'electrolyte', label: 'Electrolyte Discovery', icon: <Zap className="w-4 h-4" /> },
    { id: 'architecture', label: 'Architecture & Ledger', icon: <GitBranch className="w-4 h-4" /> },
    { id: 'readiness', label: 'Research Readiness', icon: <ShieldCheck className="w-4 h-4" /> },
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
              <span className="text-xs text-slate-400 font-mono">v2.4</span>
              <span className="text-xs bg-slate-100 text-slate-600 px-2 py-0.5 rounded border border-slate-200 font-mono">
                {branch.replace('integration/', '')}
              </span>
            </div>
            <p className="text-xs text-slate-500 hidden sm:block">
              Multimodal Scientific Decision Engine & Autonomous Discovery Control
            </p>
          </div>
        </div>

        {/* Status & Actions */}
        <div className="flex items-center gap-3">
          <ModeBadge mode={currentMode} />

          <button
            onClick={onToggleNotes}
            className="hidden md:inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium text-slate-700 bg-slate-50 hover:bg-slate-100 border border-slate-200 rounded-md transition"
            title="Open Advisor Demonstration Script & Speaker Notes"
          >
            <BookOpen className="w-3.5 h-3.5 text-slate-500" />
            <span>Speaker Notes</span>
          </button>

          <button
            onClick={onLaunchPresenter}
            className="inline-flex items-center gap-2 px-3.5 py-1.5 text-xs font-semibold text-white bg-emerald-700 hover:bg-emerald-600 rounded-md shadow-xs transition cursor-pointer"
          >
            <Presentation className="w-4 h-4 text-emerald-200" />
            <span>Presenter Mode</span>
          </button>
        </div>
      </div>

      {/* Navigation Tabs */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 flex overflow-x-auto no-scrollbar border-t border-slate-100">
        <nav className="flex space-x-1" aria-label="Tabs">
          {tabs.map((tab) => {
            const isActive = currentTab === tab.id;
            return (
              <button
                key={tab.id}
                onClick={() => onSelectTab(tab.id)}
                className={`flex items-center gap-2 px-3.5 py-2.5 text-xs font-medium border-b-2 whitespace-nowrap transition cursor-pointer ${
                  isActive
                    ? 'border-emerald-600 text-emerald-800 font-semibold bg-emerald-50/40'
                    : 'border-transparent text-slate-600 hover:text-slate-900 hover:border-slate-300'
                }`}
              >
                <span className={isActive ? 'text-emerald-700' : 'text-slate-400'}>{tab.icon}</span>
                <span>{tab.label}</span>
              </button>
            );
          })}
        </nav>
      </div>
    </header>
  );
};
