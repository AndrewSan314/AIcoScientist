import React from 'react';
import { Compass, Eye, Microscope, Cpu, Award, ChevronLeft, ChevronRight } from 'lucide-react';

interface SceneNavigatorProps {
  currentScene: number;
  onSceneSelect: (sceneNumber: number) => void;
}

const SCENE_DEFS = [
  { id: 1, label: 'Laboratory Overview', shortLabel: 'Overview', icon: Compass },
  { id: 2, label: 'Connected Production Line', shortLabel: 'Line', icon: Eye },
  { id: 3, label: 'Inside the Electrode', shortLabel: 'Microstructure', icon: Microscope },
  { id: 4, label: 'AI Optimization Studio', shortLabel: 'AI Studio', icon: Cpu },
  { id: 5, label: 'Scientific Evidence', shortLabel: 'Evidence', icon: Award }
];

export const SceneNavigator: React.FC<SceneNavigatorProps> = ({
  currentScene,
  onSceneSelect
}) => {
  const canPrev = currentScene > 1;
  const canNext = currentScene < 5;

  return (
    <nav className="absolute bottom-6 left-1/2 -translate-x-1/2 z-30 pointer-events-auto">
      <div className="bg-white/90 backdrop-blur-md px-3 py-2 rounded-2xl shadow-lg border border-[#DCE8EC] flex items-center space-x-1.5">
        {/* Previous Button */}
        <button
          onClick={() => canPrev && onSceneSelect(currentScene - 1)}
          disabled={!canPrev}
          aria-label="Previous Scene"
          className={`p-2 rounded-xl transition-all ${
            canPrev
              ? 'text-slate-700 hover:bg-slate-100 hover:text-slate-900 cursor-pointer'
              : 'text-slate-300 cursor-not-allowed opacity-50'
          }`}
        >
          <ChevronLeft className="w-5 h-5" />
        </button>

        {/* Scene Tabs */}
        <div className="flex items-center space-x-1 px-1">
          {SCENE_DEFS.map((s) => {
            const Icon = s.icon;
            const isActive = currentScene === s.id;
            return (
              <button
                key={s.id}
                onClick={() => onSceneSelect(s.id)}
                className={`flex items-center space-x-2 px-3.5 py-2 rounded-xl text-xs font-semibold transition-all duration-200 cursor-pointer ${
                  isActive
                    ? 'bg-[#087F8C] text-white shadow-xs'
                    : 'text-slate-600 hover:text-slate-900 hover:bg-slate-100'
                }`}
              >
                <Icon className={`w-4 h-4 ${isActive ? 'text-white' : 'text-slate-500'}`} />
                <span className="hidden md:inline">{s.label}</span>
                <span className="md:hidden">{s.shortLabel}</span>
              </button>
            );
          })}
        </div>

        {/* Next Button */}
        <button
          onClick={() => canNext && onSceneSelect(currentScene + 1)}
          disabled={!canNext}
          aria-label="Next Scene"
          className={`p-2 rounded-xl transition-all ${
            canNext
              ? 'text-slate-700 hover:bg-slate-100 hover:text-slate-900 cursor-pointer'
              : 'text-slate-300 cursor-not-allowed opacity-50'
          }`}
        >
          <ChevronRight className="w-5 h-5" />
        </button>
      </div>
    </nav>
  );
};
