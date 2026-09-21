import React from 'react';
import { Battery, Zap, RotateCcw, ShieldCheck, BookOpen, Layers } from 'lucide-react';
import { ScenarioId, ExhibitionScenario } from '../../data/types';

interface NavbarProps {
  scenarios: Record<ScenarioId, ExhibitionScenario>;
  activeScenarioId: ScenarioId;
  currentScene: number;
  onScenarioChange: (id: ScenarioId) => void;
  onSceneSelect: (sceneNumber: number) => void;
  onResetCamera: () => void;
  onToggleEvidenceDrawer: () => void;
}

export const Navbar: React.FC<NavbarProps> = ({
  scenarios,
  activeScenarioId,
  currentScene,
  onScenarioChange,
  onSceneSelect,
  onResetCamera,
  onToggleEvidenceDrawer
}) => {
  const activeScenario = scenarios[activeScenarioId];

  return (
    <header className="absolute top-0 left-0 right-0 z-30 pointer-events-auto px-6 py-4 flex items-center justify-between">
      {/* Brand & Scientific Identity */}
      <div className="flex items-center space-x-3 bg-white/85 backdrop-blur-md px-4 py-2 rounded-2xl shadow-xs border border-[#DCE8EC]">
        <div className="w-9 h-9 rounded-xl bg-[#087F8C] flex items-center justify-center text-white shadow-xs">
          <Battery className="w-5 h-5 text-white" />
        </div>
        <div>
          <div className="flex items-center space-x-2">
            <span className="font-bold text-lg tracking-tight text-[#142A35]">AIcoScientist</span>
            <span className="text-[10px] font-semibold tracking-wider uppercase px-2 py-0.5 rounded-full bg-[#E8F4F2] text-[#087F8C] border border-[#087F8C]/20">
              Battery Process Lab
            </span>
          </div>
          <p className="text-xs text-slate-500 font-medium">Multi-Stage Process Optimization & Digital Twin</p>
        </div>
      </div>

      {/* Scenario Switcher (Graphite vs NMC622) */}
      <div className="bg-white/85 backdrop-blur-md p-1.5 rounded-2xl shadow-xs border border-[#DCE8EC] flex items-center space-x-1">
        <button
          onClick={() => onScenarioChange('warwick_nmc622_calendering')}
          className={`flex items-center space-x-2 px-4 py-2 rounded-xl text-xs font-semibold transition-all duration-200 cursor-pointer ${
            activeScenarioId === 'warwick_nmc622_calendering'
              ? 'bg-[#142A35] text-white shadow-xs'
              : 'text-slate-600 hover:text-slate-900 hover:bg-slate-100'
          }`}
        >
          <Zap className={`w-3.5 h-3.5 ${activeScenarioId === 'warwick_nmc622_calendering' ? 'text-[#F59E42]' : 'text-slate-400'}`} />
          <span>Warwick NMC622 Cathode</span>
          <span className="text-[10px] font-mono px-1.5 py-0.2 rounded bg-white/20 text-white">Pilot 5C</span>
        </button>

        <button
          onClick={() => onScenarioChange('drakopoulos_graphite')}
          className={`flex items-center space-x-2 px-4 py-2 rounded-xl text-xs font-semibold transition-all duration-200 cursor-pointer ${
            activeScenarioId === 'drakopoulos_graphite'
              ? 'bg-[#142A35] text-white shadow-xs'
              : 'text-slate-600 hover:text-slate-900 hover:bg-slate-100'
          }`}
        >
          <Layers className={`w-3.5 h-3.5 ${activeScenarioId === 'drakopoulos_graphite' ? 'text-[#087F8C]' : 'text-slate-400'}`} />
          <span>Drakopoulos Graphite Anode</span>
          <span className="text-[10px] font-mono px-1.5 py-0.2 rounded bg-white/20 text-white">D30</span>
        </button>
      </div>

      {/* Control Actions & Provenance Button */}
      <div className="flex items-center space-x-2 bg-white/85 backdrop-blur-md p-1.5 rounded-2xl shadow-xs border border-[#DCE8EC]">
        <button
          onClick={onResetCamera}
          title="Reset Camera View"
          className="p-2 text-slate-600 hover:text-[#142A35] hover:bg-slate-100 rounded-xl transition-colors cursor-pointer"
        >
          <RotateCcw className="w-4 h-4" />
        </button>

        <button
          onClick={onToggleEvidenceDrawer}
          className="flex items-center space-x-1.5 px-3 py-1.5 text-xs font-semibold text-[#087F8C] bg-[#E8F4F2] hover:bg-[#D4ECE8] rounded-xl border border-[#087F8C]/20 transition-all cursor-pointer"
        >
          <ShieldCheck className="w-4 h-4 text-[#087F8C]" />
          <span>Scientific Provenance</span>
        </button>
      </div>
    </header>
  );
};
