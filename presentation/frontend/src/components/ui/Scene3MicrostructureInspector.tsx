import React from 'react';
import { Play, Pause, RotateCcw, ArrowRight, Gauge, Layers, Activity, AlertTriangle } from 'lucide-react';
import { ExhibitionScenario } from '../../data/types';

interface Scene3MicrostructureInspectorProps {
  scenario: ExhibitionScenario;
  compression: number; // 0 to 1
  isAutoMorphing: boolean;
  onCompressionChange: (val: number) => void;
  onToggleAutoMorph: () => void;
  onResetCompression: () => void;
  onProceedToOptimization: () => void;
}

export const Scene3MicrostructureInspector: React.FC<Scene3MicrostructureInspectorProps> = ({
  scenario,
  compression,
  isAutoMorphing,
  onCompressionChange,
  onToggleAutoMorph,
  onResetCompression,
  onProceedToOptimization
}) => {
  const ms = scenario.microstructure;

  // Calculated current physical metrics based on compression progress
  const currentThickness = (ms.initialThicknessUm * (1 - compression * ms.compressionRatio)).toFixed(1);
  const currentPorosity = (ms.initialPorosityPct * (1 - compression * 0.35)).toFixed(1);
  const isOvercompacted = Number(currentPorosity) < 28.0;

  return (
    <div className="absolute top-24 left-8 max-w-lg z-20 pointer-events-auto">
      <div className="bg-white/92 backdrop-blur-xl p-6 rounded-3xl shadow-xl border border-[#DCE8EC] space-y-5 animate-in fade-in slide-in-from-left-4 duration-500">
        {/* Header & Chemistry */}
        <div className="space-y-1">
          <div className="flex items-center justify-between">
            <span className="text-xs font-bold text-[#087F8C] uppercase tracking-wider">
              3D Nanoscale Microstructure
            </span>
            <span className="text-[10px] font-mono px-2 py-0.5 rounded-full bg-slate-100 text-slate-600 border border-slate-200">
              Drag to Rotate 360°
            </span>
          </div>
          <h2 className="text-xl font-bold text-[#142A35]">
            {ms.chemistryLabel}
          </h2>
          <p className="text-xs text-slate-500">
            {ms.substrateFoilName} • {ms.activeParticleType}
          </p>
        </div>

        {/* Compression Morph Controls */}
        <div className="p-4 rounded-2xl bg-slate-50 border border-slate-100 space-y-3">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-1.5 text-xs font-semibold text-[#142A35]">
              <Gauge className="w-4 h-4 text-[#087F8C]" />
              <span>Calender Nip Compaction</span>
            </div>
            <span className="text-xs font-mono font-bold text-[#087F8C]">
              {(compression * 100).toFixed(0)}%
            </span>
          </div>

          {/* Interactive Range Slider */}
          <input
            type="range"
            min="0"
            max="1"
            step="0.01"
            value={compression}
            onChange={(e) => onCompressionChange(parseFloat(e.target.value))}
            className="w-full h-2 bg-slate-200 rounded-lg appearance-none cursor-pointer accent-[#087F8C]"
          />

          <div className="flex items-center justify-between text-[11px] text-slate-500">
            <span>Uncalendered (0%)</span>
            <span>Fully Compressed (100%)</span>
          </div>

          {/* Animation & Reset Buttons */}
          <div className="flex items-center space-x-2 pt-1">
            <button
              onClick={onToggleAutoMorph}
              className={`flex-1 flex items-center justify-center space-x-1.5 py-2 rounded-xl text-xs font-semibold transition-all cursor-pointer ${
                isAutoMorphing
                  ? 'bg-[#F59E42] text-white shadow-xs'
                  : 'bg-white text-slate-700 border border-slate-200 hover:bg-slate-50'
              }`}
            >
              {isAutoMorphing ? <Pause className="w-3.5 h-3.5" /> : <Play className="w-3.5 h-3.5" />}
              <span>{isAutoMorphing ? 'Pause Morph Loop' : 'Play Morph Loop'}</span>
            </button>

            <button
              onClick={onResetCompression}
              title="Reset Compaction"
              className="p-2 bg-white text-slate-600 border border-slate-200 hover:bg-slate-50 rounded-xl transition-all cursor-pointer"
            >
              <RotateCcw className="w-3.5 h-3.5" />
            </button>
          </div>
        </div>

        {/* Real-Time Physical Microstructure Metrics */}
        <div className="grid grid-cols-2 gap-2.5">
          <div className="p-3 rounded-2xl bg-white border border-slate-100 shadow-xs">
            <div className="text-[11px] text-slate-500 font-medium">Layer Thickness</div>
            <div className="text-base font-bold font-mono text-[#142A35] mt-0.5">
              {currentThickness} <span className="text-xs text-slate-400 font-normal">µm</span>
            </div>
            <div className="text-[10px] text-slate-400 mt-0.5">
              Initial: {ms.initialThicknessUm} µm
            </div>
          </div>

          <div className="p-3 rounded-2xl bg-white border border-slate-100 shadow-xs">
            <div className="text-[11px] text-slate-500 font-medium">Inter-Particle Porosity</div>
            <div className={`text-base font-bold font-mono mt-0.5 ${isOvercompacted ? 'text-amber-600' : 'text-[#087F8C]'}`}>
              {currentPorosity}%
            </div>
            <div className="text-[10px] text-slate-400 mt-0.5">
              Target: ~31.9%
            </div>
          </div>
        </div>

        {/* Scientific Insight Card */}
        <div className="p-3.5 rounded-2xl bg-[#E8F4F2]/60 border border-[#087F8C]/20 text-xs text-slate-700 leading-relaxed">
          <div className="font-semibold text-[#087F8C] mb-1 flex items-center space-x-1.5">
            <Layers className="w-3.5 h-3.5 text-[#087F8C]" />
            <span>Transport Physics Rationale</span>
          </div>
          <p>{ms.scientificInsight}</p>
        </div>

        {/* Over-compaction Warning if applicable */}
        {isOvercompacted && (
          <div className="p-3 rounded-2xl bg-amber-50 border border-amber-200 flex items-start space-x-2 text-xs text-amber-800">
            <AlertTriangle className="w-4 h-4 text-amber-600 shrink-0 mt-0.5" />
            <span>Over-compaction risk: Pore tortuosity may impede liquid electrolyte replenishment during high-rate discharge.</span>
          </div>
        )}

        {/* CTA to Optimization Studio */}
        <div className="pt-1 border-t border-slate-100">
          <button
            onClick={onProceedToOptimization}
            className="w-full flex items-center justify-center space-x-2 px-4 py-3.5 bg-[#087F8C] hover:bg-[#076a75] text-white rounded-2xl text-xs font-semibold shadow-md transition-all cursor-pointer"
          >
            <span>Proceed to AI Optimization Studio</span>
            <ArrowRight className="w-4 h-4" />
          </button>
        </div>
      </div>
    </div>
  );
};
