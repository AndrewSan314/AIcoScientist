import React, { useState, useEffect } from 'react';
import { ArrowLeft, ArrowRight, Sparkles, CheckCircle, TrendingUp, AlertCircle, Award, ShieldAlert, Cpu, X, Play, Pause, RotateCw, Eye, EyeOff } from 'lucide-react';
import { ExhibitionScenario, ReplayStep } from '../../data/types';

interface Scene4OptimizationStudioProps {
  scenario: ExhibitionScenario;
  replayStep: number; // 0 = initial designs, 1-5 = sequential steps
  onStepChange: (step: number) => void;
  onSelectCandidate: (candId: string) => void;
  selectedCandidateId?: string;
  onProceedToEvidence: () => void;
  isOrbiting?: boolean;
  onToggleOrbit?: () => void;
}

export const Scene4OptimizationStudio: React.FC<Scene4OptimizationStudioProps> = ({
  scenario,
  replayStep,
  onStepChange,
  onSelectCandidate,
  selectedCandidateId,
  onProceedToEvidence,
  isOrbiting = false,
  onToggleOrbit
}) => {
  const isInitial = replayStep === 0;
  const currentStepData: ReplayStep | undefined = !isInitial ? scenario.replaySteps[replayStep - 1] : undefined;
  const totalSteps = scenario.replaySteps.length;

  const [isPlaying, setIsPlaying] = useState<boolean>(false);
  const [isInspectorOpen, setIsInspectorOpen] = useState<boolean>(true);

  // Auto-play stepper for video recording & exhibition presentations
  useEffect(() => {
    if (!isPlaying) return;
    const timer = setInterval(() => {
      onStepChange(replayStep >= totalSteps ? 0 : replayStep + 1);
    }, 2800);
    return () => clearInterval(timer);
  }, [isPlaying, replayStep, totalSteps, onStepChange]);

  const selectedCandidate = scenario.candidates.find((c) => c.id === selectedCandidateId);
  const isSelectedInit = selectedCandidate ? scenario.replayInitialIds.includes(selectedCandidate.id) : false;
  const isSelectedAcquired = selectedCandidate ? scenario.replaySteps.slice(0, replayStep).some((s) => s.selectedCandidateId === selectedCandidate.id) : false;
  const isSelectedCurrentProposal = selectedCandidate && currentStepData?.selectedCandidateId === selectedCandidate.id;
  const isSelectedRevealed = isSelectedInit || isSelectedAcquired || isSelectedCurrentProposal;

  // Precalculate convergence history curve
  const initialCandidates = scenario.candidates.filter((c) => scenario.replayInitialIds.includes(c.id));
  const initialBest = Math.max(...initialCandidates.map((c) => c.revealedTarget.value));
  const convergenceHistory = [
    { step: 0, label: 'Init', val: initialBest, isOptimal: false },
    ...scenario.replaySteps.map((s) => ({
      step: s.step,
      label: `S${s.step}`,
      val: s.bestSoFar,
      isOptimal: s.isOptimal,
    })),
  ];
  const maxVal = Math.max(...convergenceHistory.map((h) => h.val));
  const minVal = Math.min(...convergenceHistory.map((h) => h.val));
  const valRange = maxVal - minVal || 1;

  return (
    <div className="absolute top-20 left-8 right-8 bottom-24 z-20 pointer-events-none flex justify-between gap-6">
      {/* Left Column: Sequential AI Decision Cockpit */}
      <div className="w-full max-w-lg pointer-events-auto bg-white/94 backdrop-blur-xl p-6 rounded-3xl shadow-xl border border-[#DCE8EC] flex flex-col justify-between overflow-y-auto animate-in fade-in slide-in-from-left-4 duration-500">
        <div className="space-y-4">
          {/* Header & Replay Mode Badge */}
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-2">
              <span className="text-xs font-bold text-[#087F8C] uppercase tracking-wider">
                Recorded Replay · Seed 11
              </span>
              <span className="text-[10px] font-mono px-2 py-0.5 rounded-full bg-slate-100 text-slate-600 border border-slate-200">
                OFFLINE_REPLAY
              </span>
            </div>

            {/* Orbit & AutoPlay Controls */}
            <div className="flex items-center space-x-1.5">
              {onToggleOrbit && (
                <button
                  onClick={onToggleOrbit}
                  className={`px-2.5 py-1 rounded-lg text-xs font-semibold flex items-center space-x-1 transition-all cursor-pointer ${
                    isOrbiting ? 'bg-[#087F8C] text-white shadow-xs' : 'bg-slate-100 hover:bg-slate-200 text-slate-700'
                  }`}
                  title="Toggle 360° turntable orbit"
                >
                  <RotateCw className={`w-3 h-3 ${isOrbiting ? 'animate-spin' : ''}`} />
                  <span>{isOrbiting ? 'Orbiting' : '3D Orbit'}</span>
                </button>
              )}

              <button
                onClick={() => setIsPlaying(!isPlaying)}
                className={`p-1.5 rounded-lg transition-all cursor-pointer ${
                  isPlaying ? 'bg-amber-500 text-white shadow-xs' : 'bg-slate-100 hover:bg-slate-200 text-slate-700'
                }`}
                title={isPlaying ? 'Pause auto-step' : 'Auto-play sequential steps'}
              >
                {isPlaying ? <Pause className="w-3.5 h-3.5" /> : <Play className="w-3.5 h-3.5" />}
              </button>
            </div>
          </div>

          {/* Stepper Bar */}
          <div className="flex items-center space-x-1 bg-slate-100 p-1 rounded-xl">
            <button
              onClick={() => onStepChange(0)}
              className={`px-2.5 py-1 rounded-lg text-xs font-semibold transition-all cursor-pointer flex-1 text-center ${
                replayStep === 0 ? 'bg-white text-[#142A35] shadow-xs' : 'text-slate-500 hover:text-slate-900'
              }`}
            >
              Init (3)
            </button>
            {scenario.replaySteps.map((st) => (
              <button
                key={st.step}
                onClick={() => onStepChange(st.step)}
                className={`px-2.5 py-1 rounded-lg text-xs font-semibold transition-all cursor-pointer flex-1 text-center ${
                  replayStep === st.step
                    ? st.isOptimal
                      ? 'bg-[#F59E42] text-white shadow-xs'
                      : 'bg-[#087F8C] text-white shadow-xs'
                    : 'text-slate-500 hover:text-slate-900'
                }`}
              >
                Step {st.step}
              </button>
            ))}
          </div>

          {/* Current Step Title / State */}
          {isInitial ? (
            <div className="p-4 rounded-2xl bg-sky-50/70 border border-sky-100 space-y-2">
              <div className="flex items-center space-x-2 text-xs font-bold text-sky-800">
                <Cpu className="w-4 h-4 text-sky-600" />
                <span>Initial Design Cold-Start (3 Historical Observations)</span>
              </div>
              <p className="text-xs text-sky-700 leading-relaxed">
                Three recorded seed observations initialize this replay. Gray points have unknown outcomes. Teal/orange heights show revealed measurements only.
              </p>
              <div className="text-xs font-mono font-semibold text-slate-700 pt-1">
                Initial Best-so-Far: <span className="text-[#087F8C] font-bold">{Math.max(...scenario.candidates.filter(c => scenario.replayInitialIds.includes(c.id)).map(c => c.revealedTarget.value)).toFixed(4)} {scenario.targetUnit}</span>
              </div>
            </div>
          ) : currentStepData && (
            <div className="space-y-3">
              {/* Proposal Banner */}
              <div className={`p-4 rounded-2xl border transition-all ${
                currentStepData.isOptimal
                  ? 'bg-amber-50/80 border-amber-200'
                  : 'bg-slate-50 border-slate-200/80'
              }`}>
                <div className="flex items-center justify-between">
                  <span className="text-xs font-semibold text-slate-500">
                    Step 0{currentStepData.step} Acquisition
                  </span>
                  {currentStepData.isOptimal && (
                    <span className="inline-flex items-center space-x-1 px-2.5 py-0.5 rounded-full bg-[#F59E42] text-white text-[10px] font-bold shadow-xs">
                      <Award className="w-3 h-3" />
                      <span>Optimal Target Rediscovered!</span>
                    </span>
                  )}
                </div>

                <div className="text-lg font-bold text-[#142A35] mt-1">
                  {currentStepData.selectedCandidateDisplay}
                </div>

                {/* Controls Summary */}
                <div className="grid grid-cols-2 gap-2 mt-2.5 pt-2 border-t border-slate-200/60 text-xs">
                  {Object.entries(currentStepData.controlsSummary).map(([k, v]) => (
                    <div key={k} className="flex items-center justify-between text-slate-600">
                      <span className="text-slate-400">{k}:</span>
                      <span className="font-mono font-semibold text-slate-800">{v}</span>
                    </div>
                  ))}
                </div>
              </div>

              {/* Surrogate Prediction vs Ground Truth Observation */}
              <div className="grid grid-cols-2 gap-3">
                {/* Model Prediction */}
                <div className="p-3.5 rounded-2xl bg-white border border-[#DCE8EC] shadow-xs space-y-1">
                  <div className="text-[11px] text-slate-500 font-medium">Model Prediction (μ ± σ)</div>
                  <div className="text-base font-bold font-mono text-[#087F8C]">
                    {currentStepData.predictedMean.toFixed(2)}
                    <span className="text-xs text-slate-400 font-normal"> ± {currentStepData.predictedStd.toFixed(2)}</span>
                  </div>
                  <div className="text-[10px] text-slate-400">
                    Acquisition Score: <span className="font-mono font-semibold text-slate-600">{currentStepData.acquisitionValue.toFixed(4)}</span>
                  </div>
                </div>

                {/* Historical Revealed Evaluation */}
                <div className="p-3.5 rounded-2xl bg-white border border-[#DCE8EC] shadow-xs space-y-1">
                  <div className="text-[11px] text-slate-500 font-medium">Revealed Historical Mean</div>
                  <div className={`text-base font-bold font-mono ${currentStepData.isOptimal ? 'text-[#F59E42]' : 'text-[#142A35]'}`}>
                    {currentStepData.revealedTarget.toFixed(4)}
                  </div>
                  <div className="text-[10px] text-slate-500">
                    Best-so-Far: <span className="font-mono font-semibold text-[#087F8C]">{currentStepData.bestSoFar.toFixed(4)}</span>
                  </div>
                </div>
              </div>

              {/* Decision Explanation Text */}
              <div className="p-3.5 rounded-2xl bg-[#E8F4F2]/60 border border-[#087F8C]/20 text-xs text-slate-700 leading-relaxed">
                <div className="font-semibold text-[#087F8C] mb-1 flex items-center space-x-1.5">
                  <Sparkles className="w-3.5 h-3.5 text-[#087F8C]" />
                  <span>Recorded Decision Context</span>
                </div>
                <p>{currentStepData.explanation}</p>
              </div>
            </div>
          )}

          {/* Objective Convergence Curve */}
          <div className="p-3.5 rounded-2xl bg-white border border-[#DCE8EC] shadow-xs space-y-2">
            <div className="flex items-center justify-between text-[11px] font-semibold text-slate-700">
              <span className="flex items-center space-x-1.5">
                <TrendingUp className="w-3.5 h-3.5 text-[#087F8C]" />
                <span>Objective Convergence Trajectory</span>
              </span>
              <span className="font-mono text-[10px] text-slate-500">
                Best-so-Far: <span className="font-bold text-[#087F8C]">{(currentStepData ? currentStepData.bestSoFar : initialBest).toFixed(4)}</span>
              </span>
            </div>

            {/* SVG Sparkline */}
            <div className="h-16 w-full pt-1">
              <svg className="w-full h-full overflow-visible" viewBox="0 0 280 64">
                <defs>
                  <linearGradient id="curveGrad" x1="0%" y1="0%" x2="100%" y2="0%">
                    <stop offset="0%" stopColor="#087F8C" />
                    <stop offset="100%" stopColor="#F59E42" />
                  </linearGradient>
                </defs>
                <line x1="15" y1="50" x2="265" y2="50" stroke="#E2E8F0" strokeWidth="1" strokeDasharray="3 3" />
                <polyline
                  fill="none"
                  stroke="url(#curveGrad)"
                  strokeWidth="2.5"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  points={convergenceHistory.map((h, i) => {
                    const x = 20 + (i / (convergenceHistory.length - 1)) * 240;
                    const y = 50 - ((h.val - minVal) / valRange) * 38;
                    return `${x},${y}`;
                  }).join(' ')}
                />
                {convergenceHistory.map((h, i) => {
                  const x = 20 + (i / (convergenceHistory.length - 1)) * 240;
                  const y = 50 - ((h.val - minVal) / valRange) * 38;
                  const isActive = replayStep >= i;
                  const isCurrent = replayStep === i;
                  return (
                    <g key={i}>
                      <circle
                        cx={x}
                        cy={y}
                        r={isCurrent ? 4.5 : 3}
                        fill={isActive ? (h.isOptimal ? '#F59E42' : '#087F8C') : '#CBD5E1'}
                        stroke="#FFFFFF"
                        strokeWidth="1.5"
                      />
                      <text
                        x={x}
                        y={62}
                        fontSize="8"
                        textAnchor="middle"
                        fill={isActive ? '#475569' : '#94A3B8'}
                        fontFamily="monospace"
                      >
                        {h.label}
                      </text>
                    </g>
                  );
                })}
              </svg>
            </div>
          </div>
        </div>

        {/* Step Navigation Controls */}
        <div className="pt-4 border-t border-slate-100 flex items-center justify-between gap-3">
          <button
            onClick={() => onStepChange(Math.max(0, replayStep - 1))}
            disabled={replayStep === 0}
            className={`flex items-center space-x-1.5 px-4 py-2.5 rounded-xl text-xs font-semibold transition-all ${
              replayStep > 0
                ? 'bg-slate-100 hover:bg-slate-200 text-slate-700 cursor-pointer'
                : 'text-slate-300 opacity-50 cursor-not-allowed'
            }`}
          >
            <ArrowLeft className="w-4 h-4" />
            <span>Prev Step</span>
          </button>

          {replayStep < totalSteps ? (
            <button
              onClick={() => onStepChange(replayStep + 1)}
              className="flex-1 flex items-center justify-center space-x-1.5 px-5 py-2.5 bg-[#087F8C] hover:bg-[#076a75] text-white rounded-xl text-xs font-semibold shadow-xs transition-all cursor-pointer"
            >
              <span>Reveal Recorded Selection (Step {replayStep + 1})</span>
              <ArrowRight className="w-4 h-4" />
            </button>
          ) : (
            <button
              onClick={onProceedToEvidence}
              className="flex-1 flex items-center justify-center space-x-1.5 px-5 py-2.5 bg-[#142A35] hover:bg-[#1f3b49] text-white rounded-xl text-xs font-semibold shadow-xs transition-all cursor-pointer"
            >
              <span>View Scientific Evidence</span>
              <ArrowRight className="w-4 h-4" />
            </button>
          )}
        </div>
      </div>

      {/* Right Column: Candidate Pool Table & Inspector */}
      <div className={`pointer-events-auto bg-white/94 backdrop-blur-xl rounded-3xl shadow-xl border border-[#DCE8EC] flex flex-col justify-between overflow-hidden animate-in fade-in slide-in-from-right-4 duration-500 transition-all ${
        isInspectorOpen ? 'w-96 p-5' : 'w-14 p-2.5 items-center'
      }`}>
        {!isInspectorOpen ? (
          <div className="flex flex-col items-center justify-between h-full py-2">
            <button
              onClick={() => setIsInspectorOpen(true)}
              className="p-2 rounded-xl bg-slate-100 hover:bg-slate-200 text-[#087F8C] cursor-pointer"
              title="Expand candidate pool"
            >
              <Eye className="w-4 h-4" />
            </button>
            <span className="[writing-mode:vertical-lr] text-xs font-bold text-slate-500 tracking-wider">
              CANDIDATES ({scenario.candidates.length})
            </span>
            <div />
          </div>
        ) : (
          <>
            <div className="flex flex-col flex-1 min-h-0">
              <div className="flex items-center justify-between pb-3 border-b border-slate-100 shrink-0">
                <div>
                  <div className="text-xs font-bold text-[#142A35]">Candidate Search Space</div>
                  <div className="text-[11px] text-slate-500">
                    {scenario.candidates.length} candidates · height appears after reveal
                  </div>
                </div>
                <div className="flex items-center space-x-1.5">
                  <span className="text-[10px] font-mono px-2 py-0.5 rounded-md bg-slate-100 text-slate-600">
                    {scenario.targetUnit}
                  </span>
                  <button
                    onClick={() => setIsInspectorOpen(false)}
                    className="p-1 rounded-md text-slate-400 hover:text-slate-600 hover:bg-slate-100 cursor-pointer"
                    title="Collapse panel"
                  >
                    <EyeOff className="w-3.5 h-3.5" />
                  </button>
                </div>
              </div>

              <p className="text-[10px] text-slate-500 py-2">
                {scenario.id === 'warwick_nmc622_calendering'
                  ? 'X: roll temperature · Z: target density, separated by loading regime · Y: revealed 5C/0.2C ratio'
                  : 'X: coating speed · Z: coating gap, offset by calendering status · Y: revealed D30 (mAh/g)'}
                {' · Gray: unknown · Blue: initial · Teal: observed · Orange: recorded best. Lines join acquired observations in replay order, not a fitted response.'}
              </p>
          {/* Candidate Inspector Card (When candidate is selected) */}
          {selectedCandidate && (
            <div className="my-3 p-3.5 rounded-2xl bg-[#F0F7F7] border border-[#087F8C]/30 text-xs space-y-2 shrink-0 animate-in fade-in zoom-in-95 duration-200">
              <div className="flex items-center justify-between">
                <div className="flex items-center space-x-1.5 truncate mr-2">
                  <span className="font-bold text-[#142A35] truncate">{selectedCandidate.displayId}</span>
                  <span className={`text-[9px] font-mono px-2 py-0.5 rounded-md font-semibold shrink-0 ${
                    isSelectedCurrentProposal
                      ? 'bg-amber-100 text-amber-800'
                      : isSelectedInit
                      ? 'bg-sky-100 text-sky-800'
                      : isSelectedAcquired
                      ? 'bg-teal-100 text-teal-800'
                      : 'bg-slate-200 text-slate-700'
                  }`}>
                    {isSelectedCurrentProposal ? 'ACTIVE PROPOSAL' : isSelectedInit ? 'INITIAL PRIOR' : isSelectedAcquired ? 'ACQUIRED' : 'HELD OUT (POOL)'}
                  </span>
                </div>
                <button
                  onClick={() => onSelectCandidate('')}
                  className="text-slate-400 hover:text-slate-600 p-1 rounded-md cursor-pointer shrink-0"
                  title="Close Inspector"
                >
                  <X className="w-3.5 h-3.5" />
                </button>
              </div>

              {/* Controls */}
              <div className="grid grid-cols-2 gap-1.5 pt-1 border-t border-[#087F8C]/15 text-[11px]">
                {Object.entries(selectedCandidate.controls).map(([k, v]) => (
                  <div key={k} className="flex items-center justify-between bg-white/80 px-2 py-1 rounded-lg border border-slate-100 truncate">
                    <span className="text-slate-500 truncate mr-1">{k}:</span>
                    <span className="font-mono font-semibold text-slate-800 shrink-0">{String(v)}</span>
                  </div>
                ))}
              </div>

              {/* Target Outcome / Horizon Check */}
              <div className="pt-1.5 border-t border-[#087F8C]/15 flex items-center justify-between text-xs">
                <span className="text-slate-500 font-medium">
                  {isSelectedRevealed ? 'Revealed Target:' : 'Target Evaluation:'}
                </span>
                <span className={`font-mono font-bold ${
                  isSelectedRevealed ? 'text-[#087F8C]' : 'text-slate-400 italic text-[11px]'
                }`}>
                  {isSelectedRevealed
                    ? `${selectedCandidate.revealedTarget.value.toFixed(scenario.id === 'warwick_nmc622_calendering' ? 4 : 2)} ${scenario.targetUnit}`
                    : 'Withheld (Pre-decision horizon)'}
                </span>
              </div>

              {/* Metrology details if revealed */}
              {isSelectedRevealed && selectedCandidate.metrology && (
                <div className="pt-1 text-[10px] text-slate-500 grid grid-cols-2 gap-1">
                  {selectedCandidate.metrology.calenderedThicknessUm !== undefined && (
                    <div>Thickness: <span className="font-mono font-semibold text-slate-700">{selectedCandidate.metrology.calenderedThicknessUm} µm</span></div>
                  )}
                  {selectedCandidate.metrology.calenderedPorosityPct !== undefined && (
                    <div>Porosity: <span className="font-mono font-semibold text-slate-700">{selectedCandidate.metrology.calenderedPorosityPct}%</span></div>
                  )}
                  {selectedCandidate.metrology.calenderedDensityGCm3 !== undefined && (
                    <div>Density: <span className="font-mono font-semibold text-slate-700">{selectedCandidate.metrology.calenderedDensityGCm3} g/cm³</span></div>
                  )}
                  {selectedCandidate.metrology.activeMassMg !== undefined && (
                    <div>Active Mass: <span className="font-mono font-semibold text-slate-700">{selectedCandidate.metrology.activeMassMg} mg</span></div>
                  )}
                </div>
              )}
            </div>
          )}

          {/* Candidate List */}
          <div className={`space-y-1.5 overflow-y-auto pr-1 flex-1 ${
            selectedCandidate ? 'max-h-[calc(100vh-500px)]' : 'mt-3 max-h-[calc(100vh-340px)]'
          }`}>
            {scenario.candidates.map((cand) => {
              const isSelected = selectedCandidateId === cand.id;
              const isInit = scenario.replayInitialIds.includes(cand.id);
              const isAcquired = scenario.replaySteps.slice(0, replayStep).some((s) => s.selectedCandidateId === cand.id);
              const isCurrentProposal = currentStepData?.selectedCandidateId === cand.id;
              const isRevealed = isInit || isAcquired || isCurrentProposal;

              return (
                <div
                  key={cand.id}
                  onClick={() => onSelectCandidate(selectedCandidateId === cand.id ? '' : cand.id)}
                  className={`p-2.5 rounded-xl border text-xs transition-all cursor-pointer flex items-center justify-between ${
                    isCurrentProposal
                      ? 'bg-amber-50 border-amber-300 ring-1 ring-amber-300'
                      : isSelected
                      ? 'bg-[#E8F4F2] border-[#087F8C]'
                      : isAcquired
                      ? 'bg-teal-50/50 border-teal-100'
                      : 'bg-slate-50/70 border-slate-100 hover:bg-slate-100'
                  }`}
                >
                  <div className="truncate mr-2">
                    <div className="font-semibold text-slate-800 truncate">{cand.displayId}</div>
                    <div className="text-[10px] text-slate-400 font-mono truncate">
                      {Object.entries(cand.controls).map(([k, v]) => `${k}:${v}`).slice(0, 2).join(' ')}
                    </div>
                  </div>

                  <div className="text-right shrink-0">
                    <div className={`font-mono ${isRevealed ? 'font-bold text-slate-900' : 'font-medium text-slate-400'}`}>
                      {isRevealed
                        ? cand.revealedTarget.value.toFixed(scenario.id === 'warwick_nmc622_calendering' ? 4 : 2)
                        : '—'}
                    </div>
                    <div className="text-[9px] font-mono text-slate-400">
                      {isInit ? 'INITIAL' : isCurrentProposal ? 'ACTIVE' : isAcquired ? 'VISITED' : 'POOL'}
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        </div>

        <div className="pt-3 border-t border-slate-100 text-[11px] text-slate-400 flex items-center justify-between shrink-0">
          <span>Click candidate or 3D point to inspect</span>
          <button
            onClick={onProceedToEvidence}
            className="text-xs font-semibold text-[#087F8C] hover:underline cursor-pointer"
          >
            Evidence →
          </button>
        </div>
          </>
        )}
      </div>
    </div>
  );
};
