import React from 'react';
import { ArrowLeft, ArrowRight, CheckCircle, Cpu, Play, RotateCcw, Sparkles } from 'lucide-react';
import type { ExhibitionScenario } from '../../data/types';
import { pendingDecision, type ProcessExperiencePhase, visibleDecision } from '../../data/processExperience';

interface ProcessDecisionExperienceProps {
  scenario: ExhibitionScenario;
  phase: ProcessExperiencePhase;
  revealedSteps: number;
  onBeginDecision: () => void;
  onRunProcess: () => void;
  onNextDecision: () => void;
  onRepeatDecision: () => void;
  onOpenAdvanced: () => void;
}

const LABELS: Record<string, string> = {
  coating_speed_m_per_min: 'Web speed',
  coating_gap_um: 'Coating gap',
  drying_temperature_c: 'Drying temperature',
  calendering_applied: 'Calendering',
  roll_temperature_c: 'Roll temperature',
  target_density_g_cm3: 'Target density',
  target_coating_weight_gsm: 'Target coating weight',
  loading_regime: 'Loading regime',
};

const formatControl = (key: string, value: string | number | boolean) => {
  if (key === 'coating_speed_m_per_min') return `${value} m/min`;
  if (key === 'coating_gap_um') return `${value} µm`;
  if (key === 'drying_temperature_c' || key === 'roll_temperature_c') return `${value} °C`;
  if (key === 'target_density_g_cm3') return `${value} g/cm³`;
  if (key === 'target_coating_weight_gsm') return `${value} g/m²`;
  if (key === 'calendering_applied') return value ? 'Applied' : 'Not applied';
  return String(value);
};

export const ProcessDecisionExperience: React.FC<ProcessDecisionExperienceProps> = ({
  scenario,
  phase,
  revealedSteps,
  onBeginDecision,
  onRunProcess,
  onNextDecision,
  onRepeatDecision,
  onOpenAdvanced,
}) => {
  const proposal = pendingDecision(scenario, revealedSteps);
  const decision = visibleDecision(scenario, revealedSteps, phase);
  const isResult = phase === 'RESULT_REVEAL';
  const isRunning = phase === 'ILLUSTRATED_PROCESS_RUN';
  const unitLabel = scenario.id === 'drakopoulos_graphite' ? 'mAh/g' : '5C/0.2C ratio';

  if (!proposal && !isResult) {
    return (
      <div className="rounded-2xl border border-emerald-200 bg-emerald-50/90 p-4 text-xs text-emerald-900">
        All recorded decisions are revealed. Open the advanced view to inspect the full trajectory.
        <button onClick={onOpenAdvanced} className="mt-3 block font-semibold underline">Open advanced analysis</button>
      </div>
    );
  }

  if (phase === 'PROCESS_EXPLORE') {
    return (
      <button
        data-testid="begin-process-decision"
        onClick={onBeginDecision}
        className="w-full rounded-2xl bg-[#087F8C] px-5 py-3.5 text-sm font-semibold text-white shadow-lg shadow-teal-900/10 transition hover:bg-[#076a75]"
      >
        <span className="flex items-center justify-center gap-2"><Sparkles className="h-4 w-4" />{scenario.id === 'drakopoulos_graphite' ? 'Let AI select a recipe' : 'Let AI choose the calendering conditions'}</span>
      </button>
    );
  }

  if (!decision) return null;

  return (
    <section data-testid="process-decision" className="space-y-3 rounded-2xl border border-[#B7DDE0] bg-white/94 p-4 shadow-xl backdrop-blur-xl">
      <div className="flex items-center justify-between gap-3">
        <div>
          <div className="flex items-center gap-1.5 text-[10px] font-bold uppercase tracking-[0.16em] text-[#087F8C]"><Cpu className="h-3.5 w-3.5" />Recorded AI decision · seed 11</div>
          <div className="mt-1 text-lg font-bold text-[#142A35]">{decision.selectedCandidateDisplay}</div>
        </div>
        <span className="rounded-full bg-slate-100 px-2.5 py-1 font-mono text-[10px] text-slate-500">{Math.min(revealedSteps + (isResult ? 0 : 1), scenario.replaySteps.length)}/{scenario.replaySteps.length}</span>
      </div>

      <div className="grid grid-cols-2 gap-2">
        {Object.entries(decision.controlsSummary).map(([key, value]) => (
          <div key={key} className="rounded-xl border border-slate-100 bg-slate-50 px-3 py-2">
            <div className="text-[10px] text-slate-500">{LABELS[key] || key}</div>
            <div className="mt-0.5 font-mono text-[11px] font-semibold text-[#142A35]">{formatControl(key, value)}</div>
          </div>
        ))}
      </div>

      {!isResult && (
        <div className="rounded-xl border border-sky-100 bg-sky-50/80 px-3 py-2 text-[11px] text-sky-900">
          Pre-reveal model estimate: <span className="font-mono font-semibold">{decision.predictedMean.toFixed(scenario.id === 'drakopoulos_graphite' ? 2 : 4)} ± {decision.predictedStd.toFixed(scenario.id === 'drakopoulos_graphite' ? 2 : 4)} {unitLabel}</span>
        </div>
      )}

      {isRunning && (
        <div className="flex items-center gap-3 rounded-xl bg-[#142A35] px-3 py-3 text-xs text-white">
          <span className="h-2.5 w-2.5 animate-pulse rounded-full bg-[#F59E42]" />
          <span>{scenario.id === 'drakopoulos_graphite' ? 'Illustrative coating pass → verified drying condition' : 'Illustrative roller and electrode-web response'}</span>
        </div>
      )}

      {phase === 'AI_DECISION' && (
        <button data-testid="run-process" onClick={onRunProcess} className="w-full rounded-xl bg-[#142A35] px-4 py-3 text-xs font-semibold text-white hover:bg-[#1f3b49]">
          <span className="flex items-center justify-center gap-2"><Play className="h-4 w-4" />Run illustrated process response</span>
        </button>
      )}

      {isResult && (
        <>
          <div data-testid="process-result" className="rounded-xl border border-amber-200 bg-amber-50 px-3 py-3">
            <div className="flex items-center gap-1.5 text-[10px] font-bold uppercase tracking-wider text-amber-700"><CheckCircle className="h-3.5 w-3.5" />Historical measurement revealed</div>
            <div className="mt-1 font-mono text-xl font-bold text-[#142A35]">{decision.revealedTarget.toFixed(scenario.id === 'drakopoulos_graphite' ? 2 : 4)} <span className="text-xs font-medium">{unitLabel}</span></div>
            <div className="mt-1 text-[11px] text-slate-600">Best so far: <span className="font-mono font-semibold">{decision.bestSoFar.toFixed(scenario.id === 'drakopoulos_graphite' ? 2 : 4)}</span></div>
          </div>
          <div className="grid grid-cols-2 gap-2">
            <button onClick={onRepeatDecision} className="rounded-xl border border-slate-200 px-3 py-2.5 text-xs font-semibold text-slate-700 hover:bg-slate-50"><span className="flex items-center justify-center gap-1.5"><RotateCcw className="h-3.5 w-3.5" />Repeat</span></button>
            {revealedSteps < scenario.replaySteps.length ? (
              <button onClick={onNextDecision} className="rounded-xl bg-[#087F8C] px-3 py-2.5 text-xs font-semibold text-white"><span className="flex items-center justify-center gap-1.5">Next decision<ArrowRight className="h-3.5 w-3.5" /></span></button>
            ) : (
              <button onClick={onOpenAdvanced} className="rounded-xl bg-[#087F8C] px-3 py-2.5 text-xs font-semibold text-white">Advanced analysis</button>
            )}
          </div>
          <button onClick={onOpenAdvanced} className="flex items-center gap-1 text-[11px] font-semibold text-[#087F8C] hover:underline"><ArrowLeft className="h-3 w-3 rotate-180" />Inspect candidate table and methodology</button>
        </>
      )}
    </section>
  );
};
