import React, { useState } from 'react';
import {
  SnapshotData,
  CampaignStep,
  ScoredActionRecord,
  RevealPhase,
  CandidateViewMode,
  HeatmapMetricMode
} from '../types/mission_control';
import { HypothesisBeliefTrajectoryChart } from '../components/charts/HypothesisBeliefTrajectoryChart';
import { PredictiveDistributionChart } from '../components/charts/PredictiveDistributionChart';
import { CandidateModalityHeatmap } from '../components/charts/CandidateModalityHeatmap';
import { ScoreWaterfallChart } from '../components/charts/ScoreWaterfallChart';
import { TradeoffScatterChart } from '../components/charts/TradeoffScatterChart';
import { StarkHologramSphere } from '../components/StarkHologramSphere';
import {
  Sparkles,
  Lock,
  Eye,
  RefreshCw,
  Award,
  CheckCircle,
  HelpCircle,
  TrendingUp,
  Activity,
  Layers,
  FileSpreadsheet
} from 'lucide-react';

interface Props {
  data: SnapshotData;
  controlledStepIndex?: number;
  onStepChange?: (step: number) => void;
}

export const DiscoveryLabWorkspace: React.FC<Props> = ({
  data,
  controlledStepIndex = 1,
  onStepChange
}) => {
  const steps = data.flagship_campaign?.steps || [];
  const totalSteps = steps.length;
  const [stepIndex, setStepIndex] = useState<number>(controlledStepIndex);

  // Synchronize when controlledStepIndex prop changes
  React.useEffect(() => {
    if (controlledStepIndex && controlledStepIndex !== stepIndex) {
      setStepIndex(controlledStepIndex);
    }
  }, [controlledStepIndex]);

  const handleStepSelect = (s: number) => {
    const clamped = Math.max(1, Math.min(s, totalSteps));
    setStepIndex(clamped);
    setRevealPhase('A_SCORED');
    onStepChange?.(clamped);
  };

  const currentStep: CampaignStep | null = steps[stepIndex - 1] || steps[0] || null;

  // Selected candidate and modality for inspection
  const winnerAction = (currentStep?.preregistration as unknown as ScoredActionRecord) || currentStep?.top_actions?.[0] || null;
  const [selectedCandidateId, setSelectedCandidateId] = useState<string>(
    winnerAction?.action?.candidate_id || 'controlled-3'
  );
  const [selectedModality, setSelectedModality] = useState<string>(
    winnerAction?.action?.action_type || 'XRD'
  );

  // Sync candidate selection when step changes
  React.useEffect(() => {
    if (winnerAction?.action) {
      setSelectedCandidateId(winnerAction.action.candidate_id);
      setSelectedModality(winnerAction.action.action_type);
    }
  }, [winnerAction]);

  // Workflow Reveal Phase (State A -> B -> C -> D)
  const [revealPhase, setRevealPhase] = useState<RevealPhase>('A_SCORED');

  // Primary chart switcher mode
  const [primaryChartTab, setPrimaryChartTab] = useState<'trajectory' | 'predictive' | 'tradeoff'>('trajectory');

  // Candidate explorer switcher mode
  const [candidateViewMode, setCandidateViewMode] = useState<CandidateViewMode>('heatmap');

  // Heatmap metric mode
  const [heatmapMetric, setHeatmapMetric] = useState<HeatmapMetricMode>('composite');

  const allActions = currentStep?.all_scored_actions || currentStep?.top_actions || [];

  // Currently inspected action record
  const inspectedAction = allActions.find(
    (a) => a.action?.candidate_id === selectedCandidateId && a.action?.action_type === selectedModality
  ) || winnerAction;

  // Beliefs before and after
  const priorBeliefs = currentStep?.preregistration?.beliefs_before || {
    H1_PHASE_PURITY_LIMITED: 0.3333,
    H2_COMPOSITION_HOMOGENEITY_LIMITED: 0.3333,
    H3_MORPHOLOGY_KINETICS_LIMITED: 0.3333
  };

  const posteriorBeliefs = currentStep?.belief_update?.beliefs_after || priorBeliefs;
  const displayBeliefs = revealPhase === 'D_UPDATED' ? posteriorBeliefs : priorBeliefs;

  const observedMeasurement = currentStep?.observation?.observed_measurement;
  const isBlinded = revealPhase === 'A_SCORED' || revealPhase === 'B_PREREGISTERED';

  return (
    <div className="space-y-6 pb-12">
      {/* 1. Header & Research Question Banner */}
      <section className="rounded-3xl border border-slate-200 bg-white p-5 sm:p-6 shadow-xs">
        <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-4">
          <div>
            <div className="flex flex-wrap items-center gap-2 mb-1.5">
              <span className="inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full text-2xs font-mono font-bold bg-emerald-50 text-emerald-800 border border-emerald-200 uppercase tracking-wider whitespace-nowrap">
                <Sparkles className="w-3 h-3 text-emerald-600 shrink-0" />
                Discovery Lab • Active Decision Engine
              </span>
              <span className="hidden sm:inline text-2xs font-mono text-slate-400">|</span>
              <span className="text-2xs font-mono text-slate-500">Domain: Solid-State Inorganic Synthesis (A-Lab)</span>
            </div>
            <h1 className="text-xl sm:text-2xl font-extrabold text-slate-900 tracking-tight">
              Autonomous Multimodal Decision & Epistemic Inference Loop
            </h1>
            <p className="text-xs sm:text-sm text-slate-600 mt-1 max-w-4xl">
              AIcoScientist jointly selects <strong>which material</strong> and <strong>which measurement tool</strong> to apply next, optimizing Expected Hypothesis Information Gain per unit experimental cost.
            </p>
          </div>

          {/* Step Selector Controls */}
          <div className="flex items-center gap-2 bg-slate-50 p-2 rounded-xl border border-slate-200 shrink-0">
            <span className="text-xs font-mono text-slate-500 font-semibold pl-1">Campaign Step:</span>
            <div className="flex items-center gap-1">
              {steps.map((s) => (
                <button
                  key={s.step}
                  onClick={() => handleStepSelect(s.step)}
                  className={`px-3 py-1.5 rounded-lg text-xs font-mono font-bold transition-all cursor-pointer ${
                    stepIndex === s.step
                      ? 'bg-emerald-600 text-white shadow-xs'
                      : 'bg-white text-slate-700 hover:bg-slate-100 border border-slate-200'
                  }`}
                >
                  Step {s.step}
                </button>
              ))}
            </div>
          </div>
        </div>

        {/* 2. The 4-State Preregistration & Replay Banner */}
        <div className="mt-5 pt-4 border-t border-slate-100">
          <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-3 mb-3">
            <div className="flex items-center gap-2">
              <Activity className="w-4 h-4 text-emerald-600" />
              <span className="text-xs font-bold text-slate-900 uppercase tracking-wider font-mono">
                The Scientific Preregistration & Bayesian Loop
              </span>
            </div>
            <span className="text-2xs font-mono text-slate-500">
              State transitions guarantee zero hindsight bias or observation leakage
            </span>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-4 gap-2 text-xs font-mono">
            {/* State A */}
            <button
              onClick={() => setRevealPhase('A_SCORED')}
              className={`p-3 rounded-xl border text-left transition cursor-pointer flex flex-col justify-between ${
                revealPhase === 'A_SCORED'
                  ? 'bg-emerald-50/80 border-emerald-500 ring-2 ring-emerald-500/20 shadow-2xs'
                  : 'bg-white border-slate-200 hover:bg-slate-50'
              }`}
            >
              <div className="flex items-center justify-between">
                <span className="font-bold text-slate-800">State A: Scored</span>
                <span className={`w-2 h-2 rounded-full ${revealPhase === 'A_SCORED' ? 'bg-emerald-600' : 'bg-slate-300'}`} />
              </div>
              <p className="text-2xs text-slate-500 mt-1">
                Hypotheses scored; observations strictly firewalled.
              </p>
            </button>

            {/* State B */}
            <button
              onClick={() => setRevealPhase('B_PREREGISTERED')}
              className={`p-3 rounded-xl border text-left transition cursor-pointer flex flex-col justify-between ${
                revealPhase === 'B_PREREGISTERED'
                  ? 'bg-amber-50/80 border-amber-500 ring-2 ring-amber-500/20 shadow-2xs'
                  : 'bg-white border-slate-200 hover:bg-slate-50'
              }`}
            >
              <div className="flex items-center justify-between">
                <span className="font-bold text-slate-800">State B: Preregistered</span>
                <Lock className={`w-3.5 h-3.5 ${revealPhase === 'B_PREREGISTERED' ? 'text-amber-600' : 'text-slate-400'}`} />
              </div>
              <p className="text-2xs text-slate-500 mt-1">
                Predictive distributions locked in immutable evidence ledger.
              </p>
            </button>

            {/* State C */}
            <button
              onClick={() => setRevealPhase('C_REVEALED')}
              className={`p-3 rounded-xl border text-left transition cursor-pointer flex flex-col justify-between ${
                revealPhase === 'C_REVEALED'
                  ? 'bg-blue-50/80 border-blue-500 ring-2 ring-blue-500/20 shadow-2xs'
                  : 'bg-white border-slate-200 hover:bg-slate-50'
              }`}
            >
              <div className="flex items-center justify-between">
                <span className="font-bold text-slate-800">State C: Evidence Reveal</span>
                <Eye className={`w-3.5 h-3.5 ${revealPhase === 'C_REVEALED' ? 'text-blue-600' : 'text-slate-400'}`} />
              </div>
              <p className="text-2xs text-slate-500 mt-1">
                Canonical physical measurement revealed; update pending.
              </p>
            </button>

            {/* State D */}
            <button
              onClick={() => setRevealPhase('D_UPDATED')}
              className={`p-3 rounded-xl border text-left transition cursor-pointer flex flex-col justify-between ${
                revealPhase === 'D_UPDATED'
                  ? 'bg-violet-50/80 border-violet-500 ring-2 ring-violet-500/20 shadow-2xs'
                  : 'bg-white border-slate-200 hover:bg-slate-50'
              }`}
            >
              <div className="flex items-center justify-between">
                <span className="font-bold text-slate-800">State D: Belief Shift</span>
                <TrendingUp className={`w-3.5 h-3.5 ${revealPhase === 'D_UPDATED' ? 'text-violet-600' : 'text-slate-400'}`} />
              </div>
              <p className="text-2xs text-slate-500 mt-1">
                Bayesian posterior computed; log Bayes factors recorded.
              </p>
            </button>
          </div>
        </div>
      </section>

      {/* 3. Upper Analytical Grid: Chart-First View (7:5 Split) */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left 7 Columns: Primary Scientific Visualization Panel */}
        <div className="lg:col-span-7 rounded-3xl border border-slate-200 bg-white p-5 sm:p-6 shadow-xs flex flex-col min-h-[460px]">
          {/* Primary View Switcher Tabs */}
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 border-b border-slate-100 pb-3 mb-4">
            <div className="flex items-center gap-1.5 bg-slate-100 p-1 rounded-xl border border-slate-200 text-xs font-mono">
              <button
                onClick={() => setPrimaryChartTab('trajectory')}
                className={`px-3 py-1.5 rounded-lg transition-all cursor-pointer flex items-center gap-1.5 ${
                  primaryChartTab === 'trajectory'
                    ? 'bg-white text-emerald-800 font-bold shadow-2xs'
                    : 'text-slate-600 hover:text-slate-900'
                }`}
              >
                <TrendingUp className="w-3.5 h-3.5 text-emerald-600" />
                <span>Belief Trajectory</span>
              </button>
              <button
                onClick={() => setPrimaryChartTab('predictive')}
                className={`px-3 py-1.5 rounded-lg transition-all cursor-pointer flex items-center gap-1.5 ${
                  primaryChartTab === 'predictive'
                    ? 'bg-white text-emerald-800 font-bold shadow-2xs'
                    : 'text-slate-600 hover:text-slate-900'
                }`}
              >
                <Activity className="w-3.5 h-3.5 text-emerald-600" />
                <span>Predictive Distributions</span>
              </button>
              <button
                onClick={() => setPrimaryChartTab('tradeoff')}
                className={`px-3 py-1.5 rounded-lg transition-all cursor-pointer flex items-center gap-1.5 ${
                  primaryChartTab === 'tradeoff'
                    ? 'bg-white text-emerald-800 font-bold shadow-2xs'
                    : 'text-slate-600 hover:text-slate-900'
                }`}
              >
                <Sparkles className="w-3.5 h-3.5 text-emerald-600" />
                <span>HIG–Discovery Trade-off</span>
              </button>
            </div>

            <div className="text-2xs font-mono text-slate-400">
              Interactive View • Click to synchronise
            </div>
          </div>

          {/* Chart View Content */}
          <div className="flex-1 w-full min-h-[360px]">
            {primaryChartTab === 'trajectory' && (
              <HypothesisBeliefTrajectoryChart
                data={data}
                currentStepIndex={stepIndex}
                revealPhase={revealPhase}
                onSelectStep={handleStepSelect}
              />
            )}
            {primaryChartTab === 'predictive' && (
              <PredictiveDistributionChart
                currentStep={currentStep}
                revealPhase={revealPhase}
                selectedCandidateId={selectedCandidateId}
                selectedModality={selectedModality}
              />
            )}
            {primaryChartTab === 'tradeoff' && (
              <TradeoffScatterChart
                actions={allActions}
                winnerActionId={winnerAction?.action?.action_id}
                selectedCandidateId={selectedCandidateId}
                selectedModality={selectedModality}
                onSelectAction={(cId, mod) => {
                  setSelectedCandidateId(cId);
                  setSelectedModality(mod);
                }}
              />
            )}
          </div>
        </div>

        {/* Right 5 Columns: Hypothesis Observatory, Score Waterfall & Reveal Status */}
        <div className="lg:col-span-5 space-y-6">
          {/* Card A: Hypothesis Observatory */}
          <div className="rounded-3xl border border-slate-200 bg-white p-5 shadow-xs">
            <div className="flex items-center justify-between mb-3">
              <div className="flex items-center gap-2">
                <span className="flex h-6 w-6 items-center justify-center rounded-md bg-emerald-600 text-xs font-black text-white">
                  Ψ
                </span>
                <h3 className="text-sm font-bold text-slate-900 tracking-tight">
                  Hypothesis Observatory
                </h3>
              </div>
              <span className="text-2xs font-mono bg-slate-100 text-slate-600 px-2 py-0.5 rounded border border-slate-200">
                {revealPhase === 'D_UPDATED' ? 'Posterior State' : 'Prior State'}
              </span>
            </div>

            <div className="space-y-3 font-mono text-xs">
              {/* H1 */}
              <div>
                <div className="flex items-center justify-between mb-1">
                  <span className="font-semibold text-emerald-800 flex items-center gap-1.5">
                    <span className="w-2 h-2 rounded-full bg-emerald-600 inline-block" />
                    H₁: Phase Purity Limited
                  </span>
                  <span className="font-bold text-emerald-700">
                    {((displayBeliefs['H1_PHASE_PURITY_LIMITED'] ?? 0.3333) * 100).toFixed(1)}%
                  </span>
                </div>
                <div className="w-full bg-slate-100 rounded-full h-2 overflow-hidden border border-slate-200">
                  <div
                    className="bg-emerald-600 h-2 rounded-full transition-all duration-500"
                    style={{
                      width: `${(displayBeliefs['H1_PHASE_PURITY_LIMITED'] ?? 0.3333) * 100}%`
                    }}
                  />
                </div>
              </div>

              {/* H2 */}
              <div>
                <div className="flex items-center justify-between mb-1">
                  <span className="font-semibold text-amber-800 flex items-center gap-1.5">
                    <span className="w-2 h-2 rounded-full bg-amber-600 inline-block" />
                    H₂: Homogeneity Limited
                  </span>
                  <span className="font-bold text-amber-700">
                    {((displayBeliefs['H2_COMPOSITION_HOMOGENEITY_LIMITED'] ?? 0.3333) * 100).toFixed(1)}%
                  </span>
                </div>
                <div className="w-full bg-slate-100 rounded-full h-2 overflow-hidden border border-slate-200">
                  <div
                    className="bg-amber-500 h-2 rounded-full transition-all duration-500"
                    style={{
                      width: `${(displayBeliefs['H2_COMPOSITION_HOMOGENEITY_LIMITED'] ?? 0.3333) * 100}%`
                    }}
                  />
                </div>
              </div>

              {/* H3 */}
              <div>
                <div className="flex items-center justify-between mb-1">
                  <span className="font-semibold text-violet-800 flex items-center gap-1.5">
                    <span className="w-2 h-2 rounded-full bg-violet-600 inline-block" />
                    H₃: Kinetics Limited
                  </span>
                  <span className="font-bold text-violet-700">
                    {((displayBeliefs['H3_MORPHOLOGY_KINETICS_LIMITED'] ?? 0.3333) * 100).toFixed(1)}%
                  </span>
                </div>
                <div className="w-full bg-slate-100 rounded-full h-2 overflow-hidden border border-slate-200">
                  <div
                    className="bg-violet-500 h-2 rounded-full transition-all duration-500"
                    style={{
                      width: `${(displayBeliefs['H3_MORPHOLOGY_KINETICS_LIMITED'] ?? 0.3333) * 100}%`
                    }}
                  />
                </div>
              </div>
            </div>

            <p className="mt-3 pt-2 border-t border-slate-100 text-3xs text-slate-400 italic">
              Epistemic Notice: Belief weights represent relative explanatory likelihoods among simplified models, not absolute metaphysical truth.
            </p>
          </div>

          {/* Card B: Score Waterfall Chart */}
          <div className="rounded-3xl border border-slate-200 bg-white p-5 shadow-xs">
            <ScoreWaterfallChart action={inspectedAction} />
          </div>

          {/* Card C: Evidence Card */}
          <div className="rounded-3xl border border-slate-200 bg-white p-5 shadow-xs">
            <div className="flex items-center justify-between mb-2">
              <h4 className="text-xs font-bold text-slate-900 tracking-tight flex items-center gap-1.5">
                <Eye className="w-3.5 h-3.5 text-blue-600" />
                <span>Canonical Measurement Observation</span>
              </h4>
              <span className="text-3xs font-mono text-slate-400">
                {isBlinded ? 'FIREWALLED' : 'REVEALED'}
              </span>
            </div>

            {isBlinded ? (
              <div className="bg-slate-50 border border-slate-200 rounded-xl p-4 text-center">
                <Lock className="w-6 h-6 text-slate-400 mx-auto mb-1.5" />
                <p className="text-xs font-bold text-slate-700">Observation Data Blinding Active</p>
                <p className="text-2xs text-slate-500 mt-0.5">
                  Physical measurements are firewalled until preregistration is committed.
                </p>
                <button
                  onClick={() => setRevealPhase('C_REVEALED')}
                  className="mt-3 inline-flex items-center gap-1.5 px-3 py-1.5 bg-slate-900 hover:bg-slate-800 text-white rounded-md text-xs font-semibold cursor-pointer transition"
                >
                  <Eye className="w-3.5 h-3.5" />
                  <span>Reveal Observation (State C)</span>
                </button>
              </div>
            ) : (
              <div className="space-y-2 font-mono text-xs">
                <div className="p-3 bg-blue-50/60 rounded-xl border border-blue-200 text-blue-950">
                  <div className="flex items-center justify-between text-2xs text-blue-700 mb-1">
                    <span>Observed Observable:</span>
                    <span>{observedMeasurement?.modality} {observedMeasurement?.candidate_id}</span>
                  </div>
                  <div className="text-sm font-bold">
                    Values: {Array.isArray(observedMeasurement?.value) ? observedMeasurement.value.map((v) => Number(v).toFixed(3)).join(', ') : observedMeasurement?.value}
                  </div>
                </div>

                {revealPhase === 'C_REVEALED' && (
                  <button
                    onClick={() => setRevealPhase('D_UPDATED')}
                    className="w-full py-2 bg-violet-600 hover:bg-violet-700 text-white rounded-lg text-xs font-bold cursor-pointer transition shadow-xs flex items-center justify-center gap-1.5"
                  >
                    <RefreshCw className="w-3.5 h-3.5" />
                    <span>Execute Bayesian Update (State D)</span>
                  </button>
                )}

                {revealPhase === 'D_UPDATED' && (
                  <div className="flex items-center justify-between px-3 py-2 bg-emerald-50 border border-emerald-200 rounded-lg text-2xs text-emerald-900">
                    <div className="flex items-center gap-1.5">
                      <CheckCircle className="w-3.5 h-3.5 text-emerald-600" />
                      <span>Posterior updated. Realized entropy reduction: <strong>{currentStep?.observation?.realized_entropy_reduction_nats?.toFixed(3) ?? '0.210'} nats</strong>.</span>
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>
        </div>
      </div>

      {/* 4. Lower Action Space Explorer: Matrix, Hologram & Counterfactuals */}
      <section className="rounded-3xl border border-slate-200 bg-white p-5 sm:p-6 shadow-xs">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 border-b border-slate-100 pb-3 mb-4">
          <div>
            <div className="flex items-center gap-2">
              <span className="flex h-6 w-6 items-center justify-center rounded-md bg-emerald-600 text-xs font-black text-white">
                Ω
              </span>
              <h3 className="text-base font-bold text-slate-900 tracking-tight">
                Candidate Action Space Explorer
              </h3>
            </div>
            <p className="text-xs text-slate-500">
              Explore the 12 inorganic candidates across XRD and Rietveld Refinement modalities
            </p>
          </div>

          {/* Explorer View Switcher */}
          <div className="flex items-center gap-1 bg-slate-100 p-1 rounded-xl border border-slate-200 text-xs font-mono">
            <button
              onClick={() => setCandidateViewMode('heatmap')}
              className={`px-3 py-1.5 rounded-lg transition cursor-pointer flex items-center gap-1.5 ${
                candidateViewMode === 'heatmap'
                  ? 'bg-white text-emerald-800 font-bold shadow-2xs'
                  : 'text-slate-600 hover:text-slate-900'
              }`}
            >
              <Layers className="w-3.5 h-3.5 text-emerald-600" />
              <span>Action Matrix</span>
            </button>
            <button
              onClick={() => setCandidateViewMode('hologram')}
              className={`px-3 py-1.5 rounded-lg transition cursor-pointer flex items-center gap-1.5 ${
                candidateViewMode === 'hologram'
                  ? 'bg-white text-emerald-800 font-bold shadow-2xs'
                  : 'text-slate-600 hover:text-slate-900'
              }`}
            >
              <Sparkles className="w-3.5 h-3.5 text-emerald-600" />
              <span>3D Stark Hologram</span>
            </button>
            <button
              onClick={() => setCandidateViewMode('table')}
              className={`px-3 py-1.5 rounded-lg transition cursor-pointer flex items-center gap-1.5 ${
                candidateViewMode === 'table'
                  ? 'bg-white text-emerald-800 font-bold shadow-2xs'
                  : 'text-slate-600 hover:text-slate-900'
              }`}
            >
              <FileSpreadsheet className="w-3.5 h-3.5 text-emerald-600" />
              <span>Counterfactuals (24 Actions)</span>
            </button>
          </div>
        </div>

        {/* View Mode Content */}
        {candidateViewMode === 'heatmap' && (
          <CandidateModalityHeatmap
            actions={allActions}
            winnerActionId={winnerAction?.action?.action_id}
            selectedCandidateId={selectedCandidateId}
            selectedModality={selectedModality}
            activeMetric={heatmapMetric}
            onChangeMetric={setHeatmapMetric}
            onSelectAction={(cId, mod) => {
              setSelectedCandidateId(cId);
              setSelectedModality(mod);
            }}
          />
        )}

        {candidateViewMode === 'hologram' && (
          <div className="rounded-2xl border border-slate-200 overflow-hidden bg-slate-950 p-2">
            <StarkHologramSphere
              candidates={data.flagship_campaign?.candidates || []}
              selectedCandidateId={selectedCandidateId}
              onSelectCandidate={(cId) => setSelectedCandidateId(cId)}
              currentStep={currentStep || undefined}
            />
          </div>
        )}

        {candidateViewMode === 'table' && (
          <div className="overflow-x-auto border border-slate-200 rounded-xl bg-white shadow-2xs">
            <table className="w-full text-left border-collapse text-xs font-mono">
              <thead>
                <tr className="bg-slate-50 border-b border-slate-200 text-2xs uppercase text-slate-500">
                  <th className="py-2.5 px-3">Rank</th>
                  <th className="py-2.5 px-3">Candidate</th>
                  <th className="py-2.5 px-3">Modality</th>
                  <th className="py-2.5 px-3 text-right">Net Score S(a)</th>
                  <th className="py-2.5 px-3 text-right">Score Gap ΔS</th>
                  <th className="py-2.5 px-3 text-right">Raw HIG (nats)</th>
                  <th className="py-2.5 px-3 text-right">Cost</th>
                  <th className="py-2.5 px-3">Status</th>
                </tr>
              </thead>
              <tbody>
                {allActions.map((a, idx) => {
                  const isWinner = a.action?.action_id === winnerAction?.action?.action_id;
                  const isSelected =
                    a.action?.candidate_id === selectedCandidateId &&
                    a.action?.action_type === selectedModality;
                  const score = a.total_action_score ?? 0;
                  const winnerScore = winnerAction?.total_action_score ?? 0;
                  const gap = winnerScore - score;
                  const rawHig = a.raw_expected_hig_nats ?? a.expected_hig_nats ?? 0;
                  const cost = a.raw_estimated_cost ?? (a.action?.estimated_cost || 1.0);

                  return (
                    <tr
                      key={a.action?.action_id || idx}
                      onClick={() => {
                        if (a.action?.candidate_id && a.action?.action_type) {
                          setSelectedCandidateId(a.action.candidate_id);
                          setSelectedModality(a.action.action_type);
                        }
                      }}
                      className={`border-b border-slate-100 last:border-b-0 cursor-pointer transition ${
                        isSelected
                          ? 'bg-emerald-50/80 font-semibold'
                          : isWinner
                          ? 'bg-emerald-50/40'
                          : 'hover:bg-slate-50/60'
                      }`}
                    >
                      <td className="py-2 px-3">
                        {isWinner ? (
                          <span className="inline-flex items-center gap-1 font-bold text-emerald-800">
                            <Award className="w-3.5 h-3.5 text-emerald-600" />
                            #1 (Selected)
                          </span>
                        ) : (
                          `#${idx + 1}`
                        )}
                      </td>
                      <td className="py-2 px-3 font-bold text-slate-800">
                        {a.action?.candidate_id}
                      </td>
                      <td className="py-2 px-3">
                        <span className={`px-2 py-0.5 rounded text-2xs font-bold ${
                          a.action?.action_type === 'XRD'
                            ? 'bg-emerald-100 text-emerald-800'
                            : 'bg-violet-100 text-violet-800'
                        }`}>
                          {a.action?.action_type}
                        </span>
                      </td>
                      <td className="py-2 px-3 text-right font-bold">
                        {score >= 0 ? `+${score.toFixed(4)}` : score.toFixed(4)}
                      </td>
                      <td className="py-2 px-3 text-right text-slate-500">
                        {gap <= 1e-4 ? '0.0000' : `-${gap.toFixed(4)}`}
                      </td>
                      <td className="py-2 px-3 text-right text-emerald-700">
                        {rawHig.toFixed(3)} nats
                      </td>
                      <td className="py-2 px-3 text-right text-slate-600">
                        {cost.toFixed(1)}
                      </td>
                      <td className="py-2 px-3 text-2xs text-slate-500">
                        {isWinner ? (
                          <span className="text-emerald-700 font-bold">Recommended Action</span>
                        ) : gap > 0.4 ? (
                          'Deprioritized (Cost penalty)'
                        ) : (
                          'Suboptimal HIG'
                        )}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </section>
    </div>
  );
};
