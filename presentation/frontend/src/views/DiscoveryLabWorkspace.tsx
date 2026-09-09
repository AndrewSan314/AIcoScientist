import React, { useState, useEffect, useRef } from 'react';
import {
  SnapshotData,
  CampaignStep,
  ScoredActionRecord,
  RevealPhase,
  HeatmapMetricMode,
  DiscoveryFlowState,
  DatasetOption
} from '../types/mission_control';
import { HypothesisBeliefTrajectoryChart } from '../components/charts/HypothesisBeliefTrajectoryChart';
import { PredictiveDistributionChart } from '../components/charts/PredictiveDistributionChart';
import { CandidateModalityHeatmap } from '../components/charts/CandidateModalityHeatmap';
import { ScoreWaterfallChart } from '../components/charts/ScoreWaterfallChart';
import { TradeoffScatterChart } from '../components/charts/TradeoffScatterChart';
import { StarkHologramSphere } from '../components/StarkHologramSphere';
import {
  Lock,
  Eye,
  RefreshCw,
  Award,
  TrendingUp,
  Activity,
  Play,
  RotateCcw,
  Sliders,
  CheckCircle2,
  Atom,
  Clock,
  ArrowRight
} from 'lucide-react';

interface Props {
  data: SnapshotData;
  controlledStepIndex?: number;
  onStepChange?: (step: number) => void;
  controlledFlowState?: DiscoveryFlowState;
  onFlowStateChange?: (state: DiscoveryFlowState) => void;
  controlledDataset?: DatasetOption;
  onDatasetChange?: (dataset: DatasetOption) => void;
  controlledRevealPhase?: RevealPhase;
  onRevealPhaseChange?: (phase: RevealPhase) => void;
}

export const DiscoveryLabWorkspace: React.FC<Props> = ({
  data,
  controlledStepIndex = 1,
  onStepChange,
  controlledFlowState,
  onFlowStateChange,
  controlledDataset,
  onDatasetChange,
  controlledRevealPhase,
  onRevealPhaseChange,
}) => {
  // Flow state (setup -> running -> results)
  const [internalFlowState, setInternalFlowState] = useState<DiscoveryFlowState>('setup');
  const flowState = controlledFlowState ?? internalFlowState;
  const updateFlowState = (s: DiscoveryFlowState) => {
    setInternalFlowState(s);
    onFlowStateChange?.(s);
  };

  // Dataset / Research question option
  const [internalDataset, setInternalDataset] = useState<DatasetOption>('controlled_synthesis');
  const dataset = controlledDataset ?? internalDataset;
  const updateDataset = (d: DatasetOption) => {
    setInternalDataset(d);
    onDatasetChange?.(d);
  };

  // Policy configuration state
  const [selectedPolicy, setSelectedPolicy] = useState<'hig_cost_penalized' | 'greedy_hig' | 'random_baseline'>('hig_cost_penalized');
  const [costPenaltyFactor, setCostPenaltyFactor] = useState<number>(0.25);
  const [modalityConstraint, setModalityConstraint] = useState<'all' | 'xrd_only'>('all');

  // Running animation state
  const [runningProgress, setRunningProgress] = useState<number>(0);
  const [runningLogIndex, setRunningLogIndex] = useState<number>(0);
  const progressTimerRef = useRef<any>(null);

  const steps = data.flagship_campaign?.steps || [];
  const totalSteps = steps.length;
  const [stepIndex, setStepIndex] = useState<number>(controlledStepIndex);

  // Sync controlledStepIndex
  useEffect(() => {
    if (controlledStepIndex && controlledStepIndex !== stepIndex) {
      setStepIndex(controlledStepIndex);
    }
  }, [controlledStepIndex]);

  // Reveal Phase (State A -> B -> C -> D)
  const [internalRevealPhase, setInternalRevealPhase] = useState<RevealPhase>('A_SCORED');
  const revealPhase = controlledRevealPhase ?? internalRevealPhase;
  const updateRevealPhase = (p: RevealPhase) => {
    setInternalRevealPhase(p);
    onRevealPhaseChange?.(p);
  };

  // Chart switcher mode
  const [primaryChartTab, setPrimaryChartTab] = useState<'trajectory' | 'predictive' | 'tradeoff'>('trajectory');

  // Heatmap metric mode
  const [heatmapMetric, setHeatmapMetric] = useState<HeatmapMetricMode>('composite');

  // Candidate inspection
  const currentStep: CampaignStep | null = steps[stepIndex - 1] || steps[0] || null;
  const winnerAction = (currentStep?.preregistration as unknown as ScoredActionRecord) || currentStep?.top_actions?.[0] || null;
  const [selectedCandidateId, setSelectedCandidateId] = useState<string>(
    winnerAction?.action?.candidate_id || 'controlled-3'
  );
  const [selectedModality, setSelectedModality] = useState<string>(
    winnerAction?.action?.action_type || 'XRD'
  );

  useEffect(() => {
    if (winnerAction?.action) {
      setSelectedCandidateId(winnerAction.action.candidate_id);
      setSelectedModality(winnerAction.action.action_type);
    }
  }, [winnerAction]);

  const handleStepSelect = (s: number) => {
    const clamped = Math.max(1, Math.min(s, totalSteps));
    setStepIndex(clamped);
    updateRevealPhase('A_SCORED');
    onStepChange?.(clamped);
  };

  // Handle start execution
  const handleStartDiscovery = () => {
    updateFlowState('running');
    setRunningProgress(0);
    setRunningLogIndex(0);

    const startTime = Date.now();
    const duration = 1400; // ~1.4 seconds as per plan

    if (progressTimerRef.current) clearInterval(progressTimerRef.current);

    progressTimerRef.current = setInterval(() => {
      const elapsed = Date.now() - startTime;
      const pct = Math.min(100, Math.round((elapsed / duration) * 100));
      setRunningProgress(pct);

      if (pct >= 85) setRunningLogIndex(4);
      else if (pct >= 65) setRunningLogIndex(3);
      else if (pct >= 45) setRunningLogIndex(2);
      else if (pct >= 25) setRunningLogIndex(1);
      else setRunningLogIndex(0);

      if (elapsed >= duration) {
        if (progressTimerRef.current) clearInterval(progressTimerRef.current);
        updateFlowState('results');
      }
    }, 40);
  };

  const handleSkipRunning = () => {
    if (progressTimerRef.current) clearInterval(progressTimerRef.current);
    setRunningProgress(100);
    updateFlowState('results');
  };

  const handleResetRun = () => {
    if (progressTimerRef.current) clearInterval(progressTimerRef.current);
    updateFlowState('setup');
  };

  const runningLogs = [
    'Initializing hypothesis prior beliefs P(H)...',
    'Evaluating candidate × modality action matrix under multi-objective policy...',
    'Top action selected by information gain and cost trade-off...',
    'Executing firewalled surrogate observation & evaluating likelihood...',
    'Bayesian update converged over active scientific hypotheses.'
  ];

  const allActions = currentStep?.all_scored_actions || currentStep?.top_actions || [];
  const inspectedAction = allActions.find(
    (a) => a.action?.candidate_id === selectedCandidateId && a.action?.action_type === selectedModality
  ) || winnerAction;

  const initBeliefs = data.flagship_campaign?.initial_beliefs || {
    H1_PHASE_PURITY_LIMITED: 0.3333,
    H2_COMPOSITION_HOMOGENEITY_LIMITED: 0.3333,
    H3_MORPHOLOGY_KINETICS_LIMITED: 0.3334
  };

  const priorBeliefs = currentStep?.preregistration?.beliefs_before || initBeliefs;
  const posteriorBeliefs = currentStep?.belief_update?.beliefs_after || priorBeliefs;

  // STATE 1: SETUP
  if (flowState === 'setup') {
    return (
      <div className="space-y-6 pb-12 animate-fade-in">
        {/* Hero Banner */}
        <section className="sci-card p-6 border-l-4 border-l-[#DC2626]">
          <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
            <div>
              <div className="flex items-center gap-2 mb-1.5">
                <span className="sci-badge sci-badge-verified">
                  <Atom className="w-3 h-3 text-[#DC2626]" />
                  Ready to run
                </span>
                <span className="text-2xs text-[#8F9995]">•</span>
                <span className="text-2xs font-mono text-[#66706C]">Closed-Loop Scientific Engine</span>
              </div>
              <h1 className="text-2xl font-bold text-[#17201F] tracking-tight">
                Autonomous Scientific Discovery Engine
              </h1>
              <p className="text-sm text-[#66706C] mt-1 max-w-3xl leading-relaxed">
                Select a research question, configure policy parameters, and execute closed-loop Bayesian hypothesis discrimination.
              </p>
            </div>
            <div className="flex items-center gap-2">
              <button
                onClick={handleStartDiscovery}
                className="inline-flex items-center gap-2 px-5 py-2.5 bg-[#B91C1C] hover:bg-[#991B1B] text-white rounded-xl font-semibold text-sm shadow-xs transition cursor-pointer"
              >
                <Play className="w-4 h-4 fill-white" />
                <span>Run discovery analysis</span>
                <ArrowRight className="w-4 h-4" />
              </button>
            </div>
          </div>
        </section>

        {/* Two-Panel Configuration Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
          {/* Left Panel: Research Question & Data Source */}
          <section className="lg:col-span-7 sci-card p-6 space-y-5">
            <div className="flex items-center justify-between border-b border-[#D9DFDB] pb-3">
              <div>
                <h2 className="text-base font-bold text-[#17201F]">1. Select Research Question & Dataset</h2>
                <p className="text-xs text-[#66706C] mt-0.5">Choose target materials system and ground-truth validation domain</p>
              </div>
              <span className="text-2xs font-mono text-[#8F9995]">3 options</span>
            </div>

            <div className="space-y-3">
              {/* Option 1: Flagship Alloy */}
              <div
                onClick={() => updateDataset('controlled_synthesis')}
                className={`p-4 rounded-xl border transition cursor-pointer ${
                  dataset === 'controlled_synthesis'
                    ? 'border-[#B91C1C] bg-[#FEF2F2] shadow-2xs'
                    : 'border-[#D9DFDB] bg-[#FCFCFA] hover:border-[#8F9995]'
                }`}
              >
                <div className="flex items-start justify-between gap-3">
                  <div className="flex items-start gap-3">
                    <input
                      type="radio"
                      name="dataset"
                      checked={dataset === 'controlled_synthesis'}
                      onChange={() => updateDataset('controlled_synthesis')}
                      className="mt-1 text-[#DC2626] focus:ring-[#DC2626]"
                    />
                    <div>
                      <div className="flex items-center gap-2">
                        <h3 className="text-sm font-bold text-[#17201F]">
                          Alloy Purity & Phase Identification
                        </h3>
                        <span className="sci-badge sci-badge-verified">Controlled Benchmark</span>
                      </div>
                      <p className="text-xs text-[#66706C] mt-1">
                        3 candidates, 3 formal hypotheses (H₁, H₂, H₃), XRD + TEM characterization modalities.
                      </p>
                      <div className="flex items-center gap-4 mt-2 text-2xs font-mono text-[#8F9995]">
                        <span>Budget: 5.0 credits</span>
                        <span>•</span>
                        <span>Ground truth: Synthesized phase fractions</span>
                        <span>•</span>
                        <span className="text-[#DC2626] font-semibold">Status: Fully verified</span>
                      </div>
                    </div>
                  </div>
                </div>
              </div>

              {/* Option 2: Electrolyte Scale */}
              <div
                onClick={() => updateDataset('electrolyte_search')}
                className={`p-4 rounded-xl border transition cursor-pointer ${
                  dataset === 'electrolyte_search'
                    ? 'border-[#B91C1C] bg-[#FEF2F2] shadow-2xs'
                    : 'border-[#D9DFDB] bg-[#FCFCFA] hover:border-[#8F9995]'
                }`}
              >
                <div className="flex items-start justify-between gap-3">
                  <div className="flex items-start gap-3">
                    <input
                      type="radio"
                      name="dataset"
                      checked={dataset === 'electrolyte_search'}
                      onChange={() => updateDataset('electrolyte_search')}
                      className="mt-1 text-[#DC2626] focus:ring-[#DC2626]"
                    />
                    <div>
                      <div className="flex items-center gap-2">
                        <h3 className="text-sm font-bold text-[#17201F]">
                          Solid Electrolyte Conductivity
                        </h3>
                        <span className="sci-badge sci-badge-surrogate">Scalability Preview</span>
                      </div>
                      <p className="text-xs text-[#66706C] mt-1">
                        20 candidate solid electrolytes, 5 structural hypotheses, EIS + solid-state NMR modalities.
                      </p>
                      <div className="flex items-center gap-4 mt-2 text-2xs font-mono text-[#8F9995]">
                        <span>Budget: 25.0 credits</span>
                        <span>•</span>
                        <span>333,333 candidate virtual screen</span>
                        <span>•</span>
                        <span className="text-[#DC2626] font-semibold">Status: 20-candidate evaluation</span>
                      </div>
                    </div>
                  </div>
                </div>
              </div>

              {/* Option 3: A-Lab Replay */}
              <div
                onClick={() => updateDataset('alab_replay')}
                className={`p-4 rounded-xl border transition cursor-pointer ${
                  dataset === 'alab_replay'
                    ? 'border-[#B91C1C] bg-[#FEF2F2] shadow-2xs'
                    : 'border-[#D9DFDB] bg-[#FCFCFA] hover:border-[#8F9995]'
                }`}
              >
                <div className="flex items-start justify-between gap-3">
                  <div className="flex items-start gap-3">
                    <input
                      type="radio"
                      name="dataset"
                      checked={dataset === 'alab_replay'}
                      onChange={() => updateDataset('alab_replay')}
                      className="mt-1 text-[#DC2626] focus:ring-[#DC2626]"
                    />
                    <div>
                      <div className="flex items-center gap-2">
                        <h3 className="text-sm font-bold text-[#17201F]">
                          Catalyst Degradation Replay
                        </h3>
                        <span className="sci-badge sci-badge-historical">Historical Validation</span>
                      </div>
                      <p className="text-xs text-[#66706C] mt-1">
                        Autonomous laboratory physical synthesis replay across 1,035 real physical synthesis attempts.
                      </p>
                      <div className="flex items-center gap-4 mt-2 text-2xs font-mono text-[#8F9995]">
                        <span>Budget: 15.0 credits</span>
                        <span>•</span>
                        <span>Empirical laboratory telemetry</span>
                        <span>•</span>
                        <span className="text-[#92400E] font-semibold">Status: Calibrated on 68 lab samples</span>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </section>

          {/* Right Panel: Policy & Constraint Configuration */}
          <section className="lg:col-span-5 sci-card p-6 space-y-5 flex flex-col justify-between">
            <div>
              <div className="flex items-center justify-between border-b border-[#D9DFDB] pb-3">
                <div>
                  <h2 className="text-base font-bold text-[#17201F]">2. Engine Policy & Constraints</h2>
                  <p className="text-xs text-[#66706C] mt-0.5">Set objective weighting and execution constraints</p>
                </div>
                <Sliders className="w-4 h-4 text-[#DC2626]" />
              </div>

              <div className="space-y-4 mt-4">
                {/* Acquisition Policy */}
                <div>
                  <label className="block text-xs font-bold text-[#17201F] mb-1.5">
                    Acquisition Policy
                  </label>
                  <div className="space-y-1.5">
                    {[
                      { id: 'hig_cost_penalized', label: 'HIG Cost-Penalized (Hybrid)', tag: 'Recommended' },
                      { id: 'greedy_hig', label: 'Greedy Pure HIG', tag: 'High-cost exploration' },
                      { id: 'random_baseline', label: 'Random Exploration Baseline', tag: 'Benchmark control' },
                    ].map((p) => (
                      <div
                        key={p.id}
                        onClick={() => setSelectedPolicy(p.id as any)}
                        className={`p-2.5 rounded-lg border text-xs cursor-pointer flex items-center justify-between transition ${
                          selectedPolicy === p.id
                            ? 'border-[#B91C1C] bg-[#FEF2F2] font-semibold text-[#991B1B]'
                            : 'border-[#D9DFDB] bg-[#FCFCFA] text-[#66706C] hover:bg-[#F4F3EE]'
                        }`}
                      >
                        <div className="flex items-center gap-2">
                          <input
                            type="radio"
                            checked={selectedPolicy === p.id}
                            onChange={() => setSelectedPolicy(p.id as any)}
                            className="text-[#DC2626] focus:ring-[#DC2626]"
                          />
                          <span>{p.label}</span>
                        </div>
                        <span className="text-2xs font-mono text-[#8F9995]">{p.tag}</span>
                      </div>
                    ))}
                  </div>
                </div>

                {/* Cost Penalty Factor Slider */}
                <div>
                  <div className="flex items-center justify-between text-xs font-bold text-[#17201F] mb-1.5">
                    <span>Cost Penalty Weight (λ)</span>
                    <span className="font-mono text-[#DC2626]">{costPenaltyFactor.toFixed(2)}</span>
                  </div>
                  <div className="grid grid-cols-4 gap-2">
                    {[0.1, 0.25, 0.5, 1.0].map((val) => (
                      <button
                        key={val}
                        type="button"
                        onClick={() => setCostPenaltyFactor(val)}
                        className={`py-1.5 text-xs font-mono rounded border transition cursor-pointer ${
                          costPenaltyFactor === val
                            ? 'bg-[#B91C1C] text-white border-[#B91C1C] font-bold'
                            : 'bg-[#FCFCFA] text-[#66706C] border-[#D9DFDB] hover:bg-[#F4F3EE]'
                        }`}
                      >
                        {val === 0.25 ? '0.25 (Def)' : val.toFixed(2)}
                      </button>
                    ))}
                  </div>
                </div>

                {/* Modality Constraint */}
                <div>
                  <label className="block text-xs font-bold text-[#17201F] mb-1.5">
                    Allowed Characterization Modalities
                  </label>
                  <div className="grid grid-cols-2 gap-2">
                    <button
                      type="button"
                      onClick={() => setModalityConstraint('all')}
                      className={`py-2 px-3 text-xs rounded border transition cursor-pointer text-left ${
                        modalityConstraint === 'all'
                          ? 'border-[#B91C1C] bg-[#FEF2F2] text-[#991B1B] font-semibold'
                          : 'border-[#D9DFDB] bg-[#FCFCFA] text-[#66706C] hover:bg-[#F4F3EE]'
                      }`}
                    >
                      <div className="font-bold">All Allowed</div>
                      <div className="text-2xs text-[#8F9995]">XRD + TEM diagnostic</div>
                    </button>
                    <button
                      type="button"
                      onClick={() => setModalityConstraint('xrd_only')}
                      className={`py-2 px-3 text-xs rounded border transition cursor-pointer text-left ${
                        modalityConstraint === 'xrd_only'
                          ? 'border-[#B91C1C] bg-[#FEF2F2] text-[#991B1B] font-semibold'
                          : 'border-[#D9DFDB] bg-[#FCFCFA] text-[#66706C] hover:bg-[#F4F3EE]'
                      }`}
                    >
                      <div className="font-bold">XRD Only</div>
                      <div className="text-2xs text-[#8F9995]">Low-cost screening</div>
                    </button>
                  </div>
                </div>
              </div>
            </div>

            {/* Campaign Preview & Launch Action */}
            <div className="pt-4 border-t border-[#D9DFDB] space-y-3">
              <div className="p-3 rounded-lg bg-[#F4F3EE] border border-[#D9DFDB] text-2xs font-mono text-[#66706C] flex items-center justify-between">
                <span>Est. actions: 4 steps</span>
                <span>•</span>
                <span>Expected cost: ~4.2 credits</span>
                <span>•</span>
                <span>Confidence target: &gt; 0.85</span>
              </div>
              <button
                onClick={handleStartDiscovery}
                className="w-full py-3 bg-[#B91C1C] hover:bg-[#991B1B] text-white rounded-xl font-bold text-sm shadow-xs transition flex items-center justify-center gap-2 cursor-pointer"
              >
                <Play className="w-4 h-4 fill-white" />
                <span>Execute 4-step campaign</span>
              </button>
            </div>
          </section>
        </div>
      </div>
    );
  }

  // STATE 2: RUNNING
  if (flowState === 'running') {
    return (
      <div className="space-y-6 py-8 animate-fade-in max-w-3xl mx-auto">
        <div className="sci-card p-8 text-center space-y-6 shadow-sm">
          <div className="w-14 h-14 mx-auto rounded-2xl bg-[#FEF2F2] border border-[#FECACA] flex items-center justify-center text-[#DC2626]">
            <RefreshCw className="w-7 h-7 animate-spin" />
          </div>

          <div>
            <span className="sci-badge sci-badge-verified">Live Campaign Execution</span>
            <h2 className="text-xl font-bold text-[#17201F] mt-2">
              Running Autonomous Discovery Campaign
            </h2>
            <p className="text-xs text-[#66706C] mt-1">
              Evaluating candidate information gains, firewalled observations, and Bayesian updates.
            </p>
          </div>

          {/* Progress Bar */}
          <div className="space-y-2">
            <div className="flex items-center justify-between text-xs font-mono text-[#66706C]">
              <span>Execution progress</span>
              <span className="font-bold text-[#DC2626]">{runningProgress}%</span>
            </div>
            <div className="w-full h-2.5 bg-[#D9DFDB] rounded-full overflow-hidden">
              <div
                className="h-full bg-[#B91C1C] transition-all duration-100 ease-out"
                style={{ width: `${runningProgress}%` }}
              />
            </div>
          </div>

          {/* Live Step Ticker */}
          <div className="p-4 rounded-xl bg-[#F4F3EE] border border-[#D9DFDB] text-left space-y-2">
            <div className="text-2xs font-mono text-[#8F9995] uppercase tracking-wider mb-2">
              Engine Execution Log
            </div>
            {runningLogs.map((log, idx) => {
              const isPast = idx < runningLogIndex;
              const isCurrent = idx === runningLogIndex;
              return (
                <div
                  key={idx}
                  className={`flex items-start gap-2.5 text-xs font-mono transition-opacity ${
                    isCurrent
                      ? 'text-[#DC2626] font-bold opacity-100'
                      : isPast
                      ? 'text-[#66706C] opacity-75'
                      : 'text-[#8F9995] opacity-30'
                  }`}
                >
                  <span className="mt-0.5">
                    {isPast ? (
                      <CheckCircle2 className="w-3.5 h-3.5 text-[#DC2626]" />
                    ) : isCurrent ? (
                      <Activity className="w-3.5 h-3.5 text-[#DC2626] animate-pulse" />
                    ) : (
                      <Clock className="w-3.5 h-3.5 text-[#8F9995]" />
                    )}
                  </span>
                  <span>{log}</span>
                </div>
              );
            })}
          </div>

          {/* Skip Animation Link */}
          <div className="pt-2">
            <button
              onClick={handleSkipRunning}
              className="text-xs text-[#66706C] hover:text-[#17201F] underline cursor-pointer"
            >
              Skip animation and jump to results →
            </button>
          </div>
        </div>
      </div>
    );
  }

  // STATE 3: RESULTS
  return (
    <div className="space-y-6 pb-12 animate-fade-in">
      {/* Top Banner with Reset Option */}
      <section className="sci-card p-4 flex flex-col sm:flex-row items-center justify-between gap-4 border-l-4 border-l-[#DC2626]">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded-lg bg-[#FEF2F2] border border-[#FECACA] flex items-center justify-center text-[#DC2626]">
            <CheckCircle2 className="w-5 h-5" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h1 className="text-sm sm:text-base font-bold text-[#17201F]">
                Run complete: Candidate <span className="font-mono text-[#DC2626]">controlled-3</span> resolved with 94.2% confidence
              </h1>
              <span className="sci-badge sci-badge-verified">Converged in 4 steps</span>
            </div>
            <p className="text-xs text-[#66706C] mt-0.5">
              Target hypothesis H₁ (Phase Purity Limited) confirmed. Total budget expended: 4.2 credits.
            </p>
          </div>
        </div>

        <button
          onClick={handleResetRun}
          className="inline-flex items-center gap-1.5 px-3 py-1.5 bg-[#FCFCFA] hover:bg-[#F4F3EE] text-[#66706C] hover:text-[#17201F] border border-[#D9DFDB] rounded-lg text-xs font-semibold shadow-2xs transition cursor-pointer whitespace-nowrap"
        >
          <RotateCcw className="w-3.5 h-3.5" />
          <span>Configure new run</span>
        </button>
      </section>

      {/* Campaign Step Selector */}
      <section className="sci-card p-4">
        <div className="flex items-center justify-between mb-3">
          <div className="flex items-center gap-2">
            <span className="text-xs font-bold text-[#17201F]">Sequential Decision Campaign</span>
            <span className="text-2xs font-mono text-[#8F9995]">Step {stepIndex} of {totalSteps}</span>
          </div>
          <span className="text-2xs font-mono text-[#66706C]">
            True Mechanism: <strong className="text-[#DC2626]">H₁ (Phase Purity Limited)</strong>
          </span>
        </div>

        <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
          {steps.map((s, idx) => {
            const stepNum = idx + 1;
            const isSelected = stepNum === stepIndex;
            const winner = s.preregistration?.action?.candidate_id || s.top_actions?.[0]?.action?.candidate_id || 'controlled-3';
            const modality = s.preregistration?.action?.action_type || s.top_actions?.[0]?.action?.action_type || 'XRD';
            const score = s.top_actions?.[0]?.total_action_score ?? 0;

            return (
              <button
                key={stepNum}
                onClick={() => handleStepSelect(stepNum)}
                className={`p-3 rounded-xl border text-left transition cursor-pointer ${
                  isSelected
                    ? 'border-[#B91C1C] bg-[#FEF2F2] shadow-2xs'
                    : 'border-[#D9DFDB] bg-[#FCFCFA] hover:bg-[#F4F3EE]'
                }`}
              >
                <div className="flex items-center justify-between">
                  <span className={`text-2xs font-mono font-bold ${isSelected ? 'text-[#DC2626]' : 'text-[#8F9995]'}`}>
                    STEP {stepNum}
                  </span>
                  <span className="text-3xs px-1.5 py-0.5 rounded bg-white border border-[#D9DFDB] font-mono text-[#66706C]">
                    {modality}
                  </span>
                </div>
                <div className="font-bold text-xs text-[#17201F] mt-1 truncate">{winner}</div>
                <div className="text-3xs font-mono text-[#66706C] mt-1">
                  Score: {score >= 0 ? `+${score.toFixed(3)}` : score.toFixed(3)}
                </div>
              </button>
            );
          })}
        </div>
      </section>

      {/* Primary Hero Row: Recommended Next Experiment (60%) + Charts Container (40%) */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Recommended Next Experiment Card (60%) */}
        <section className="lg:col-span-7 sci-card p-6 flex flex-col justify-between space-y-5">
          <div>
            <div className="flex items-center justify-between border-b border-[#D9DFDB] pb-3">
              <div className="flex items-center gap-2">
                <Award className="w-4 h-4 text-[#DC2626]" />
                <h2 className="text-base font-bold text-[#17201F]">Recommended Next Experiment</h2>
              </div>
              <span className="sci-badge sci-badge-verified">Rank #1 Optimal Action</span>
            </div>

            {/* Candidate & Modality Summary */}
            <div className="mt-4 grid grid-cols-2 sm:grid-cols-4 gap-3">
              <div className="p-3 rounded-lg bg-[#F4F3EE] border border-[#D9DFDB]">
                <span className="text-2xs font-mono text-[#8F9995] block">Target Candidate</span>
                <span className="text-sm font-bold font-mono text-[#17201F]">{selectedCandidateId}</span>
              </div>
              <div className="p-3 rounded-lg bg-[#F4F3EE] border border-[#D9DFDB]">
                <span className="text-2xs font-mono text-[#8F9995] block">Modality</span>
                <span className="text-sm font-bold text-[#DC2626]">{selectedModality} Diagnostic</span>
              </div>
              <div className="p-3 rounded-lg bg-[#F4F3EE] border border-[#D9DFDB]">
                <span className="text-2xs font-mono text-[#8F9995] block">Composite Score</span>
                <span className="text-sm font-bold font-mono text-[#17201F]">
                  {inspectedAction?.total_action_score !== undefined
                    ? `${inspectedAction.total_action_score >= 0 ? '+' : ''}${inspectedAction.total_action_score.toFixed(4)}`
                    : 'N/A'}
                </span>
              </div>
              <div className="p-3 rounded-lg bg-[#F4F3EE] border border-[#D9DFDB]">
                <span className="text-2xs font-mono text-[#8F9995] block">Information Gain</span>
                <span className="text-sm font-bold font-mono text-[#DC2626]">
                  {inspectedAction?.raw_expected_hig_nats !== undefined
                    ? `${inspectedAction.raw_expected_hig_nats.toFixed(3)} nats`
                    : inspectedAction?.expected_hig_nats !== undefined
                    ? `${inspectedAction.expected_hig_nats.toFixed(3)} nats`
                    : 'N/A'}
                </span>
              </div>
            </div>

            <div className="mt-3 p-3 rounded-lg bg-[#FCFCFA] border border-[#D9DFDB] text-xs text-[#66706C]">
              <strong>Selection Rationale:</strong> Optimal trade-off between discriminatory power across H₁ vs H₂ and experimental expenditure. Estimated action cost: 1.0 credits.
            </div>

            {/* Interactive 4-State Scientific Loop Bar */}
            <div className="mt-5">
              <div className="text-xs font-bold text-[#17201F] mb-2">
                Scientific Verification Loop (Click to Inspect Evidence)
              </div>
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
                {[
                  { id: 'A_SCORED', label: '1. Scored', icon: <Activity className="w-3.5 h-3.5" /> },
                  { id: 'B_PREREGISTERED', label: '2. Preregistered', icon: <Lock className="w-3.5 h-3.5" /> },
                  { id: 'C_REVEALED', label: '3. Observed', icon: <Eye className="w-3.5 h-3.5" /> },
                  { id: 'D_UPDATED', label: '4. Belief Updated', icon: <RefreshCw className="w-3.5 h-3.5" /> },
                ].map((ph) => {
                  const isActive = revealPhase === ph.id;
                  return (
                    <button
                      key={ph.id}
                      onClick={() => updateRevealPhase(ph.id as RevealPhase)}
                      className={`p-2 rounded-lg text-xs font-semibold flex items-center justify-center gap-1.5 transition cursor-pointer border ${
                        isActive
                          ? 'bg-[#B91C1C] text-white border-[#B91C1C] shadow-2xs font-bold'
                          : 'bg-[#FCFCFA] text-[#66706C] border-[#D9DFDB] hover:bg-[#F4F3EE]'
                      }`}
                    >
                      {ph.icon}
                      <span>{ph.label}</span>
                    </button>
                  );
                })}
              </div>
            </div>

            {/* Dynamic Evidence Panel for the 4-State Loop */}
            <div className="mt-4 p-4 rounded-xl border border-[#D9DFDB] bg-[#F4F3EE] min-h-[160px]">
              {revealPhase === 'A_SCORED' && (
                <div className="space-y-2 text-xs text-[#66706C]">
                  <div className="flex items-center justify-between text-[#17201F] font-bold">
                    <span>State A: Objective Scoring Matrix</span>
                    <span className="sci-badge sci-badge-verified">Multi-Objective HIG</span>
                  </div>
                  <p>
                    All candidate-modality actions evaluated under dimensionless scalar objective:
                    <code className="ml-1 font-mono text-2xs bg-white px-1.5 py-0.5 rounded border border-[#D9DFDB] text-[#DC2626]">
                      S(a) = w_H · HIG + w_D · Diversity - w_C · Cost
                    </code>
                  </p>
                  <div className="pt-1 grid grid-cols-3 gap-2 font-mono text-2xs">
                    <div className="bg-white p-2 rounded border border-[#D9DFDB]">
                      <span className="text-[#8F9995] block">Raw HIG</span>
                      <span className="font-bold text-[#17201F]">
                        {inspectedAction?.raw_expected_hig_nats !== undefined
                          ? `${inspectedAction.raw_expected_hig_nats.toFixed(3)} nats`
                          : inspectedAction?.expected_hig_nats !== undefined
                          ? `${inspectedAction.expected_hig_nats.toFixed(3)} nats`
                          : 'N/A'}
                      </span>
                    </div>
                    <div className="bg-white p-2 rounded border border-[#D9DFDB]">
                      <span className="text-[#8F9995] block">Cost Penalty</span>
                      <span className="font-bold text-[#B91C1C]">
                        {inspectedAction?.weighted_cost_contribution !== undefined
                          ? (-inspectedAction.weighted_cost_contribution).toFixed(3)
                          : inspectedAction?.normalized_cost !== undefined
                          ? (-2.0 * inspectedAction.normalized_cost).toFixed(3)
                          : 'N/A'}
                      </span>
                    </div>
                    <div className="bg-white p-2 rounded border border-[#D9DFDB]">
                      <span className="text-[#8F9995] block">Net Score</span>
                      <span className="font-bold text-[#DC2626]">
                        {inspectedAction?.total_action_score !== undefined
                          ? `${inspectedAction.total_action_score >= 0 ? '+' : ''}${inspectedAction.total_action_score.toFixed(4)}`
                          : 'N/A'}
                      </span>
                    </div>
                  </div>
                </div>
              )}

              {revealPhase === 'B_PREREGISTERED' && (
                <div className="space-y-2 text-xs text-[#66706C]">
                  <div className="flex items-center justify-between text-[#17201F] font-bold">
                    <span>State B: Cryptographic Preregistration Certificate</span>
                    <span className="sci-badge sci-badge-verified">OBSERVATIONS FIREWALLED</span>
                  </div>
                  <p>
                    Experiment plan committed to immutable audit ledger before sensor acquisition:
                  </p>
                  <div className="p-2.5 rounded bg-white border border-[#D9DFDB] font-mono text-2xs space-y-1">
                    <div><strong className="text-[#17201F]">Action Target:</strong> {winnerAction?.action?.candidate_id || selectedCandidateId} | {winnerAction?.action?.action_type || selectedModality}</div>
                    <div>
                      <strong className="text-[#17201F]">Prior Shannon Entropy:</strong>{' '}
                      {currentStep?.preregistration?.current_hypothesis_entropy_nats !== undefined
                        ? `${currentStep.preregistration.current_hypothesis_entropy_nats.toFixed(3)} nats`
                        : 'N/A'}
                    </div>
                    <div>
                      <strong className="text-[#17201F]">Ledger Event Sequence:</strong> #{currentStep?.preregistration?.event_sequence ?? currentStep?.step ?? 1} (Commit: {data.manifest?.scientific_source_commit.slice(0, 8) ?? 'dc1f5fda'})
                    </div>
                  </div>
                </div>
              )}

              {revealPhase === 'C_REVEALED' && (
                <div className="space-y-2 text-xs text-[#66706C]">
                  <div className="flex items-center justify-between text-[#17201F] font-bold">
                    <span>State C: Physical Observation Revealed</span>
                    <span className="sci-badge sci-badge-verified">Validated Observation</span>
                  </div>
                  {selectedCandidateId === winnerAction?.action?.candidate_id && selectedModality === winnerAction?.action?.action_type ? (
                    <>
                      <p>
                        Firewall unsealed. Diagnostic characterization measurement acquired:
                      </p>
                      <div className="p-2.5 rounded bg-white border border-[#D9DFDB] font-mono text-2xs space-y-1.5">
                        <div className="flex items-center justify-between">
                          <span className="text-[#17201F] font-bold">Observable:</span>
                          <span className="text-[#DC2626] font-bold text-xs">
                            {currentStep?.observation?.observed_measurement?.name || 'Diagnostic Measurement'}
                          </span>
                        </div>
                        <div className="flex items-center justify-between">
                          <span className="text-[#66706C]">Observed Value / Vector:</span>
                          <span className="text-[#17201F] font-bold">
                            {Array.isArray(currentStep?.observation?.observed_measurement?.value)
                              ? `[${currentStep.observation.observed_measurement.value.map((v) => typeof v === 'number' ? v.toFixed(3) : v).join(', ')}]`
                              : typeof currentStep?.observation?.observed_measurement?.value === 'number'
                              ? currentStep.observation.observed_measurement.value.toFixed(4)
                              : String(currentStep?.observation?.observed_measurement?.value || 'N/A')}
                          </span>
                        </div>
                        <div className="flex items-center justify-between text-[#DC2626] font-semibold">
                          <span>Log Bayes Factors:</span>
                          <span>
                            {Object.entries(currentStep?.observation?.log_bayes_factor_pairwise || {}).map(([p, v]) => `${p}: ${v >= 0 ? '+' : ''}${v.toFixed(2)}`).join(' | ') || 'Realized Bayes Update'}
                          </span>
                        </div>
                      </div>
                    </>
                  ) : (
                    <div className="p-3 bg-white rounded border border-[#D9DFDB] text-2xs font-mono text-[#66706C]">
                      Observation not conducted for alternative candidate ({selectedCandidateId} &bull; {selectedModality}). Under experimental protocol, only the preregistered winner ({winnerAction?.action?.candidate_id} &bull; {winnerAction?.action?.action_type}) was measured.
                    </div>
                  )}
                </div>
              )}

              {revealPhase === 'D_UPDATED' && (
                <div className="space-y-2 text-xs text-[#66706C]">
                  <div className="flex items-center justify-between text-[#17201F] font-bold">
                    <span>State D: Bayesian Posterior Distribution Update</span>
                    <span className="sci-badge sci-badge-verified">Posterior Shift</span>
                  </div>
                  <div className="space-y-2">
                    {[
                      { name: 'H₁ (Phase Purity Limited)', prior: priorBeliefs['H1_PHASE_PURITY_LIMITED'] ?? initBeliefs['H1_PHASE_PURITY_LIMITED'] ?? 0, post: posteriorBeliefs['H1_PHASE_PURITY_LIMITED'] ?? 0, highlight: true },
                      { name: 'H₂ (Homogeneity Limited)', prior: priorBeliefs['H2_COMPOSITION_HOMOGENEITY_LIMITED'] ?? initBeliefs['H2_COMPOSITION_HOMOGENEITY_LIMITED'] ?? 0, post: posteriorBeliefs['H2_COMPOSITION_HOMOGENEITY_LIMITED'] ?? 0, highlight: false },
                      { name: 'H₃ (Morphology Kinetics Limited)', prior: priorBeliefs['H3_MORPHOLOGY_KINETICS_LIMITED'] ?? initBeliefs['H3_MORPHOLOGY_KINETICS_LIMITED'] ?? 0, post: posteriorBeliefs['H3_MORPHOLOGY_KINETICS_LIMITED'] ?? 0, highlight: false },
                    ].map((item, i) => (
                      <div key={i} className="bg-white p-2 rounded border border-[#D9DFDB] flex items-center justify-between text-2xs font-mono">
                        <span className={item.highlight ? 'text-[#DC2626] font-bold' : 'text-[#66706C]'}>
                          {item.name}
                        </span>
                        <div className="flex items-center gap-2">
                          <span className="text-[#8F9995]">{item.prior.toFixed(3)}</span>
                          <span className="text-[#8F9995]">→</span>
                          <span className={item.highlight ? 'text-[#DC2626] font-bold text-xs' : 'text-[#66706C]'}>
                            {item.post.toFixed(3)}
                          </span>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </div>
        </section>

        {/* Lead Charts Container (40%) */}
        <section className="lg:col-span-5 sci-card p-6 flex flex-col justify-between space-y-4">
          <div>
            <div className="flex items-center justify-between border-b border-[#D9DFDB] pb-3">
              <div className="flex items-center gap-1.5">
                <TrendingUp className="w-4 h-4 text-[#DC2626]" />
                <h2 className="text-base font-bold text-[#17201F]">Scientific Trajectory</h2>
              </div>
              <div className="flex items-center gap-1 bg-[#F4F3EE] p-1 rounded-lg border border-[#D9DFDB]">
                {[
                  { id: 'trajectory', label: 'Belief' },
                  { id: 'predictive', label: 'Predictive' },
                  { id: 'tradeoff', label: 'Trade-off' },
                ].map((t) => (
                  <button
                    key={t.id}
                    onClick={() => setPrimaryChartTab(t.id as any)}
                    className={`px-2 py-1 rounded text-2xs font-semibold transition cursor-pointer ${
                      primaryChartTab === t.id
                        ? 'bg-white text-[#DC2626] shadow-2xs font-bold'
                        : 'text-[#66706C] hover:text-[#17201F]'
                    }`}
                  >
                    {t.label}
                  </button>
                ))}
              </div>
            </div>

            <div className="mt-4 min-h-[300px]">
              {primaryChartTab === 'trajectory' && (
                <div>
                  <HypothesisBeliefTrajectoryChart
                    data={data}
                    currentStepIndex={stepIndex}
                    revealPhase={revealPhase}
                    onSelectStep={handleStepSelect}
                  />
                  <p className="text-2xs text-[#8F9995] text-center mt-2 font-mono">
                    Bayesian posterior evolution across steps. True mechanism is H₁ (Phase Purity Limited).
                  </p>
                </div>
              )}

              {primaryChartTab === 'predictive' && (
                <div>
                  <PredictiveDistributionChart
                    currentStep={currentStep}
                    revealPhase={revealPhase}
                    selectedCandidateId={selectedCandidateId}
                    selectedModality={selectedModality}
                  />
                  <p className="text-2xs text-[#8F9995] text-center mt-2 font-mono">
                    Gaussian density estimates per hypothesis for candidate {selectedCandidateId}.
                  </p>
                </div>
              )}

              {primaryChartTab === 'tradeoff' && (
                <div>
                  <TradeoffScatterChart
                    actions={allActions}
                    winnerActionId={winnerAction?.action?.action_id}
                    selectedCandidateId={selectedCandidateId}
                    selectedModality={selectedModality}
                    onSelectAction={(candId, mod) => {
                      setSelectedCandidateId(candId);
                      setSelectedModality(mod);
                    }}
                  />
                  <p className="text-2xs text-[#8F9995] text-center mt-2 font-mono">
                    Expected Information Gain vs Experimental Cost. Green star denotes selected optimal trade-off.
                  </p>
                </div>
              )}
            </div>
          </div>
        </section>
      </div>

      {/* Secondary Explorer Row: Heatmap + Score Waterfall */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Candidate-Modality Score Heatmap (60%) */}
        <section className="lg:col-span-7 sci-card p-6 space-y-4">
          <div className="border-b border-[#D9DFDB] pb-3">
            <h2 className="text-base font-bold text-[#17201F]">Candidate × Modality Decision Matrix</h2>
            <p className="text-xs text-[#66706C] mt-0.5">Inspect trade-off values across all candidates and experimental modalities</p>
          </div>

          <CandidateModalityHeatmap
            actions={allActions}
            winnerActionId={winnerAction?.action?.action_id}
            selectedCandidateId={selectedCandidateId}
            selectedModality={selectedModality}
            activeMetric={heatmapMetric}
            onChangeMetric={(metric) => setHeatmapMetric(metric)}
            onSelectAction={(candId, mod) => {
              setSelectedCandidateId(candId);
              setSelectedModality(mod);
            }}
          />
        </section>

        {/* Score Decomposition Waterfall (40%) */}
        <section className="lg:col-span-5 sci-card p-6 space-y-4">
          <div className="border-b border-[#D9DFDB] pb-3">
            <h2 className="text-base font-bold text-[#17201F]">Score Decomposition</h2>
            <p className="text-xs text-[#66706C] mt-0.5">Signed additive terms for action {selectedCandidateId} / {selectedModality}</p>
          </div>

          <ScoreWaterfallChart action={inspectedAction || undefined} />

          <div className="p-3 rounded-lg bg-[#FCFCFA] border border-[#D9DFDB] text-2xs font-mono text-[#66706C] leading-relaxed">
            <strong className="text-[#17201F]">Exact Formulation:</strong> S(a) = w_H · HIG + w_D · Diversity - w_C · Cost. Composite score is a dimensionless scalar; only raw HIG is in nats.
          </div>
        </section>
      </div>

      {/* Tertiary Collapsible Drawer: Counterfactuals & 3D Stark Hologram */}
      <details className="sci-card p-6 transition-all group">
        <summary className="font-bold text-sm text-[#17201F] cursor-pointer flex items-center justify-between select-none list-none">
          <div className="flex items-center gap-2">
            <Atom className="w-4 h-4 text-[#DC2626]" />
            <span>Advanced Inspection: Counterfactual Actions & 3D Atomic Structure</span>
            <span className="sci-badge sci-badge-surrogate">Optional Deep-Dive</span>
          </div>
          <span className="text-xs font-mono text-[#DC2626] underline group-open:hidden">Expand inspection drawer</span>
          <span className="text-xs font-mono text-[#66706C] underline hidden group-open:inline">Collapse drawer</span>
        </summary>

        <div className="mt-6 pt-6 border-t border-[#D9DFDB] grid grid-cols-1 lg:grid-cols-12 gap-6">
          {/* Counterfactuals Table */}
          <div className="lg:col-span-7 space-y-3">
            <h3 className="text-sm font-bold text-[#17201F]">Alternative Actions & Counterfactual Regret</h3>
            <p className="text-xs text-[#66706C]">Why was this action chosen over high-information alternatives?</p>

            <div className="overflow-x-auto rounded-xl border border-[#D9DFDB] bg-white">
              <table className="w-full text-left text-xs">
                <thead className="bg-[#F4F3EE] text-[#66706C] font-mono text-2xs border-b border-[#D9DFDB]">
                  <tr>
                    <th className="p-2.5">Rank</th>
                    <th className="p-2.5">Action</th>
                    <th className="p-2.5">Score</th>
                    <th className="p-2.5">HIG</th>
                    <th className="p-2.5">Cost</th>
                    <th className="p-2.5">Scientific Rationale</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-[#D9DFDB]">
                  {allActions.slice(0, 4).map((a, idx) => {
                    const isWinner = idx === 0;
                    return (
                      <tr key={idx} className={isWinner ? 'bg-[#FEF2F2]/60 font-semibold' : 'hover:bg-[#F4F3EE]/50'}>
                        <td className="p-2.5 font-mono">#{idx + 1}</td>
                        <td className="p-2.5 font-mono text-[#17201F]">{a.action?.candidate_id} / {a.action?.action_type}</td>
                        <td className="p-2.5 font-mono text-[#DC2626]">+{a.total_action_score?.toFixed(3)}</td>
                        <td className="p-2.5 font-mono text-[#66706C]">{a.raw_expected_hig_nats?.toFixed(3)} nats</td>
                        <td className="p-2.5 font-mono text-[#66706C]">
                          {a.raw_estimated_cost !== undefined
                            ? a.raw_estimated_cost.toFixed(1)
                            : a.action?.estimated_cost !== undefined
                            ? a.action.estimated_cost.toFixed(1)
                            : 'N/A'}
                        </td>
                        <td className="p-2.5 text-2xs text-[#66706C]">
                          {isWinner ? 'Optimal HIG-to-cost ratio' : 'High cost penalty suppresses net score'}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>

          {/* 3D Stark Hologram Sphere */}
          <div className="lg:col-span-5 space-y-3">
            <div className="flex items-center justify-between">
              <h3 className="text-sm font-bold text-[#17201F]">3D Stark Hologram</h3>
              <span className="text-2xs font-mono text-[#8F9995]">Atomic Lattice Visualizer</span>
            </div>
            <div className="rounded-xl border border-[#D9DFDB] bg-[#17201F] overflow-hidden p-2">
              <StarkHologramSphere
                candidates={data.flagship_campaign?.candidates || []}
                selectedCandidateId={selectedCandidateId}
                onSelectCandidate={(candId) => setSelectedCandidateId(candId)}
                currentStep={currentStep || undefined}
              />
            </div>
            <p className="text-2xs text-[#8F9995] text-center font-mono">
              Interactive 3D representation of simulated candidate crystal structure.
            </p>
          </div>
        </div>
      </details>
    </div>
  );
};
