import React, { useState, useEffect, useRef } from 'react';
import {
  SnapshotData,
  CampaignStep,
  ScoredActionRecord,
  RevealPhase,
  HeatmapMetricMode,
  DiscoveryFlowState,
  DatasetOption,
  SurrogateOptimizationView,
} from '../types/mission_control';
import { HypothesisBeliefTrajectoryChart } from '../components/charts/HypothesisBeliefTrajectoryChart';
import { PredictiveDistributionChart } from '../components/charts/PredictiveDistributionChart';
import { CandidateModalityHeatmap } from '../components/charts/CandidateModalityHeatmap';
import { ScoreWaterfallChart } from '../components/charts/ScoreWaterfallChart';
import { TradeoffScatterChart } from '../components/charts/TradeoffScatterChart';
import { StarkHologramSphere } from '../components/StarkHologramSphere';
import { ElectrolyteOptimizationChart } from '../components/charts/ElectrolyteOptimizationChart';
import { resolveCampaign, visibleCampaignStep } from '../utils/campaignResolver';
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
  ArrowRight,
  Database,
  FlaskConical,
  Zap,
  Info
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

const sourceNumber = (value: unknown, digits = 4) =>
  typeof value === 'number' && Number.isFinite(value) ? value.toFixed(digits) : 'Not recorded';

const SurrogateOptimizationPanel: React.FC<{
  view: SurrogateOptimizationView;
  simulation: SnapshotData['electrolyte_simulation'];
  onBackToSetup?: () => void;
}> = ({ view, simulation, onBackToSetup }) => {
  const run = view.simulationRun;
  return (
    <div className="space-y-6 pb-12 animate-fade-in">
      <section className="sci-card p-6 border-l-4 border-l-[#2563EB]">
        <div className="flex flex-col md:flex-row md:items-start md:justify-between gap-4">
          <div>
            <span className="sci-badge sci-badge-verified">SIMULATED SURROGATE</span>
            <h1 className="text-2xl font-bold text-[#17201F] mt-2">{view.displayName}</h1>
            <p className="text-sm text-[#66706C] mt-1 max-w-3xl">{view.banner.description}</p>
          </div>
          <div className="flex flex-col items-end gap-2">
            <div className="text-right text-xs font-mono text-[#66706C]">
              <div>Policy: {view.policy === 'HYBRID_DEFAULT' ? 'Hybrid Policy' : view.policy}</div>
              <div className="text-[#DC2626] font-semibold">{view.banner.badge}</div>
            </div>
            {onBackToSetup && (
              <button
                onClick={onBackToSetup}
                className="inline-flex items-center gap-1.5 px-3 py-1.5 bg-[#FCFCFA] hover:bg-[#F4F3EE] text-[#17201F] border border-[#D9DFDB] rounded-lg text-xs font-semibold shadow-2xs transition cursor-pointer"
              >
                <RotateCcw className="w-3.5 h-3.5 text-[#DC2626]" />
                <span>Change configuration</span>
              </button>
            )}
          </div>
        </div>
      </section>
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        {[
          ['Source target', view.scientificTargetName || 'Not recorded'],
          ['Best selected latent', sourceNumber(run.best_selected_latent_capacity)],
          ['Latent simple regret', sourceNumber(run.simple_regret_latent)],
          ['Queries recorded', String(run.queried_candidate_ids.length)],
        ].map(([label, value]) => (
          <div key={label} className="sci-card p-4">
            <div className="text-2xs uppercase tracking-wider text-[#8F9995]">{label}</div>
            <div className="text-lg font-mono font-semibold text-[#17201F] mt-1">{value}</div>
          </div>
        ))}
      </div>
      <section className="sci-card p-5">
        <div className="flex items-center justify-between mb-3">
          <h2 className="text-sm font-bold text-[#17201F]">Recorded surrogate trajectory</h2>
          <span className="text-xs font-mono text-[#66706C]">No physical measurements</span>
        </div>
        <ElectrolyteOptimizationChart
          simulationData={{
            ...simulation,
            detailed_policy_seed_runs: { [view.policy]: [view.simulationRun] },
            screeningDiagnostics: view.screeningDiagnostics,
          }}
          selectedPolicy={view.policy}
          selectedSeed={view.seed}
        />
      </section>
      <section className="sci-card p-5 text-xs text-[#66706C] space-y-2">
        {view.disclosures.map((disclosure) => <p key={disclosure}>• {disclosure}</p>)}
      </section>
    </div>
  );
};

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
  const registryEntries = data.dataset_registry?.datasets || [];
  const controlledInfo = registryEntries.find((entry) => entry.id === 'controlled_multimodal_alloy');
  const electrolyteInfo = registryEntries.find((entry) => entry.id === 'anode_free_electrolyte_screening');
  const alabInfo = registryEntries.find((entry) => entry.id === 'alab_precursor_genome');
  const controlledPolicyOptions = [
    { id: 'HYBRID', sourceId: 'HYBRID', label: 'Hybrid policy (HIG + Cost + Discovery)', tag: 'HYBRID' },
    { id: 'PURE_HIG', sourceId: 'PURE_HIG', label: 'Pure HIG policy (Information Gain Only)', tag: 'PURE_HIG' },
    { id: 'DISCOVERY_ONLY', sourceId: 'DISCOVERY_ONLY', label: 'Discovery-only policy (Exploitation)', tag: 'DISCOVERY_ONLY' },
  ].filter((option) => (controlledInfo?.availableConfigurations || []).some((configuration) => configuration.policy === option.sourceId));
  const surrogatePolicyOptions = [
    { id: 'HYBRID_DEFAULT', sourceId: 'HYBRID_DEFAULT', label: 'Hybrid surrogate policy (Uncertainty + Target)', tag: 'HYBRID_DEFAULT' },
    { id: 'PURE_FALSIFICATION', sourceId: 'PURE_FALSIFICATION', label: 'Pure falsification policy', tag: 'PURE_FALSIFICATION' },
    { id: 'RANDOM', sourceId: 'RANDOM', label: 'Random surrogate policy', tag: 'RANDOM' },
  ].filter((option) => (electrolyteInfo?.availableConfigurations || []).some((configuration) => configuration.policy === option.sourceId));

  // Flow state (setup -> running -> results)
  const [internalFlowState, setInternalFlowState] = useState<DiscoveryFlowState>('setup');
  const flowState = controlledFlowState ?? internalFlowState;
  const updateFlowState = (s: DiscoveryFlowState) => {
    setInternalFlowState(s);
    onFlowStateChange?.(s);
  };

  // Policy configuration state - reset on dataset change to prevent cross-domain leakage
  const [selectedPolicy, setSelectedPolicy] = useState<string>('HYBRID');

  // Dataset / Research question option
  const [internalDataset, setInternalDataset] = useState<DatasetOption>('controlled_synthesis');
  const dataset = controlledDataset ?? internalDataset;
  const updateDataset = (d: DatasetOption) => {
    setInternalDataset(d);
    onDatasetChange?.(d);
    if (d === 'electrolyte_search' || d === 'anode_free_electrolyte_screening') {
      setSelectedPolicy('HYBRID_DEFAULT');
    } else if (d === 'alab_replay' || d === 'alab_precursor_genome') {
      setSelectedPolicy(String(alabInfo?.defaultConfiguration?.policy || 'HYBRID'));
    } else {
      setSelectedPolicy('HYBRID');
    }
  };

  // Running animation state
  const [runningProgress, setRunningProgress] = useState<number>(0);
  const [runningLogIndex, setRunningLogIndex] = useState<number>(0);
  const progressTimerRef = useRef<any>(null);

  // Resolve only source-backed views; an unavailable configuration stays unavailable.
  const resolution = resolveCampaign(dataset, data, selectedPolicy);
  const resolved = resolution.ok ? resolution.value : null;
  const steps = resolved && 'steps' in resolved ? resolved.steps : [];
  const totalSteps = steps.length;
  const [stepIndex, setStepIndex] = useState<number>(controlledStepIndex);

  // Reset step index and ensure correct policy domain when dataset changes
  useEffect(() => {
    setStepIndex(1);
    updateRevealPhase('A_SCORED');
    if (dataset === 'electrolyte_search' || dataset === 'anode_free_electrolyte_screening') {
      if (selectedPolicy !== 'HYBRID_DEFAULT' && selectedPolicy !== 'PURE_FALSIFICATION' && selectedPolicy !== 'RANDOM') {
        setSelectedPolicy('HYBRID_DEFAULT');
      }
    } else if (dataset === 'alab_replay' || dataset === 'alab_precursor_genome') {
      // alab fixed replay
    } else {
      if (selectedPolicy !== 'HYBRID' && selectedPolicy !== 'PURE_HIG' && selectedPolicy !== 'DISCOVERY_ONLY') {
        setSelectedPolicy('HYBRID');
      }
    }
  }, [dataset]);

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
  const rawCurrentStep: CampaignStep | null = steps[stepIndex - 1] || steps[0] || null;
  const currentStep: CampaignStep | null = rawCurrentStep ? visibleCampaignStep(rawCurrentStep, revealPhase) : null;
  const winnerAction: ScoredActionRecord | null = currentStep?.top_actions?.[0] || null;
  const candidates = resolved && 'candidates' in resolved ? resolved.candidates : [];
  const modalities = resolved && 'modalities' in resolved ? resolved.modalities : [];
  const [selectedCandidateId, setSelectedCandidateId] = useState<string>(
    winnerAction?.action?.candidate_id || candidates[0]?.candidate_id || ''
  );
  const [selectedModality, setSelectedModality] = useState<string>(
    winnerAction?.action?.action_type || modalities[0] || ''
  );

  useEffect(() => {
    if (winnerAction?.action) {
      setSelectedCandidateId(winnerAction.action.candidate_id);
      setSelectedModality(winnerAction.action.action_type);
    } else if (candidates.length > 0) {
      setSelectedCandidateId(candidates[0].candidate_id);
      setSelectedModality(modalities[0] || '');
    }
  }, [winnerAction, dataset, stepIndex, candidates, modalities]);

  if (!resolved) {
    if (resolution.ok) return null;
    return (
      <section className="sci-card p-6 border-l-4 border-l-[#B91C1C]">
        <h1 className="text-lg font-bold text-[#17201F]">Campaign unavailable</h1>
        <p className="text-sm text-[#66706C] mt-2">{resolution.message}</p>
        {Boolean(resolution.availableOptions) && (
          <p className="text-xs font-mono text-[#8F9995] mt-3">Available: {String(JSON.stringify(resolution.availableOptions))}</p>
        )}
      </section>
    );
  }

  const handleStepSelect = (s: number) => {
    const clamped = Math.max(1, Math.min(s, totalSteps));
    setStepIndex(clamped);
    updateRevealPhase('A_SCORED');
    onStepChange?.(clamped);
  };

  // Presentation-only loading animation; it never executes a new experiment.
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

  const runningLogs =
    dataset === 'alab_replay' || dataset === 'alab_precursor_genome'
      ? [
          `Loading source-linked A-Lab catalog (${alabInfo?.candidateCount ?? 'not recorded'} samples)...`,
          `Resolving recorded replay ${alabInfo?.defaultConfiguration?.runId ?? 'not recorded'}...`,
          `Loading featured samples ${(alabInfo?.featuredCandidateIds || []).join(', ') || 'not recorded'}...`,
          'Keeping unlinked modalities out of the candidate × modality action space...',
          'Historical replay ready; original laboratory policy is not inferred.'
        ]
      : dataset === 'electrolyte_search' || dataset === 'anode_free_electrolyte_screening'
      ? [
          `Loading ${electrolyteInfo?.candidateCount ?? 'not recorded'} virtual electrolyte candidates...`,
          'Loading the source screening diagnostic and its tranche metadata...',
          `Selecting the source working set (${electrolyteInfo?.screenedWorkingSetCount ?? 'not recorded'} candidates)...`,
          'Loading surrogate optimization trajectory...',
          'Surrogate trajectory ready; no physical battery measurement is implied.'
        ]
      : [
          `Loading ${controlledInfo?.hypotheses?.length ?? 'not recorded'} source hypotheses and ${controlledInfo?.candidateCount ?? 'not recorded'} source candidates...`,
          'Loading the selected recorded policy trajectory...',
          'Showing the preregistered action before revealing its observation...',
          'Revealing the source synthetic measurement...',
          'Updating the recorded posterior without claiming physical confirmation.'
        ];

  const allActions = currentStep?.all_scored_actions || currentStep?.top_actions || [];
  const inspectedAction: ScoredActionRecord | null = allActions.find(
    (a) => a.action?.candidate_id === selectedCandidateId && a.action?.action_type === selectedModality
  ) || null;

  const initBeliefs = ('campaign' in resolved ? resolved.campaign.initial_beliefs : {}) || {};
  const selectedHistoricalSample = resolved.kind === 'historical_replay'
    ? resolved.alabSamples.find((sample) => sample.sample_id === selectedCandidateId)
    : undefined;

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
                   Source analysis available
                </span>
                <span className="text-2xs text-[#8F9995]">•</span>
                <span className="text-2xs font-mono text-[#66706C]">Closed-Loop Scientific Engine</span>
              </div>
              <h1 className="text-2xl font-bold text-[#17201F] tracking-tight">
                 Explore Source-Backed Scientific Workflows
              </h1>
              <p className="text-sm text-[#66706C] mt-1 max-w-3xl leading-relaxed">
                 Explore recorded scientific decision workflows across controlled, retrospective, and surrogate evidence regimes.
              </p>
            </div>
            <div className="flex items-center gap-2">
              <button
                onClick={handleStartDiscovery}
                className="inline-flex items-center gap-2 px-5 py-2.5 bg-[#B91C1C] hover:bg-[#991B1B] text-white rounded-xl font-semibold text-sm shadow-xs transition cursor-pointer"
              >
                <Play className="w-4 h-4 fill-white" />
                 <span>Open recorded analysis</span>
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
                 <p className="text-xs text-[#66706C] mt-0.5">Choose a source-backed evidence domain</p>
              </div>
              <span className="text-2xs font-mono text-[#8F9995]">3 options</span>
            </div>

            <div className="space-y-3">
              {/* Option 1: Controlled Multimodal Alloy Benchmark */}
              <div
                onClick={() => updateDataset('controlled_synthesis')}
                className={`p-4 rounded-xl border transition cursor-pointer ${
                  dataset === 'controlled_synthesis' || dataset === 'controlled_multimodal_alloy'
                    ? 'border-[#B91C1C] bg-[#FEF2F2] shadow-2xs'
                    : 'border-[#D9DFDB] bg-[#FCFCFA] hover:border-[#8F9995]'
                }`}
              >
                <div className="flex items-start justify-between gap-3">
                  <div className="flex items-start gap-3">
                    <input
                      type="radio"
                      name="dataset"
                      checked={dataset === 'controlled_synthesis' || dataset === 'controlled_multimodal_alloy'}
                      onChange={() => updateDataset('controlled_synthesis')}
                      className="mt-1 text-[#DC2626] focus:ring-[#DC2626]"
                    />
                    <div>
                      <div className="flex items-center gap-2">
                        <h3 className="text-sm font-bold text-[#17201F]">
                          Controlled Multimodal Alloy Benchmark
                        </h3>
                        <span className="sci-badge sci-badge-verified">Controlled Benchmark</span>
                      </div>
                      <p className="text-xs text-[#66706C] mt-1">
                        {controlledInfo?.summary || 'Source summary unavailable.'}
                      </p>
                      <div className="flex flex-wrap items-center gap-2 sm:gap-4 mt-2 text-2xs font-mono text-[#8F9995]">
                        <span>Source candidates: {controlledInfo?.candidateCount ?? 'N/A'}</span>
                        <span>•</span>
                        <span>Ground truth: In-silico underlying world</span>
                        <span>•</span>
                        <span className="text-[#DC2626] font-semibold">Status: {controlledInfo?.statusBadge || 'N/A'}</span>
                      </div>
                    </div>
                  </div>
                </div>
              </div>

              {/* Option 2: Electrolyte Screening & Surrogate Optimization */}
              <div
                onClick={() => updateDataset('electrolyte_search')}
                className={`p-4 rounded-xl border transition cursor-pointer ${
                  dataset === 'electrolyte_search' || dataset === 'anode_free_electrolyte_screening'
                    ? 'border-[#B91C1C] bg-[#FEF2F2] shadow-2xs'
                    : 'border-[#D9DFDB] bg-[#FCFCFA] hover:border-[#8F9995]'
                }`}
              >
                <div className="flex items-start justify-between gap-3">
                  <div className="flex items-start gap-3">
                    <input
                      type="radio"
                      name="dataset"
                      checked={dataset === 'electrolyte_search' || dataset === 'anode_free_electrolyte_screening'}
                      onChange={() => updateDataset('electrolyte_search')}
                      className="mt-1 text-[#DC2626] focus:ring-[#DC2626]"
                    />
                    <div>
                      <div className="flex items-center gap-2">
                        <h3 className="text-sm font-bold text-[#17201F]">
                          Anode-Free Electrolyte Screening & Surrogate Optimization
                        </h3>
                        <span className="sci-badge sci-badge-surrogate">Screening & Surrogate</span>
                      </div>
                      <p className="text-xs text-[#66706C] mt-1">
                        {electrolyteInfo?.summary || 'Source summary unavailable.'}
                      </p>
                      <div className="flex flex-wrap items-center gap-2 sm:gap-4 mt-2 text-2xs font-mono text-[#8F9995]">
                        <span>Working set: {electrolyteInfo?.screenedWorkingSetCount ?? 'N/A'}</span>
                        <span>•</span>
                        <span>Target: {electrolyteInfo?.scientificTargetName || electrolyteInfo?.targetObservable || 'N/A'}</span>
                        <span>•</span>
                        <span className="text-[#DC2626] font-semibold">Status: {electrolyteInfo?.statusBadge || 'N/A'}</span>
                      </div>
                    </div>
                  </div>
                </div>
              </div>

              {/* Option 3: A-Lab Precursor Genome Replay */}
              <div
                onClick={() => updateDataset('alab_replay')}
                className={`p-4 rounded-xl border transition cursor-pointer ${
                  dataset === 'alab_replay' || dataset === 'alab_precursor_genome'
                    ? 'border-[#B91C1C] bg-[#FEF2F2] shadow-2xs'
                    : 'border-[#D9DFDB] bg-[#FCFCFA] hover:border-[#8F9995]'
                }`}
              >
                <div className="flex items-start justify-between gap-3">
                  <div className="flex items-start gap-3">
                    <input
                      type="radio"
                      name="dataset"
                      checked={dataset === 'alab_replay' || dataset === 'alab_precursor_genome'}
                      onChange={() => updateDataset('alab_replay')}
                      className="mt-1 text-[#DC2626] focus:ring-[#DC2626]"
                    />
                    <div>
                      <div className="flex items-center gap-2">
                        <h3 className="text-sm font-bold text-[#17201F]">
                          A-Lab Precursor Genome Retrospective Replay
                        </h3>
                        <span className="sci-badge sci-badge-historical">Historical Validation</span>
                      </div>
                      <p className="text-xs text-[#66706C] mt-1">
                        {alabInfo?.summary || 'Source summary unavailable.'}
                      </p>
                      <div className="flex flex-wrap items-center gap-2 sm:gap-4 mt-2 text-2xs font-mono text-[#8F9995]">
                        <span>Source samples: {alabInfo?.candidateCount ?? 'N/A'}</span>
                        <span>•</span>
                        <span>Featured targets: {alabInfo?.featuredCandidateIds?.length ?? 'Not recorded'} characterized benchmarks</span>
                        <span>•</span>
                        <span className="text-[#DC2626] font-semibold">Status: {alabInfo?.statusBadge || 'N/A'}</span>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </section>

          {/* Right Panel: Recorded Policy Configuration */}
          <section className="lg:col-span-5 sci-card p-6 space-y-5 flex flex-col justify-between">
            <div>
              <div className="flex items-center justify-between border-b border-[#D9DFDB] pb-3">
                <div>
                  <h2 className="text-base font-bold text-[#17201F]">2. Recorded Policy Configuration</h2>
                  <p className="text-xs text-[#66706C] mt-0.5">Choose a recorded source policy; run configuration and modality availability remain source-defined</p>
                </div>
                <Sliders className="w-4 h-4 text-[#DC2626]" />
              </div>

              {dataset === 'alab_replay' || dataset === 'alab_precursor_genome' ? (
                <div className="mt-4 p-4 rounded-xl bg-[#FEF2F2] border border-[#FECACA] space-y-2">
                  <div className="flex items-center gap-2 text-[#991B1B] font-bold text-xs">
                    <Info className="w-4 h-4 text-[#DC2626]" />
                    <span>Fixed Historical Laboratory Replay</span>
                  </div>
                  <p className="text-xs text-[#66706C] leading-relaxed">
                    This campaign represents an immutable retrospective replay across source-linked laboratory synthesis samples. Replay evaluates preregistered autonomous action decisions against experimental outcomes.
                  </p>
                  <div className="pt-2 text-2xs text-[#8F9995] space-y-1">
                    <div>• Preregistered Policy: <strong className="text-[#17201F]">{alabInfo?.defaultConfiguration?.policy || 'Hybrid Policy'}</strong></div>
                    <div>• Available Modalities: {Object.entries(alabInfo?.modalities || {}).filter(([, modality]) => modality.available).map(([name, modality]) => `${name} (${modality.linkedCandidateCount ?? 'N/A'} linked)`).join(', ') || 'N/A'}</div>
                    <div>• Disclosure: {alabInfo?.disclosures?.find((text) => text.includes('SEM')) || 'Source linkage limitations unavailable.'}</div>
                  </div>
                </div>
              ) : dataset === 'electrolyte_search' || dataset === 'anode_free_electrolyte_screening' ? (
                <div className="space-y-4 mt-4">
                  <div>
                    <label className="block text-xs font-bold text-[#17201F] mb-1.5">
                      Surrogate Acquisition Policy
                    </label>
                    <div className="space-y-1.5">
                      {surrogatePolicyOptions.map((p) => (
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
                          {selectedPolicy === p.id && (
                            <span className="text-3xs font-semibold px-1.5 py-0.5 rounded bg-[#DC2626] text-white">Active</span>
                          )}
                        </div>
                      ))}
                    </div>
                  </div>
                  <div className="p-3 rounded-xl bg-[#FCFCFA] border border-[#D9DFDB] text-2xs text-[#66706C] space-y-1">
                    <div className="text-[#17201F] font-bold">Surrogate Optimization Framework:</div>
                    <div>• Optimization Model: Ensemble Surrogate Regressor</div>
                    <div>• Target Property: {electrolyteInfo?.scientificTargetName || 'Not recorded'}</div>
                    <div>• Candidate Search Space: {electrolyteInfo?.candidateCount ? electrolyteInfo.candidateCount.toLocaleString() : 'Not recorded'} virtual candidates (Screened: {electrolyteInfo?.screenedWorkingSetCount ?? 'Not recorded'})</div>
                  </div>
                </div>
              ) : (
                <div className="space-y-4 mt-4">
                  {/* Acquisition Policy */}
                  <div>
                    <label className="block text-xs font-bold text-[#17201F] mb-1.5">
                      Acquisition Policy
                    </label>
                    <div className="space-y-1.5">
                      {controlledPolicyOptions.map((p) => (
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
                          {selectedPolicy === p.id && (
                            <span className="text-3xs font-semibold px-1.5 py-0.5 rounded bg-[#DC2626] text-white">Active</span>
                          )}
                        </div>
                      ))}
                    </div>
                  </div>

                </div>
              )}
            </div>

            {/* Campaign Preview & Launch Action */}
            <div className="pt-4 border-t border-[#D9DFDB] space-y-3">
              <div className="p-3 rounded-lg bg-[#F4F3EE] border border-[#D9DFDB] text-2xs font-mono text-[#66706C] flex items-center justify-between">
                <span>Recorded actions: {totalSteps} {dataset === 'electrolyte_search' || dataset === 'anode_free_electrolyte_screening' ? 'queries' : 'steps'}</span>
                <span>•</span>
                <span>Expended budget: {resolved.banner.budgetExpended} {resolved.banner.budgetUnits}</span>
           <span>•</span>
           <span>{resolved.statusBadge}</span>
              </div>
              <button
                onClick={handleStartDiscovery}
                className="w-full py-3 bg-[#B91C1C] hover:bg-[#991B1B] text-white rounded-xl font-bold text-sm shadow-xs transition flex items-center justify-center gap-2 cursor-pointer"
              >
                <Play className="w-4 h-4 fill-white" />
                <span>
                  {dataset === 'alab_replay' || dataset === 'alab_precursor_genome'
                    ? 'Replay recorded decision sequence'
                    : dataset === 'electrolyte_search' || dataset === 'anode_free_electrolyte_screening'
                    ? 'Open surrogate trajectory'
                    : 'Open recorded analysis'}
                </span>
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
            <span className="sci-badge sci-badge-verified">Recorded Run Playback</span>
            <h2 className="text-xl font-bold text-[#17201F] mt-2">
              Loading Source-Backed Research Run
            </h2>
            <p className="text-xs text-[#66706C] mt-1">
              Preparing the selected recorded trajectory for inspection; this control does not execute a new experiment.
            </p>
          </div>

          {/* Progress Bar */}
          <div className="space-y-2">
            <div className="flex items-center justify-between text-xs font-mono text-[#66706C]">
               <span>Presentation loading</span>
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
               Source Playback Log
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
               Skip loading animation and inspect the record →
            </button>
          </div>
        </div>
      </div>
    );
  }

  // STATE 3: RESULTS
  if (resolved.kind === 'surrogate_optimization') {
    return (
      <SurrogateOptimizationPanel
        view={resolved}
        simulation={data.electrolyte_simulation}
        onBackToSetup={() => updateFlowState('setup')}
      />
    );
  }

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
                {resolved.banner.title}
              </h1>
              <span className="sci-badge sci-badge-verified">{resolved.banner.badge}</span>
            </div>
            <p className="text-xs text-[#66706C] mt-0.5">
              {resolved.banner.description}
            </p>
          </div>
        </div>

        <button
          onClick={handleResetRun}
          className="inline-flex items-center gap-1.5 px-3 py-1.5 bg-[#FCFCFA] hover:bg-[#F4F3EE] text-[#66706C] hover:text-[#17201F] border border-[#D9DFDB] rounded-lg text-xs font-semibold shadow-2xs transition cursor-pointer whitespace-nowrap"
        >
          <RotateCcw className="w-3.5 h-3.5" />
           <span>Change source configuration</span>
        </button>
      </section>

      {/* Campaign Step Selector */}
      <section className="sci-card p-4">
        <div className="flex items-center justify-between mb-3">
          <div className="flex items-center gap-2">
            <span className="text-xs font-bold text-[#17201F]">
               {dataset === 'electrolyte_search' || dataset === 'anode_free_electrolyte_screening'
                 ? 'Recorded Surrogate Query Sequence'
                 : 'Recorded Decision Sequence'}
            </span>
            <span className="text-2xs font-mono text-[#8F9995]">Step {stepIndex} of {totalSteps}</span>
          </div>
          <span className="text-xs text-[#66706C]">
            {dataset === 'controlled_synthesis' || dataset === 'controlled_multimodal_alloy' ? (
               <>True Mechanism: <strong className="text-[#DC2626]">H₁ (Phase Purity Limited)</strong></>
            ) : dataset === 'alab_replay' || dataset === 'alab_precursor_genome' ? (
               <>Benchmark: <strong className="text-[#DC2626]">A-Lab Solid-State Synthesis</strong></>
            ) : (
               <>Optimization Framework: <strong className="text-[#DC2626]">Ensemble Surrogate Regressor</strong></>
            )}
          </span>
        </div>

        <div className={`grid gap-2 ${totalSteps > 8 ? 'grid-cols-3 sm:grid-cols-5 md:grid-cols-8' : 'grid-cols-2 sm:grid-cols-4 md:grid-cols-6'}`}>
          {steps.map((s, idx) => {
            const stepNum = idx + 1;
            const isSelected = stepNum === stepIndex;
            const winner = s.preregistration?.action?.candidate_id || s.top_actions?.[0]?.action?.candidate_id || 'N/A';
            const modality = s.preregistration?.action?.action_type || s.top_actions?.[0]?.action?.action_type || 'Not recorded';
            const score = s.top_actions?.[0]?.total_action_score;

            return (
              <button
                key={stepNum}
                onClick={() => handleStepSelect(stepNum)}
                className={`p-2.5 rounded-xl border text-left transition cursor-pointer ${
                  isSelected
                    ? 'border-[#B91C1C] bg-[#FEF2F2] shadow-2xs'
                    : 'border-[#D9DFDB] bg-[#FCFCFA] hover:bg-[#F4F3EE]'
                }`}
              >
                <div className="flex items-center justify-between">
                  <span className={`text-2xs font-mono font-bold ${isSelected ? 'text-[#DC2626]' : 'text-[#8F9995]'}`}>
                    {dataset === 'electrolyte_search' || dataset === 'anode_free_electrolyte_screening' ? `Q${stepNum}` : `STEP ${stepNum}`}
                  </span>
                  <span className="text-3xs px-1 py-0.5 rounded bg-white border border-[#D9DFDB] font-mono text-[#66706C] truncate max-w-[60px]">
                    {modality}
                  </span>
                </div>
                <div className="font-bold text-xs text-[#17201F] mt-1 truncate">{winner}</div>
                <div className="text-3xs font-mono text-[#66706C] mt-0.5">
                  Score: {typeof score === 'number' ? (score >= 0 ? `+${score.toFixed(3)}` : score.toFixed(3)) : 'Not recorded'}
                </div>
              </button>
            );
          })}
        </div>
      </section>

       {/* Primary Hero Row: recorded action + charts container */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
         {/* Recorded action card */}
        <section className="lg:col-span-6 sci-card p-6 flex flex-col justify-between space-y-5">
          <div>
            <div className="flex items-center justify-between border-b border-[#D9DFDB] pb-3">
              <div className="flex items-center gap-2">
                <Award className="w-4 h-4 text-[#DC2626]" />
                 <h2 className="text-base font-bold text-[#17201F]">Recorded Action Under Inspection</h2>
              </div>
               <span className="sci-badge sci-badge-verified">Source-ranked action</span>
            </div>

            {/* Candidate & Modality Summary */}
            {inspectedAction ? (
              <>
                <div className="mt-4 grid grid-cols-2 sm:grid-cols-4 gap-3">
                  <div className="p-3 rounded-lg bg-[#F4F3EE] border border-[#D9DFDB]">
                    <span className="text-2xs font-mono text-[#8F9995] block">Recorded Candidate</span>
                    <span className="text-sm font-bold font-mono text-[#17201F]">{selectedCandidateId}</span>
                  </div>
                  <div className="p-3 rounded-lg bg-[#F4F3EE] border border-[#D9DFDB]">
                    <span className="text-2xs font-mono text-[#8F9995] block">Modality</span>
                    <span className="text-sm font-bold text-[#DC2626]">{selectedModality}</span>
                  </div>
                  <div className="p-3 rounded-lg bg-[#F4F3EE] border border-[#D9DFDB]">
                    <span className="text-2xs font-mono text-[#8F9995] block">Composite Score</span>
                    <span className="text-sm font-bold font-mono text-[#17201F]">
                      {inspectedAction.total_action_score !== undefined
                        ? `${inspectedAction.total_action_score >= 0 ? '+' : ''}${inspectedAction.total_action_score.toFixed(4)}`
                        : 'N/A'}
                    </span>
                  </div>
                  <div className="p-3 rounded-lg bg-[#F4F3EE] border border-[#D9DFDB]">
                    <span className="text-2xs font-mono text-[#8F9995] block">Information Gain</span>
                    <span className="text-sm font-bold font-mono text-[#DC2626]">
                      {(inspectedAction.raw_expected_hig_nats ?? inspectedAction.expected_hig_nats) !== undefined
                        ? `${(inspectedAction.raw_expected_hig_nats ?? inspectedAction.expected_hig_nats)?.toFixed(3)} nats`
                        : 'N/A'}
                    </span>
                  </div>
                </div>

                <div className="mt-3 p-3 rounded-lg bg-[#FCFCFA] border border-[#D9DFDB] text-xs text-[#66706C]">
                  <strong>Recorded score context:</strong> {inspectedAction.action.action_type} on {inspectedAction.action.candidate_id} has source score {sourceNumber(inspectedAction.total_action_score)} and recorded cost {sourceNumber(inspectedAction.action.estimated_cost)}.
                </div>
              </>
            ) : (
              <div className="mt-4 p-4 rounded-xl border border-[#D9DFDB] bg-[#F4F3EE] text-xs text-[#66706C] flex items-center gap-3">
                <Info className="w-5 h-5 text-[#8F9995] shrink-0" />
                <div>
                  <div className="font-bold text-[#17201F]">Action not scored in this recorded step</div>
                  <div className="mt-0.5 text-[#66706C]">
                    Unavailable / infeasible action: {selectedCandidateId} × {selectedModality} was not evaluated in step {stepIndex}.
                  </div>
                </div>
              </div>
            )}

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
                  {inspectedAction ? (
                    <>
                      <p>
                        Source action-score records are shown as persisted; no presentation-layer weights or normalized components are reconstructed.
                      </p>
                      <div className="pt-1 grid grid-cols-3 gap-2 font-mono text-2xs">
                        <div className="bg-white p-2 rounded border border-[#D9DFDB]">
                          <span className="text-[#8F9995] block">Raw HIG</span>
                          <span className="font-bold text-[#17201F]">
                            {(inspectedAction.raw_expected_hig_nats ?? inspectedAction.expected_hig_nats) !== undefined
                              ? `${(inspectedAction.raw_expected_hig_nats ?? inspectedAction.expected_hig_nats)?.toFixed(3)} nats`
                              : 'N/A'}
                          </span>
                        </div>
                        <div className="bg-white p-2 rounded border border-[#D9DFDB]">
                          <span className="text-[#8F9995] block">Recorded Cost</span>
                          <span className="font-bold text-[#B91C1C]">
                            {inspectedAction.action?.estimated_cost !== undefined
                              ? inspectedAction.action.estimated_cost.toFixed(3)
                              : 'N/A'}
                          </span>
                        </div>
                        <div className="bg-white p-2 rounded border border-[#D9DFDB]">
                          <span className="text-[#8F9995] block">Net Score</span>
                          <span className="font-bold text-[#DC2626]">
                            {inspectedAction.total_action_score !== undefined
                              ? `${inspectedAction.total_action_score >= 0 ? '+' : ''}${inspectedAction.total_action_score.toFixed(4)}`
                              : 'N/A'}
                          </span>
                        </div>
                      </div>
                    </>
                  ) : (
                    <div className="p-3 bg-white rounded border border-[#D9DFDB] text-[#66706C]">
                      Action not scored in this recorded step. No source HIG, cost, or net score was evaluated for {selectedCandidateId} × {selectedModality}.
                    </div>
                  )}
                </div>
              )}

              {revealPhase === 'B_PREREGISTERED' && (
                <div className="space-y-2 text-xs text-[#66706C]">
                  <div className="flex items-center justify-between text-[#17201F] font-bold">
                    <span>State B: Cryptographic Preregistration Certificate</span>
                    <span className="sci-badge sci-badge-verified">OBSERVATIONS FIREWALLED</span>
                  </div>
                  <p>
                    Recorded plan appears before the source observation reveal:
                  </p>
                  <div className="p-2.5 rounded bg-white border border-[#D9DFDB] font-mono text-2xs space-y-1">
                    <div><strong className="text-[#17201F]">Action Target:</strong> {currentStep?.preregistration?.action?.candidate_id || selectedCandidateId} | {currentStep?.preregistration?.action?.action_type || selectedModality}</div>
                    <div>
                      <strong className="text-[#17201F]">Prior Shannon Entropy:</strong>{' '}
                      {currentStep?.preregistration?.current_hypothesis_entropy_nats !== undefined
                        ? `${currentStep.preregistration.current_hypothesis_entropy_nats.toFixed(3)} nats`
                        : 'N/A'}
                    </div>
                    <div>
                      <strong className="text-[#17201F]">Preregistration Status:</strong>{' '}
                      <span className="text-[#DC2626] font-semibold">Locked & Cryptographically Verified</span>
                    </div>
                  </div>
                </div>
              )}

              {revealPhase === 'C_REVEALED' && (
                <div className="space-y-2 text-xs text-[#66706C]">
                  <div className="flex items-center justify-between text-[#17201F] font-bold">
                    <span>State C: Source Observation Revealed</span>
                    <span className="sci-badge sci-badge-verified">Recorded Observation</span>
                  </div>
                  {selectedCandidateId === winnerAction?.action?.candidate_id && selectedModality === winnerAction?.action?.action_type ? (
                    <>
                      <p>
                         Source observation revealed:
                      </p>
                      <div className="p-2.5 rounded bg-white border border-[#D9DFDB] font-mono text-2xs space-y-1.5">
                        <div className="flex items-center justify-between">
                          <span className="text-[#17201F] font-bold">Observable:</span>
                          <span className="text-[#DC2626] font-bold text-xs">
                             {currentStep?.observation?.observed_measurement?.name || 'Source measurement record'}
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
                    <span className="sci-badge sci-badge-verified">
                      {dataset === 'alab_replay' || dataset === 'alab_precursor_genome'
                        ? 'Historical Telemetry'
                        : dataset === 'electrolyte_search' || dataset === 'anode_free_electrolyte_screening'
                        ? 'Surrogate Evaluation'
                        : 'Posterior Shift'}
                    </span>
                  </div>
                  {dataset === 'alab_replay' || dataset === 'alab_precursor_genome' ? (
                    <div className="bg-white p-3 rounded-lg border border-[#D9DFDB] space-y-2 font-mono text-2xs">
                      <div className="flex items-center justify-between">
                        <span className="text-[#66706C]">Landmark Synthesis Target:</span>
                        <span className="text-[#17201F] font-bold">{selectedCandidateId}</span>
                      </div>
                      <div className="flex items-center justify-between">
                        <span className="text-[#66706C]">Reaction category:</span>
                        <span className="text-[#DC2626] font-bold">{selectedHistoricalSample?.reaction_category || 'Not recorded'}</span>
                      </div>
                      <div className="flex items-center justify-between text-[#8F9995] pt-1 border-t border-[#D9DFDB]">
                        <span>Validation Provenance:</span>
                        <span>{selectedHistoricalSample?.source_archive || 'Source archive not recorded'}</span>
                      </div>
                    </div>
                  ) : dataset === 'electrolyte_search' || dataset === 'anode_free_electrolyte_screening' ? (
                    <div className="bg-white p-3 rounded-lg border border-[#D9DFDB] space-y-2 font-mono text-2xs">
                      <div className="flex items-center justify-between">
                        <span className="text-[#66706C]">Queried Candidate:</span>
                        <span className="text-[#17201F] font-bold">{selectedCandidateId}</span>
                      </div>
                      <div className="flex items-center justify-between">
                        <span className="text-[#66706C]">Normalized Retention Capacity (Cycle 20):</span>
                        <span className="text-[#DC2626] font-bold">
                          {inspectedAction?.raw_discovery_utility !== undefined
                            ? inspectedAction.raw_discovery_utility.toFixed(4)
                            : 'N/A'}
                        </span>
                      </div>
                      <div className="flex items-center justify-between">
                        <span className="text-[#66706C]">Cumulative Information Gain:</span>
                        <span className="text-[#17201F] font-bold">
                          {inspectedAction?.raw_expected_hig_nats !== undefined
                            ? `${inspectedAction.raw_expected_hig_nats.toFixed(4)} nats`
                            : 'N/A'}
                        </span>
                      </div>
                      <div className="flex items-center justify-between text-[#8F9995] pt-1 border-t border-[#D9DFDB]">
                        <span>Surrogate Model:</span>
                        <span className="font-semibold text-[#17201F]">Ensemble Surrogate Regressor</span>
                      </div>
                    </div>
                  ) : (
                    <div className="space-y-2">
                      {Array.from(new Set([...Object.keys(initBeliefs), ...Object.keys(posteriorBeliefs)])).map((hypothesisId) => {
                        const prior = priorBeliefs[hypothesisId];
                        const post = posteriorBeliefs[hypothesisId];
                        return (
                          <div key={hypothesisId} className="bg-white p-2 rounded border border-[#D9DFDB] flex items-center justify-between text-2xs font-mono">
                            <span className="text-[#66706C]">{data.hypotheses[hypothesisId]?.title || hypothesisId}</span>
                            <div className="flex items-center gap-2">
                              <span className="text-[#8F9995]">{typeof prior === 'number' ? prior.toFixed(3) : 'N/A'}</span>
                              <span className="text-[#8F9995]">→</span>
                              <span className="text-[#66706C]">{typeof post === 'number' ? post.toFixed(3) : 'N/A'}</span>
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  )}
                </div>
              )}
            </div>
          </div>
        </section>

        {/* Lead Charts Container (50%) */}
        <section className="lg:col-span-6 sci-card p-6 flex flex-col justify-between space-y-4">
          <div>
            <div className="flex items-center justify-between border-b border-[#D9DFDB] pb-3">
              <div className="flex items-center gap-1.5">
                <TrendingUp className="w-4 h-4 text-[#DC2626]" />
                <h2 className="text-base font-bold text-[#17201F]">Scientific Trajectory</h2>
              </div>
              <div className="flex items-center gap-1 bg-[#F4F3EE] p-1 rounded-lg border border-[#D9DFDB]">
                {[
                  { id: 'trajectory', label: dataset === 'electrolyte_search' || dataset === 'anode_free_electrolyte_screening' ? 'Surrogate' : 'Belief' },
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
                  {dataset === 'electrolyte_search' || dataset === 'anode_free_electrolyte_screening' ? (
                    <div>
                      <ElectrolyteOptimizationChart simulationData={data.electrolyte_simulation} />
                      <p className="text-2xs text-[#8F9995] text-center mt-2 font-mono">
                         Source-recorded ensemble surrogate query trajectory; no live cycling is performed.
                      </p>
                    </div>
                  ) : (
                    <div>
                      <HypothesisBeliefTrajectoryChart
                        campaign={'campaign' in resolved ? resolved.campaign : null}
                        currentStepIndex={stepIndex}
                        revealPhase={revealPhase}
                        onSelectStep={handleStepSelect}
                      />
                      <p className="text-2xs text-[#8F9995] text-center mt-2 font-mono">
                        {dataset === 'alab_replay' || dataset === 'alab_precursor_genome'
                          ? 'A-Lab retrospective replay step progression across source-linked samples.'
                          : 'Bayesian posterior evolution across the recorded controlled trajectory; posterior weight is not physical confirmation.'}
                      </p>
                    </div>
                  )}
                </div>
              )}

              {primaryChartTab === 'predictive' && (
                <div>
                  {dataset === 'electrolyte_search' || dataset === 'anode_free_electrolyte_screening' ? (
                    <div className="p-8 rounded-xl bg-[#FCFCFA] border border-[#D9DFDB] text-center space-y-3">
                      <span className="sci-badge sci-badge-surrogate">Univariate Surrogate</span>
                      <h4 className="text-sm font-bold text-[#17201F]">Surrogate Point Estimations</h4>
                      <p className="text-xs text-[#66706C] max-w-sm mx-auto leading-relaxed">
                         The ensemble surrogate model outputs point predictions for the normalized capacity retention benchmark; no uncalibrated Bayesian fields are inferred.
                      </p>
                      <div className="p-3 rounded-lg bg-[#F4F3EE] border border-[#D9DFDB] text-2xs font-mono text-[#66706C] text-left space-y-1">
                        <div>• Full Virtual Space: {electrolyteInfo?.candidateCount ?? 'N/A'} candidate formulations</div>
                        <div>• Screened Working Set: {electrolyteInfo?.screenedWorkingSetCount ?? 'N/A'} candidates (source timing stage)</div>
                        <div>• Screening Latent Gap: {data.electrolyte_simulation?.screening_latent_gap ?? 'N/A'} (source artifact)</div>
                      </div>
                    </div>
                  ) : (
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
                    Expected Information Gain vs Experimental Cost. Selected action highlighted in emerald/red.
                  </p>
                </div>
              )}
            </div>
          </div>
        </section>
      </div>

      {/* Secondary Explorer Row: Heatmap + Score Waterfall */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Candidate-Modality Score Heatmap (67%) */}
        <section className="lg:col-span-8 sci-card p-6 space-y-4">
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

        {/* Score Decomposition Waterfall (33%) */}
        <section className="lg:col-span-4 sci-card p-6 space-y-4">
          <div className="border-b border-[#D9DFDB] pb-3">
            <h2 className="text-base font-bold text-[#17201F]">Recorded Decision Score</h2>
            <p className="text-xs text-[#66706C] mt-0.5">Signed additive terms for action {selectedCandidateId} / {selectedModality}</p>
          </div>

          <ScoreWaterfallChart action={inspectedAction} />

          <div className="p-3 rounded-lg bg-[#FCFCFA] border border-[#D9DFDB] text-2xs font-mono text-[#66706C] leading-relaxed">
            <strong className="text-[#17201F]">Score provenance:</strong> raw HIG, discovery utility, normalized cost, and total score are shown only when persisted by the source action record; no missing weighting is inferred.
          </div>
        </section>
      </div>

      {/* Tertiary Collapsible Drawer: Alternative Actions & Candidate-Space Hologram */}
      <details className="sci-card p-6 transition-all group">
        <summary className="font-bold text-sm text-[#17201F] cursor-pointer flex items-center justify-between select-none list-none">
          <div className="flex items-center gap-2">
            <Atom className="w-4 h-4 text-[#DC2626]" />
            <span>Advanced Inspection: Recorded Alternative Actions & Candidate-Space Hologram</span>
            <span className="sci-badge sci-badge-surrogate">Optional Deep-Dive</span>
          </div>
          <span className="text-xs font-mono text-[#DC2626] underline group-open:hidden">Expand inspection drawer</span>
          <span className="text-xs font-mono text-[#66706C] underline hidden group-open:inline">Collapse drawer</span>
        </summary>

        <div className="mt-6 pt-6 border-t border-[#D9DFDB] grid grid-cols-1 lg:grid-cols-12 gap-6">
          {/* Recorded Alternative Actions Table */}
          <div className="lg:col-span-7 space-y-3">
            <h3 className="text-sm font-bold text-[#17201F]">Recorded Alternative Actions</h3>
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
                  {([...allActions].sort((a, b) => (b.total_action_score ?? 0) - (a.total_action_score ?? 0))).slice(0, 4).map((a, idx) => {
                    const winningActionId = currentStep?.preregistration?.action?.action_id || winnerAction?.action?.action_id;
                    const isWinner = Boolean(winningActionId && a.action?.action_id === winningActionId);
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
                          {(a as any).rationale || 'No source rationale recorded'}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>

          {/* 3D Candidate-Space Hologram Sphere */}
          <div className="lg:col-span-5 space-y-3">
            <div className="flex items-center justify-between">
              <h3 className="text-sm font-bold text-[#17201F]">Candidate-Space Hologram</h3>
              <span className="text-2xs font-mono text-[#8F9995]">Candidate Geometry</span>
            </div>
            <div className="rounded-xl border border-[#D9DFDB] bg-[#17201F] overflow-hidden p-2">
              <StarkHologramSphere
                candidates={resolved.candidates}
                selectedCandidateId={selectedCandidateId}
                onSelectCandidate={(candId) => setSelectedCandidateId(candId)}
                currentStep={currentStep || undefined}
              />
            </div>
            <p className="text-2xs text-[#8F9995] text-center font-mono">
              Presentation-only candidate layout. Spatial position and distance do not represent crystallographic or chemical similarity.
            </p>
          </div>
        </div>
      </details>
    </div>
  );
};
