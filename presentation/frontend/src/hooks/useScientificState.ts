import { useState, useMemo, useEffect, useCallback } from 'react';
import {
  SnapshotData,
  ScientificWorkspaceState,
  RevealPhase,
  CandidateViewMode,
  HeatmapMetricMode,
  CampaignStep,
  ScoredActionRecord
} from '../types/mission_control';

export interface UseScientificStateReturn {
  state: ScientificWorkspaceState;
  currentStep: CampaignStep | null;
  totalSteps: number;
  winnerAction: ScoredActionRecord | null;
  currentCandidatePool: ScoredActionRecord[];
  activeObservableNames: string[];
  setStepIndex: (step: number) => void;
  setSelectedCandidate: (candidateId: string) => void;
  setSelectedModality: (modality: string) => void;
  setRevealPhase: (phase: RevealPhase) => void;
  setActiveMetric: (metric: HeatmapMetricMode) => void;
  setCandidateViewMode: (mode: CandidateViewMode) => void;
  setPrimaryChartMode: (mode: 'trajectory' | 'predictive' | 'tradeoff') => void;
  advanceRevealPhase: () => void;
  resetWorkflow: () => void;
}

export const useScientificState = (
  data: SnapshotData,
  controlledStepIndex?: number
): UseScientificStateReturn => {
  const steps = useMemo(() => {
    return data.flagship_campaign?.steps || [];
  }, [data]);

  const totalSteps = steps.length;

  const [stepIndex, setStepIndexState] = useState<number>(1);
  const [selectedCandidateId, setSelectedCandidateId] = useState<string>(
    steps[0]?.preregistration?.action?.candidate_id || ''
  );
  const [selectedModality, setSelectedModality] = useState<string>(
    steps[0]?.preregistration?.action?.action_type || ''
  );
  const [revealPhase, setRevealPhase] = useState<RevealPhase>('A_SCORED');
  const [activeMetric, setActiveMetric] = useState<HeatmapMetricMode>('composite');
  const [candidateViewMode, setCandidateViewMode] = useState<CandidateViewMode>('heatmap');
  const [primaryChartMode, setPrimaryChartMode] = useState<'trajectory' | 'predictive' | 'tradeoff'>('trajectory');
  const [comparisonPolicy] = useState<string>('PURE_HIG');

  // Sync with external controlledStepIndex (e.g. from Presenter Mode)
  useEffect(() => {
    if (controlledStepIndex !== undefined && controlledStepIndex >= 1 && controlledStepIndex <= totalSteps) {
      setStepIndexState(controlledStepIndex);
    }
  }, [controlledStepIndex, totalSteps]);

  // Current active campaign step
  const currentStep = useMemo(() => {
    if (totalSteps === 0) return null;
    const clamped = Math.max(1, Math.min(stepIndex, totalSteps));
    return steps[clamped - 1] || null;
  }, [steps, stepIndex, totalSteps]);

  // Current recorded recommendation (winner)
  const winnerAction = useMemo(() => {
    if (!currentStep) return null;
    return currentStep.top_actions?.[0] || null;
  }, [currentStep]);

  // Candidate pool (all 24 actions for current step)
  const currentCandidatePool = useMemo(() => {
    if (!currentStep) return [];
    return currentStep.all_scored_actions || currentStep.top_actions || [];
  }, [currentStep]);

  // Automatically update candidate and modality when step changes to match the winner action
  useEffect(() => {
    if (winnerAction?.action) {
      setSelectedCandidateId(winnerAction.action.candidate_id);
      setSelectedModality(winnerAction.action.action_type);
      setRevealPhase('A_SCORED');
    }
  }, [winnerAction]);

  // Active observable names for the current modality
  const activeObservableNames = useMemo(() => {
    if (!currentStep) return [];
    const dists = currentStep.preregistration?.predictive_distributions;
    if (dists) {
      for (const hid of Object.keys(dists)) {
        if (dists[hid]?.observable_names?.length) {
          return dists[hid].observable_names || [];
        }
      }
    }
    const obs = currentStep.observation?.observed_measurement?.observable_names;
    if (obs?.length) return obs;
    return [];
  }, [currentStep, selectedModality]);

  const setStepIndex = useCallback((s: number) => {
    setStepIndexState(Math.max(1, Math.min(s, totalSteps)));
  }, [totalSteps]);

  const setSelectedCandidate = useCallback((cId: string) => {
    setSelectedCandidateId(cId);
  }, []);

  const setSelectedModalityHandler = useCallback((mod: string) => {
    setSelectedModality(mod);
  }, []);

  const advanceRevealPhase = useCallback(() => {
    setRevealPhase((prev) => {
      if (prev === 'A_SCORED') return 'B_PREREGISTERED';
      if (prev === 'B_PREREGISTERED') return 'C_REVEALED';
      if (prev === 'C_REVEALED') return 'D_UPDATED';
      return 'A_SCORED';
    });
  }, []);

  const resetWorkflow = useCallback(() => {
    setRevealPhase('A_SCORED');
  }, []);

  const state: ScientificWorkspaceState = {
    campaignKind: 'flagship_synthetic',
    stepIndex,
    selectedCandidateId,
    selectedModality,
    recordedRecommendation: winnerAction,
    revealPhase,
    activeMetric,
    comparisonPolicy,
    candidateViewMode,
    primaryChartMode
  };

  return {
    state,
    currentStep,
    totalSteps,
    winnerAction,
    currentCandidatePool,
    activeObservableNames,
    setStepIndex,
    setSelectedCandidate,
    setSelectedModality: setSelectedModalityHandler,
    setRevealPhase,
    setActiveMetric,
    setCandidateViewMode,
    setPrimaryChartMode,
    advanceRevealPhase,
    resetWorkflow
  };
};
