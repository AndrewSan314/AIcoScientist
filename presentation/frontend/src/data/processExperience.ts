import type { ExhibitionScenario, ReplayStep, ScenarioId } from './types';

export type ProcessExperiencePhase =
  | 'OVERVIEW'
  | 'PROCESS_ENTERING'
  | 'PROCESS_EXPLORE'
  | 'AI_DECISION'
  | 'ILLUSTRATED_PROCESS_RUN'
  | 'RESULT_REVEAL'
  | 'PROCESS_EXIT';

const DECISION_STAGE: Record<ScenarioId, string> = {
  drakopoulos_graphite: 'coating',
  warwick_nmc622_calendering: 'calendering',
};

export const isDecisionStage = (scenario: ExhibitionScenario, stageId: string) =>
  DECISION_STAGE[scenario.id] === stageId;

export const pendingDecision = (scenario: ExhibitionScenario, revealedSteps: number): ReplayStep | undefined =>
  scenario.replaySteps[revealedSteps];

export const visibleDecision = (
  scenario: ExhibitionScenario,
  revealedSteps: number,
  phase: ProcessExperiencePhase,
): ReplayStep | undefined => phase === 'RESULT_REVEAL'
  ? scenario.replaySteps[Math.max(0, revealedSteps - 1)]
  : pendingDecision(scenario, revealedSteps);

export const revealNextDecision = (scenario: ExhibitionScenario, revealedSteps: number) =>
  Math.min(revealedSteps + 1, scenario.replaySteps.length);
