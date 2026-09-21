import { describe, expect, it } from 'vitest';
import { DRAKOPOULOS_SCENARIO, WARWICK_SCENARIO } from './exhibitionData';
import { isDecisionStage, pendingDecision, revealNextDecision, visibleDecision } from './processExperience';

describe('process-native replay state', () => {
  it('keeps proposals hidden until the process run reveals the exact record', () => {
    const proposal = pendingDecision(WARWICK_SCENARIO, 0)!;
    expect(proposal.selectedCandidateDisplay).toBe('EXP_07');
    expect(visibleDecision(WARWICK_SCENARIO, 0, 'AI_DECISION')?.revealedTarget).toBe(proposal.revealedTarget);
    const revealed = revealNextDecision(WARWICK_SCENARIO, 0);
    expect(revealed).toBe(1);
    expect(visibleDecision(WARWICK_SCENARIO, revealed, 'RESULT_REVEAL')).toEqual(proposal);
  });

  it('synchronizes navigation by using one revealed-step counter', () => {
    expect(pendingDecision(DRAKOPOULOS_SCENARIO, 3)?.selectedCandidateDisplay).toBe('Recipe c1c280');
    expect(revealNextDecision(DRAKOPOULOS_SCENARIO, 5)).toBe(5);
  });

  it('isolates the two process decision contexts and source values', () => {
    expect(isDecisionStage(WARWICK_SCENARIO, 'calendering')).toBe(true);
    expect(isDecisionStage(WARWICK_SCENARIO, 'coating')).toBe(false);
    expect(isDecisionStage(DRAKOPOULOS_SCENARIO, 'coating')).toBe(true);
    expect(pendingDecision(WARWICK_SCENARIO, 4)?.revealedTarget).toBeCloseTo(0.79473999128743);
    expect(pendingDecision(DRAKOPOULOS_SCENARIO, 3)?.revealedTarget).toBeCloseTo(402.249050156032);
  });
});
