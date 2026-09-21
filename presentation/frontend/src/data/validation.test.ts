import { describe, it, expect } from 'vitest';
import { DRAKOPOULOS_SCENARIO, WARWICK_SCENARIO } from './exhibitionData';
import { validateScenario, validateAllScenarios } from './validation';

describe('Exhibition Data Layer Integrity', () => {
  it('validates Drakopoulos Graphite scenario without errors', () => {
    const res = validateScenario(DRAKOPOULOS_SCENARIO);
    expect(res.errors).toEqual([]);
    expect(res.valid).toBe(true);
  });

  it('validates Warwick NMC622 scenario without errors', () => {
    const res = validateScenario(WARWICK_SCENARIO);
    expect(res.errors).toEqual([]);
    expect(res.valid).toBe(true);
  });

  it('validates all exhibition scenarios in registry', () => {
    const res = validateAllScenarios({
      drakopoulos_graphite: DRAKOPOULOS_SCENARIO,
      warwick_nmc622_calendering: WARWICK_SCENARIO
    });
    expect(res.errors).toEqual([]);
    expect(res.valid).toBe(true);
  });

  it('rejects cross-dataset candidate contamination', () => {
    const contaminatedScenario = {
      ...DRAKOPOULOS_SCENARIO,
      candidates: [
        ...DRAKOPOULOS_SCENARIO.candidates,
        {
          id: 'EXP_99_LEAKED',
          displayId: 'Leaked Warwick Condition',
          controls: {},
          revealedTarget: { name: 'D30', value: 300, unit: 'mAh/g' },
          replicateCount: 1,
          cellIds: ['FAKE']
        }
      ]
    };
    const res = validateScenario(contaminatedScenario);
    expect(res.valid).toBe(false);
    expect(res.errors.some((e) => e.includes('leaked into Drakopoulos'))).toBe(true);
  });

  it('rejects unit mismatch', () => {
    const badUnitScenario = {
      ...WARWICK_SCENARIO,
      targetUnit: 'mAh/g' // Warwick must be dimensionless_ratio!
    };
    const res = validateScenario(badUnitScenario);
    expect(res.valid).toBe(false);
    expect(res.errors.some((e) => e.includes('dimensionless_ratio'))).toBe(true);
  });

  it('enforces monotonic best-so-far in optimization replay', () => {
    const badReplayScenario = {
      ...DRAKOPOULOS_SCENARIO,
      replaySteps: DRAKOPOULOS_SCENARIO.replaySteps.map((step, idx) => 
        idx === 2 ? { ...step, bestSoFar: 10.0 } : step
      )
    };
    const res = validateScenario(badReplayScenario);
    expect(res.valid).toBe(false);
    expect(res.errors.some((e) => e.includes('bestSoFar decreased'))).toBe(true);
  });
});
