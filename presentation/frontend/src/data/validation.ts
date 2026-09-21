import { ExhibitionScenario, ScenarioId } from './types';

export interface ValidationResult {
  valid: boolean;
  errors: string[];
  warnings: string[];
}

export function validateScenario(scenario: ExhibitionScenario): ValidationResult {
  const errors: string[] = [];
  const warnings: string[] = [];

  // 1. Scenario ID & Basic Metadata
  if (!scenario.id) {
    errors.push('Scenario ID is required.');
  }

  // 2. Units Consistency
  if (scenario.id === 'drakopoulos_graphite' && scenario.targetUnit !== 'mAh/g') {
    errors.push(`Drakopoulos target unit must be mAh/g, got: ${scenario.targetUnit}`);
  }
  if (scenario.id === 'warwick_nmc622_calendering' && scenario.targetUnit !== 'dimensionless_ratio') {
    errors.push(`Warwick target unit must be dimensionless_ratio, got: ${scenario.targetUnit}`);
  }

  // 3. Candidate Pool Integrity
  const candidateIds = new Set<string>();
  scenario.candidates.forEach((c) => {
    if (candidateIds.has(c.id)) {
      errors.push(`Duplicate candidate ID found: ${c.id}`);
    }
    candidateIds.add(c.id);

    if (!Number.isFinite(c.revealedTarget.value)) {
      errors.push(`Candidate ${c.id} has non-finite revealed target value: ${c.revealedTarget.value}`);
    }

    if (c.revealedTarget.unit !== scenario.targetUnit) {
      errors.push(`Candidate ${c.id} target unit ${c.revealedTarget.unit} does not match scenario ${scenario.targetUnit}`);
    }
  });

  // 4. Initial Replay Candidates
  scenario.replayInitialIds.forEach((initId) => {
    if (!candidateIds.has(initId)) {
      errors.push(`Initial design candidate ${initId} is not in the candidate pool.`);
    }
  });

  // 5. Replay Trajectory Step Validation
  let prevBestSoFar = -Infinity;
  const acquiredCandidates = new Set<string>(scenario.replayInitialIds);

  scenario.replaySteps.forEach((step, idx) => {
    if (step.step !== idx + 1) {
      errors.push(`Replay step sequence error: expected ${idx + 1}, got ${step.step}`);
    }

    if (!candidateIds.has(step.selectedCandidateId)) {
      errors.push(`Step ${step.step} selected unknown candidate: ${step.selectedCandidateId}`);
    }

    if (!Number.isFinite(step.predictedMean) || !Number.isFinite(step.predictedStd)) {
      errors.push(`Step ${step.step} has non-finite prediction statistics.`);
    }

    if (!Number.isFinite(step.revealedTarget) || !Number.isFinite(step.bestSoFar)) {
      errors.push(`Step ${step.step} has non-finite revealed/best-so-far values.`);
    }

    // Monotonic best-so-far invariant for maximization
    if (step.bestSoFar < prevBestSoFar - 1e-6) {
      errors.push(`Step ${step.step} bestSoFar decreased from ${prevBestSoFar} to ${step.bestSoFar}`);
    }
    prevBestSoFar = step.bestSoFar;

    // Duplicate acquisition guard
    if (acquiredCandidates.has(step.selectedCandidateId)) {
      errors.push(`Candidate ${step.selectedCandidateId} was re-selected at step ${step.step}`);
    }
    acquiredCandidates.add(step.selectedCandidateId);
  });

  // 6. Microstructure Bounds
  const ms = scenario.microstructure;
  if (ms.initialThicknessUm <= 0 || ms.calenderedThicknessUm <= 0) {
    errors.push('Microstructure thickness values must be strictly positive.');
  }
  if (ms.calenderedThicknessUm > ms.initialThicknessUm) {
    errors.push('Calendered thickness cannot exceed uncalendered thickness.');
  }
  if (ms.initialPorosityPct <= 0 || ms.initialPorosityPct >= 100 || ms.calenderedPorosityPct <= 0 || ms.calenderedPorosityPct >= 100) {
    errors.push('Porosity percentages must be between 0 and 100%.');
  }

  // 7. Scenario Isolation Guardrail
  if (scenario.id === 'drakopoulos_graphite') {
    scenario.candidates.forEach((c) => {
      if (c.id.startsWith('EXP_')) {
        errors.push(`Warwick condition ${c.id} leaked into Drakopoulos scenario!`);
      }
    });
  } else if (scenario.id === 'warwick_nmc622_calendering') {
    scenario.candidates.forEach((c) => {
      if (c.id.startsWith('protocol-')) {
        errors.push(`Drakopoulos recipe ${c.id} leaked into Warwick scenario!`);
      }
    });
  }

  return {
    valid: errors.length === 0,
    errors,
    warnings
  };
}

export function validateAllScenarios(scenarios: Record<ScenarioId, ExhibitionScenario>): ValidationResult {
  const allErrors: string[] = [];
  const allWarnings: string[] = [];

  Object.values(scenarios).forEach((sc) => {
    const res = validateScenario(sc);
    if (!res.valid) {
      allErrors.push(...res.errors.map((e) => `[${sc.id}] ${e}`));
    }
    allWarnings.push(...res.warnings.map((w) => `[${sc.id}] ${w}`));
  });

  return {
    valid: allErrors.length === 0,
    errors: allErrors,
    warnings: allWarnings
  };
}
