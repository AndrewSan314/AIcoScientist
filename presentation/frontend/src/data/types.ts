/**
 * TypeScript Data Models for AIcoScientist Exhibition Experience
 * 
 * Enforces strict scientific provenance, non-leakage information horizons,
 * and deterministic scenario isolation.
 */

export type ScenarioId = 'drakopoulos_graphite' | 'warwick_nmc622_calendering';

export type EvidenceKind = 
  | 'PHYSICAL_HISTORICAL'
  | 'PILOT_LINE_HISTORICAL'
  | 'OFFLINE_REPLAY'
  | 'SIMULATED_PHYSICS';

export interface ProcessStageInfo {
  id: string;
  order: number;
  label: string;
  shortDescription: string;
  detailedDescription: string;
  isOptimizationTarget: boolean;
  controllableParameters: Array<{
    name: string;
    label: string;
    unit: string;
    typicalRange: string;
  }>;
  measuredProperties: Array<{
    name: string;
    label: string;
    unit: string;
  }>;
}

export interface CandidateRecord {
  id: string;
  displayId: string;
  controls: Record<string, number | string | boolean>;
  prediction?: {
    mean: number;
    std: number;
    unit: string;
  };
  revealedTarget: {
    name: string;
    value: number;
    unit: string;
  };
  metrology?: {
    preCalenderingThicknessUm?: number;
    calenderedThicknessUm?: number;
    calenderedDensityGCm3?: number;
    calenderedPorosityPct?: number;
    tensileStrengthKPa?: number;
    activeMassMg?: number;
  };
  replicateCount: number;
  cellIds: string[];
}

export interface ReplayStep {
  step: number;
  selectedCandidateId: string;
  selectedCandidateDisplay: string;
  controlsSummary: Record<string, string | number | boolean>;
  predictedMean: number;
  predictedStd: number;
  acquisitionValue: number;
  revealedTarget: number;
  bestSoFar: number;
  simpleRegret: number;
  isOptimal: boolean;
  explanation: string;
  intermediateMetrology?: {
    density?: string;
    porosity?: string;
    thickness?: string;
  };
}

export interface BenchmarkComparison {
  policyId: string;
  policyName: string;
  hitAt1Pct: number;
  hitAt3Pct: number;
  hitAt5Pct: number;
  randomBaselineHitAt5Pct: number;
  simpleRegretAt5: number;
  seedsEvaluated: number;
  supportedClaim: string;
  allowedWording: string;
  knownLimitations: string[];
}

export interface MicrostructureSpec {
  chemistryLabel: string;
  substrateFoilName: string;
  substrateColorHex: string;
  activeParticleType: string;
  particleMorphology: 'FLAKES_OBLATE' | 'POLYCRYSTALLINE_SPHERICAL';
  particleColorHex: string;
  binderColorHex: string;
  initialThicknessUm: number;
  calenderedThicknessUm: number;
  initialPorosityPct: number;
  calenderedPorosityPct: number;
  compressionRatio: number;
  scientificInsight: string;
}

export interface ProvenanceDetails {
  authors: string;
  title: string;
  journal: string;
  year: number;
  doi: string;
  license: string;
  facility: string;
  replicateSummary: string;
}

export interface ExhibitionScenario {
  id: ScenarioId;
  title: string;
  subtitle: string;
  targetDescription: string;
  targetUnit: string;
  evidenceKind: EvidenceKind;
  provenance: ProvenanceDetails;
  stages: ProcessStageInfo[];
  candidates: CandidateRecord[];
  replayInitialIds: string[];
  replaySteps: ReplayStep[];
  benchmark: BenchmarkComparison;
  microstructure: MicrostructureSpec;
}
