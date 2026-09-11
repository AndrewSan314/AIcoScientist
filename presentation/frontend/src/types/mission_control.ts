export type EvidenceMode =
  | 'CONTROLLED_SYNTHETIC'
  | 'HISTORICAL_REPLAY'
  | 'SIMULATED_SURROGATE'
  | 'LIVE_COMPUTED'
  | 'NOT_AVAILABLE';

export type DataMode = EvidenceMode;

export type WorkspaceTab = 'discovery' | 'benchmarks' | 'system';
export type LegacyNavTab = 'overview' | 'cockpit' | 'alab' | 'benchmarks' | 'electrolyte' | 'architecture' | 'readiness';
export type RevealPhase = 'A_SCORED' | 'B_PREREGISTERED' | 'C_REVEALED' | 'D_UPDATED';
export type CandidateViewMode = 'heatmap' | 'tradeoff' | 'hologram' | 'table';
export type HeatmapMetricMode = 'composite' | 'raw_hig' | 'norm_hig' | 'discovery' | 'cost';

export type DiscoveryFlowState = 'setup' | 'running' | 'results';

export type CanonicalDatasetId =
  | 'controlled_multimodal_alloy'
  | 'alab_precursor_genome'
  | 'anode_free_electrolyte_screening';

export type LegacyDatasetId = 'controlled_synthesis' | 'alab_replay' | 'electrolyte_search';

export type DatasetOption = CanonicalDatasetId | LegacyDatasetId;

export interface DatasetModalityInfo {
  cost?: number | null;
  diagnostic?: boolean;
  units?: string;
  available: boolean;
  reason?: string;
  linkedCandidateCount?: number;
  coverage?: number;
  linkageQuality?: string;
  missingness?: string;
  source?: string;
}

export interface DatasetProvenance {
  sourceType: string;
  sourcePaths: string[];
  citation: string;
  doi: string | null;
  license: string;
}

export interface DatasetCapabilities {
  modelHypothesesAvailable: boolean;
  posteriorModelWeightsAvailable: boolean;
  mutuallyExclusivePhysicalMechanismsClaimed: boolean;
  prospectiveMechanismIdentification: boolean;
  candidateScreening: boolean;
  preregistrationReplay: boolean;
  closedLoopExecution: boolean;
  surrogateSimulation: boolean;
  evidenceKind: EvidenceMode;
}

export type RecordedConfigurationMode =
  | 'CONTROLLED_SYNTHETIC'
  | 'HISTORICAL_REPLAY'
  | 'SIMULATED_SURROGATE';

export interface RecordedCampaignConfiguration {
  configurationId: string;
  runId?: string;
  world?: string;
  seed?: number;
  policy: string;
  mode: RecordedConfigurationMode;
  stepCount: number;
  candidateIds?: string[];
  modalities?: string[];
}

export interface ScientificDatasetRegistryEntry {
  id: CanonicalDatasetId;
  legacy_id: LegacyDatasetId;
  displayName: string;
  domain: string;
  provenance: DatasetProvenance;
  candidateCount: number;
  candidateIds: string[];
  featuredCandidateIds?: string[];
  availableConfigurations: RecordedCampaignConfiguration[];
  screenedWorkingSetCount?: number;
  targetObservable?: string;
  scientificTargetName?: string;
  targetObservableDescription?: string;
  modalities: Record<string, DatasetModalityInfo>;
  hypotheses: string[];
  defaultConfiguration: Record<string, any>;
  capabilities: DatasetCapabilities;
  summary: string;
  statusBadge: string;
  disclosures: string[];
  limitations?: string[];
}

export interface DatasetRegistry {
  registry_schema_version: string;
  datasets: ScientificDatasetRegistryEntry[];
}

export interface ResolvedViewShell {
  kind: 'controlled_multimodal' | 'historical_replay' | 'surrogate_optimization';
  datasetId: CanonicalDatasetId;
  displayName: string;
  domain: string;
  statusBadge: string;
  evidenceKind: EvidenceMode;
  selectedConfiguration: Record<string, unknown>;
  sourceManifest: {
    sourcePaths: string[];
    sourceArtifactHashes?: Record<string, string>;
    scientificSourceCommit?: string | null;
    scientificSourceCommitStatus?: string;
  };
  availableViews: string[];
  limitations: string[];
  banner: {
    title: string;
    badge: string;
    description: string;
    confidenceOrUtilityLabel: string;
    budgetExpended: number | string | null;
    budgetUnits: string;
  };
  capabilities: DatasetCapabilities;
  disclosures: string[];
}

export interface ControlledMultimodalView extends ResolvedViewShell {
  kind: 'controlled_multimodal';
  campaign: FlagshipCampaign;
  steps: CampaignStep[];
  totalSteps: number;
  candidates: Candidate[];
  hypotheses: string[];
  modalities: string[];
}

export interface HistoricalReplayView extends ResolvedViewShell {
  kind: 'historical_replay';
  campaign: ReplayCampaign;
  steps: CampaignStep[];
  totalSteps: number;
  candidates: Candidate[];
  hypotheses: string[];
  modalities: string[];
  replaySampleIds: string[];
  alabSamples: SampleItem[];
}

export interface SurrogateTrajectoryStep {
  queryIndex: number;
  candidateId: string;
  revealedNoisyValue?: number;
  selectedLatentValue?: number;
  bestSelectedLatentValue?: number;
}

export interface SurrogateOptimizationView extends ResolvedViewShell {
  kind: 'surrogate_optimization';
  scientificTargetName?: string;
  trajectory: SurrogateTrajectoryStep[];
  policy: string;
  seed: number;
  simulationRun: SurrogateRun;
  screeningDiagnostics: ElectrolyteScreeningData;
}

export type ResolvedResearchView =
  | ControlledMultimodalView
  | HistoricalReplayView
  | SurrogateOptimizationView;

export type ResolvedCampaignView = ResolvedResearchView;

export type ResolutionFailureReason =
  | 'UNKNOWN_DATASET'
  | 'UNKNOWN_POLICY'
  | 'UNKNOWN_SEED'
  | 'UNKNOWN_WORLD'
  | 'RUN_NOT_AVAILABLE'
  | 'SOURCE_ARTIFACT_MISSING'
  | 'INVALID_SOURCE_DATA'
  | 'UNSUPPORTED_CONFIGURATION';

export type ResolveResult<T> =
  | { ok: true; value: T }
  | { ok: false; reason: ResolutionFailureReason; message: string; availableOptions?: unknown };

export interface CampaignResolutionConfiguration {
  runId?: string;
  world?: string;
  seed?: number;
  policy?: string;
}

export interface PresenterSceneState {
  workspace: WorkspaceTab;
  discoveryFlowState?: DiscoveryFlowState;
  datasetOption?: DatasetOption;
  benchmarkQuestionId?: number;
  campaignStep?: number;
  revealPhase?: RevealPhase;
  subtab?: string;
}

export interface ScientificWorkspaceState {
  campaignKind: 'flagship_synthetic' | 'alab_replay';
  stepIndex: number;
  selectedCandidateId: string;
  selectedModality: string;
  recordedRecommendation: ScoredActionRecord | null;
  revealPhase: RevealPhase;
  activeMetric: HeatmapMetricMode;
  comparisonPolicy: string;
  candidateViewMode: CandidateViewMode;
  primaryChartMode: 'trajectory' | 'predictive' | 'tradeoff';
}

export interface ProvenanceInfo {
  head_commit: string;
  branch: string;
  total_ledger_events: number;
  scientific_source_commit?: string | null;
  scientific_source_commit_status?: 'VERIFIED' | 'UNVERIFIED';
  scientific_source_provenance_manifest?: string;
}

export interface HypothesisDefinition {
  hypothesis_id: string;
  title: string;
  assumptions: string[];
  predicted_observables: Record<string, string[]>;
  falsification_signature: {
    strongly_supporting_patterns: string[];
    strongly_falsifying_patterns: string[];
    ambiguous_patterns: string[];
  };
  training_count?: number;
}

export interface Candidate {
  candidate_id: string;
  composition_label: string;
  target_system?: string;
  target_formula?: string | null;
  target_stoichiometry?: string | null;
  precursors?: string[];
  heating_temperature_c?: number | null;
  heating_time_minutes?: number | null;
  reaction_energy_ev_per_atom?: number | null;
  reaction_category?: string | null;
  xrd_available?: boolean;
  refinement_available?: boolean;
  outcome_available?: boolean;
  refinement_rwp?: number | null;
  refinement_phases?: Array<{ name: string; weight_percent?: number | null }>;
  refinement_cases?: RefinementCase[];
  selected_refinement_case_id?: string | null;
  refinement_selection_rule?: string | null;
  source_record_identifier?: string | null;
  status?: 'unobserved' | 'characterized' | 'outcome_tested' | 'selected';
}

export interface DisplayLayout {
  kind: 'PRESENTATION_ONLY';
  algorithm: 'GRID' | 'FIBONACCI_SPHERE' | 'INDEX_LAYOUT';
  x: number;
  y: number;
  z?: number;
}

export interface ActionMetadata {
  cost_units: string;
  prerequisites: string[];
}

export interface ScientificActionItem {
  action_id: string;
  candidate_id: string;
  action_type: string;
  estimated_cost: number;
  requested_at_step: number;
  metadata: ActionMetadata;
}

export interface HigDiagnostics {
  raw_hig_mc_nats: number;
  clipped_hig_nats: number;
  current_entropy_nats: number;
  posterior_entropy_mc_mean: number;
  posterior_entropy_mc_std?: number;
  hig_mc_standard_error?: number;
  mc_samples: number;
  hig_bound_k?: number;
  hig_numeric_epsilon_nats?: number;
  hig_bound_epsilon_nats?: number;
  raw_hig_lower_bound_ok?: boolean;
  raw_hig_upper_bound_ok?: boolean;
  predictive_variance_by_hypothesis?: Record<string, number[]>;
}

export interface ScoredActionRecord {
  event: string;
  event_sequence: number;
  step: number;
  timestamp: string;
  action: ScientificActionItem;
  expected_hig_nats: number;
  discovery_utility: number;
  normalized_cost: number;
  total_action_score: number;
  policy_name: string;
  current_hypothesis_entropy_nats: number;
  dominant_hypothesis_disagreement?: string[];
  hig_diagnostics?: HigDiagnostics;
  run_id?: string;
  // Exact decomposition and extrema fields
  raw_expected_hig_nats?: number;
  normalized_hig?: number;
  raw_discovery_utility?: number;
  normalized_discovery?: number;
  raw_estimated_cost?: number;
  w_hig?: number;
  w_discovery?: number;
  w_cost?: number;
  weighted_hig_contribution?: number;
  weighted_discovery_contribution?: number;
  weighted_cost_contribution?: number;
  step_max_hig?: number;
  step_max_discovery?: number;
  step_max_cost?: number;
  predictive_distribution_available?: boolean;
  predictive_distribution_unavailability_reason?: string;
}

export type HeatmapCellStatus = 'FEASIBLE_SCORED' | 'FEASIBLE_ZERO' | 'INFEASIBLE' | 'UNAVAILABLE';

export interface PreregisteredActionRecord {
  event: string;
  step: number;
  event_sequence: number;
  timestamp: string;
  measurement_revealed: boolean;
  action: ScientificActionItem;
  beliefs_before: Record<string, number>;
  predictive_distributions: Record<string, {
    hypothesis_id: string;
    candidate_id: string;
    modality: string;
    mean?: number[];
    variance?: number[];
    observable_names?: string[];
    distribution_kind?: string;
    categories?: string[];
    probabilities?: number[];
  }>;
  expected_hig_nats: number;
  discovery_utility: number;
  normalized_cost: number;
  total_action_score: number;
  current_hypothesis_entropy_nats: number;
  hig_diagnostics?: HigDiagnostics;
  falsification_signatures?: Record<string, any>;
}

export interface ObservedMeasurement {
  observable_id: string;
  candidate_id: string;
  modality: string;
  name: string;
  value: number | number[] | string;
  uncertainty?: number | number[] | null;
  units?: string | null;
  timestamp?: string | null;
  observable_type?: string;
  observable_names?: string[];
  raw_artifact_ref?: string | null;
  extractor_name?: string | null;
  provenance?: Record<string, any>;
}

export interface MeasurementRevealedRecord {
  event: string;
  step: number;
  event_sequence: number;
  timestamp: string;
  action: ScientificActionItem;
  observed_measurement: ObservedMeasurement;
  likelihood_under_hypothesis: Record<string, number>;
  log_bayes_factor_pairwise: Record<string, number>;
  beliefs_before: Record<string, number>;
  beliefs_after: Record<string, number>;
  posterior_delta: Record<string, number>;
  realized_entropy_reduction_nats: number;
}

export interface BeliefUpdateRecord {
  event: string;
  step: number;
  event_sequence: number;
  timestamp: string;
  action: ScientificActionItem;
  beliefs_before: Record<string, number>;
  beliefs_after: Record<string, number>;
  likelihood_under_hypothesis: Record<string, number>;
}

export interface CampaignStep {
  step: number;
  tested_candidates_before?: string[];
  preregistration: PreregisteredActionRecord | null;
  observation: MeasurementRevealedRecord | null;
  belief_update: BeliefUpdateRecord | null;
  all_scored_actions?: ScoredActionRecord[];
  top_actions: ScoredActionRecord[];
  total_actions_evaluated: number;
  step_max_hig?: number;
  step_max_discovery?: number;
  step_max_cost?: number;
}

export interface FlagshipCampaign {
  run_id: string;
  world: string;
  seed: number;
  policy: string;
  policy_weights?: {
    w_hig: number;
    w_discovery: number;
    w_cost: number;
  };
  initial_beliefs: Record<string, number>;
  candidates: Candidate[];
  steps: CampaignStep[];
}

export interface ReplayCampaign {
  run_id: string;
  policy: string;
  seed: number;
  mode: string;
  initial_beliefs?: Record<string, number>;
  replay_candidate_ids?: string[];
  steps: CampaignStep[];
}

export interface ObservableCalibrationMetric {
  MAE: number;
  RMSE: number;
  NLL: number;
  coverage50: number;
  coverage90: number;
  calibration_error: number;
  NRMSE?: number;
  normalization?: {
    method: string;
    scale: number;
  };
}

export interface CalibrationData {
  acceptance_thresholds: {
    coverage50_abs_error_max: number;
    coverage90_abs_error_max: number;
  };
  XRD: Record<string, ObservableCalibrationMetric>;
  REFINEMENT: Record<string, ObservableCalibrationMetric>;
  interpretation_annotations?: Record<string, any>;
}

export interface ValidationGateInfo {
  status: string;
  scientific_methodology_status: string;
  release_readiness: string;
  readiness: Record<string, string>;
  gates: Record<string, string>;
  gate_evidence: {
    ledger_event_count: number;
    boolean_gate_count: number;
    boolean_gate_pass_count: number;
    hig_raw_bound_violation_count: number;
    controlled_worlds_clean: string[];
    controlled_worlds_stress: string[];
    required_seeds: number[];
    unsupported_retrospective_modalities: string[];
  };
}

export interface BenchmarkSummaryData {
  status: string;
  trajectory_count: number;
  world_types: string[];
  worlds: string[];
  policies: string[];
  seeds: number[];
  design: Record<string, any>;
  summary_by_world_policy: Record<string, Record<string, any>>;
  policy_validation?: Record<string, any>;
}

export interface SensitivityData {
  status: string;
  trajectory_count: number;
  design: Record<string, any>;
  aggregate_by_world_policy: Record<string, any>;
}

export interface ElectrolyteScreeningData {
  search_space_size: number;
  full_search_space_latent_max: number;
  evidence_mode: string;
  historical_observation_count: number;
  screening_method: string;
  chosen_default_working_set_size: number;
  working_set_trials: Record<string, any>;
  reference_comparison?: Record<string, any>;
}

export interface ElectrolyteSimulationData {
  status?: string;
  search_space_size?: number;
  working_set_size?: number;
  oracle_kind?: string;
  policies?: Record<string, any>;
  comparison_table?: any[];
  detailed_policy_seed_runs?: Record<string, SurrogateRun[]>;
  simulation_policies?: Record<string, Record<string, number>>;
  evaluated_seeds?: number[];
}

export interface SurrogateRun {
  seed: number;
  queried_candidate_ids: string[];
  revealed_noisy_values: number[];
  selected_latent_values: number[];
  best_latent_curve: number[];
  best_selected_latent_capacity?: number;
  best_noisy_observed_capacity?: number;
  simple_regret_latent?: number;
  simple_regret_vs_full_latent?: number;
  cumulative_raw_hig_nats?: number;
  realized_entropy_reduction?: number;
  queried_count?: number;
  regret_vs_oracle_max?: number;
  [key: string]: unknown;
}

export interface SampleItem {
  sample_id: string;
  target_formula: string;
  target_stoichiometry?: string | null;
  precursors: string[];
  heating_temperature_c?: number | null;
  heating_time_minutes?: number | null;
  heating_time_hours?: number | null;
  reaction_energy_ev_per_atom?: number | null;
  reaction_category?: string | null;
  xrd_available: boolean;
  refinement_available: boolean;
  outcome_available: boolean;
  sem_available: boolean;
  eds_available: boolean;
  sem_availability_reason?: string;
  eds_availability_reason?: string;
  refinement_rwp?: number | null;
  refinement_phases?: Array<{ name: string; weight_percent?: number | null }>;
  refinement_cases?: RefinementCase[];
  selected_refinement_case_id?: string | null;
  refinement_selection_rule?: string | null;
  canonical_descriptors?: Record<string, number>;
  refinement_observables?: Record<string, number>;
  source_archive: string;
  source_record_identifier?: string;
  extractor_name?: string;
  extractor_version?: string;
  extractor_provenance?: string;
}

export interface RefinementCase {
  scan_id: string;
  case_id: string;
  rwp?: number | null;
  phases?: Record<string, number> | null;
  source_index: number;
  scan_index: number;
}

export interface CoreAbstraction {
  name: string;
  role: string;
  file: string;
}

export interface DomainItem {
  domain_id: string;
  name: string;
  status: string;
  candidates_count: number;
  modalities: string[];
  purpose: string;
}

export interface ArchitectureData {
  core_abstractions: CoreAbstraction[];
  domains: DomainItem[];
}

export interface SnapshotManifest {
  snapshot_schema_version: string;
  generated_at_utc: string;
  presentation_build_commit: string;
  snapshot_generator_commit: string;
  scientific_source_commit: string | null;
  scientific_source_commit_status: 'VERIFIED' | 'UNVERIFIED';
  scientific_source_provenance_manifest: string;
  source_branch: string;
  source_artifact_hashes: Record<string, string>;
  source_dataset_manifest_hash: string;
  campaign_run_id: string;
  campaign_world: string;
  campaign_seed: number;
  campaign_policy: string;
  campaign_step_count: number;
  total_real_samples: number;
  total_audit_events: number;
  validation_gate_pass_count: number;
  validation_gate_total_count: number;
  transformation_version?: string;
}

export interface SnapshotData {
  version: string;
  generated_at: string;
  provenance: ProvenanceInfo;
  manifest?: SnapshotManifest;
  dataset_registry?: DatasetRegistry;
  flagship_campaign: FlagshipCampaign;
  alab_replay_campaign: ReplayCampaign;
  hypotheses: Record<string, HypothesisDefinition>;
  validation: ValidationGateInfo;
  calibration: CalibrationData;
  benchmarks: BenchmarkSummaryData;
  sensitivity: SensitivityData;
  electrolyte_screening: ElectrolyteScreeningData;
  electrolyte_simulation: any;
  campaign_runs?: Array<FlagshipCampaign | ReplayCampaign>;
  alab_audit: any;
  modality_inventory: any;
  samples: SampleItem[];
  architecture: ArchitectureData;
  ledger_sample_events: any[];
}
