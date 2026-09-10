import type {
  CampaignResolutionConfiguration,
  CampaignStep,
  Candidate,
  DatasetOption,
  EvidenceMode,
  HistoricalReplayView,
  ResolvedCampaignView,
  ResolveResult,
  SnapshotData,
  SurrogateOptimizationView,
  SurrogateRun,
  ControlledMultimodalView,
  FlagshipCampaign,
  ReplayCampaign,
  ScientificDatasetRegistryEntry,
  ResolvedViewShell,
  SampleItem,
} from '../types/mission_control';

const POLICY_ALIASES: Record<string, string> = {
  hig_cost_penalized: 'HYBRID',
  greedy_hig: 'PURE_HIG',
  discovery_only: 'DISCOVERY_ONLY',
  random_baseline: 'RANDOM_ACTION',
};

const SURROGATE_POLICY_ALIASES: Record<string, string> = {
  hig_cost_penalized: 'HYBRID_DEFAULT',
  greedy_hig: 'PURE_FALSIFICATION',
  random_baseline: 'RANDOM',
};

export function normalizeDatasetId(option?: DatasetOption | string):
  | 'controlled_multimodal_alloy'
  | 'alab_precursor_genome'
  | 'anode_free_electrolyte_screening'
  | null {
  if (option === 'controlled_synthesis' || option === 'controlled_multimodal_alloy') {
    return 'controlled_multimodal_alloy';
  }
  if (option === 'alab_replay' || option === 'alab_precursor_genome') {
    return 'alab_precursor_genome';
  }
  if (option === 'electrolyte_search' || option === 'anode_free_electrolyte_screening') {
    return 'anode_free_electrolyte_screening';
  }
  return null;
}

function failure<T>(
  reason: Extract<ResolveResult<T>, { ok: false }>['reason'],
  message: string,
  availableOptions?: unknown,
): ResolveResult<T> {
  return { ok: false, reason, message, availableOptions };
}

function registryEntry(snapshot: SnapshotData, datasetId: string):
  | { ok: true; value: ScientificDatasetRegistryEntry }
  | { ok: false; result: ResolveResult<ResolvedCampaignView> } {
  const entry = snapshot.dataset_registry?.datasets.find((dataset) => dataset.id === datasetId);
  if (!entry) {
    return {
      ok: false,
      result: failure('SOURCE_ARTIFACT_MISSING', `Dataset registry entry is missing for ${datasetId}.`),
    };
  }
  return { ok: true, value: entry };
}

function sourceManifest(snapshot: SnapshotData, entry: ScientificDatasetRegistryEntry) {
  return {
    sourcePaths: entry.provenance.sourcePaths,
    sourceArtifactHashes: snapshot.manifest?.source_artifact_hashes,
    scientificSourceCommit: snapshot.manifest?.scientific_source_commit,
    scientificSourceCommitStatus: snapshot.manifest?.scientific_source_commit_status,
  };
}

function availableAlternatives(entry: ScientificDatasetRegistryEntry, mode: string) {
  const configurations = entry.availableConfigurations.filter((configuration) => configuration.mode === mode);
  return {
    configurations,
    policies: [...new Set(configurations.map((configuration) => configuration.policy))].sort(),
    seeds: [...new Set(configurations.map((configuration) => configuration.seed).filter((seed): seed is number => seed !== undefined))].sort((a, b) => a - b),
    worlds: [...new Set(configurations.map((configuration) => configuration.world).filter((world): world is string => Boolean(world)))].sort(),
  };
}

function candidatePool(snapshot: SnapshotData, run: FlagshipCampaign | ReplayCampaign): Candidate[] {
  const ids = new Set<string>();
  for (const step of run.steps) {
    for (const action of step.all_scored_actions || []) {
      if (action.action?.candidate_id) ids.add(action.action.candidate_id);
    }
    const candidateId = step.preregistration?.action?.candidate_id;
    if (candidateId) ids.add(candidateId);
  }
  const samples = new Map((snapshot.samples || []).map((sample) => [sample.sample_id, sample]));
  return [...ids].map((candidate_id) => {
    const sample: SampleItem | undefined = 'mode' in run && run.mode === 'HISTORICAL_REPLAY' ? samples.get(candidate_id) : undefined;
    return sample ? {
      candidate_id,
      composition_label: sample.target_formula || candidate_id,
      target_system: 'A-Lab Precursor Genome',
      target_formula: sample.target_formula,
      target_stoichiometry: sample.target_stoichiometry,
      precursors: sample.precursors,
      heating_temperature_c: sample.heating_temperature_c,
      heating_time_minutes: sample.heating_time_minutes,
      reaction_energy_ev_per_atom: sample.reaction_energy_ev_per_atom,
      reaction_category: sample.reaction_category,
      xrd_available: sample.xrd_available,
      refinement_available: sample.refinement_available,
      outcome_available: sample.outcome_available,
      refinement_rwp: sample.refinement_rwp,
      refinement_phases: sample.refinement_phases,
      refinement_cases: sample.refinement_cases,
      selected_refinement_case_id: sample.selected_refinement_case_id,
      refinement_selection_rule: sample.refinement_selection_rule,
      source_record_identifier: sample.source_record_identifier,
    } : {
      candidate_id,
      composition_label: candidate_id,
      target_system: 'Controlled synthetic world',
    };
  });
}

function actionModalities(run: FlagshipCampaign | ReplayCampaign): string[] {
  return [...new Set(run.steps.flatMap((step) =>
    (step.all_scored_actions || []).map((action) => action.action?.action_type).filter(Boolean) as string[],
  ))];
}

function selectedCost(run: FlagshipCampaign | ReplayCampaign): number | null {
  const costs = run.steps.map((step) => step.preregistration?.action?.estimated_cost);
  if (!costs.length || costs.some((cost) => typeof cost !== 'number' || !Number.isFinite(cost))) return null;
  return costs.reduce<number>((sum, cost) => sum + Number(cost), 0);
}

function recordedRun(snapshot: SnapshotData, entry: ScientificDatasetRegistryEntry, config: CampaignResolutionConfiguration, defaultConfig: Record<string, any>, expectedMode: 'CONTROLLED_SYNTHETIC' | 'HISTORICAL_REPLAY'):
  | { ok: true; value: FlagshipCampaign | ReplayCampaign }
  | { ok: false; result: ResolveResult<ResolvedCampaignView> } {
  const runs = (snapshot.campaign_runs || []).filter((candidate) => ('mode' in candidate ? candidate.mode : 'CONTROLLED_SYNTHETIC') === expectedMode);
  const alternatives = availableAlternatives(entry, expectedMode);
  const requested = {
    runId: config.runId,
    world: config.world !== undefined ? config.world : defaultConfig.world,
    seed: config.seed !== undefined ? config.seed : defaultConfig.seed,
    policy: config.policy !== undefined ? config.policy : defaultConfig.policy,
  };
  const matchesExplicit = (run: FlagshipCampaign | ReplayCampaign) => {
    if (requested.runId !== undefined && run.run_id !== requested.runId) return false;
    if (config.world !== undefined && (run as FlagshipCampaign).world !== config.world) return false;
    if (config.seed !== undefined && run.seed !== config.seed) return false;
    if (config.policy !== undefined && run.policy !== config.policy) return false;
    return true;
  };
  const run = requested.runId
    ? runs.find((candidate) => candidate.run_id === requested.runId && matchesExplicit(candidate))
    : runs.find((candidate) =>
        (expectedMode === 'HISTORICAL_REPLAY' ? config.world === undefined : (candidate as FlagshipCampaign).world === requested.world)
        && candidate.seed === requested.seed
        && candidate.policy === requested.policy,
      );
  if (!run) {
    const values = new Set(alternatives.configurations.map((configuration) => configuration.policy));
    const seeds = new Set(alternatives.configurations.map((configuration) => configuration.seed));
    const worlds = new Set(alternatives.configurations.map((configuration) => configuration.world));
    const reason = config.policy !== undefined && !values.has(config.policy) ? 'UNKNOWN_POLICY'
      : config.seed !== undefined && !seeds.has(config.seed) ? 'UNKNOWN_SEED'
      : config.world !== undefined && !worlds.has(config.world) ? 'UNKNOWN_WORLD'
      : 'RUN_NOT_AVAILABLE';
    return { ok: false, result: failure(reason, 'No recorded campaign run matches the requested exact configuration tuple.', {
      requested,
      ...availableAlternatives(entry, expectedMode),
    }) };
  }
  return { ok: true, value: run };
}

function shell(
  snapshot: SnapshotData,
  entry: ScientificDatasetRegistryEntry,
  kind: 'controlled_multimodal' | 'historical_replay' | 'surrogate_optimization',
  evidenceKind: EvidenceMode,
  selectedConfiguration: Record<string, unknown>,
): Omit<ResolvedViewShell, 'kind'> {
  return {
    datasetId: entry.id,
    displayName: entry.displayName,
    domain: entry.domain,
    statusBadge: entry.statusBadge,
    evidenceKind,
    selectedConfiguration,
    sourceManifest: sourceManifest(snapshot, entry),
    availableViews: kind === 'surrogate_optimization'
      ? ['screening', 'trajectory', 'policy_metrics']
      : ['trajectory', 'preregistration', 'observation', 'posterior'],
    limitations: entry.limitations || entry.disclosures,
    banner: {
      title: entry.displayName,
      badge: entry.statusBadge,
      description: entry.summary,
      confidenceOrUtilityLabel: 'Not recorded',
      budgetExpended: null,
      budgetUnits: 'units',
    },
    capabilities: entry.capabilities,
    disclosures: entry.disclosures,
  };
}

function controlledView(
  snapshot: SnapshotData,
  entry: ScientificDatasetRegistryEntry,
  run: FlagshipCampaign,
): ControlledMultimodalView {
  const finalBeliefs = run.steps.at(-1)?.belief_update?.beliefs_after;
  const ranked = finalBeliefs && Object.entries(finalBeliefs).sort(([, a], [, b]) => b - a)[0];
  const cost = selectedCost(run);
  const selected = run.steps.at(-1)?.preregistration?.action;
  const campaign = { ...run, candidates: candidatePool(snapshot, run) };
  return {
    kind: 'controlled_multimodal',
    ...shell(snapshot, entry, 'controlled_multimodal', 'CONTROLLED_SYNTHETIC', {
      runId: run.run_id,
      world: run.world,
      seed: run.seed,
      policy: run.policy,
    }),
    campaign,
    steps: campaign.steps,
    totalSteps: campaign.steps.length,
    candidates: campaign.candidates,
    hypotheses: Object.keys(snapshot.hypotheses),
    modalities: actionModalities(campaign),
    banner: {
      title: `Controlled Multimodal Campaign • ${run.policy === 'HYBRID' ? 'Hybrid Policy' : run.policy === 'PURE_HIG' ? 'Pure HIG Policy' : run.policy === 'DISCOVERY_ONLY' ? 'Discovery-Only Policy' : run.policy}`,
      badge: `${campaign.steps.length} steps recorded`,
      description: selected
        ? `Selected action ${selected.candidate_id} / ${selected.action_type} is sourced from the recorded trajectory.`
        : 'Selected action is unavailable in this recorded trajectory.',
      confidenceOrUtilityLabel: ranked
        ? `Highest final model weight: ${ranked[0]} ${(ranked[1] * 100).toFixed(2)}% (not physical confirmation)`
        : 'Final model weights not recorded',
      budgetExpended: cost,
      budgetUnits: 'simulated cost units',
    },
  };
}

function historicalView(
  snapshot: SnapshotData,
  entry: ScientificDatasetRegistryEntry,
  run: ReplayCampaign,
): HistoricalReplayView {
  const campaign = { ...run, candidates: candidatePool(snapshot, run) };
  const replaySampleIds = [...new Set(campaign.steps.map((step) => step.preregistration?.action?.candidate_id).filter(Boolean) as string[])];
  const cost = selectedCost(campaign);
  const finalBeliefs = campaign.steps.at(-1)?.belief_update?.beliefs_after;
  const ranked = finalBeliefs && Object.entries(finalBeliefs).sort(([, a], [, b]) => b - a)[0];
  return {
    kind: 'historical_replay',
    ...shell(snapshot, entry, 'historical_replay', 'HISTORICAL_REPLAY', {
      runId: run.run_id,
      seed: run.seed,
      policy: run.policy,
      mode: run.mode,
    }),
    campaign,
    steps: campaign.steps,
    totalSteps: campaign.steps.length,
    candidates: campaign.candidates,
    hypotheses: Object.keys(snapshot.hypotheses),
    modalities: actionModalities(campaign),
    replaySampleIds,
    alabSamples: snapshot.samples.filter((sample) => replaySampleIds.includes(sample.sample_id)),
    banner: {
      title: 'A-Lab Synthesis Retrospective Replay • Autonomous Loop',
      badge: `${campaign.steps.length} recorded steps`,
      description: 'Historical observations are replayed from source-linked artifacts; this is not the original laboratory policy log.',
      confidenceOrUtilityLabel: ranked
        ? `Highest retrospective model weight: ${ranked[0]} ${(ranked[1] * 100).toFixed(2)}%`
        : 'Retrospective model weight not recorded',
      budgetExpended: cost,
      budgetUnits: 'recorded action cost units',
    },
  };
}

function validSurrogateRun(run: SurrogateRun): boolean {
  const lengths = [
    run.queried_candidate_ids?.length,
    run.revealed_noisy_values?.length,
    run.selected_latent_values?.length,
    run.best_latent_curve?.length,
  ];
  return lengths.every((length) => typeof length === 'number' && length === lengths[0]) && lengths[0] > 0
    && [...run.revealed_noisy_values, ...run.selected_latent_values, ...run.best_latent_curve].every(Number.isFinite);
}

function surrogateView(
  snapshot: SnapshotData,
  entry: ScientificDatasetRegistryEntry,
  run: SurrogateRun,
  screening: any,
  policy: string,
): SurrogateOptimizationView {
  const trajectory = run.queried_candidate_ids.map((candidateId, index) => ({
    queryIndex: index + 1,
    candidateId,
    revealedNoisyValue: run.revealed_noisy_values[index],
    selectedLatentValue: run.selected_latent_values[index],
    bestSelectedLatentValue: run.best_latent_curve[index],
  }));
  const summary = snapshot.electrolyte_simulation.simulation_policies?.[policy];
  return {
    kind: 'surrogate_optimization',
    ...shell(snapshot, entry, 'surrogate_optimization', 'SIMULATED_SURROGATE', { policy, seed: run.seed }),
    scientificTargetName: entry.scientificTargetName || entry.targetObservable || entry.targetObservableDescription,
    policy,
    seed: run.seed,
    trajectory,
    simulationRun: run,
    screeningDiagnostics: screening,
    banner: {
      title: `Electrolyte Surrogate Optimization • ${policy === 'HYBRID_DEFAULT' ? 'Hybrid Policy' : policy === 'PURE_HIG' ? 'Pure HIG Policy' : policy === 'DISCOVERY_ONLY' ? 'Discovery-Only Policy' : policy}`,
      badge: `${run.queried_candidate_ids.length} queries recorded`,
      description: 'This view replays a frozen in-silico surrogate trajectory; no physical battery measurement was performed.',
      confidenceOrUtilityLabel: run.best_selected_latent_capacity !== undefined && run.simple_regret_latent !== undefined
        ? `Best selected latent value: ${run.best_selected_latent_capacity.toFixed(4)}; latent simple regret: ${run.simple_regret_latent.toFixed(4)}`
        : 'Best latent value or regret not recorded',
      budgetExpended: run.queried_candidate_ids.length,
      budgetUnits: 'surrogate queries',
    },
    disclosures: [
      ...entry.disclosures,
      summary ? `Recorded policy summary is sourced from simulation_policies.${policy}.` : 'Policy summary is not recorded for this run.',
    ],
  };
}

export function resolveCampaign(
  datasetOption: DatasetOption | string = 'controlled_multimodal_alloy',
  snapshot: SnapshotData,
  selectedPolicyId?: string,
  configuration: CampaignResolutionConfiguration = {},
): ResolveResult<ResolvedCampaignView> {
  const canonicalId = normalizeDatasetId(datasetOption);
  if (!canonicalId) return failure('UNKNOWN_DATASET', `Unknown dataset ID: ${String(datasetOption)}.`);

  const entryResult = registryEntry(snapshot, canonicalId);
  if (!entryResult.ok) return entryResult.result;
  const entry = entryResult.value;

  if (canonicalId === 'anode_free_electrolyte_screening') {
    const simulation = snapshot.electrolyte_simulation;
    const runs = simulation?.detailed_policy_seed_runs;
    if (!runs || typeof runs !== 'object') return failure('SOURCE_ARTIFACT_MISSING', 'Surrogate trajectory artifact is missing detailed policy runs.');
    if (configuration.runId !== undefined || configuration.world !== undefined) {
      return failure('UNSUPPORTED_CONFIGURATION', 'Surrogate trajectories accept only an exact source policy and seed tuple.');
    }
    const requestedPolicyRaw = configuration.policy ?? selectedPolicyId;
    if (requestedPolicyRaw && ['HYBRID', 'PURE_HIG', 'DISCOVERY_ONLY', 'RANDOM_ACTION'].includes(requestedPolicyRaw)) {
      return failure('UNKNOWN_POLICY', `Policy ${requestedPolicyRaw} is a controlled multimodal policy and cannot be applied to electrolyte surrogate screening.`, availableAlternatives(entry, 'SIMULATED_SURROGATE'));
    }
    const requestedPolicy = requestedPolicyRaw ?? String(entry.defaultConfiguration.policy);
    const policy = SURROGATE_POLICY_ALIASES[requestedPolicy] || requestedPolicy;
    if (!Object.prototype.hasOwnProperty.call(runs, policy)) {
      return failure('UNKNOWN_POLICY', `No surrogate policy is recorded as ${requestedPolicy}.`, availableAlternatives(entry, 'SIMULATED_SURROGATE'));
    }
    const requestedSeed = configuration.seed ?? Number(entry.defaultConfiguration.seed);
    const activeRun = (runs[policy] as SurrogateRun[]).find((run) => run.seed === requestedSeed);
    if (!activeRun) {
      return failure('RUN_NOT_AVAILABLE', `No surrogate trajectory is recorded for ${policy} / seed ${requestedSeed}.`, availableAlternatives(entry, 'SIMULATED_SURROGATE'));
    }
    if (!validSurrogateRun(activeRun)) return failure('INVALID_SOURCE_DATA', `Surrogate trajectory ${policy} / seed ${requestedSeed} has mismatched or non-finite arrays.`);
    return { ok: true, value: surrogateView(snapshot, entry, activeRun, snapshot.electrolyte_screening, policy) };
  }

  const policyInput = configuration.policy ?? selectedPolicyId;
  if (policyInput && ['HYBRID_DEFAULT', 'PURE_FALSIFICATION', 'RANDOM'].includes(policyInput)) {
    return failure('UNKNOWN_POLICY', `Policy ${policyInput} is an electrolyte surrogate policy and cannot be applied to ${canonicalId}.`, availableAlternatives(entry, canonicalId === 'alab_precursor_genome' ? 'HISTORICAL_REPLAY' : 'CONTROLLED_SYNTHETIC'));
  }
  const requestedPolicy = policyInput ? POLICY_ALIASES[policyInput] || policyInput : undefined;
  const expectedMode = canonicalId === 'alab_precursor_genome' ? 'HISTORICAL_REPLAY' : 'CONTROLLED_SYNTHETIC';
  const runResult = recordedRun(snapshot, entry, { ...configuration, policy: requestedPolicy }, entry.defaultConfiguration, expectedMode);
  if (!runResult.ok) return runResult.result;
  const run = runResult.value;
  if (canonicalId === 'alab_precursor_genome') {
    const replayRun = run as ReplayCampaign;
    const referencedSampleIds = [...new Set([
      ...(replayRun.replay_candidate_ids || []),
      ...replayRun.steps.flatMap((step) => (step.all_scored_actions || []).map((score) => score.action?.candidate_id).filter(Boolean) as string[]),
      ...replayRun.steps.map((step) => step.preregistration?.action?.candidate_id).filter(Boolean) as string[],
    ])];
    const missingSampleIds = referencedSampleIds.filter((id) => !snapshot.samples.some((sample) => sample.sample_id === id));
    if (missingSampleIds.length) return failure('SOURCE_ARTIFACT_MISSING', `Replay ${replayRun.run_id} references missing A-Lab source samples.`, { runId: replayRun.run_id, missingSampleIds });
    return { ok: true, value: historicalView(snapshot, entry, replayRun) };
  }
  return { ok: true, value: controlledView(snapshot, entry, run as FlagshipCampaign) };
}

export function visibleCampaignStep(step: CampaignStep, phase: 'A_SCORED' | 'B_PREREGISTERED' | 'C_REVEALED' | 'D_UPDATED'): CampaignStep {
  if (phase === 'A_SCORED' || phase === 'B_PREREGISTERED') {
    return { ...step, observation: null, belief_update: null };
  }
  if (phase === 'C_REVEALED') return { ...step, belief_update: null };
  return step;
}
