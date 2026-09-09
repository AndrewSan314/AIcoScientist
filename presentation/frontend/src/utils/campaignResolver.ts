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
} from '../types/mission_control';

const POLICY_ALIASES: Record<string, string> = {
  hig_cost_penalized: 'HYBRID',
  greedy_hig: 'PURE_HIG',
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
  };
}

function candidatePool(run: FlagshipCampaign | ReplayCampaign): Candidate[] {
  const ids = new Set<string>();
  for (const step of run.steps) {
    for (const action of step.all_scored_actions || []) {
      if (action.action?.candidate_id) ids.add(action.action.candidate_id);
    }
    const candidateId = step.preregistration?.action?.candidate_id;
    if (candidateId) ids.add(candidateId);
  }
  return [...ids].map((candidate_id) => ({
    candidate_id,
    composition_label: candidate_id,
    target_system: 'mode' in run && run.mode === 'HISTORICAL_REPLAY' ? 'A-Lab Precursor Genome' : 'Controlled synthetic world',
  }));
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

function recordedRun(snapshot: SnapshotData, config: CampaignResolutionConfiguration, defaultConfig: Record<string, any>):
  | { ok: true; value: FlagshipCampaign | ReplayCampaign }
  | { ok: false; result: ResolveResult<ResolvedCampaignView> } {
  const runs = snapshot.campaign_runs || [];
  const requestedRunId = config.runId;
  const requestedWorld = config.world ?? defaultConfig.world;
  const requestedSeed = config.seed ?? defaultConfig.seed;
  const requestedPolicy = config.policy ?? defaultConfig.policy;

  const run = requestedRunId
    ? runs.find((candidate) => candidate.run_id === requestedRunId)
    : runs.find((candidate) => {
        const campaign = candidate as FlagshipCampaign;
        return campaign.world === requestedWorld && campaign.seed === requestedSeed && campaign.policy === requestedPolicy;
      });

  if (!run) {
    const reason = requestedRunId
      ? 'RUN_NOT_AVAILABLE'
      : requestedWorld && !runs.some((candidate) => (candidate as FlagshipCampaign).world === requestedWorld)
      ? 'UNKNOWN_WORLD'
      : requestedSeed !== undefined && !runs.some((candidate) => (candidate as FlagshipCampaign).seed === requestedSeed)
      ? 'UNKNOWN_SEED'
      : requestedPolicy && !runs.some((candidate) => (candidate as FlagshipCampaign).policy === requestedPolicy)
      ? 'UNKNOWN_POLICY'
      : 'RUN_NOT_AVAILABLE';
    return {
      ok: false,
      result: failure(reason, 'No recorded campaign run matches the requested configuration.', runs.map((item) => item.run_id)),
    };
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
  const campaign = { ...run, candidates: candidatePool(run) };
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
      title: `Recorded run ${run.run_id}`,
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
  const campaign = { ...run, candidates: candidatePool(run) };
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
      title: `Recorded historical replay ${run.run_id}`,
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
    simpleRegretLatent: index === run.queried_candidate_ids.length - 1 ? run.simple_regret_latent : undefined,
    cumulativeRawHigNats: undefined,
    realizedEntropyReductionNats: undefined,
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
      title: `Recorded surrogate trajectory ${policy} / seed ${run.seed}`,
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
    const requestedPolicy = configuration.policy || selectedPolicyId || String(entry.defaultConfiguration.policy);
    const policy = SURROGATE_POLICY_ALIASES[requestedPolicy] || requestedPolicy;
    if (!Object.prototype.hasOwnProperty.call(runs, policy)) {
      return failure('UNKNOWN_POLICY', `No surrogate policy is recorded as ${requestedPolicy}.`, Object.keys(runs));
    }
    const requestedSeed = configuration.seed ?? Number(entry.defaultConfiguration.seed);
    const activeRun = (runs[policy] as SurrogateRun[]).find((run) => run.seed === requestedSeed);
    if (!activeRun) {
      return failure('RUN_NOT_AVAILABLE', `No surrogate trajectory is recorded for ${policy} / seed ${requestedSeed}.`, (runs[policy] as SurrogateRun[]).map((run) => run.seed));
    }
    if (!validSurrogateRun(activeRun)) return failure('INVALID_SOURCE_DATA', `Surrogate trajectory ${policy} / seed ${requestedSeed} has mismatched or non-finite arrays.`);
    return { ok: true, value: surrogateView(snapshot, entry, activeRun, snapshot.electrolyte_screening, policy) };
  }

  const requestedPolicy = configuration.policy || (selectedPolicyId ? POLICY_ALIASES[selectedPolicyId] || selectedPolicyId : undefined);
  const runResult = recordedRun(snapshot, { ...configuration, policy: requestedPolicy }, entry.defaultConfiguration);
  if (!runResult.ok) return runResult.result;
  const run = runResult.value;
  if (canonicalId === 'alab_precursor_genome') {
    if (!('mode' in run) || run.mode !== 'HISTORICAL_REPLAY') return failure('UNSUPPORTED_CONFIGURATION', `Run ${run.run_id} is not a historical replay.`);
    return { ok: true, value: historicalView(snapshot, entry, run as ReplayCampaign) };
  }
  if ('mode' in run && run.mode === 'HISTORICAL_REPLAY') return failure('UNSUPPORTED_CONFIGURATION', `Historical run ${run.run_id} cannot resolve as a controlled campaign.`);
  return { ok: true, value: controlledView(snapshot, entry, run as FlagshipCampaign) };
}

export function visibleCampaignStep(step: CampaignStep, phase: 'A_SCORED' | 'B_PREREGISTERED' | 'C_REVEALED' | 'D_UPDATED'): CampaignStep {
  if (phase === 'A_SCORED' || phase === 'B_PREREGISTERED') {
    return { ...step, observation: null, belief_update: null };
  }
  if (phase === 'C_REVEALED') return { ...step, belief_update: null };
  return step;
}
