/** Client-side fail-closed validation for the source-backed presentation snapshot. */

export interface ValidationResult {
  valid: boolean;
  errors: string[];
  warnings: string[];
}

const finite = (value: unknown): value is number => typeof value === 'number' && Number.isFinite(value);

function validateCampaign(campaign: any, errors: string[], label: string) {
  if (!campaign?.run_id || !Array.isArray(campaign.steps) || !campaign.steps.length) {
    errors.push(`${label} has no recorded run or steps.`);
    return;
  }
  const initialBeliefs = campaign.initial_beliefs || {};
  const initialSum = Object.values(initialBeliefs).reduce((sum: number, value: any) => sum + Number(value), 0);
  if (Math.abs(initialSum - 1) > 1e-4) errors.push(`${label} initial beliefs do not sum to 1 (${initialSum}).`);
  let previousBeliefs = initialBeliefs;
  campaign.steps.forEach((step: any, index: number) => {
    const name = `${label} step ${step?.step ?? index + 1}`;
    if (step?.step !== index + 1) errors.push(`${name} index is not sequential.`);
    const prereg = step?.preregistration;
    const observation = step?.observation;
    const update = step?.belief_update;
    if (!prereg || !observation || !update) {
      errors.push(`${name} is missing a source event.`);
      return;
    }
    if (!(prereg.event_sequence < observation.event_sequence && observation.event_sequence < update.event_sequence)) {
      errors.push(`${name} event order is not preregistration < reveal < update.`);
    }
    if (prereg.action?.candidate_id !== observation.action?.candidate_id || prereg.action?.action_type !== observation.action?.action_type) {
      errors.push(`${name} selected action does not match its observation.`);
    }
    for (const [hypothesisId, value] of Object.entries(prereg.beliefs_before || {})) {
      if (Math.abs(Number(value) - Number(previousBeliefs[hypothesisId] ?? NaN)) > 1e-4) errors.push(`${name} belief continuity is broken for ${hypothesisId}.`);
    }
    const posterior = update.beliefs_after || {};
    const posteriorSum = Object.values(posterior).reduce((sum: number, value: any) => sum + Number(value), 0);
    if (Math.abs(posteriorSum - 1) > 1e-4) errors.push(`${name} posterior does not sum to 1.`);
    previousBeliefs = posterior;
    if (!prereg.predictive_distributions || !Object.keys(prereg.predictive_distributions).length) errors.push(`${name} has no predictive distributions.`);
    for (const [hypothesisId, distribution] of Object.entries<any>(prereg.predictive_distributions || {})) {
      if (distribution.distribution_kind === 'categorical') {
        const sum = (distribution.probabilities || []).reduce((total: number, value: any) => total + Number(value), 0);
        if (Math.abs(sum - 1) > 1e-4) errors.push(`${name} ${hypothesisId} categorical probabilities are invalid.`);
      } else if (!(distribution.variance || []).every((value: any) => finite(value) && value > 0)) {
        errors.push(`${name} ${hypothesisId} predictive variance is invalid.`);
      }
    }
    const scores = step.all_scored_actions || [];
    if (!scores.length) errors.push(`${name} has no source action scores.`);
    scores.forEach((score: any) => {
      if (!score.action?.action_id) errors.push(`${name} has a score without a source action ID.`);
      for (const field of ['expected_hig_nats', 'discovery_utility', 'normalized_cost', 'total_action_score']) {
        if (!finite(score[field])) errors.push(`${name} source score field ${field} is not finite.`);
      }
    });
  });
}

export function validateSnapshot(data: any): ValidationResult {
  const errors: string[] = [];
  const warnings: string[] = [];
  if (!data || typeof data !== 'object') return { valid: false, errors: ['Snapshot data is not an object.'], warnings };

  const manifest = data.manifest;
  const manifestKeys = ['snapshot_schema_version', 'generated_at_utc', 'scientific_source_commit', 'snapshot_generator_commit', 'presentation_build_commit', 'source_branch', 'source_artifact_hashes', 'campaign_run_id', 'campaign_world', 'total_real_samples', 'validation_gate_pass_count'];
  if (!manifest) errors.push('Snapshot is missing its embedded manifest.');
  else {
    manifestKeys.forEach((key) => { if (manifest[key] === undefined || manifest[key] === null || manifest[key] === '') errors.push(`Manifest is missing ${key}.`); });
    if (Object.keys(manifest.source_artifact_hashes || {}).length < 10) errors.push('Manifest tracks fewer than 10 source artifacts.');
  }

  validateCampaign(data.flagship_campaign, errors, 'Flagship campaign');
  const campaignRuns = data.campaign_runs || [];
  if (!Array.isArray(campaignRuns) || campaignRuns.length < 3) errors.push('Snapshot has too few complete campaign runs.');
  if (new Set(campaignRuns.map((run: any) => run.run_id)).size !== campaignRuns.length) errors.push('Snapshot campaign run IDs are not unique.');
  campaignRuns.forEach((run: any) => validateCampaign(run, errors, `Campaign ${run.run_id || 'unknown'}`));

  const samples = data.samples || [];
  if (samples.length !== 1035 || new Set(samples.map((sample: any) => sample.sample_id)).size !== samples.length) errors.push('A-Lab sample catalog is not the unique 1,035-record source catalog.');

  const benchmarks = data.benchmarks;
  if (!benchmarks || benchmarks.status !== 'METHODOLOGY_VALID' || !Number.isInteger(benchmarks.trajectory_count) || benchmarks.trajectory_count <= 0) errors.push('Benchmark summary is missing or invalid.');

  const registry = data.dataset_registry;
  if (!registry || !Array.isArray(registry.datasets) || registry.datasets.length !== 3) errors.push('Dataset registry must contain the three canonical datasets.');
  (registry?.datasets || []).forEach((dataset: any) => {
    if (Array.isArray(dataset.candidateIds) && dataset.candidateIds.length !== dataset.candidateCount) errors.push(`${dataset.id} candidate catalog is truncated.`);
    if (dataset.id === 'anode_free_electrolyte_screening' && /cycle 3/i.test(dataset.targetObservableDescription || '')) errors.push('Electrolyte registry contains the incorrect cycle-3 target description.');
  });

  const simulation = data.electrolyte_simulation;
  if (simulation?.oracle_kind !== 'SIMULATED_SURROGATE' || simulation?.physical_synthesis !== false) errors.push('Electrolyte simulation is not explicitly surrogate-only.');
  Object.entries<any>(simulation?.detailed_policy_seed_runs || {}).forEach(([policy, runs]) => {
    (runs || []).forEach((run: any) => {
      const lengths = ['queried_candidate_ids', 'revealed_noisy_values', 'selected_latent_values', 'best_latent_curve'].map((key) => run[key]?.length || 0);
      if (!lengths[0] || new Set(lengths).size !== 1) errors.push(`Surrogate ${policy}/${run.seed} trajectory arrays are truncated or mismatched.`);
    });
  });

  return { valid: errors.length === 0, errors, warnings };
}
