/**
 * presentation/frontend/src/utils/snapshotValidation.ts
 *
 * Client-side fail-closed validation for AIcoScientist presentation snapshot data.
 */

export interface ValidationResult {
  valid: boolean;
  errors: string[];
  warnings: string[];
}

export function validateSnapshot(data: any): ValidationResult {
  const errors: string[] = [];
  const warnings: string[] = [];

  if (!data || typeof data !== 'object') {
    return { valid: false, errors: ['Snapshot data is null or not an object.'], warnings: [] };
  }

  // 1. Manifest Validation
  const manifest = data.manifest;
  if (!manifest) {
    errors.push('Snapshot missing required embedded manifest.');
  } else {
    const requiredManifestKeys = [
      'snapshot_schema_version',
      'generated_at_utc',
      'scientific_source_commit',
      'snapshot_generator_commit',
      'presentation_build_commit',
      'source_branch',
      'source_artifact_hashes',
      'campaign_run_id',
      'campaign_world',
      'total_real_samples',
      'validation_gate_pass_count',
    ];

    for (const key of requiredManifestKeys) {
      if (!manifest[key]) {
        errors.push(`Manifest missing required field: ${key}`);
      }
    }

    const hashes = manifest.source_artifact_hashes || {};
    if (Object.keys(hashes).length < 10) {
      errors.push(`Manifest tracks fewer than 10 source artifacts (${Object.keys(hashes).length}).`);
    }
  }

  // 2. Flagship Campaign Invariants
  const campaign = data.flagship_campaign;
  if (!campaign) {
    errors.push('Snapshot missing flagship_campaign.');
  } else {
    const initBeliefs = campaign.initial_beliefs || {};
    const initProbSum = Object.values(initBeliefs).reduce((acc: number, v: any) => acc + Number(v), 0);
    if (Math.abs(initProbSum - 1.0) > 1e-4) {
      errors.push(`Initial beliefs do not sum to 1.0 (sum: ${initProbSum}).`);
    }

    const weights = campaign.policy_weights || {};
    const w_h = Number(weights.w_hig);
    const w_d = Number(weights.w_discovery);
    const w_c = Number(weights.w_cost);

    if (isNaN(w_h) || isNaN(w_d) || isNaN(w_c)) {
      errors.push('Campaign policy weights (w_hig, w_discovery, w_cost) must be valid numbers.');
    }

    const steps = campaign.steps || [];
    if (steps.length === 0) {
      errors.push('Flagship campaign contains 0 steps.');
    }

    let prevBeliefs: Record<string, number> = initBeliefs;

    for (let i = 0; i < steps.length; i++) {
      const step = steps[i];
      const sNum = step.step;
      if (sNum !== i + 1) {
        errors.push(`Step index mismatch at position ${i}: expected step ${i + 1}, got ${sNum}.`);
      }

      const prereg = step.preregistration;
      const obs = step.observation;
      const upd = step.belief_update;
      const scores = step.all_scored_actions || [];

      if (!prereg || !obs || !upd) {
        errors.push(`Step ${sNum} missing preregistration, observation, or belief_update record.`);
        continue;
      }

      // Event sequence monotonicity: prereg < obs < upd
      if (!(prereg.event_sequence < obs.event_sequence && obs.event_sequence < upd.event_sequence)) {
        errors.push(
          `Step ${sNum} event ordering violated: prereg(${prereg.event_sequence}) < obs(${obs.event_sequence}) < upd(${upd.event_sequence})`
        );
      }

      // Action consistency
      if (prereg.action?.candidate_id !== obs.action?.candidate_id) {
        errors.push(`Step ${sNum} candidate mismatch between preregistration and observation.`);
      }
      if (prereg.action?.action_type !== obs.action?.action_type) {
        errors.push(`Step ${sNum} modality mismatch between preregistration and observation.`);
      }

      // Belief continuity: prior(t) === prev_posterior(t-1)
      const priors = prereg.beliefs_before || {};
      for (const [hId, pVal] of Object.entries(priors)) {
        const prevP = prevBeliefs[hId] ?? 0.0;
        if (Math.abs(Number(pVal) - prevP) > 1e-4) {
          errors.push(`Step ${sNum} belief continuity broken for ${hId}: prior(${pVal}) !== prev_posterior(${prevP}).`);
        }
      }

      // Posterior validity
      const postBeliefs = upd.beliefs_after || {};
      const postSum = Object.values(postBeliefs).reduce((acc: number, v: any) => acc + Number(v), 0);
      if (Math.abs(postSum - 1.0) > 1e-4) {
        errors.push(`Step ${sNum} posterior belief sum invalid: ${postSum}.`);
      }

      prevBeliefs = postBeliefs;

      // Predictive distribution check
      const dists = prereg.predictive_distributions || {};
      for (const [hId, dist] of Object.entries<any>(dists)) {
        const kind = dist.distribution_kind || 'gaussian';
        if (kind === 'gaussian') {
          const variances = dist.variance || [];
          for (const v of variances) {
            if (typeof v !== 'number' || isNaN(v) || !isFinite(v) || v <= 0) {
              errors.push(`Step ${sNum} ${hId} invalid non-positive predictive variance: ${v}.`);
            }
          }
        } else if (kind === 'categorical') {
          const probs = dist.probabilities || [];
          const sumProb = probs.reduce((a: number, b: number) => a + b, 0);
          if (Math.abs(sumProb - 1.0) > 1e-4) {
            errors.push(`Step ${sNum} ${hId} categorical probabilities do not sum to 1.0 (sum: ${sumProb}).`);
          }
        }
      }

      // Score decomposition verification
      for (const act of scores) {
        const total = Number(act.total_action_score);
        const normHig = Number(act.normalized_hig);
        const normDisc = Number(act.normalized_discovery);
        const normCost = Number(act.normalized_cost);

        if (!isNaN(normHig) && !isNaN(normDisc) && !isNaN(normCost)) {
          const expectedScore = w_h * normHig + w_d * normDisc - w_c * normCost;
          if (Math.abs(expectedScore - total) > 1e-4) {
            errors.push(
              `Step ${sNum} action ${act.action?.action_id} score mismatch: computed=${expectedScore.toFixed(4)}, stored=${total.toFixed(4)}`
            );
          }
        }
      }
    }
  }

  // 3. Samples catalog validation
  const samples = data.samples || [];
  if (samples.length !== 1035) {
    errors.push(`Samples catalog count mismatch: expected 1,035, found ${samples.length}.`);
  }

  // 4. Benchmarks validation
  const benchmarks = data.benchmarks;
  if (!benchmarks) {
    errors.push('Snapshot missing benchmarks.');
  } else {
    if (benchmarks.status !== 'METHODOLOGY_VALID') {
      errors.push(`Benchmarks status is ${benchmarks.status}, expected METHODOLOGY_VALID.`);
    }
    if (benchmarks.trajectory_count !== 180) {
      errors.push(`Benchmarks trajectory count is ${benchmarks.trajectory_count}, expected 180.`);
    }

    const swp = benchmarks.summary_by_world_policy || {};
    for (const [world, policies] of Object.entries<any>(swp)) {
      for (const [pol, stats] of Object.entries<any>(policies)) {
        for (const mKey of ['mean_steps_to_MAP', 'mean_steps_to_posterior_gt_0.5', 'mean_steps_to_posterior_gt_0.8']) {
          const val = stats[mKey];
          if (val !== null && val !== undefined && val > 100) {
            errors.push(`Sentinel value leaked into benchmark: ${world}/${pol}/${mKey} = ${val}`);
          }
        }
      }
    }
  }

  return {
    valid: errors.length === 0,
    errors,
    warnings,
  };
}
