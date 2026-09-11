import { describe, expect, it } from 'vitest';
import { readFileSync } from 'node:fs';
import { resolveCampaign, normalizeDatasetId } from './campaignResolver';
import type { SnapshotData } from '../types/mission_control';

const snapshot = JSON.parse(readFileSync(new URL('../../../data/snapshot.json', import.meta.url), 'utf8')) as SnapshotData;

describe('source-backed campaign resolver', () => {
  it('rejects unknown dataset IDs instead of mapping them to the controlled default', () => {
    expect(normalizeDatasetId('not-a-dataset')).toBeNull();
    const result = resolveCampaign('not-a-dataset', snapshot);
    expect(result).toMatchObject({ ok: false, reason: 'UNKNOWN_DATASET' });
  });

  it('selects the exact controlled policy-comparison run', () => {
    const result = resolveCampaign('controlled_synthesis', snapshot, 'hig_cost_penalized');
    expect(result.ok).toBe(true);
    if (!result.ok) return;
    expect(result.value.kind).toBe('controlled_multimodal');
    if (result.value.kind !== 'controlled_multimodal') return;
    expect(result.value.campaign.run_id).toBe('policy_comparison:WORLD_H1_PHASE_PURITY:42:HYBRID');
    expect(result.value.campaign.policy_weights).toBeUndefined();
    expect(result.value.steps[0].all_scored_actions?.[0].total_action_score).toBe(
      snapshot.flagship_campaign.steps[0].all_scored_actions?.[0].total_action_score,
    );
  });

  it('fails closed when a requested controlled policy has no scored source run', () => {
    const result = resolveCampaign('controlled_synthesis', snapshot, 'random_baseline');
    expect(result).toMatchObject({ ok: false, reason: 'UNKNOWN_POLICY' });
  });

  it('resolves an exact controlled tuple and rejects a valid-but-unrecorded combination', () => {
    const exact = resolveCampaign('controlled_synthesis', snapshot, undefined, {
      world: 'CLEAN_WORLD_H1_PHASE_PURITY', seed: 7, policy: 'PURE_HIG',
    });
    expect(exact).toMatchObject({ ok: true });
    const missing = resolveCampaign('controlled_synthesis', snapshot, undefined, {
      world: 'CLEAN_WORLD_H1_PHASE_PURITY', seed: 42, policy: 'HYBRID',
    });
    expect(missing).toMatchObject({ ok: false, reason: 'RUN_NOT_AVAILABLE' });
    if (!missing.ok) expect(missing.availableOptions).toMatchObject({ configurations: expect.any(Array) });
  });

  it('does not cross-resolve historical and controlled run modes', () => {
    expect(resolveCampaign('controlled_synthesis', snapshot, undefined, { runId: 'replay:HYBRID:42:1' })).toMatchObject({ ok: false, reason: 'RUN_NOT_AVAILABLE' });
    expect(resolveCampaign('alab_replay', snapshot, undefined, { runId: 'policy_comparison:WORLD_H1_PHASE_PURITY:42:HYBRID' })).toMatchObject({ ok: false, reason: 'RUN_NOT_AVAILABLE' });
  });

  it('keeps A-Lab replay distinct and source-linked', () => {
    const result = resolveCampaign('alab_replay', snapshot);
    expect(result.ok).toBe(true);
    if (!result.ok) return;
    expect(result.value.kind).toBe('historical_replay');
    if (result.value.kind !== 'historical_replay') return;
    expect(result.value.campaign.run_id).toBe('replay:HYBRID:42:1');
    expect(result.value.alabSamples.map((sample) => sample.sample_id)).toEqual(['PG_0209', 'PG_0214', 'PG_0309']);
    expect(snapshot.dataset_registry?.datasets.find((dataset) => dataset.id === 'alab_precursor_genome')?.candidateIds).toHaveLength(1035);
    const candidates = result.value.candidates;
    for (const sampleId of ['PG_0309', 'PG_0214', 'PG_0209']) {
      const sample = snapshot.samples.find((item) => item.sample_id === sampleId);
      const candidate = candidates.find((item) => item.candidate_id === sampleId);
      expect(candidate).toMatchObject({
        target_formula: sample?.target_formula,
        target_stoichiometry: sample?.target_stoichiometry,
        precursors: sample?.precursors,
        source_record_identifier: sampleId,
      });
    }
  });

  it('selects an exact surrogate policy/seed trajectory without creating campaign events', () => {
    const result = resolveCampaign('electrolyte_search', snapshot, 'hig_cost_penalized', { seed: 101 });
    expect(result.ok).toBe(true);
    if (!result.ok) return;
    expect(result.value.kind).toBe('surrogate_optimization');
    if (result.value.kind !== 'surrogate_optimization') return;
    expect(result.value.policy).toBe('HYBRID_DEFAULT');
    expect(result.value.seed).toBe(101);
    expect(result.value.trajectory).toHaveLength(15);
    expect((result.value as any).steps).toBeUndefined();
    expect(result.value.trajectory[0]).not.toHaveProperty('simpleRegretLatent');
    expect(result.value.trajectory[0]).not.toHaveProperty('cumulativeRawHigNats');
    expect(result.value.trajectory[0]).not.toHaveProperty('realizedEntropyReductionNats');
    expect(result.value.policy).toBe('HYBRID_DEFAULT');
  });

  it('rejects unavailable surrogate seeds instead of falling back to seed 42', () => {
    const result = resolveCampaign('electrolyte_search', snapshot, 'hig_cost_penalized', { seed: 999 });
    expect(result).toMatchObject({ ok: false, reason: 'RUN_NOT_AVAILABLE' });
  });

  it('rejects unavailable surrogate policies without falling back to HYBRID_DEFAULT', () => {
    const result = resolveCampaign('electrolyte_search', snapshot, undefined, { policy: 'NOT_A_SOURCE_POLICY', seed: 42 });
    expect(result).toMatchObject({ ok: false, reason: 'UNKNOWN_POLICY' });
  });

  it('publishes exact source configuration matrices', () => {
    for (const dataset of snapshot.dataset_registry?.datasets || []) {
      expect(dataset.availableConfigurations.length).toBeGreaterThan(0);
      expect(new Set(dataset.availableConfigurations.map((configuration) => configuration.configurationId)).size).toBe(dataset.availableConfigurations.length);
    }
    expect(snapshot.dataset_registry?.datasets.find((dataset) => dataset.id === 'controlled_multimodal_alloy')?.availableConfigurations).toHaveLength(33);
    expect(snapshot.dataset_registry?.datasets.find((dataset) => dataset.id === 'alab_precursor_genome')?.availableConfigurations).toHaveLength(3);
    expect(snapshot.dataset_registry?.datasets.find((dataset) => dataset.id === 'anode_free_electrolyte_screening')?.availableConfigurations).toHaveLength(18);
  });
});
