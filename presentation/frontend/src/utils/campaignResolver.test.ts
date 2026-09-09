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

  it('keeps A-Lab replay distinct and source-linked', () => {
    const result = resolveCampaign('alab_replay', snapshot);
    expect(result.ok).toBe(true);
    if (!result.ok) return;
    expect(result.value.kind).toBe('historical_replay');
    if (result.value.kind !== 'historical_replay') return;
    expect(result.value.campaign.run_id).toBe('replay:HYBRID:42:1');
    expect(result.value.alabSamples.map((sample) => sample.sample_id)).toEqual(['PG_0209', 'PG_0214', 'PG_0309']);
    expect(snapshot.dataset_registry?.datasets.find((dataset) => dataset.id === 'alab_precursor_genome')?.candidateIds).toHaveLength(1035);
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
  });

  it('rejects unavailable surrogate seeds instead of falling back to seed 42', () => {
    const result = resolveCampaign('electrolyte_search', snapshot, 'hig_cost_penalized', { seed: 999 });
    expect(result).toMatchObject({ ok: false, reason: 'RUN_NOT_AVAILABLE' });
  });
});
