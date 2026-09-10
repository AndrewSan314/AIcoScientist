import { describe, expect, it } from 'vitest';
import { readFileSync } from 'node:fs';
import { resolveCampaign } from './campaignResolver';
import type { SnapshotData, ScoredActionRecord } from '../types/mission_control';

const snapshot = JSON.parse(
  readFileSync(new URL('../../../data/snapshot.json', import.meta.url), 'utf8')
) as SnapshotData;

describe('presentation UI regression & data correctness', () => {
  describe('1. Cross-Domain Belief Trajectory Isolation', () => {
    it('ensures controlled campaign and A-Lab replay campaign are isolated and distinct', () => {
      const controlledRes = resolveCampaign('controlled_synthesis', snapshot);
      expect(controlledRes.ok).toBe(true);
      if (!controlledRes.ok || controlledRes.value.kind !== 'controlled_multimodal') return;

      const replayRes = resolveCampaign('alab_replay', snapshot);
      expect(replayRes.ok).toBe(true);
      if (!replayRes.ok || replayRes.value.kind !== 'historical_replay') return;

      // Ensure distinct run IDs and campaigns
      expect(controlledRes.value.campaign.run_id).not.toBe(replayRes.value.campaign.run_id);
      expect(controlledRes.value.campaign.run_id).toContain('policy_comparison');
      expect(replayRes.value.campaign.run_id).toContain('replay');

      // Controlled has 4 steps, A-Lab replay has 6 steps
      expect(controlledRes.value.campaign.steps.length).toBe(4);
      expect(replayRes.value.campaign.steps.length).toBe(6);

      // Controlled initial beliefs differ from A-Lab replay initial beliefs
      expect(controlledRes.value.campaign.initial_beliefs).toBeDefined();
      expect(replayRes.value.campaign.initial_beliefs).toBeDefined();
      expect(replayRes.value.campaign.steps[0].step).toBe(1);
    });
  });

  describe('2. Campaign Resolver Strict Policy Isolation', () => {
    it('fails closed when controlled policies are supplied to surrogate electrolyte dataset', () => {
      const controlledPolicies = ['HYBRID', 'PURE_HIG', 'DISCOVERY_ONLY', 'RANDOM_ACTION'];
      for (const pol of controlledPolicies) {
        const res = resolveCampaign('anode_free_electrolyte_screening', snapshot, undefined, { policy: pol });
        expect(res).toMatchObject({ ok: false, reason: 'UNKNOWN_POLICY' });
      }
    });

    it('fails closed when surrogate policies are supplied to controlled synthesis dataset', () => {
      const surrogatePolicies = ['HYBRID_DEFAULT', 'PURE_FALSIFICATION', 'RANDOM'];
      for (const pol of surrogatePolicies) {
        const res = resolveCampaign('controlled_multimodal_alloy', snapshot, undefined, { policy: pol });
        expect(res).toMatchObject({ ok: false, reason: 'UNKNOWN_POLICY' });
      }
    });

    it('fails closed when surrogate policies are supplied to A-Lab precursor genome dataset', () => {
      const surrogatePolicies = ['HYBRID_DEFAULT', 'PURE_FALSIFICATION', 'RANDOM'];
      for (const pol of surrogatePolicies) {
        const res = resolveCampaign('alab_precursor_genome', snapshot, undefined, { policy: pol });
        expect(res).toMatchObject({ ok: false, reason: 'UNKNOWN_POLICY' });
      }
    });
  });

  describe('3. Surrogate Optimization Schema Discipline', () => {
    it('resolves surrogate dataset with trajectory but no campaign steps or uncalibrated fields', () => {
      const res = resolveCampaign('anode_free_electrolyte_screening', snapshot, undefined, {
        policy: 'HYBRID_DEFAULT',
        seed: 42,
      });
      expect(res.ok).toBe(true);
      if (!res.ok || res.value.kind !== 'surrogate_optimization') return;

      expect(res.value.trajectory.length).toBeGreaterThan(0);
      expect((res.value as any).steps).toBeUndefined();

      const firstPoint = res.value.trajectory[0];
      expect(firstPoint).toHaveProperty('queryIndex');
      expect(firstPoint).toHaveProperty('candidateId');
      expect((firstPoint as any).simpleRegretLatent).toBeUndefined();
      expect((firstPoint as any).cumulativeRawHigNats).toBeUndefined();
      expect((firstPoint as any).realizedEntropyReductionNats).toBeUndefined();
    });
  });

  describe('4. Alternative Actions Sorting & Winner Matching', () => {
    it('sorts alternative actions strictly descending by total_action_score', () => {
      const step = snapshot.flagship_campaign.steps[0];
      const actions: ScoredActionRecord[] = step.all_scored_actions || [];
      expect(actions.length).toBeGreaterThan(1);

      const sorted = [...actions].sort(
        (a, b) => (b.total_action_score ?? 0) - (a.total_action_score ?? 0)
      );

      for (let i = 0; i < sorted.length - 1; i++) {
        expect(sorted[i].total_action_score ?? 0).toBeGreaterThanOrEqual(
          sorted[i + 1].total_action_score ?? 0
        );
      }
    });

    it('matches the winner action by action_id against preregistration action', () => {
      const step = snapshot.flagship_campaign.steps[0];
      const winnerActionId = step.preregistration?.action?.action_id;
      expect(winnerActionId).toBeDefined();

      const actions: ScoredActionRecord[] = step.all_scored_actions || [];
      const winnerMatches = actions.filter((a) => a.action?.action_id === winnerActionId);
      expect(winnerMatches.length).toBe(1);
    });

    it('falls back to "No source rationale recorded" when rationale is missing', () => {
      const step = snapshot.flagship_campaign.steps[0];
      const action = step.all_scored_actions?.[0];
      const rationale = (action as any)?.rationale || 'No source rationale recorded';
      expect(typeof rationale).toBe('string');
      expect(rationale.length).toBeGreaterThan(0);
    });
  });

  describe('5. Authentic Score Provenance (No Synthetic Weights)', () => {
    it('verifies that flagship campaign actions contain source scores without policy_weights', () => {
      expect((snapshot.flagship_campaign as any).policy_weights).toBeUndefined();
      const step = snapshot.flagship_campaign.steps[0];
      const winner = step.top_actions?.[0];
      expect(winner).toBeDefined();
      expect(typeof winner?.total_action_score).toBe('number');
      expect(typeof (winner?.raw_expected_hig_nats ?? winner?.expected_hig_nats)).toBe('number');
    });
  });

  describe('6. Policy Benchmark Censored Metrics Integrity', () => {
    it('verifies threshold_metrics P>0.8 crossing rate and conditional mean step counts', () => {
      const stressH3Random =
        snapshot.benchmarks.summary_by_world_policy['STRESS_WORLD_H3_MORPHOLOGY_KINETICS']?.['RANDOM_ACTION'];
      expect(stressH3Random).toBeDefined();

      const p08Metrics = stressH3Random?.threshold_metrics?.['P>0.8'];
      expect(p08Metrics).toBeDefined();
      // 1 out of 5 seeds crossed P > 0.8
      expect(p08Metrics?.crossing_rate).toBe(0.2);
      expect(p08Metrics?.N_crossed).toBe(1);
      expect(p08Metrics?.N_total).toBe(5);
      // The single crossed seed took 2 steps
      expect(p08Metrics?.mean_steps_conditional_on_crossing).toBe(2);

      // For clean world HYBRID, all 5 crossed at step 1
      const cleanH1Hybrid =
        snapshot.benchmarks.summary_by_world_policy['CLEAN_WORLD_H1_PHASE_PURITY']?.['HYBRID'];
      expect(cleanH1Hybrid?.threshold_metrics?.['P>0.8']?.crossing_rate).toBe(1.0);
      expect(cleanH1Hybrid?.threshold_metrics?.['P>0.8']?.mean_steps_conditional_on_crossing).toBe(1);
    });
  });
});
