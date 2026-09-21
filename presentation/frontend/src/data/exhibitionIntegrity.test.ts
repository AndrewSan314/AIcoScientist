import { describe, expect, it } from 'vitest';
import { readFileSync } from 'node:fs';
import * as THREE from 'three';
import { OptimizationVisualization } from '../components/3d/OptimizationVisualization';
import { DRAKOPOULOS_SCENARIO, WARWICK_SCENARIO } from './exhibitionData';
import { validateScenario } from './validation';

function snapshot(view: OptimizationVisualization) {
  const result: unknown[] = [];
  view.group.traverse(obj => {
    if (obj instanceof THREE.Mesh || obj instanceof THREE.Line) {
      const material = obj.material as THREE.MeshStandardMaterial;
      result.push([obj.visible, obj.position.toArray(), obj.scale.toArray(), material.color?.getHex(), material.opacity, material.emissive?.getHex(), material.emissiveIntensity,
        Array.from(obj.geometry.attributes.position?.array ?? [])]);
    }
  });
  return result;
}

describe('Exhibition scientific integrity', () => {
  for (const scenario of [DRAKOPOULOS_SCENARIO, WARWICK_SCENARIO]) {
    it(`${scenario.id}: hidden outcomes cannot change visible geometry, style or order`, () => {
      expect(validateScenario(scenario).errors).toEqual([]);
      const view = new OptimizationVisualization(scenario);
      for (let step = 0; step <= scenario.replaySteps.length; step++) {
        const revealed = new Set([...scenario.replayInitialIds, ...scenario.replaySteps.slice(0, step).map(s => s.selectedCandidateId)]);
        const altered = structuredClone(scenario);
        altered.candidates.forEach(c => { if (!revealed.has(c.id)) c.revealedTarget.value = 999999; });
        const other = new OptimizationVisualization(altered);
        view.setStep(step); other.setStep(step);
        expect(snapshot(other)).toEqual(snapshot(view));
        const hidden = view.getHitboxes().filter(m => !revealed.has(m.userData.candidateId));
        expect(hidden.every(m => m.position.y === 0.18)).toBe(true);
      }
      view.setStep(0);
      expect(snapshot(view)).toEqual(snapshot(new OptimizationVisualization(scenario)));
    });
  }
  it('matches original recorded seed 11 selections and predictions', () => {
    const cases = [
      [DRAKOPOULOS_SCENARIO, '../../outputs/drakopoulos_rediscovery_v4/trajectories/AICOSCIENTIST_PROCESS_SURROGATE_trajectories.json'],
      [WARWICK_SCENARIO, '../../outputs/warwick_nmc622_calendering/trajectories/aicoscientist_full_process_engine_seed_11.json'],
    ] as const;
    for (const [scenario, path] of cases) {
      const data = JSON.parse(readFileSync(path, 'utf8'));
      const replay = Array.isArray(data) ? data.find(r => r.seed === 11) : data;
      expect(scenario.replayInitialIds).toEqual(replay.initial_candidate_ids);
      scenario.replaySteps.forEach((s, i) => {
        expect([s.selectedCandidateId, s.predictedMean, s.predictedStd, s.acquisitionValue, s.revealedTarget, s.bestSoFar])
          .toEqual(['selected_id', 'predicted_mean', 'predicted_std', 'acquisition_value', 'revealed_target', 'best_so_far'].map(k => replay.steps[i][k]));
      });
    }
  });
});
