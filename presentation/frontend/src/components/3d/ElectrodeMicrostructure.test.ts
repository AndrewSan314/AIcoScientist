import { expect, test } from 'vitest';
import * as THREE from 'three';
import { ElectrodeMicrostructure } from './ElectrodeMicrostructure';
import type { MicrostructureSpec } from '../../data/types';

test('packing remains separated, foil stays fixed and morph is reversible', () => {
  for (const particleMorphology of ['FLAKES_OBLATE', 'POLYCRYSTALLINE_SPHERICAL'] as const) {
    const spec = { particleMorphology, substrateColorHex: '#ad784e' } as MicrostructureSpec;
    const model = new ElectrodeMicrostructure(spec);
    const collector = model.group.children[0];
    const particles = model.group.children.filter(p => p.userData.activeParticle) as THREE.Mesh[];
    const state = () => particles.map(p => [...p.position.toArray(), ...p.quaternion.toArray()]);
    const initial = state();
    let previousHeight = Infinity;
    // Includes 0 / 25 / 50 / 75 / 100%, with dense intermediate checks.
    for (let step = 0; step <= 100; step++) {
      model.setCompressionProgress(step / 100);
      model.group.updateMatrixWorld(true);
      const bounds = particles.map(p => new THREE.Box3().setFromObject(p));
      for (let i = 0; i < bounds.length; i++) {
        expect(bounds[i].min.y).toBeGreaterThan(0.085);
        for (let j = i + 1; j < bounds.length; j++) {
          if (bounds[i].intersectsBox(bounds[j])) throw new Error(`${particleMorphology} overlap ${i}/${j} at ${step}%`);
        }
      }
      const height = Math.max(...bounds.map(b => b.max.y));
      expect(height).toBeLessThanOrEqual(previousHeight + 1e-10);
      previousHeight = height;
      expect(collector.position.toArray()).toEqual([0, 0.0425, 0]);
      expect(collector.scale.toArray()).toEqual([1, 1, 1]);
    }
    const end = state();
    for (let loop = 0; loop < 3; loop++) {
      for (let step = 100; step >= 0; step--) model.setCompressionProgress(step / 100);
      expect(state()).toEqual(initial);
      model.setCompressionProgress(1);
      expect(state()).toEqual(end);
    }
    model.setAutoMorph(true);
    model.update(24);
    expect(model.getProgress()).toBe(1);
    model.updateScenarioSpec(spec, 0.5);
    expect(model.getProgress()).toBe(0.5);
    expect(model.group.children.filter(p => p.userData.activeParticle)).toHaveLength(particles.length);
  }
});
