import * as THREE from 'three';
import { mergeGeometries } from 'three/addons/utils/BufferGeometryUtils.js';
import type { MicrostructureSpec } from '../../data/types';

interface ParticleData {
  mesh: THREE.Mesh;
  initialPos: THREE.Vector3;
  compressedPos: THREE.Vector3;
  initialRot: THREE.Quaternion;
  compressedRot: THREE.Quaternion;
}

/** Authored illustrative packing, not tomography or a pore-scale simulation. */
export class ElectrodeMicrostructure {
  public group = new THREE.Group();
  private particles: ParticleData[] = [];
  private binderLines!: THREE.LineSegments;
  private binderEdges: [number, number][] = [];
  private currentProgress = 0;
  private isAutoMorphing = false;
  private morphDirection = 1;
  private morphSpeed = 0.25;

  constructor(spec: MicrostructureSpec) {
    this.buildMicrostructure(spec);
  }

  private buildMicrostructure(spec: MicrostructureSpec) {
    const graphite = spec.particleMorphology === 'FLAKES_OBLATE';
    const collector = new THREE.Mesh(
      new THREE.BoxGeometry(6.2, 0.085, 4.1),
      new THREE.MeshStandardMaterial({ color: spec.substrateColorHex, metalness: 0.8, roughness: 0.32 }),
    );
    collector.name = 'Current collector — fixed foil';
    collector.position.y = 0.0425;
    collector.receiveShadow = true;
    collector.castShadow = true;
    this.group.add(collector);

    // Four thin irregular lamellae make each graphite particle, with real edge relief.
    const geometries: THREE.BufferGeometry[] = [];
    for (let variant = 0; variant < 5; variant++) {
      if (graphite) {
        const wafers: THREE.BufferGeometry[] = [];
        for (let wafer = 0; wafer < 4; wafer++) {
          const shape = new THREE.Shape();
          for (let edge = 0; edge < 9; edge++) {
            const a = edge / 9 * Math.PI * 2;
            const r = 0.88 + 0.09 * Math.sin(edge * 2.3 + variant + wafer * 0.7);
            const x = Math.cos(a) * 0.34 * r;
            const z = Math.sin(a) * 0.235 * r;
            if (edge === 0) shape.moveTo(x, z); else shape.lineTo(x, z);
          }
          shape.closePath();
          const g = new THREE.ExtrudeGeometry(shape, {
            depth: 0.015, bevelEnabled: true, bevelSegments: 2,
            steps: 1, bevelSize: 0.006, bevelThickness: 0.004,
          });
          g.rotateX(-Math.PI / 2);
          g.translate((wafer % 2) * 0.008, wafer * 0.022 - 0.039, 0);
          wafers.push(g);
        }
        geometries.push(mergeGeometries(wafers));
        wafers.forEach(g => g.dispose());
      } else {
        const g = new THREE.SphereGeometry(0.285, 32, 24);
        const positions = g.getAttribute('position');
        for (let v = 0; v < positions.count; v++) {
          const x = positions.getX(v), y = positions.getY(v), z = positions.getZ(v);
          // Spatial harmonics are seamless at the UV seam and bounded inside the envelope.
          const grain = Math.sin(x * 93 + variant) * Math.sin(y * 87) * Math.sin(z * 79);
          const swell = Math.sin(x * 21 + variant) * Math.cos(z * 19) * Math.sin(y * 17);
          const r = 0.92 + grain * 0.035 + swell * 0.025;
          positions.setXYZ(v, x * r, y * r, z * r);
        }
        g.computeVertexNormals();
        geometries.push(g);
      }
    }
    const materials = Array.from({ length: 5 }, (_, i) => new THREE.MeshStandardMaterial({
      color: graphite ? new THREE.Color().setHSL(0.59, 0.08, 0.16 + i * 0.018) : new THREE.Color().setHSL(0.59, .32, .095 + i * .012),
      roughness: graphite ? 0.48 : 0.56,
      metalness: graphite ? 0.24 : 0.1,
    }));
    this.particles = [];
    const layers = graphite ? 5 : 3;
    for (let layer = 0; layer < layers; layer++) {
      for (let row = 0; row < 5; row++) {
        for (let col = 0; col < 7; col++) {
          const id = this.particles.length;
          const phase = col * 2.13 + row * 3.71 + layer * 1.61;
          const x = (col - 3) * 0.82 + Math.sin(phase) * 0.025;
          const z = (row - 2) * 0.74 + Math.cos(phase) * 0.025;
          const initialPos = new THREE.Vector3(x, graphite ? 0.29 + layer * 0.54 : 0.39 + layer * 0.96, z);
          const compressedPos = new THREE.Vector3(x + Math.sin(phase) * 0.018,
            graphite ? 0.22 + layer * 0.28 : 0.375 + layer * 0.59, z + Math.cos(phase) * 0.018);
          const initialRot = new THREE.Quaternion().setFromEuler(new THREE.Euler(
            graphite ? Math.sin(phase) * 0.11 : 0, Math.sin(phase) * 0.2, graphite ? Math.cos(phase) * 0.1 : 0,
          ));
          const compressedRot = new THREE.Quaternion().setFromEuler(new THREE.Euler(0, Math.sin(phase) * 0.2, 0));
          const mesh = new THREE.Mesh(geometries[id % 5], materials[id % 5]);
          mesh.name = graphite ? 'Graphite lamellar particle' : 'Illustrative NMC secondary granule';
          mesh.userData.activeParticle = true;
          mesh.castShadow = true;
          mesh.receiveShadow = true;
          this.group.add(mesh);
          this.particles.push({ mesh, initialPos, compressedPos, initialRot, compressedRot });
        }
      }
    }
    this.buildBinderNetwork();
    this.setCompressionProgress(this.currentProgress);
  }

  private buildBinderNetwork() {
    this.binderEdges = [];
    // Sparse dark contact bridges; actual pore space remains open, not a glowing lattice.
    for (let i = 0; i < this.particles.length; i++) {
      if (i % 7 < 6 && i % 3 === 0) this.binderEdges.push([i, i + 1]);
      if (i + 35 < this.particles.length && i % 4 === 0) this.binderEdges.push([i, i + 35]);
    }
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array(this.binderEdges.length * 6), 3).setUsage(THREE.DynamicDrawUsage));
    this.binderLines = new THREE.LineSegments(geometry, new THREE.LineBasicMaterial({ color: 0x303b3c, transparent: true, opacity: 0.42 }));
    this.binderLines.name = 'Illustrative conductive binder bridges';
    this.binderLines.frustumCulled = false;
    this.group.add(this.binderLines);
  }

  public setCompressionProgress(progress: number) {
    this.currentProgress = Number.isFinite(progress) ? THREE.MathUtils.clamp(progress, 0, 1) : 0;
    const t = THREE.MathUtils.smoothstep(this.currentProgress, 0, 1);
    // Fixed cell envelopes stay disjoint at every intermediate t; no particle is scaled.
    for (const p of this.particles) {
      p.mesh.position.lerpVectors(p.initialPos, p.compressedPos, t);
      p.mesh.quaternion.slerpQuaternions(p.initialRot, p.compressedRot, t);
    }
    const positions = this.binderLines.geometry.getAttribute('position') as THREE.BufferAttribute;
    this.binderEdges.forEach(([a, b], index) => {
      const p = this.particles[a].mesh.position, q = this.particles[b].mesh.position;
      positions.setXYZ(index * 2, p.x, p.y, p.z);
      positions.setXYZ(index * 2 + 1, q.x, q.y, q.z);
    });
    positions.needsUpdate = true;
  }

  public setAutoMorph(auto: boolean) { this.isAutoMorphing = auto; }
  public toggleAutoMorph(): boolean { this.isAutoMorphing = !this.isAutoMorphing; return this.isAutoMorphing; }
  public resetOrientation() { this.group.rotation.set(0.1, 0.4, 0); }

  public updateScenarioSpec(spec: MicrostructureSpec, preservedCompression?: number) {
    if (preservedCompression !== undefined) this.currentProgress = preservedCompression;
    const geometries = new Set<THREE.BufferGeometry>();
    const materials = new Set<THREE.Material>();
    this.group.traverse(obj => {
      if (obj instanceof THREE.Mesh || obj instanceof THREE.Line) {
        geometries.add(obj.geometry);
        (Array.isArray(obj.material) ? obj.material : [obj.material]).forEach(m => materials.add(m));
      }
    });
    geometries.forEach(g => g.dispose());
    materials.forEach(m => m.dispose());
    this.group.clear();
    this.buildMicrostructure(spec);
  }

  public update(delta: number) {
    if (!this.isAutoMorphing || !Number.isFinite(delta) || delta <= 0) return;
    const phase = this.morphDirection === 1 ? this.currentProgress : 2 - this.currentProgress;
    const next = (phase + delta * this.morphSpeed) % 2;
    this.morphDirection = next < 1 ? 1 : -1;
    this.setCompressionProgress(next <= 1 ? next : 2 - next);
  }

  public getProgress(): number { return this.currentProgress; }
  public getAutoMorph(): boolean { return this.isAutoMorphing; }
}
