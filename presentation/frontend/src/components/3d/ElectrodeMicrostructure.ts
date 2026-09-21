import * as THREE from 'three';
import { MicrostructureSpec } from '../../data/types';

interface ParticleData {
  mesh: THREE.Mesh;
  initialPos: THREE.Vector3;
  compressedPos: THREE.Vector3;
  initialRot: THREE.Euler;
  compressedRot: THREE.Euler;
  radius: number;
}

interface BinderEdge {
  p1Idx: number;
  p2Idx: number;
}

export class ElectrodeMicrostructure {
  public group: THREE.Group;
  private substrateMesh!: THREE.Mesh;
  private substrateMaterial!: THREE.MeshStandardMaterial;
  private particleMaterial!: THREE.MeshStandardMaterial;
  private binderMaterial!: THREE.LineBasicMaterial;
  private binderLines!: THREE.LineSegments;
  private binderEdges: BinderEdge[] = [];
  private particles: ParticleData[] = [];
  private caliperTopLine!: THREE.Line;
  private caliperBottomLine!: THREE.Line;
  private caliperSpine!: THREE.Line;
  private caliperLabelMesh!: THREE.Group;
  private currentProgress: number = 0; // 0 = uncalendered, 1 = fully calendered
  private activeSpec!: MicrostructureSpec;
  private isAutoMorphing: boolean = false;
  private morphDirection: number = 1;
  private morphSpeed: number = 0.4; // cycles per second

  constructor(spec: MicrostructureSpec) {
    this.group = new THREE.Group();
    this.activeSpec = spec;
    this.buildMicrostructure(spec);
  }

  private buildMicrostructure(spec: MicrostructureSpec) {
    // 1. Current Collector Base Substrate Foil (Bottom plate at Y = 0)
    const subGeo = new THREE.BoxGeometry(6.4, 0.16, 6.4);
    this.substrateMaterial = new THREE.MeshStandardMaterial({
      color: new THREE.Color(spec.substrateColorHex),
      metalness: 0.92,
      roughness: 0.22
    });
    this.substrateMesh = new THREE.Mesh(subGeo, this.substrateMaterial);
    this.substrateMesh.position.set(0, 0.08, 0);
    this.substrateMesh.receiveShadow = true;
    this.group.add(this.substrateMesh);

    // Substrate foil edge border trim
    const edgeGeo = new THREE.EdgesGeometry(subGeo);
    const edgeMat = new THREE.LineBasicMaterial({ color: 0x087F8C, transparent: true, opacity: 0.35 });
    const edgeLines = new THREE.LineSegments(edgeGeo, edgeMat);
    edgeLines.position.copy(this.substrateMesh.position);
    this.group.add(edgeLines);

    // 2. Active Material Particles
    const isGraphite = spec.particleMorphology === 'FLAKES_OBLATE';
    
    // Richer PBR material with realistic industrial specular response
    this.particleMaterial = new THREE.MeshStandardMaterial({
      color: isGraphite ? new THREE.Color(0x2D3748) : new THREE.Color(0x4A5568),
      roughness: isGraphite ? 0.55 : 0.42,
      metalness: isGraphite ? 0.40 : 0.30,
      flatShading: false
    });

    const baseGeo = isGraphite
      ? new THREE.DodecahedronGeometry(0.32, 1)
      : new THREE.IcosahedronGeometry(0.28, 2);

    // Deterministic pseudo-random seed generator
    let seed = 12345;
    const rng = () => {
      seed = (seed * 9301 + 49297) % 233280;
      return seed / 233280;
    };

    this.particles = [];

    // Generate 3 layers of particles with realistic porous packing
    const layers = [
      { yMin: 0.35, yMax: 0.85, count: 55 },
      { yMin: 0.85, yMax: 1.55, count: 60 },
      { yMin: 1.55, yMax: 2.35, count: 45 }
    ];

    layers.forEach((layer, layerIdx) => {
      for (let i = 0; i < layer.count; i++) {
        const mesh = new THREE.Mesh(baseGeo, this.particleMaterial);
        mesh.castShadow = true;
        mesh.receiveShadow = true;

        const radiusScale = 0.75 + rng() * 0.5;
        if (isGraphite) {
          // Lamellar flake morphology: flattened along Y, wider along X and Z
          mesh.scale.set(radiusScale * 1.35, radiusScale * 0.45, radiusScale * 1.15);
        } else {
          // Polycrystalline spherical granule with slight irregular facet scaling
          mesh.scale.set(
            radiusScale * (0.9 + rng() * 0.2),
            radiusScale * (0.9 + rng() * 0.2),
            radiusScale * (0.9 + rng() * 0.2)
          );
        }

        // Uncompressed initial coordinate
        const initX = (rng() - 0.5) * 5.0;
        const initY = layer.yMin + rng() * (layer.yMax - layer.yMin);
        const initZ = (rng() - 0.5) * 5.0;
        const initialPos = new THREE.Vector3(initX, initY, initZ);

        // Calibrated calendered coordinate:
        // 1. Vertical compression (-35% for upper layers, -15% for bottom layer)
        const verticalCompressionFactor = 0.65 + (layerIdx === 0 ? 0.20 : 0.0);
        const compY = initY * verticalCompressionFactor;

        // 2. Lateral void rearrangement (particles slide into surrounding pores)
        const lateralJitterX = (rng() - 0.5) * 0.35;
        const lateralJitterZ = (rng() - 0.5) * 0.35;
        const compX = initX * 1.05 + lateralJitterX;
        const compZ = initZ * 1.05 + lateralJitterZ;
        const compressedPos = new THREE.Vector3(compX, compY, compZ);

        // Rotational deformation under nip shear
        const initRot = new THREE.Euler(rng() * Math.PI, rng() * Math.PI, rng() * Math.PI);
        const compRot = new THREE.Euler(
          initRot.x + (rng() - 0.5) * 0.4,
          initRot.y + (rng() - 0.5) * 0.4,
          initRot.z + (isGraphite ? -0.2 : (rng() - 0.5) * 0.3)
        );

        mesh.position.copy(initialPos);
        mesh.rotation.copy(initRot);
        this.group.add(mesh);

        this.particles.push({
          mesh,
          initialPos,
          compressedPos,
          initialRot: initRot,
          compressedRot: compRot,
          radius: radiusScale * 0.3
        });
      }
    });

    // 3. Conductive Binder Domain (CBD) Webbing Network
    this.buildBinderNetwork();

    // 4. Thickness Calipers / Scientific Height Gauge
    this.buildTechnicalCalipers(spec);

    this.setCompressionProgress(this.currentProgress);
  }

  private buildBinderNetwork() {
    this.binderEdges = [];
    // Connect nearest neighbor particle pairs within percolation threshold
    for (let i = 0; i < this.particles.length; i += 2) {
      const p1 = this.particles[i].initialPos;
      for (let j = i + 1; j < Math.min(i + 6, this.particles.length); j++) {
        const p2 = this.particles[j].initialPos;
        if (p1.distanceTo(p2) < 1.15) {
          this.binderEdges.push({ p1Idx: i, p2Idx: j });
        }
      }
    }

    const posArray = new Float32Array(this.binderEdges.length * 6);
    let ptr = 0;
    for (const edge of this.binderEdges) {
      const p1 = this.particles[edge.p1Idx].initialPos;
      const p2 = this.particles[edge.p2Idx].initialPos;
      posArray[ptr++] = p1.x;
      posArray[ptr++] = p1.y;
      posArray[ptr++] = p1.z;
      posArray[ptr++] = p2.x;
      posArray[ptr++] = p2.y;
      posArray[ptr++] = p2.z;
    }

    const binderGeo = new THREE.BufferGeometry();
    const posAttr = new THREE.Float32BufferAttribute(posArray, 3);
    posAttr.setUsage(THREE.DynamicDrawUsage);
    binderGeo.setAttribute('position', posAttr);

    this.binderMaterial = new THREE.LineBasicMaterial({
      color: 0x087F8C,
      transparent: true,
      opacity: 0.45,
      linewidth: 1
    });
    this.binderLines = new THREE.LineSegments(binderGeo, this.binderMaterial);
    this.group.add(this.binderLines);
  }

  private buildTechnicalCalipers(spec: MicrostructureSpec) {
    this.caliperLabelMesh = new THREE.Group();
    const lineMat = new THREE.LineBasicMaterial({ color: 0xF59E42 });
    const lineDimMat = new THREE.LineBasicMaterial({ color: 0x087F8C, transparent: true, opacity: 0.6 });

    // Place the gauge on the right side of the sample (X = 3.1, Z = 0) where the view is completely open
    const gaugeX = 3.1;
    const gaugeZ = 0;

    // Bottom caliper line at substrate foil
    const botPoints = [
      new THREE.Vector3(gaugeX - 0.2, 0.16, gaugeZ),
      new THREE.Vector3(gaugeX + 0.35, 0.16, gaugeZ)
    ];
    const botGeo = new THREE.BufferGeometry().setFromPoints(botPoints);
    this.caliperBottomLine = new THREE.Line(botGeo, lineMat);
    this.caliperLabelMesh.add(this.caliperBottomLine);

    // Initial reference line (uncalendered 50-52 µm height mark)
    const initPoints = [
      new THREE.Vector3(gaugeX, 2.4, gaugeZ),
      new THREE.Vector3(gaugeX + 0.25, 2.4, gaugeZ)
    ];
    const initGeo = new THREE.BufferGeometry().setFromPoints(initPoints);
    const initLine = new THREE.Line(initGeo, lineDimMat);
    this.caliperLabelMesh.add(initLine);

    // Calendered target reference line (37-39 µm target height mark)
    const targetPoints = [
      new THREE.Vector3(gaugeX, 1.6, gaugeZ),
      new THREE.Vector3(gaugeX + 0.25, 1.6, gaugeZ)
    ];
    const targetGeo = new THREE.BufferGeometry().setFromPoints(targetPoints);
    const targetLine = new THREE.Line(targetGeo, lineDimMat);
    this.caliperLabelMesh.add(targetLine);

    // Vertical height caliper spine (from foil Y = 0.16 to current active film top)
    const spinePoints = [
      new THREE.Vector3(gaugeX, 0.16, gaugeZ),
      new THREE.Vector3(gaugeX, 2.4, gaugeZ)
    ];
    const spineGeo = new THREE.BufferGeometry().setFromPoints(spinePoints);
    const spineAttr = spineGeo.attributes.position as THREE.BufferAttribute;
    spineAttr.setUsage(THREE.DynamicDrawUsage);
    this.caliperSpine = new THREE.Line(spineGeo, lineMat);
    this.caliperLabelMesh.add(this.caliperSpine);

    // Dynamic top caliper pointer line indicating active film height
    const topPoints = [
      new THREE.Vector3(gaugeX - 0.25, 2.4, gaugeZ),
      new THREE.Vector3(gaugeX + 0.45, 2.4, gaugeZ)
    ];
    const topGeo = new THREE.BufferGeometry().setFromPoints(topPoints);
    const topAttr = topGeo.attributes.position as THREE.BufferAttribute;
    topAttr.setUsage(THREE.DynamicDrawUsage);
    this.caliperTopLine = new THREE.Line(topGeo, lineMat);
    this.caliperLabelMesh.add(this.caliperTopLine);

    this.group.add(this.caliperLabelMesh);
  }

  public setCompressionProgress(progress: number) {
    this.currentProgress = Math.max(0, Math.min(1, progress));

    // 1. Lerp each particle position and orientation
    for (const p of this.particles) {
      p.mesh.position.lerpVectors(p.initialPos, p.compressedPos, this.currentProgress);

      // Lerp rotation angles
      p.mesh.rotation.x = THREE.MathUtils.lerp(p.initialRot.x, p.compressedRot.x, this.currentProgress);
      p.mesh.rotation.y = THREE.MathUtils.lerp(p.initialRot.y, p.compressedRot.y, this.currentProgress);
      p.mesh.rotation.z = THREE.MathUtils.lerp(p.initialRot.z, p.compressedRot.z, this.currentProgress);
    }

    // 2. Synchronously update conductive binder lines with moving particles
    if (this.binderLines && this.binderEdges.length > 0) {
      const posAttr = this.binderLines.geometry.attributes.position as THREE.BufferAttribute;
      const arr = posAttr.array as Float32Array;
      let ptr = 0;
      for (const edge of this.binderEdges) {
        const p1 = this.particles[edge.p1Idx].mesh.position;
        const p2 = this.particles[edge.p2Idx].mesh.position;
        arr[ptr++] = p1.x;
        arr[ptr++] = p1.y;
        arr[ptr++] = p1.z;
        arr[ptr++] = p2.x;
        arr[ptr++] = p2.y;
        arr[ptr++] = p2.z;
      }
      posAttr.needsUpdate = true;
    }

    // 3. Update thickness caliper top mark and vertical spine
    const initH = 2.4;
    const compH = 1.6;
    const currentH = THREE.MathUtils.lerp(initH, compH, this.currentProgress);

    if (this.caliperTopLine) {
      const posAttr = this.caliperTopLine.geometry.attributes.position as THREE.BufferAttribute;
      posAttr.setY(0, currentH);
      posAttr.setY(1, currentH);
      posAttr.needsUpdate = true;
    }

    if (this.caliperSpine) {
      const spineAttr = this.caliperSpine.geometry.attributes.position as THREE.BufferAttribute;
      spineAttr.setY(1, currentH);
      spineAttr.needsUpdate = true;
    }
  }

  public setAutoMorph(auto: boolean) {
    this.isAutoMorphing = auto;
  }

  public toggleAutoMorph(): boolean {
    this.isAutoMorphing = !this.isAutoMorphing;
    return this.isAutoMorphing;
  }

  public resetOrientation() {
    this.group.rotation.set(0.1, 0.4, 0);
  }

  public updateScenarioSpec(spec: MicrostructureSpec, preservedCompression?: number) {
    this.activeSpec = spec;
    if (preservedCompression !== undefined) {
      this.currentProgress = preservedCompression;
    }

    // Dispose old geometries and materials to avoid WebGL memory leak
    while (this.group.children.length > 0) {
      const obj = this.group.children[0];
      this.group.remove(obj);
      if (obj instanceof THREE.Mesh || obj instanceof THREE.Line || obj instanceof THREE.LineSegments) {
        obj.geometry.dispose();
      }
    }

    this.buildMicrostructure(spec);
    this.setCompressionProgress(this.currentProgress);
  }

  public update(delta: number) {
    if (this.isAutoMorphing) {
      let nextProg = this.currentProgress + this.morphDirection * delta * this.morphSpeed;
      if (nextProg >= 1.0) {
        nextProg = 1.0;
        this.morphDirection = -1;
      } else if (nextProg <= 0.0) {
        nextProg = 0.0;
        this.morphDirection = 1;
      }
      this.setCompressionProgress(nextProg);
    }
  }

  public getProgress(): number {
    return this.currentProgress;
  }

  public getAutoMorph(): boolean {
    return this.isAutoMorphing;
  }
}
