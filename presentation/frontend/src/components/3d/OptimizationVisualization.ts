import * as THREE from 'three';
import { ExhibitionScenario, CandidateRecord, ReplayStep } from '../../data/types';

export interface CandidateClickCallback {
  (candidateId: string): void;
}

export class OptimizationVisualization {
  public group: THREE.Group;
  private scenario: ExhibitionScenario;
  private candidateMeshes: Map<string, THREE.Mesh> = new Map();
  private candidateHitboxes: THREE.Mesh[] = [];
  private candidateStems: Map<string, THREE.Line> = new Map();
  private candidateFootprints: Map<string, THREE.Mesh> = new Map();
  private trajectoryGroup!: THREE.Group;
  private trajectoryTubeMesh: THREE.Mesh | null = null;
  private photonParticles: THREE.Mesh[] = [];
  private trajectoryCurve: THREE.CatmullRomCurve3 | null = null;
  private selectionBeacon!: THREE.Group;
  private inspectBeacon!: THREE.Group;
  private surrogateMesh: THREE.Mesh | null = null;
  private wireframeMesh: THREE.Mesh | null = null;
  private floorHeatmapMesh: THREE.Mesh | null = null;
  private coordinateCageGroup!: THREE.Group;
  private scanPlaneMesh: THREE.Mesh | null = null;
  private currentStepIndex: number = 0;
  private selectedCandidateId?: string;
  private onCandidateClick?: CandidateClickCallback;
  private isOrbiting: boolean = false;

  constructor(scenario: ExhibitionScenario, onCandidateClick?: CandidateClickCallback) {
    this.group = new THREE.Group();
    this.scenario = scenario;
    this.onCandidateClick = onCandidateClick;
    this.buildCoordinateCage();
    this.buildResponseSurface();
    this.buildCandidatePoints();
    this.buildTrajectoryPath();
    this.buildSelectionBeacon();
    this.buildInspectBeacon();
    this.setStep(0);
  }

  public setOrbit(enabled: boolean) {
    this.isOrbiting = enabled;
  }

  public setOrbiting(enabled: boolean) {
    this.isOrbiting = enabled;
  }

  public toggleOrbit(): boolean {
    this.isOrbiting = !this.isOrbiting;
    return this.isOrbiting;
  }

  public getIsOrbiting(): boolean {
    return this.isOrbiting;
  }

  private mapCoords(cand: CandidateRecord, revealed = false): THREE.Vector3 {
    const isWarwick = this.scenario.id === 'warwick_nmc622_calendering';

    if (isWarwick) {
      // Temperature: 85 -> -2.8, 120 -> 0.0, 145 -> +2.8
      const temp = Number(cand.controls.roll_temperature_c ?? 100);
      const normTemp = (temp - 85) / (145 - 85);
      const worldX = (normTemp - 0.5) * 5.8;

      // Target density: 2.70 -> -1.2, 2.95 -> 0.0, 3.20 -> +1.2
      const density = Number(cand.controls.target_density_g_cm3 ?? 2.95);
      const normDensity = (density - 2.70) / (3.20 - 2.70);
      const densityZ = (normDensity - 0.5) * 2.6;

      // Separate Loading Regimes along Z axis: Low (-1.9) vs High (+1.9)
      const isHighLoading = cand.controls.loading_regime === 'HIGH';
      const worldZ = densityZ + (isHighLoading ? 1.9 : -1.9);

      // Y: Target Rate ratio (0.09 to 0.82)
      const yVal = revealed ? cand.revealedTarget.value : 0;
      const normY = Math.max(0, Math.min(1, (yVal - 0.05) / (0.85 - 0.05)));
      const worldY = revealed ? 0.6 + normY * 3.2 : 0.18;

      return new THREE.Vector3(worldX, worldY, worldZ);
    } else {
      // Drakopoulos: Speed: 0.1 -> -2.6, 0.2 -> 0.0, 0.4 -> +2.6
      const speed = Number(cand.controls.coating_speed_m_per_min ?? 0.2);
      const normSpeed = (speed - 0.1) / (0.4 - 0.1);
      const worldX = (normSpeed - 0.5) * 5.6;

      // Coating gap: 100 -> -2.4, 150 -> 0.0, 200 -> +2.4
      const gap = Number(cand.controls.coating_gap_um ?? 150);
      const normGap = (gap - 100) / (200 - 100);
      let worldZ = (normGap - 0.5) * 5.2;

      if (cand.controls.calendering_applied) {
        worldZ += 0.35;
      }

      // Y: D30 Specific capacity (40 to 415 mAh/g)
      const yVal = revealed ? cand.revealedTarget.value : 0;
      const normY = Math.max(0, Math.min(1, (yVal - 30) / (420 - 30)));
      const worldY = revealed ? 0.6 + normY * 3.2 : 0.18;

      return new THREE.Vector3(worldX, worldY, worldZ);
    }
  }

  private createTextSprite(text: string, subtext = '', color = '#142A35', bgColor = 'rgba(255,255,255,0.92)'): THREE.Sprite {
    if (typeof document === 'undefined') {
      const mat = new THREE.SpriteMaterial({ transparent: true, opacity: 0 });
      const sprite = new THREE.Sprite(mat);
      sprite.scale.set(1.8, 0.56, 1);
      sprite.userData.rawText = `${text} ${subtext || ''}`;
      return sprite;
    }
    const canvas = document.createElement('canvas');
    canvas.width = 512;
    canvas.height = 160;
    const ctx = canvas.getContext('2d')!;

    // Rounded background
    ctx.fillStyle = bgColor;
    ctx.beginPath();
    ctx.roundRect(8, 8, 496, 144, 20);
    ctx.fill();
    ctx.strokeStyle = '#CBD5E1';
    ctx.lineWidth = 4;
    ctx.stroke();

    // Primary text
    ctx.fillStyle = color;
    ctx.font = 'bold 36px system-ui, sans-serif';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(text, 256, subtext ? 62 : 80);

    if (subtext) {
      ctx.fillStyle = '#64748B';
      ctx.font = '24px monospace';
      ctx.fillText(subtext, 256, 112);
    }

    const tex = new THREE.CanvasTexture(canvas);
    tex.colorSpace = THREE.SRGBColorSpace;
    const mat = new THREE.SpriteMaterial({ map: tex, transparent: true, depthTest: false });
    const sprite = new THREE.Sprite(mat);
    sprite.scale.set(1.8, 0.56, 1);
    sprite.renderOrder = 30;
    sprite.userData.rawText = `${text} ${subtext || ''}`;
    return sprite;
  }

  private buildCoordinateCage() {
    this.coordinateCageGroup = new THREE.Group();
    const isWarwick = this.scenario.id === 'warwick_nmc622_calendering';

    // 1. Floor Reference Grid with primary & secondary lines
    const floorGrid = new THREE.GridHelper(8.0, 16, 0x087F8C, 0xE2E8F0);
    floorGrid.position.y = 0.02;
    this.coordinateCageGroup.add(floorGrid);

    // 2. Back Wall Grid (X-Y datum elevation)
    const backGrid = new THREE.GridHelper(8.0, 8, 0x142A35, 0xF1F5F9);
    backGrid.rotation.x = Math.PI / 2;
    backGrid.position.set(0, 2.0, -3.4);
    this.coordinateCageGroup.add(backGrid);

    // 3. Side Wall Grid (Z-Y datum elevation)
    const sideGrid = new THREE.GridHelper(8.0, 8, 0x142A35, 0xF1F5F9);
    sideGrid.rotation.z = Math.PI / 2;
    sideGrid.position.set(-3.6, 2.0, 0);
    this.coordinateCageGroup.add(sideGrid);

    // 4. Bounding Cage Accent Rails
    const railMat = new THREE.LineBasicMaterial({ color: 0x94A3B8, transparent: true, opacity: 0.6 });
    const boxGeo = new THREE.BoxGeometry(7.2, 4.0, 6.8);
    const wireGeo = new THREE.EdgesGeometry(boxGeo);
    const boxWires = new THREE.LineSegments(wireGeo, railMat);
    boxWires.position.set(0, 2.0, 0);
    this.coordinateCageGroup.add(boxWires);

    // 5. 3D Axis Title Sprites
    const xTitle = this.createTextSprite(
      isWarwick ? 'X: Roll Temperature' : 'X: Coating Speed',
      isWarwick ? '85°C → 145°C' : '0.10 → 0.40 m/min',
      '#087F8C'
    );
    xTitle.position.set(0, -0.35, 3.8);
    this.coordinateCageGroup.add(xTitle);

    const zTitle = this.createTextSprite(
      isWarwick ? 'Z: Target Density / Loading' : 'Z: Coating Gap',
      isWarwick ? '2.70 → 3.20 g/cm³ · Low / High' : '100 → 200 µm · Calendered',
      '#142A35'
    );
    zTitle.position.set(4.2, -0.35, 0);
    this.coordinateCageGroup.add(zTitle);

    const yTitle = this.createTextSprite(
      isWarwick ? 'Y: Rate Capability (5C/0.2C)' : 'Y: D30 Capacity (mAh/g)',
      'Revealed Objective Performance',
      '#F59E42'
    );
    yTitle.position.set(-4.2, 3.8, 0);
    this.coordinateCageGroup.add(yTitle);

    // 6. Scientific Boundary Header Placard
    const headerSprite = this.createTextSprite(
      'ILLUSTRATIVE PARAMETER LANDSCAPE',
      'RBF Surface Interpolation · Non-Quantitative Illustration',
      '#087F8C',
      'rgba(244, 247, 247, 0.95)'
    );
    headerSprite.position.set(0, 4.45, -3.4);
    headerSprite.scale.set(2.4, 0.75, 1);
    this.coordinateCageGroup.add(headerSprite);

    this.group.add(this.coordinateCageGroup);
  }

  private buildResponseSurface() {
    // 1. Floor Field Canvas Texture (Illustrative Evaluation Field - Non-Quantitative)
    if (typeof document !== 'undefined') {
      const heatCanvas = document.createElement('canvas');
      heatCanvas.width = 512;
      heatCanvas.height = 512;
      const hCtx = heatCanvas.getContext('2d')!;

      // Clean base gradient
      const baseGrad = hCtx.createRadialGradient(256, 256, 40, 256, 256, 250);
      baseGrad.addColorStop(0, 'rgba(245, 158, 66, 0.22)');
      baseGrad.addColorStop(0.45, 'rgba(8, 127, 140, 0.15)');
      baseGrad.addColorStop(0.85, 'rgba(56, 189, 248, 0.08)');
      baseGrad.addColorStop(1, 'rgba(255, 255, 255, 0)');
      hCtx.fillStyle = baseGrad;
      hCtx.fillRect(0, 0, 512, 512);

      // Heatmap rings / acquisition isolines
      hCtx.strokeStyle = 'rgba(8, 127, 140, 0.25)';
      hCtx.lineWidth = 2;
      for (let r = 50; r <= 240; r += 35) {
        hCtx.beginPath();
        hCtx.arc(256, 256, r, 0, Math.PI * 2);
        hCtx.stroke();
      }

      const heatTex = new THREE.CanvasTexture(heatCanvas);
      heatTex.colorSpace = THREE.SRGBColorSpace;
      const floorMat = new THREE.MeshBasicMaterial({ map: heatTex, transparent: true, opacity: 0.85, depthWrite: false });
      this.floorHeatmapMesh = new THREE.Mesh(new THREE.PlaneGeometry(7.2, 6.8), floorMat);
      this.floorHeatmapMesh.rotation.x = -Math.PI / 2;
      this.floorHeatmapMesh.position.y = 0.03;
      this.group.add(this.floorHeatmapMesh);
    } else {
      const floorMat = new THREE.MeshBasicMaterial({ color: 0x087F8C, transparent: true, opacity: 0.15 });
      this.floorHeatmapMesh = new THREE.Mesh(new THREE.PlaneGeometry(7.2, 6.8), floorMat);
      this.floorHeatmapMesh.rotation.x = -Math.PI / 2;
      this.floorHeatmapMesh.position.y = 0.03;
      this.group.add(this.floorHeatmapMesh);
    }

    // 2. Smooth 3D Gaussian Process Surrogate Manifold Mesh
    // Grid resolution 36x36 covering search space
    const resX = 36;
    const resZ = 36;
    const width = 6.4;
    const depth = 6.0;
    const geo = new THREE.PlaneGeometry(width, depth, resX - 1, resZ - 1);
    geo.rotateX(-Math.PI / 2);

    const positions = geo.attributes.position;
    const colors = new Float32Array(positions.count * 3);
    geo.setAttribute('color', new THREE.BufferAttribute(colors, 3));

    const manifoldMat = new THREE.MeshPhysicalMaterial({
      vertexColors: true,
      transparent: true,
      opacity: 0.68,
      roughness: 0.22,
      metalness: 0.12,
      transmission: 0.35,
      side: THREE.DoubleSide,
      depthWrite: false,
    });

    this.surrogateMesh = new THREE.Mesh(geo, manifoldMat);
    this.surrogateMesh.receiveShadow = true;
    this.group.add(this.surrogateMesh);

    this.updateSurrogateManifold(new Set(this.scenario.replayInitialIds));

    // Contour wireframe overlay along surrogate
    const wireMat = new THREE.MeshBasicMaterial({
      color: 0x087F8C,
      wireframe: true,
      transparent: true,
      opacity: 0.12,
    });
    this.wireframeMesh = new THREE.Mesh(geo, wireMat);
    this.group.add(this.wireframeMesh);

    // Scanning Plane (visualizing AI search sweep)
    const scanGeo = new THREE.PlaneGeometry(width, 0.08);
    scanGeo.rotateX(-Math.PI / 2);
    const scanMat = new THREE.MeshBasicMaterial({
      color: 0x38BDF8,
      transparent: true,
      opacity: 0.55,
      side: THREE.DoubleSide,
    });
    this.scanPlaneMesh = new THREE.Mesh(scanGeo, scanMat);
    this.scanPlaneMesh.position.y = 0.06;
    this.group.add(this.scanPlaneMesh);
  }

  private updateSurrogateManifold(revealedIds: Set<string>) {
    if (!this.surrogateMesh) return;
    const geo = this.surrogateMesh.geometry as THREE.PlaneGeometry;
    const positions = geo.attributes.position;
    const colors = (geo.attributes.color as THREE.BufferAttribute).array as Float32Array;

    // Strict Anti-Leakage Scientific Invariant:
    // Only revealed candidates participate in GP surrogate interpolation
    const candidateNodes = this.scenario.candidates
      .filter((c) => revealedIds.has(c.id))
      .map((c) => ({
        pos: this.mapCoords(c, true),
        val: c.revealedTarget.value,
      }));

    const tealCol = new THREE.Color(0x087F8C);
    const mintCol = new THREE.Color(0x38BDF8);
    const goldCol = new THREE.Color(0xF59E42);

    for (let i = 0; i < positions.count; i++) {
      const vx = positions.getX(i);
      const vz = positions.getZ(i);

      // Weighted RBF elevation
      let totalWeight = 0;
      let weightedSum = 0;
      for (const node of candidateNodes) {
        const dx = vx - node.pos.x;
        const dz = vz - node.pos.z;
        const distSq = dx * dx + dz * dz;
        const w = Math.exp(-distSq / 1.8);
        weightedSum += w * node.pos.y;
        totalWeight += w;
      }

      // Smooth background prior
      const baselineY = 0.35 + Math.sin(vx * 0.4) * 0.15 + Math.cos(vz * 0.4) * 0.12;
      const vy = totalWeight > 0.001 ? (weightedSum / (totalWeight + 0.15)) * 0.85 + baselineY * 0.15 : baselineY;
      positions.setY(i, vy);

      // Height-based vertex color gradient
      const normH = Math.max(0, Math.min(1, (vy - 0.3) / 3.0));
      const c = new THREE.Color();
      if (normH < 0.5) {
        c.lerpColors(tealCol, mintCol, normH * 2.0);
      } else {
        c.lerpColors(mintCol, goldCol, (normH - 0.5) * 2.0);
      }
      colors[i * 3] = c.r;
      colors[i * 3 + 1] = c.g;
      colors[i * 3 + 2] = c.b;
    }

    positions.needsUpdate = true;
    (geo.attributes.color as THREE.BufferAttribute).needsUpdate = true;
    geo.computeVertexNormals();
  }

  private buildCandidatePoints() {
    const sphereGeo = new THREE.SphereGeometry(0.18, 24, 24);
    const footprintGeo = new THREE.RingGeometry(0.12, 0.22, 24);
    footprintGeo.rotateX(-Math.PI / 2);

    this.scenario.candidates.forEach((cand) => {
      const pos = this.mapCoords(cand);

      // Default candidate material
      const mat = new THREE.MeshStandardMaterial({
        color: 0x94A3B8,
        roughness: 0.25,
        metalness: 0.45,
      });
      const mesh = new THREE.Mesh(sphereGeo, mat);
      mesh.position.copy(pos);
      mesh.castShadow = true;
      mesh.userData = { candidateId: cand.id, cand };

      // Luminous dashed vertical support stem line
      const stemPoints = [new THREE.Vector3(pos.x, 0.05, pos.z), pos];
      const stemGeo = new THREE.BufferGeometry().setFromPoints(stemPoints);
      const stemMat = new THREE.LineBasicMaterial({ color: 0xCBD5E1, transparent: true, opacity: 0.55 });
      const stem = new THREE.Line(stemGeo, stemMat);
      this.group.add(stem);
      this.candidateStems.set(cand.id, stem);

      // Floor Footprint Ring
      const footMat = new THREE.MeshBasicMaterial({
        color: 0x94A3B8,
        transparent: true,
        opacity: 0.35,
        side: THREE.DoubleSide,
      });
      const footprint = new THREE.Mesh(footprintGeo, footMat);
      footprint.position.set(pos.x, 0.04, pos.z);
      this.group.add(footprint);
      this.candidateFootprints.set(cand.id, footprint);

      this.group.add(mesh);
      this.candidateMeshes.set(cand.id, mesh);
      this.candidateHitboxes.push(mesh);
    });
  }

  private buildTrajectoryPath() {
    this.trajectoryGroup = new THREE.Group();
    this.group.add(this.trajectoryGroup);

    // Spawn 14 photon particle beads that travel along the trajectory
    const photonGeo = new THREE.SphereGeometry(0.065, 16, 16);
    const photonMat = new THREE.MeshBasicMaterial({ color: 0xF59E42, transparent: true, opacity: 0.95 });
    for (let i = 0; i < 14; i++) {
      const p = new THREE.Mesh(photonGeo, photonMat);
      p.visible = false;
      this.photonParticles.push(p);
      this.trajectoryGroup.add(p);
    }
  }

  private buildSelectionBeacon() {
    this.selectionBeacon = new THREE.Group();

    // Pulsing orange beacon ring for active algorithm proposal
    const ringGeo = new THREE.RingGeometry(0.32, 0.46, 32);
    ringGeo.rotateX(Math.PI / 2);
    const ringMat = new THREE.MeshBasicMaterial({
      color: 0xF59E42,
      side: THREE.DoubleSide,
      transparent: true,
      opacity: 0.85,
    });
    const ring = new THREE.Mesh(ringGeo, ringMat);
    this.selectionBeacon.add(ring);

    // Inner glowing aura
    const auraGeo = new THREE.RingGeometry(0.12, 0.28, 32);
    auraGeo.rotateX(Math.PI / 2);
    const auraMat = new THREE.MeshBasicMaterial({
      color: 0xF59E42,
      side: THREE.DoubleSide,
      transparent: true,
      opacity: 0.45,
    });
    const aura = new THREE.Mesh(auraGeo, auraMat);
    this.selectionBeacon.add(aura);

    // Vertical spotlight indicator
    const beaconLight = new THREE.PointLight(0xF59E42, 2.2, 4.5);
    this.selectionBeacon.add(beaconLight);

    this.group.add(this.selectionBeacon);
  }

  private buildInspectBeacon() {
    this.inspectBeacon = new THREE.Group();

    // Cyan inspect ring for user-clicked candidate
    const ringGeo = new THREE.RingGeometry(0.26, 0.36, 32);
    ringGeo.rotateX(Math.PI / 2);
    const ringMat = new THREE.MeshBasicMaterial({
      color: 0x087F8C,
      side: THREE.DoubleSide,
      transparent: true,
      opacity: 0.9,
    });
    const ring = new THREE.Mesh(ringGeo, ringMat);
    this.inspectBeacon.add(ring);

    const auraGeo = new THREE.RingGeometry(0.10, 0.22, 32);
    auraGeo.rotateX(Math.PI / 2);
    const auraMat = new THREE.MeshBasicMaterial({
      color: 0x38BDF8,
      side: THREE.DoubleSide,
      transparent: true,
      opacity: 0.5,
    });
    const aura = new THREE.Mesh(auraGeo, auraMat);
    this.inspectBeacon.add(aura);

    this.inspectBeacon.visible = false;
    this.group.add(this.inspectBeacon);
  }

  public setStep(stepIndex: number) {
    stepIndex = Math.max(0, Math.min(this.scenario.replaySteps.length, Math.floor(stepIndex)));
    this.currentStepIndex = stepIndex;
    const revealedIds = new Set([
      ...this.scenario.replayInitialIds,
      ...this.scenario.replaySteps.slice(0, stepIndex).map((s) => s.selectedCandidateId),
    ]);

    this.updateSurrogateManifold(revealedIds);

    this.scenario.candidates.forEach((cand) => {
      const mesh = this.candidateMeshes.get(cand.id)!;
      const isRevealed = revealedIds.has(cand.id);
      mesh.position.copy(this.mapCoords(cand, isRevealed));

      const stem = this.candidateStems.get(cand.id)!;
      stem.geometry.setFromPoints([new THREE.Vector3(mesh.position.x, 0.05, mesh.position.z), mesh.position]);
      stem.visible = isRevealed;

      const footprint = this.candidateFootprints.get(cand.id)!;
      const footMat = footprint.material as THREE.MeshBasicMaterial;
      footMat.color.setHex(isRevealed ? 0x087F8C : 0x94A3B8);
      footMat.opacity = isRevealed ? 0.75 : 0.25;
    });

    // Reset all candidate point colors
    this.candidateMeshes.forEach((mesh, id) => {
      const mat = mesh.material as THREE.MeshStandardMaterial;
      mat.emissiveIntensity = 0;
      if (this.scenario.replayInitialIds.includes(id)) {
        mat.color.setHex(0x38BDF8); // Cyan for initial design seeds
        mat.emissive.setHex(0x0284C7);
        mat.emissiveIntensity = 0.25;
        mesh.scale.setScalar(1.2);
      } else {
        mat.color.setHex(0x94A3B8); // Slate for unvisited pool
        mat.emissive.setHex(0x000000);
        mesh.scale.setScalar(0.95);
      }
    });

    // Color revealed steps up to current
    const trajectoryPoints: THREE.Vector3[] = [];

    // Include the first initial seed in trajectory start
    if (this.scenario.replayInitialIds.length > 0) {
      const firstSeed = this.candidateMeshes.get(this.scenario.replayInitialIds[0]);
      if (firstSeed) trajectoryPoints.push(firstSeed.position.clone());
    }

    const activeSteps = this.scenario.replaySteps.slice(0, stepIndex);
    activeSteps.forEach((st) => {
      const m = this.candidateMeshes.get(st.selectedCandidateId);
      if (m) {
        const mat = m.material as THREE.MeshStandardMaterial;
        if (st.isOptimal) {
          mat.color.setHex(0xF59E42); // Energy Orange
          mat.emissive.setHex(0xF59E42);
          mat.emissiveIntensity = 0.65;
          m.scale.setScalar(1.6);
        } else {
          mat.color.setHex(0x087F8C); // Tech Teal
          mat.emissive.setHex(0x087F8C);
          mat.emissiveIntensity = 0.35;
          m.scale.setScalar(1.3);
        }
        trajectoryPoints.push(m.position.clone());
      }
    });

    // Position selection beacon at active step target
    if (stepIndex > 0 && stepIndex <= this.scenario.replaySteps.length) {
      const currentStep = this.scenario.replaySteps[stepIndex - 1];
      const targetMesh = this.candidateMeshes.get(currentStep.selectedCandidateId);
      if (targetMesh) {
        this.selectionBeacon.position.copy(targetMesh.position);
        this.selectionBeacon.visible = true;
      }
    } else if (this.scenario.replayInitialIds.length > 0) {
      const firstInit = this.candidateMeshes.get(this.scenario.replayInitialIds[0]);
      if (firstInit) {
        this.selectionBeacon.position.copy(firstInit.position);
        this.selectionBeacon.visible = true;
      }
    } else {
      this.selectionBeacon.visible = false;
    }

    this.setSelectedCandidate(this.selectedCandidateId);

    // Rebuild glowing 3D Trajectory Tube
    if (this.trajectoryTubeMesh) {
      this.trajectoryGroup.remove(this.trajectoryTubeMesh);
      this.trajectoryTubeMesh.geometry.dispose();
      (this.trajectoryTubeMesh.material as THREE.Material).dispose();
      this.trajectoryTubeMesh = null;
    }

    if (trajectoryPoints.length > 1) {
      this.trajectoryCurve = new THREE.CatmullRomCurve3(trajectoryPoints);
      const tubeGeo = new THREE.TubeGeometry(this.trajectoryCurve, 48, 0.045, 12, false);
      const tubeMat = new THREE.MeshStandardMaterial({
        color: 0xF59E42,
        emissive: 0xF59E42,
        emissiveIntensity: 0.6,
        roughness: 0.25,
        metalness: 0.5,
      });
      this.trajectoryTubeMesh = new THREE.Mesh(tubeGeo, tubeMat);
      this.trajectoryGroup.add(this.trajectoryTubeMesh);

      // Show particles
      this.photonParticles.forEach((p) => { p.visible = true; });
    } else {
      this.trajectoryCurve = null;
      this.photonParticles.forEach((p) => { p.visible = false; });
    }
  }

  public setSelectedCandidate(candidateId?: string) {
    this.selectedCandidateId = candidateId;
    if (candidateId) {
      const m = this.candidateMeshes.get(candidateId);
      if (m) {
        this.inspectBeacon.position.copy(m.position);
        this.inspectBeacon.visible = true;
        return;
      }
    }
    this.inspectBeacon.visible = false;
  }

  public setScenario(newScenario: ExhibitionScenario) {
    this.scenario = newScenario;

    this.group.traverse((obj) => {
      if (obj instanceof THREE.Mesh || obj instanceof THREE.Line) {
        obj.geometry.dispose();
        const materials = Array.isArray(obj.material) ? obj.material : [obj.material];
        materials.forEach((material) => material.dispose());
      }
    });
    this.candidateStems.clear();
    this.candidateFootprints.clear();
    this.selectedCandidateId = undefined;
    while (this.group.children.length > 0) {
      const obj = this.group.children[0];
      this.group.remove(obj);
    }
    this.candidateMeshes.clear();
    this.candidateHitboxes = [];
    this.photonParticles = [];
    this.trajectoryTubeMesh = null;
    this.trajectoryCurve = null;

    this.buildCoordinateCage();
    this.buildResponseSurface();
    this.buildCandidatePoints();
    this.buildTrajectoryPath();
    this.buildSelectionBeacon();
    this.buildInspectBeacon();
    this.setStep(0);
  }

  public getHitboxes(): THREE.Mesh[] {
    return this.candidateHitboxes;
  }

  public update(delta: number, elapsed: number) {
    // 1. Smooth Turntable Orbit (when enabled for video recording)
    if (this.isOrbiting) {
      this.group.rotation.y += delta * 0.25;
    }

    // 2. Pulse selection beacon
    if (this.selectionBeacon && this.selectionBeacon.visible) {
      const ring = this.selectionBeacon.children[0];
      if (ring) {
        const scale = 1.0 + Math.sin(elapsed * 4.0) * 0.18;
        ring.scale.set(scale, scale, 1);
      }
    }

    // 3. Pulse inspect beacon
    if (this.inspectBeacon && this.inspectBeacon.visible) {
      const ring = this.inspectBeacon.children[0];
      if (ring) {
        const scale = 1.0 + Math.cos(elapsed * 4.5) * 0.14;
        ring.scale.set(scale, scale, 1);
      }
    }

    // 4. Animate scanning plane sweep along search space
    if (this.scanPlaneMesh) {
      this.scanPlaneMesh.position.z = Math.sin(elapsed * 0.8) * 2.6;
    }

    // 5. Animate flowing photon particles along the optimization trajectory
    if (this.trajectoryCurve && this.photonParticles.length > 0) {
      const count = this.photonParticles.length;
      for (let i = 0; i < count; i++) {
        const p = this.photonParticles[i];
        if (!p.visible) continue;
        const t = ((elapsed * 0.35 + i / count) % 1.0);
        const pt = this.trajectoryCurve.getPoint(t);
        p.position.copy(pt);
        const pulse = 0.8 + Math.sin(elapsed * 6.0 + i) * 0.3;
        p.scale.setScalar(pulse);
      }
    }
  }
}
