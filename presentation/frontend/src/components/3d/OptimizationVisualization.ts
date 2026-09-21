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
  private surfaceMesh!: THREE.Mesh;
  private trajectoryLine!: THREE.Line;
  private selectionBeacon!: THREE.Group;
  private inspectBeacon!: THREE.Group;
  private currentStepIndex: number = 0;
  private selectedCandidateId?: string;
  private onCandidateClick?: CandidateClickCallback;

  constructor(scenario: ExhibitionScenario, onCandidateClick?: CandidateClickCallback) {
    this.group = new THREE.Group();
    this.scenario = scenario;
    this.onCandidateClick = onCandidateClick;
    this.buildResponseSurface();
    this.buildCandidatePoints();
    this.buildTrajectoryPath();
    this.buildSelectionBeacon();
    this.buildInspectBeacon();
    this.setStep(0);
  }

  private mapCoords(cand: CandidateRecord): THREE.Vector3 {
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

      // Separate Loading Regimes along Z axis: Low (-2.0) vs High (+2.0)
      const isHighLoading = cand.controls.loading_regime === 'HIGH';
      const worldZ = densityZ + (isHighLoading ? 1.9 : -1.9);

      // Y: Target Rate ratio (0.09 to 0.82)
      const yVal = cand.revealedTarget.value;
      const normY = Math.max(0, Math.min(1, (yVal - 0.05) / (0.85 - 0.05)));
      const worldY = 0.5 + normY * 3.4;

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

      // Subtle offset for calendered vs uncalendered
      if (cand.controls.calendering_applied) {
        worldZ += 0.35;
      }

      // Y: D30 Specific capacity (40 to 415 mAh/g)
      const yVal = cand.revealedTarget.value;
      const normY = Math.max(0, Math.min(1, (yVal - 30) / (420 - 30)));
      const worldY = 0.5 + normY * 3.4;

      return new THREE.Vector3(worldX, worldY, worldZ);
    }
  }

  private buildResponseSurface() {
    // 3D Parametric Response Surface / Gaussian Process Mean Landscape
    const res = 28;
    const geo = new THREE.PlaneGeometry(8, 8, res, res);
    geo.rotateX(-Math.PI / 2);

    const posAttr = geo.attributes.position as THREE.BufferAttribute;
    for (let i = 0; i < posAttr.count; i++) {
      const x = posAttr.getX(i);
      const z = posAttr.getZ(i);

      let y = 1.0;
      if (this.scenario.id === 'warwick_nmc622_calendering') {
        // Peak is at Low Temp (85C, x ~ -2.6), High Density (z ~ -0.8) in Low loading regime
        const distFromOpt = Math.hypot(x - (-2.6), z - (-0.8));
        y = 3.6 * Math.exp(-distFromOpt * 0.35) + 0.6;
      } else {
        // Drakopoulos: optimal is at 100 um gap (z ~ -2.4) and 0.2 m/min speed (x ~ 0.0)
        const distFromOpt = Math.hypot(x - 0.0, z - (-2.4));
        y = 3.7 * Math.exp(-distFromOpt * 0.38) + 0.5;
      }
      posAttr.setY(i, y);
    }
    geo.computeVertexNormals();

    const mat = new THREE.MeshStandardMaterial({
      color: 0x087F8C,
      roughness: 0.35,
      metalness: 0.25,
      transparent: true,
      opacity: 0.42,
      wireframe: false,
      side: THREE.DoubleSide
    });
    this.surfaceMesh = new THREE.Mesh(geo, mat);
    this.surfaceMesh.receiveShadow = true;
    this.group.add(this.surfaceMesh);

    // Grid wireframe contour overlay
    const wireMat = new THREE.MeshBasicMaterial({
      color: 0x087F8C,
      wireframe: true,
      transparent: true,
      opacity: 0.25
    });
    const wireMesh = new THREE.Mesh(geo, wireMat);
    this.group.add(wireMesh);

    // Reference Ground Grid
    const baseGrid = new THREE.GridHelper(8, 8, 0x142A35, 0xDCE8EC);
    baseGrid.position.y = 0.02;
    this.group.add(baseGrid);
  }

  private buildCandidatePoints() {
    const sphereGeo = new THREE.SphereGeometry(0.18, 20, 20);

    this.scenario.candidates.forEach((cand) => {
      const pos = this.mapCoords(cand);

      // Default candidate material
      const mat = new THREE.MeshStandardMaterial({
        color: 0x94A3B8,
        roughness: 0.35,
        metalness: 0.30
      });
      const mesh = new THREE.Mesh(sphereGeo, mat);
      mesh.position.copy(pos);
      mesh.castShadow = true;
      mesh.userData = { candidateId: cand.id, cand };

      // Vertical support stem line linking point to ground
      const stemPoints = [new THREE.Vector3(pos.x, 0.05, pos.z), pos];
      const stemGeo = new THREE.BufferGeometry().setFromPoints(stemPoints);
      const stemMat = new THREE.LineBasicMaterial({ color: 0xCBD5E1, transparent: true, opacity: 0.45 });
      const stem = new THREE.Line(stemGeo, stemMat);
      this.group.add(stem);

      this.group.add(mesh);
      this.candidateMeshes.set(cand.id, mesh);
      this.candidateHitboxes.push(mesh);
    });
  }

  private buildTrajectoryPath() {
    const pathGeo = new THREE.BufferGeometry();
    const pathMat = new THREE.LineBasicMaterial({
      color: 0x087F8C,
      linewidth: 2,
      transparent: true,
      opacity: 0.85
    });
    this.trajectoryLine = new THREE.Line(pathGeo, pathMat);
    this.group.add(this.trajectoryLine);
  }

  private buildSelectionBeacon() {
    this.selectionBeacon = new THREE.Group();

    // Pulsing orange beacon ring for active algorithm proposal
    const ringGeo = new THREE.RingGeometry(0.32, 0.46, 32);
    const ringMat = new THREE.MeshBasicMaterial({
      color: 0xF59E42,
      side: THREE.DoubleSide,
      transparent: true,
      opacity: 0.85
    });
    const ring = new THREE.Mesh(ringGeo, ringMat);
    ring.rotation.x = Math.PI / 2;
    this.selectionBeacon.add(ring);

    // Vertical spotlight indicator
    const beaconLight = new THREE.PointLight(0xF59E42, 1.8, 4.0);
    this.selectionBeacon.add(beaconLight);

    this.group.add(this.selectionBeacon);
  }

  private buildInspectBeacon() {
    this.inspectBeacon = new THREE.Group();

    // Cyan inspect ring for user-clicked candidate
    const ringGeo = new THREE.RingGeometry(0.24, 0.32, 32);
    const ringMat = new THREE.MeshBasicMaterial({
      color: 0x087F8C,
      side: THREE.DoubleSide,
      transparent: true,
      opacity: 0.85
    });
    const ring = new THREE.Mesh(ringGeo, ringMat);
    ring.rotation.x = Math.PI / 2;
    this.inspectBeacon.add(ring);
    this.inspectBeacon.visible = false;

    this.group.add(this.inspectBeacon);
  }

  public setStep(stepIndex: number) {
    this.currentStepIndex = stepIndex;

    // Reset all candidate point colors
    this.candidateMeshes.forEach((mesh, id) => {
      const mat = mesh.material as THREE.MeshStandardMaterial;
      if (this.scenario.replayInitialIds.includes(id)) {
        mat.color.setHex(0x38BDF8); // Cyan for initial design seeds
        mat.emissive.setHex(0x000000);
        mesh.scale.setScalar(1.1);
      } else {
        mat.color.setHex(0x94A3B8); // Slate for unvisited pool
        mat.emissive.setHex(0x000000);
        mesh.scale.setScalar(0.95);
      }
    });

    // Color revealed steps up to current
    const trajectoryPoints: THREE.Vector3[] = [];

    // Add initial points to trajectory
    this.scenario.replayInitialIds.forEach((id) => {
      const m = this.candidateMeshes.get(id);
      if (m) trajectoryPoints.push(m.position.clone());
    });

    const activeSteps = this.scenario.replaySteps.slice(0, stepIndex);
    activeSteps.forEach((st) => {
      const m = this.candidateMeshes.get(st.selectedCandidateId);
      if (m) {
        const mat = m.material as THREE.MeshStandardMaterial;
        if (st.isOptimal) {
          mat.color.setHex(0xF59E42); // Energy Orange
          mat.emissive.setHex(0xF59E42);
          mat.emissiveIntensity = 0.5;
          m.scale.setScalar(1.5);
        } else {
          mat.color.setHex(0x087F8C); // Tech Teal
          mat.emissive.setHex(0x087F8C);
          mat.emissiveIntensity = 0.3;
          m.scale.setScalar(1.25);
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

    // Update trajectory line geometry
    if (trajectoryPoints.length > 1) {
      this.trajectoryLine.geometry.setFromPoints(trajectoryPoints);
      this.trajectoryLine.visible = true;
    } else {
      this.trajectoryLine.visible = false;
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

    // Remove existing children
    while (this.group.children.length > 0) {
      const obj = this.group.children[0];
      this.group.remove(obj);
    }
    this.candidateMeshes.clear();
    this.candidateHitboxes = [];

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
    // Pulse selection beacon
    if (this.selectionBeacon && this.selectionBeacon.visible) {
      const ring = this.selectionBeacon.children[0];
      if (ring) {
        const scale = 1.0 + Math.sin(elapsed * 4.0) * 0.15;
        ring.scale.set(scale, scale, 1);
      }
    }
    // Pulse inspect beacon
    if (this.inspectBeacon && this.inspectBeacon.visible) {
      const ring = this.inspectBeacon.children[0];
      if (ring) {
        const scale = 1.0 + Math.cos(elapsed * 4.5) * 0.12;
        ring.scale.set(scale, scale, 1);
      }
    }
  }
}
