import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { ReplayStep, ScenarioId } from '../../data/types';
import type { ProcessExperiencePhase } from '../../data/processExperience';

export interface StageInteractionCallback {
  (stageId: string): void;
}

export class ManufacturingWorld {
  public group: THREE.Group;
  private calenderRollTop!: THREE.Mesh;
  private calenderRollBottom!: THREE.Mesh;
  private backingRoll!: THREE.Mesh;
  private mixerShaft!: THREE.Mesh;
  private foilStrip!: THREE.Mesh;
  private coatedLayer!: THREE.Mesh;
  private stageHitboxes: THREE.Mesh[] = [];
  private interactiveBeacons: THREE.Group[] = [];
  private irLamps: THREE.PointLight[] = [];
  private onStageClick?: StageInteractionCallback;
  private activeScenario: ScenarioId = 'warwick_nmc622_calendering';
  private foilMaterial!: THREE.MeshStandardMaterial;
  private coatedMaterial!: THREE.MeshStandardMaterial;
  private stageGroups = new Map<string, THREE.Group>();
  private coaterCutawayParts: THREE.Object3D[] = [];
  private calenderCutawayParts: THREE.Object3D[] = [];
  private formulationHoppers: THREE.Object3D[] = [];
  private mixerVessel?: THREE.Mesh;
  private dryerTunnel?: THREE.Mesh;
  private webMotionMarkers: THREE.Mesh[] = [];
  private coatingBead?: THREE.Mesh;
  private nipResponse?: THREE.Mesh;
  private characterizationTrace?: THREE.Group;
  private assemblyProbe?: THREE.Group;
  private annotationGroup = new THREE.Group();
  private processStageId = 'overview';
  private processPhase: ProcessExperiencePhase = 'OVERVIEW';
  private processBlend = 0;

  constructor(onStageClick?: StageInteractionCallback) {
    this.group = new THREE.Group();
    this.onStageClick = onStageClick;
    this.buildCleanroomEnvironment();
    this.buildManufacturingLine();
    this.group.add(this.annotationGroup);
  }

  private buildCleanroomEnvironment() {
    // 1. Cleanroom Architectural Epoxy Floor
    const floorGeo = new THREE.PlaneGeometry(90, 50);
    const floorMat = new THREE.MeshStandardMaterial({
      color: 0xF2F6F7,
      roughness: 0.22,
      metalness: 0.06,
    });
    const floor = new THREE.Mesh(floorGeo, floorMat);
    floor.rotation.x = -Math.PI / 2;
    floor.position.y = 0;
    floor.receiveShadow = true;
    this.group.add(floor);

    // High-tech cleanroom scale grid overlay
    const grid = new THREE.GridHelper(90, 45, 0x087F8C, 0xDCE8EC);
    grid.position.y = 0.01;
    (grid.material as THREE.Material).opacity = 0.45;
    (grid.material as THREE.Material).transparent = true;
    this.group.add(grid);

    // 2. Cleanroom Back Wall
    const wallGeo = new THREE.PlaneGeometry(90, 22);
    const wallMat = new THREE.MeshStandardMaterial({
      color: 0xEEF3F5,
      roughness: 0.55,
      metalness: 0.05
    });
    const backWall = new THREE.Mesh(wallGeo, wallMat);
    backWall.position.set(0, 11, -16);
    this.group.add(backWall);

    // Cleanroom Tech Lightstrip along the wall
    const stripGeo = new THREE.BoxGeometry(86, 0.25, 0.4);
    const stripMat = new THREE.MeshBasicMaterial({ color: 0x087F8C });
    const wallStrip = new THREE.Mesh(stripGeo, stripMat);
    wallStrip.position.set(0, 16, -15.8);
    this.group.add(wallStrip);

    // Overhead Cleanroom Gantries
    const gantryMat = new THREE.MeshStandardMaterial({ color: 0x2C3E50, roughness: 0.4, metalness: 0.6 });
    for (let x = -36; x <= 36; x += 18) {
      const trussGeo = new THREE.BoxGeometry(0.5, 6, 28);
      const truss = new THREE.Mesh(trussGeo, gantryMat);
      truss.position.set(x, 18, 0);
      this.group.add(truss);
    }
  }

  private buildManufacturingLine() {
    // 0. Stage 1: Slurry Formulation Dosing Station (X = -15.5)
    const formulationGroup = this.buildFormulationUnit();
    formulationGroup.position.set(-15.5, 0, 0);
    this.group.add(formulationGroup);
    this.stageGroups.set('formulation', formulationGroup);
    this.stageGroups.set('slurry_prep', formulationGroup);
    this.registerStageHitbox('formulation', formulationGroup, new THREE.Vector3(-15.5, 3.5, 0));

    // 1. Stage 2: Planetary High-Shear Vacuum Mixer (X = -11)
    const mixerGroup = this.buildMixerUnit();
    mixerGroup.position.set(-11, 0, 0);
    this.group.add(mixerGroup);
    this.stageGroups.set('mixing', mixerGroup);
    this.registerStageHitbox('mixing', mixerGroup, new THREE.Vector3(-11, 3.5, 0));

    // 2. Foil Uncoiler Reel (X = -6.5)
    const uncoilerGroup = this.buildUncoilerReel();
    uncoilerGroup.position.set(-6.5, 0, 0);
    this.group.add(uncoilerGroup);

    // 3. Stage 3: Precision Slot-Die Coater (Signature Asset) (X = -2.5)
    const coaterGroup = this.buildSlotDieCoater();
    coaterGroup.position.set(-2.5, 0, 0);
    this.group.add(coaterGroup);
    this.stageGroups.set('coating', coaterGroup);
    this.stageGroups.set('pilot_coating', coaterGroup);
    this.registerStageHitbox('coating', coaterGroup, new THREE.Vector3(-2.5, 3.5, 0));

    // 4. Stage 4: Convection & IR Drying Tunnel (X = 4)
    const dryerGroup = this.buildDryingTunnel();
    dryerGroup.position.set(4, 0, 0);
    this.group.add(dryerGroup);
    this.stageGroups.set('drying', dryerGroup);
    this.registerStageHitbox('drying', dryerGroup, new THREE.Vector3(4, 3.8, 0));

    // 5. Stage 5: Precision Calendering Machine (Signature Asset) (X = 11)
    const calenderGroup = this.buildPrecisionCalender();
    calenderGroup.position.set(11, 0, 0);
    this.group.add(calenderGroup);
    this.stageGroups.set('calendering', calenderGroup);
    this.registerStageHitbox('calendering', calenderGroup, new THREE.Vector3(11, 4.2, 0));

    // 6. Stage 6: Electrochemical Half-Cell Test Station (X = 17.5)
    const cyclerGroup = this.buildCyclerStation();
    cyclerGroup.position.set(17.5, 0, 0);
    this.group.add(cyclerGroup);
    this.stageGroups.set('characterization', cyclerGroup);
    this.stageGroups.set('rate_characterization', cyclerGroup);
    this.registerStageHitbox('characterization', cyclerGroup, new THREE.Vector3(17.5, 3.5, 0));

    // 7. Continuous Moving Electrode Foil Web
    this.buildElectrodeWeb();
    this.buildProcessOverlays();
  }

  private loadHeroAsset(path: string, parent: THREE.Group, fallbackGroup: THREE.Group, onLoaded?: (scene: THREE.Group) => void) {
    // 1. Immediately attach high-fidelity procedural fallback
    parent.add(fallbackGroup);

    // 2. Load GLB asynchronously
    new GLTFLoader().load(path, ({ scene }) => {
      scene.traverse((object) => {
        if (!(object instanceof THREE.Mesh)) return;
        object.castShadow = true;
        object.receiveShadow = true;

        if (object.name.includes('Upper precision roll')) this.calenderRollTop = object;
        if (object.name.includes('Lower precision roll')) this.calenderRollBottom = object;
        if (object.name.includes('Backing roll')) this.backingRoll = object;
        if (object.name.includes('Agitator drive shaft')) this.mixerShaft = object;
        if (object.name.includes('Mixer vessel cutaway shell')) this.mixerVessel = object;
        if (object.name.includes('Dryer removable housing')) this.dryerTunnel = object;

        object.userData.restPosition = object.position.clone();
        object.geometry.computeBoundingBox();
        const size = object.geometry.boundingBox?.getSize(new THREE.Vector3()) ?? new THREE.Vector3();
        object.userData.rotationAxis = size.x > size.y && size.x > size.z ? 'x' : size.y > size.z ? 'y' : 'z';

        if (path.includes('coater') && /(Slot die manifold|Die cross slide|Micrometer carriage)/.test(object.name)) {
          this.coaterCutawayParts.push(object);
        }
        if (path.includes('calender') && /(Split bearing chock|Hydraulic loading ram|Chrome loading piston)/.test(object.name)) {
          this.calenderCutawayParts.push(object);
        }
        if (path.includes('mixer') && /(Active material vessel|Conductive additive vessel|Binder vessel)/.test(object.name)) {
          this.formulationHoppers.push(object);
        }
      });

      if (onLoaded) {
        onLoaded(scene);
      }

      // Smoothly replace fallback group
      parent.remove(fallbackGroup);
      parent.add(scene);
    }, undefined, (error) => {
      console.warn(`[3D] Failed to load ${path}, retaining procedural fallback:`, error);
    });
  }

  private createGaugeMesh(radius: number, title: string, valStr: string, min = 0, max = 30): THREE.Group {
    const group = new THREE.Group();
    const bezelMat = new THREE.MeshStandardMaterial({ color: 0xCFD8DC, metalness: 0.9, roughness: 0.15 });
    const bezel = new THREE.Mesh(new THREE.CylinderGeometry(radius, radius, 0.05, 32), bezelMat);
    bezel.rotation.x = Math.PI / 2;
    group.add(bezel);

    const rimMat = new THREE.MeshStandardMaterial({ color: 0xD4AF37, metalness: 0.85, roughness: 0.25 });
    const rim = new THREE.Mesh(new THREE.TorusGeometry(radius, 0.012, 12, 32), rimMat);
    group.add(rim);

    const canvas = document.createElement('canvas');
    canvas.width = 256;
    canvas.height = 256;
    const ctx = canvas.getContext('2d')!;
    ctx.fillStyle = '#FFFFFF';
    ctx.beginPath();
    ctx.arc(128, 128, 120, 0, Math.PI * 2);
    ctx.fill();

    ctx.strokeStyle = '#142A35';
    ctx.lineWidth = 3;
    const startAngle = Math.PI * 0.75;
    const endAngle = Math.PI * 2.25;
    for (let i = 0; i <= 20; i++) {
      const angle = startAngle + (i / 20) * (endAngle - startAngle);
      const r1 = i % 5 === 0 ? 95 : 105;
      ctx.beginPath();
      ctx.moveTo(128 + Math.cos(angle) * r1, 128 + Math.sin(angle) * r1);
      ctx.lineTo(128 + Math.cos(angle) * 115, 128 + Math.sin(angle) * 115);
      ctx.stroke();
    }

    ctx.strokeStyle = '#E53935';
    ctx.lineWidth = 6;
    ctx.beginPath();
    ctx.arc(128, 128, 108, startAngle + 0.75 * (endAngle - startAngle), endAngle);
    ctx.stroke();

    ctx.fillStyle = '#142A35';
    ctx.font = 'bold 20px system-ui, sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText(title, 128, 85);
    ctx.font = 'bold 22px monospace';
    ctx.fillText(valStr, 128, 175);

    ctx.strokeStyle = '#E53935';
    ctx.lineWidth = 4;
    ctx.beginPath();
    ctx.moveTo(128, 128);
    const needleAngle = startAngle + 0.62 * (endAngle - startAngle);
    ctx.lineTo(128 + Math.cos(needleAngle) * 85, 128 + Math.sin(needleAngle) * 85);
    ctx.stroke();
    ctx.fillStyle = '#142A35';
    ctx.beginPath();
    ctx.arc(128, 128, 10, 0, Math.PI * 2);
    ctx.fill();

    const tex = new THREE.CanvasTexture(canvas);
    tex.colorSpace = THREE.SRGBColorSpace;
    const face = new THREE.Mesh(new THREE.CircleGeometry(radius * 0.94, 32), new THREE.MeshBasicMaterial({ map: tex }));
    face.position.z = 0.028;
    group.add(face);

    const glass = new THREE.Mesh(new THREE.CircleGeometry(radius * 0.95, 32), new THREE.MeshPhysicalMaterial({
      color: 0xFFFFFF,
      transparent: true,
      opacity: 0.25,
      roughness: 0.05,
      transmission: 0.9,
    }));
    glass.position.z = 0.035;
    group.add(glass);

    return group;
  }

  private createDigitalDisplayMesh(width: number, height: number, title: string, valStr: string, subStr: string): THREE.Mesh {
    const canvas = document.createElement('canvas');
    canvas.width = 512;
    canvas.height = 256;
    const ctx = canvas.getContext('2d')!;

    const grad = ctx.createLinearGradient(0, 0, 0, 256);
    grad.addColorStop(0, '#0E1720');
    grad.addColorStop(1, '#070B0E');
    ctx.fillStyle = grad;
    ctx.fillRect(0, 0, 512, 256);

    ctx.strokeStyle = '#087F8C';
    ctx.lineWidth = 8;
    ctx.strokeRect(4, 4, 504, 248);

    ctx.fillStyle = '#087F8C';
    ctx.fillRect(8, 8, 496, 42);
    ctx.fillStyle = '#FFFFFF';
    ctx.font = 'bold 22px monospace';
    ctx.textBaseline = 'middle';
    ctx.fillText(title, 24, 28);

    ctx.fillStyle = '#38B7C4';
    ctx.font = 'bold 54px monospace';
    ctx.fillText(valStr, 24, 115);

    ctx.fillStyle = '#F59E42';
    ctx.font = '22px monospace';
    ctx.fillText(subStr, 24, 175);

    ctx.strokeStyle = '#38B7C4';
    ctx.lineWidth = 3;
    ctx.beginPath();
    for (let x = 0; x < 200; x += 10) {
      const y = 220 - Math.sin(x * 0.1) * 15 - (x % 20 === 0 ? 5 : 0);
      if (x === 0) ctx.moveTo(280 + x, y);
      else ctx.lineTo(280 + x, y);
    }
    ctx.stroke();

    const tex = new THREE.CanvasTexture(canvas);
    tex.colorSpace = THREE.SRGBColorSpace;
    return new THREE.Mesh(new THREE.PlaneGeometry(width, height), new THREE.MeshBasicMaterial({ map: tex }));
  }

  private createWarningLabelMesh(width: number, height: number, caution: string, sub: string): THREE.Mesh {
    const canvas = document.createElement('canvas');
    canvas.width = 384;
    canvas.height = 160;
    const ctx = canvas.getContext('2d')!;

    ctx.fillStyle = '#FCD34D';
    ctx.fillRect(0, 0, 384, 160);

    const stripeW = 20;
    ctx.fillStyle = '#142A35';
    for (let x = -30; x < 420; x += stripeW * 2) {
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x + stripeW, 0);
      ctx.lineTo(x + stripeW - 15, 20);
      ctx.lineTo(x - 15, 20);
      ctx.fill();

      ctx.beginPath();
      ctx.moveTo(x, 140);
      ctx.lineTo(x + stripeW, 140);
      ctx.lineTo(x + stripeW - 15, 160);
      ctx.lineTo(x - 15, 160);
      ctx.fill();
    }

    ctx.fillStyle = '#142A35';
    ctx.fillRect(20, 26, 344, 40);
    ctx.fillStyle = '#FCD34D';
    ctx.font = 'bold 26px system-ui, sans-serif';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText('CAUTION', 192, 46);

    ctx.fillStyle = '#142A35';
    ctx.font = 'bold 22px system-ui, sans-serif';
    ctx.fillText(caution, 192, 92);
    ctx.font = '16px system-ui, sans-serif';
    ctx.fillText(sub, 192, 122);

    const tex = new THREE.CanvasTexture(canvas);
    tex.colorSpace = THREE.SRGBColorSpace;
    return new THREE.Mesh(new THREE.PlaneGeometry(width, height), new THREE.MeshBasicMaterial({ map: tex }));
  }

  private createStackLightMesh(): THREE.Group {
    const stack = new THREE.Group();
    const stemMat = new THREE.MeshStandardMaterial({ color: 0x1E293B, metalness: 0.6, roughness: 0.3 });
    const stem = new THREE.Mesh(new THREE.CylinderGeometry(0.025, 0.025, 0.3, 16), stemMat);
    stem.position.y = 0.15;
    stack.add(stem);

    const colors = [0x10B981, 0xF59E42, 0x087F8C];
    colors.forEach((col, idx) => {
      const segGeo = new THREE.CylinderGeometry(0.04, 0.04, 0.07, 16);
      const segMat = new THREE.MeshStandardMaterial({
        color: col,
        emissive: col,
        emissiveIntensity: 0.8,
        roughness: 0.2
      });
      const seg = new THREE.Mesh(segGeo, segMat);
      seg.position.y = 0.34 + idx * 0.075;
      stack.add(seg);
    });
    return stack;
  }

  private buildFormulationUnit(): THREE.Group {
    const group = new THREE.Group();
    const frameMat = new THREE.MeshStandardMaterial({ color: 0x1E293B, roughness: 0.35, metalness: 0.45 });
    const stainlessMat = new THREE.MeshStandardMaterial({ color: 0xDFE7E9, metalness: 0.90, roughness: 0.18 });
    const accentMat = new THREE.MeshStandardMaterial({ color: 0x087F8C, metalness: 0.6, roughness: 0.25 });
    const hopperMat = new THREE.MeshStandardMaterial({ color: 0xF1F5F9, metalness: 0.3, roughness: 0.2 });

    // Heavy machine bench
    const bench = new THREE.Mesh(new THREE.BoxGeometry(3.4, 0.45, 2.8), frameMat);
    bench.position.y = 0.225;
    bench.receiveShadow = true;
    group.add(bench);

    // Active Material Hopper (Tapered Cone)
    const silo1 = new THREE.Mesh(new THREE.CylinderGeometry(0.52, 0.24, 1.8, 24), hopperMat);
    silo1.name = 'Active material vessel';
    silo1.position.set(-0.95, 1.75, -0.4);
    silo1.castShadow = true;
    group.add(silo1);

    // Carbon Black Silo
    const silo2 = new THREE.Mesh(new THREE.CylinderGeometry(0.42, 0.20, 1.6, 24), hopperMat);
    silo2.name = 'Conductive additive vessel';
    silo2.position.set(0.0, 1.65, -0.4);
    silo2.castShadow = true;
    group.add(silo2);

    // Binder Silo
    const silo3 = new THREE.Mesh(new THREE.CylinderGeometry(0.35, 0.16, 1.5, 24), stainlessMat);
    silo3.name = 'Binder vessel';
    silo3.position.set(0.9, 1.6, -0.4);
    silo3.castShadow = true;
    group.add(silo3);

    [silo1, silo2, silo3].forEach((silo) => {
      silo.userData.restPosition = silo.position.clone();
      this.formulationHoppers.push(silo);
    });

    // Digital Gravimetric Load-Cell Platform
    const scale = new THREE.Mesh(new THREE.BoxGeometry(2.8, 0.18, 1.2), accentMat);
    scale.position.set(0, 0.52, 0.5);
    group.add(scale);

    // Digital Control Console Display
    const consoleUnit = new THREE.Mesh(new THREE.BoxGeometry(0.7, 0.9, 0.4), frameMat);
    consoleUnit.position.set(-1.2, 1.25, 0.7);
    group.add(consoleUnit);

    const disp = this.createDigitalDisplayMesh(0.55, 0.35, 'DOSING SCALE', '34.82 kg', 'RATE: 1.4 kg/min');
    disp.position.set(-1.2, 1.35, 0.91);
    group.add(disp);

    // Pneumatic Transfer Chute leading into Planetary Mixer
    const chuteCurve = new THREE.CatmullRomCurve3([
      new THREE.Vector3(1.2, 1.1, 0),
      new THREE.Vector3(2.4, 1.6, 0),
      new THREE.Vector3(4.2, 2.1, 0)
    ]);
    const chute = new THREE.Mesh(new THREE.TubeGeometry(chuteCurve, 32, 0.1, 16, false), stainlessMat);
    chute.castShadow = true;
    group.add(chute);

    return group;
  }

  private buildMixerUnit(): THREE.Group {
    const group = new THREE.Group();
    const steelMat = new THREE.MeshStandardMaterial({ color: 0xDFE7E9, metalness: 0.92, roughness: 0.15 });
    const frameMat = new THREE.MeshStandardMaterial({ color: 0x1E293B, roughness: 0.35, metalness: 0.45 });
    const brassMat = new THREE.MeshStandardMaterial({ color: 0xD4AF37, metalness: 0.85, roughness: 0.22 });
    const glassMat = new THREE.MeshPhysicalMaterial({ color: 0xFFFFFF, transparent: true, opacity: 0.3, roughness: 0.05, transmission: 0.9 });

    // Machine Base Pedestal
    const base = new THREE.Mesh(new THREE.BoxGeometry(3.4, 0.45, 3.4), frameMat);
    base.position.y = 0.225;
    base.receiveShadow = true;
    group.add(base);

    // Cylindrical Vacuum Mixing Vessel
    const vessel = new THREE.Mesh(new THREE.CylinderGeometry(1.22, 1.05, 2.4, 36), steelMat);
    vessel.name = 'Mixer vessel cutaway shell';
    vessel.position.y = 1.65;
    vessel.castShadow = true;
    group.add(vessel);
    this.mixerVessel = vessel;

    // Domed Top Cover & Clamp Rim
    const clampRim = new THREE.Mesh(new THREE.TorusGeometry(1.24, 0.06, 16, 36), steelMat);
    clampRim.rotation.x = Math.PI / 2;
    clampRim.position.y = 2.85;
    group.add(clampRim);

    const dome = new THREE.Mesh(new THREE.SphereGeometry(1.22, 36, 18, 0, Math.PI * 2, 0, Math.PI / 2), steelMat);
    dome.position.y = 2.85;
    group.add(dome);

    // Dual Circular Glass Sight Windows
    for (const angle of [Math.PI * 0.3, Math.PI * 0.7]) {
      const sx = Math.cos(angle) * 1.18;
      const sz = Math.sin(angle) * 1.18;
      const port = new THREE.Mesh(new THREE.CylinderGeometry(0.18, 0.18, 0.12, 24), brassMat);
      port.rotation.z = Math.PI / 2;
      port.position.set(sx, 1.9, sz);
      group.add(port);
      const glass = new THREE.Mesh(new THREE.CylinderGeometry(0.14, 0.14, 0.13, 24), glassMat);
      glass.rotation.z = Math.PI / 2;
      glass.position.set(sx, 1.9, sz);
      group.add(glass);
    }

    // Overhead Heavy-Duty Gear Motor with Cooling Fins
    const motor = new THREE.Mesh(new THREE.CylinderGeometry(0.58, 0.58, 1.4, 28), frameMat);
    motor.position.y = 4.0;
    motor.castShadow = true;
    group.add(motor);

    for (let r = 0; r < 6; r++) {
      const fin = new THREE.Mesh(new THREE.BoxGeometry(1.25, 0.04, 1.25), frameMat);
      fin.position.y = 3.5 + r * 0.18;
      group.add(fin);
    }

    // Agitator Drive Shaft
    this.mixerShaft = new THREE.Mesh(new THREE.CylinderGeometry(0.14, 0.14, 2.1, 20), steelMat);
    this.mixerShaft.position.y = 2.0;
    group.add(this.mixerShaft);

    // Vacuum Gauge on extraction line
    const vacGauge = this.createGaugeMesh(0.16, 'VACUUM', '-0.098 MPa');
    vacGauge.position.set(1.1, 3.2, 0.8);
    vacGauge.rotation.y = Math.PI * 0.25;
    group.add(vacGauge);

    // Slurry Feed Pipe leading toward coater
    const pipeCurve = new THREE.CatmullRomCurve3([
      new THREE.Vector3(1.1, 0.9, 0),
      new THREE.Vector3(2.5, 0.8, 0),
      new THREE.Vector3(4.0, 1.8, 0),
      new THREE.Vector3(7.0, 2.2, 0)
    ]);
    const pipeMat = new THREE.MeshStandardMaterial({ color: 0xCBD5E1, metalness: 0.88, roughness: 0.18 });
    const pipe = new THREE.Mesh(new THREE.TubeGeometry(pipeCurve, 36, 0.09, 16, false), pipeMat);
    group.add(pipe);

    return group;
  }

  private buildUncoilerReel(): THREE.Group {
    const group = new THREE.Group();
    const frameMat = new THREE.MeshStandardMaterial({ color: 0x1E293B, roughness: 0.35, metalness: 0.45 });
    const foilRollMat = new THREE.MeshStandardMaterial({ color: 0xE2EBED, metalness: 0.96, roughness: 0.12 });
    const chromeMat = new THREE.MeshStandardMaterial({ color: 0xFFFFFF, metalness: 0.98, roughness: 0.05 });

    // Dual Cast A-Frame Pedestals
    for (const z of [-1.35, 1.35]) {
      const stand = new THREE.Mesh(new THREE.BoxGeometry(0.24, 2.2, 0.32), frameMat);
      stand.position.set(0, 1.1, z);
      stand.castShadow = true;
      group.add(stand);

      const foot = new THREE.Mesh(new THREE.BoxGeometry(0.6, 0.14, 0.44), frameMat);
      foot.position.set(0, 0.07, z);
      group.add(foot);
    }

    // Master Foil Coil
    const roll = new THREE.Mesh(new THREE.CylinderGeometry(0.68, 0.68, 2.2, 36), foilRollMat);
    roll.rotation.x = Math.PI / 2;
    roll.position.set(0, 1.8, 0);
    roll.castShadow = true;
    group.add(roll);

    // Expanding Air Shaft Core Chucks
    for (const z of [-1.25, 1.25]) {
      const chuck = new THREE.Mesh(new THREE.CylinderGeometry(0.22, 0.22, 0.3, 24), chromeMat);
      chuck.rotation.x = Math.PI / 2;
      chuck.position.set(0, 1.8, z);
      group.add(chuck);
    }

    // Pneumatic Tension Brake Disc
    const brakeDisc = new THREE.Mesh(new THREE.CylinderGeometry(0.42, 0.42, 0.06, 28), chromeMat);
    brakeDisc.rotation.x = Math.PI / 2;
    brakeDisc.position.set(0, 1.8, -1.38);
    group.add(brakeDisc);

    // Dancer Tension Arm
    const dancerArm = new THREE.Mesh(new THREE.BoxGeometry(0.8, 0.08, 0.12), frameMat);
    dancerArm.position.set(0.6, 1.4, 1.25);
    group.add(dancerArm);

    const dancerRoller = new THREE.Mesh(new THREE.CylinderGeometry(0.08, 0.08, 2.2, 24), chromeMat);
    dancerRoller.rotation.x = Math.PI / 2;
    dancerRoller.position.set(1.0, 1.4, 0);
    group.add(dancerRoller);

    return group;
  }

  private buildSlotDieCoater(): THREE.Group {
    const group = new THREE.Group();
    const frameMat = new THREE.MeshStandardMaterial({ color: 0x14222B, roughness: 0.40, metalness: 0.55 });
    const steelMat = new THREE.MeshStandardMaterial({ color: 0xE2E8F0, metalness: 0.90, roughness: 0.16 });
    const chromeMat = new THREE.MeshStandardMaterial({ color: 0xF8FAFC, metalness: 0.98, roughness: 0.04 });
    const brassMat = new THREE.MeshStandardMaterial({ color: 0xD4AF37, metalness: 0.88, roughness: 0.18 });
    const chockMat = new THREE.MeshStandardMaterial({ color: 0x20303C, roughness: 0.35, metalness: 0.60 });
    const hoseMat = new THREE.MeshStandardMaterial({ color: 0x94A3B8, metalness: 0.75, roughness: 0.28 });
    const panelMat = new THREE.MeshStandardMaterial({ color: 0xDFE7EC, roughness: 0.25, metalness: 0.25 });

    // 1. Machine Bed & Catch Pan
    const base = new THREE.Mesh(new THREE.BoxGeometry(4.8, 0.48, 3.6), frameMat);
    base.position.y = 0.24;
    base.receiveShadow = true;
    group.add(base);

    // Leveling feet
    for (const x of [-2.1, 2.1]) {
      for (const z of [-1.5, 1.5]) {
        const foot = new THREE.Mesh(new THREE.BoxGeometry(0.42, 0.14, 0.42), frameMat);
        foot.position.set(x, 0.07, z);
        group.add(foot);
      }
    }

    const dripTray = new THREE.Mesh(new THREE.BoxGeometry(3.8, 0.08, 2.7), steelMat);
    dripTray.position.set(0, 0.52, 0);
    group.add(dripTray);

    // 2. Precision Ground Backing Roll (Aligned along Z)
    this.backingRoll = new THREE.Mesh(new THREE.CylinderGeometry(0.55, 0.55, 2.35, 48), chromeMat);
    this.backingRoll.name = 'Backing roll';
    this.backingRoll.rotation.x = Math.PI / 2;
    this.backingRoll.position.set(0, 1.8, 0);
    this.backingRoll.castShadow = true;
    group.add(this.backingRoll);

    // Heavy Cast Bearing Pedestals & Split Chocks
    for (const z of [-1.45, 1.45]) {
      const ped = new THREE.Mesh(new THREE.BoxGeometry(0.95, 0.95, 0.48), frameMat);
      ped.position.set(0, 1.1, z);
      group.add(ped);

      const chock = new THREE.Mesh(new THREE.BoxGeometry(0.74, 0.74, 0.36), chockMat);
      chock.position.set(0, 1.8, z);
      group.add(chock);

      const flange = new THREE.Mesh(new THREE.CylinderGeometry(0.32, 0.32, 0.08, 24), steelMat);
      flange.rotation.x = Math.PI / 2;
      flange.position.set(0, 1.8, z + (z > 0 ? 0.19 : -0.19));
      group.add(flange);

      const seal = new THREE.Mesh(new THREE.CylinderGeometry(0.22, 0.22, 0.05, 24), brassMat);
      seal.rotation.x = Math.PI / 2;
      seal.position.set(0, 1.8, z + (z > 0 ? 0.24 : -0.24));
      group.add(seal);

      // Vertical Portal Columns for Slot-Die Traverse
      const tower = new THREE.Mesh(new THREE.BoxGeometry(0.38, 2.8, 0.42), frameMat);
      tower.position.set(0.78, 2.1, z);
      tower.castShadow = true;
      group.add(tower);

      // Hardened Machined Way Rails
      const rail = new THREE.Mesh(new THREE.BoxGeometry(0.08, 2.5, 0.10), steelMat);
      rail.position.set(0.60, 2.1, z);
      group.add(rail);

      // Cross-Slide Carriage
      const slide = new THREE.Mesh(new THREE.BoxGeometry(1.45, 0.26, 0.34), chockMat);
      slide.name = `Die cross slide ${z}`;
      slide.position.set(0.25, 2.88, z);
      group.add(slide);

      const micrometer = new THREE.Mesh(new THREE.CylinderGeometry(0.07, 0.07, 0.42, 24), brassMat);
      micrometer.position.set(-0.14, 3.10, z);
      group.add(micrometer);
    }

    // Overhead Tie Beam
    const crossBeam = new THREE.Mesh(new THREE.BoxGeometry(0.38, 0.28, 3.32), frameMat);
    crossBeam.position.set(0.78, 3.45, 0);
    group.add(crossBeam);

    // 3. Precision Slot-Die Coating Head (Upper & Lower Bodies)
    const dieLower = new THREE.Mesh(new THREE.BoxGeometry(0.62, 0.32, 2.42), steelMat);
    dieLower.position.set(-0.25, 2.36, 0);
    dieLower.castShadow = true;
    group.add(dieLower);

    const dieUpper = new THREE.Mesh(new THREE.BoxGeometry(0.58, 0.32, 2.42), steelMat);
    dieUpper.name = 'Slot die manifold';
    dieUpper.position.set(-0.25, 2.68, 0);
    dieUpper.castShadow = true;
    group.add(dieUpper);

    // Precision Ground Mirror Chrome Die Lips
    const dieLipUpper = new THREE.Mesh(new THREE.BoxGeometry(0.18, 0.10, 2.38), chromeMat);
    dieLipUpper.position.set(-0.06, 2.54, 0);
    group.add(dieLipUpper);

    const dieLipLower = new THREE.Mesh(new THREE.BoxGeometry(0.18, 0.10, 2.38), chromeMat);
    dieLipLower.position.set(-0.06, 2.48, 0);
    group.add(dieLipLower);

    // Fluid Distribution Manifold on Die Back
    const manifoldBlock = new THREE.Mesh(new THREE.BoxGeometry(0.22, 0.24, 2.2), steelMat);
    manifoldBlock.position.set(-0.58, 2.68, 0);
    group.add(manifoldBlock);

    // 14 Differential Lip Adjustment Micrometer Screws
    for (let i = 0; i < 14; i++) {
      const zSc = -1.02 + (i / 13) * 2.04;
      const screw = new THREE.Mesh(new THREE.CylinderGeometry(0.028, 0.028, 0.14, 16), brassMat);
      screw.rotation.z = Math.PI / 2;
      screw.position.set(-0.62, 2.76, zSc);
      group.add(screw);
    }

    // Stainless Slurry Feed Pipe Line (Realistic smooth CatmullRom spline)
    const hoseCurve = new THREE.CatmullRomCurve3([
      new THREE.Vector3(-0.60, 2.68, -0.7),
      new THREE.Vector3(-1.10, 2.80, -1.2),
      new THREE.Vector3(-1.20, 2.20, -1.8),
      new THREE.Vector3(1.10, 1.40, -2.1)
    ]);
    const slurryHose = new THREE.Mesh(new THREE.TubeGeometry(hoseCurve, 32, 0.055, 16, false), hoseMat);
    group.add(slurryHose);

    // Slurry Feed Pressure Gauge
    const slurryGauge = this.createGaugeMesh(0.15, 'DIE PRESSURE', '2.4 bar');
    slurryGauge.position.set(-0.62, 3.10, 0);
    slurryGauge.rotation.y = -Math.PI / 2;
    group.add(slurryGauge);

    // Optical / Laser Triangulation Gap Micrometer Display facing camera
    const laserSensor = this.createDigitalDisplayMesh(0.65, 0.40, 'LASER GAP MICROMETER', '85.2 µm', 'TOLERANCE: ±0.5 µm');
    laserSensor.position.set(0.78, 2.75, 1.68);
    laserSensor.rotation.y = 0;
    group.add(laserSensor);

    // Metering Pump Cabinet in back
    const pumpCab = new THREE.Mesh(new THREE.BoxGeometry(1.3, 1.3, 0.8), panelMat);
    pumpCab.position.set(1.4, 0.95, -2.1);
    group.add(pumpCab);

    const pumpMotor = new THREE.Mesh(new THREE.CylinderGeometry(0.18, 0.18, 0.55, 24), frameMat);
    pumpMotor.rotation.x = Math.PI / 2;
    pumpMotor.position.set(1.4, 1.35, -1.6);
    group.add(pumpMotor);

    // 4. Infeed & Outfeed Precision Web Guide Rollers (Machined tool steel with brass collars)
    for (const [x, y] of [[-2.1, 1.84], [2.1, 1.84]] as const) {
      const guide = new THREE.Mesh(new THREE.CylinderGeometry(0.12, 0.12, 2.35, 32), steelMat);
      guide.rotation.x = Math.PI / 2;
      guide.position.set(x, y, 0);
      guide.castShadow = true;
      group.add(guide);

      for (const z of [-1.22, 1.22]) {
        const collar = new THREE.Mesh(new THREE.CylinderGeometry(0.15, 0.15, 0.05, 24), brassMat);
        collar.rotation.x = Math.PI / 2;
        collar.position.set(x, y, z);
        group.add(collar);

        const bracket = new THREE.Mesh(new THREE.BoxGeometry(0.16, 0.42, 0.16), frameMat);
        bracket.position.set(x, y - 0.22, z);
        group.add(bracket);
      }
    }

    // 5. Operator Console & Controls
    const consoleUnit = new THREE.Mesh(new THREE.BoxGeometry(0.82, 0.62, 0.22), panelMat);
    consoleUnit.position.set(2.2, 1.85, 1.85);
    group.add(consoleUnit);

    const disp = this.createDigitalDisplayMesh(0.62, 0.38, 'SLOT-DIE COATER', 'GAP: 85.0 µm', 'WEB: 25.0 m/min');
    disp.position.set(2.2, 1.85, 1.97);
    group.add(disp);

    const stackLight = this.createStackLightMesh();
    stackLight.position.set(1.95, 2.22, 1.85);
    group.add(stackLight);

    const cautionPlate = this.createWarningLabelMesh(0.85, 0.38, 'CAUTION: PINCH POINT', 'HIGH PRECISION DIE LIPS');
    cautionPlate.position.set(0, 0.26, 1.81);
    group.add(cautionPlate);

    return group;
  }

  private buildDryingTunnel(): THREE.Group {
    const group = new THREE.Group();
    const ovenHousingMat = new THREE.MeshStandardMaterial({ color: 0xE8ECEE, roughness: 0.28, metalness: 0.25 });
    const frameMat = new THREE.MeshStandardMaterial({ color: 0x1E293B, roughness: 0.35, metalness: 0.45 });
    const steelMat = new THREE.MeshStandardMaterial({ color: 0xDFE7E9, metalness: 0.90, roughness: 0.18 });
    const chromeMat = new THREE.MeshStandardMaterial({ color: 0xFFFFFF, metalness: 0.98, roughness: 0.05 });
    const windowMat = new THREE.MeshPhysicalMaterial({
      color: 0x243E4C,
      transparent: true,
      opacity: 0.45,
      roughness: 0.08,
      metalness: 0.1,
      transmission: 0.85
    });
    const quartzGlowMat = new THREE.MeshStandardMaterial({
      color: 0xFF8833,
      emissive: 0xFF6600,
      emissiveIntensity: 2.2,
      roughness: 0.15
    });

    // Main 3-Zone Enclosure (7.2m long)
    const tunnel = new THREE.Mesh(new THREE.BoxGeometry(7.2, 2.4, 2.8), ovenHousingMat);
    tunnel.name = 'Dryer removable housing';
    tunnel.position.set(0, 1.9, 0);
    tunnel.castShadow = true;
    tunnel.receiveShadow = true;
    group.add(tunnel);
    this.dryerTunnel = tunnel;

    // Structural perimeter partition bands
    for (const x of [-3.6, -1.2, 1.2, 3.6]) {
      const band = new THREE.Mesh(new THREE.BoxGeometry(0.14, 2.5, 2.88), frameMat);
      band.position.set(x, 1.9, 0);
      group.add(band);
    }

    // Heavy Support Legs
    for (const x of [-3.2, -1.2, 1.2, 3.2]) {
      for (const z of [-1.2, 1.2]) {
        const leg = new THREE.Mesh(new THREE.BoxGeometry(0.26, 0.72, 0.26), frameMat);
        leg.position.set(x, 0.36, z);
        group.add(leg);

        const pad = new THREE.Mesh(new THREE.BoxGeometry(0.44, 0.12, 0.44), frameMat);
        pad.position.set(x, 0.06, z);
        group.add(pad);
      }
    }

    // 3 Recessed Inspection Viewports with Glowing IR Quartz Heating Elements Inside
    for (const x of [-2.4, 0.0, 2.4]) {
      const winFrame = new THREE.Mesh(new THREE.BoxGeometry(1.4, 0.9, 0.12), steelMat);
      winFrame.position.set(x, 2.0, 1.42);
      group.add(winFrame);

      const win = new THREE.Mesh(new THREE.PlaneGeometry(1.2, 0.72), windowMat);
      win.position.set(x, 2.0, 1.49);
      group.add(win);

      // Glowing Quartz IR Heater Tubes inside each zone
      for (const yIr of [1.75, 2.05, 2.35]) {
        const tube = new THREE.Mesh(new THREE.CylinderGeometry(0.035, 0.035, 1.6, 16), quartzGlowMat);
        tube.rotation.z = Math.PI / 2;
        tube.position.set(x, yIr, 0.4);
        group.add(tube);

        const reflector = new THREE.Mesh(new THREE.BoxGeometry(1.6, 0.08, 0.25), chromeMat);
        reflector.position.set(x, yIr, 0.25);
        group.add(reflector);
      }

      // Radiant PointLight shining through window
      const irLight = new THREE.PointLight(0xF59E42, 2.4, 7.0);
      irLight.position.set(x, 2.0, 0.2);
      this.irLamps.push(irLight);
      group.add(irLight);

      // Cleanroom door latch handle
      const latch = new THREE.Mesh(new THREE.BoxGeometry(0.08, 0.24, 0.06), chromeMat);
      latch.position.set(x + 0.8, 2.0, 1.48);
      group.add(latch);
    }

    // Top Exhaust Chimneys with Centrifugal Blower Housings
    for (const x of [-1.8, 1.8]) {
      const blower = new THREE.Mesh(new THREE.CylinderGeometry(0.45, 0.45, 0.48, 24), frameMat);
      blower.rotation.x = Math.PI / 2;
      blower.position.set(x, 3.4, 0);
      group.add(blower);

      const motor = new THREE.Mesh(new THREE.CylinderGeometry(0.22, 0.22, 0.55, 24), new THREE.MeshStandardMaterial({ color: 0x243E4C, roughness: 0.35, metalness: 0.45 }));
      motor.rotation.x = Math.PI / 2;
      motor.position.set(x, 3.4, 0.45);
      group.add(motor);

      const duct = new THREE.Mesh(new THREE.CylinderGeometry(0.32, 0.32, 1.2, 24), steelMat);
      duct.position.set(x, 4.0, 0);
      group.add(duct);

      const damper = new THREE.Mesh(new THREE.CylinderGeometry(0.36, 0.36, 0.08, 24), frameMat);
      damper.position.set(x, 4.6, 0);
      group.add(damper);
    }

    // Hazard Caution Plate
    const caution = this.createWarningLabelMesh(1.1, 0.45, 'IR RADIATION HAZARD', 'HIGH TEMPERATURE ZONE 95°C');
    caution.position.set(0, 2.7, 1.42);
    group.add(caution);

    return group;
  }

  private buildPrecisionCalender(): THREE.Group {
    const group = new THREE.Group();
    const chromeMat = new THREE.MeshStandardMaterial({ color: 0xF8FAFC, metalness: 0.98, roughness: 0.04 });
    const frameMat = new THREE.MeshStandardMaterial({ color: 0x14222B, roughness: 0.40, metalness: 0.55 });
    const chockMat = new THREE.MeshStandardMaterial({ color: 0x20303C, roughness: 0.35, metalness: 0.60 });
    const panelMat = new THREE.MeshStandardMaterial({ color: 0xDFE7EC, roughness: 0.25, metalness: 0.25 });
    const brassMat = new THREE.MeshStandardMaterial({ color: 0xD4AF37, metalness: 0.88, roughness: 0.18 });
    const steelMat = new THREE.MeshStandardMaterial({ color: 0xE2E8F0, metalness: 0.92, roughness: 0.12 });
    const ramMat = new THREE.MeshStandardMaterial({ color: 0x20303C, roughness: 0.30, metalness: 0.65 });

    // Machine Bed & Fascia
    const bed = new THREE.Mesh(new THREE.BoxGeometry(3.8, 0.48, 3.6), frameMat);
    bed.position.y = 0.24;
    bed.receiveShadow = true;
    group.add(bed);

    for (const z of [-1.81, 1.81]) {
      const fascia = new THREE.Mesh(new THREE.BoxGeometry(3.5, 0.28, 0.04), panelMat);
      fascia.position.set(0, 0.26, z);
      group.add(fascia);
    }

    // Leveling feet
    for (const x of [-1.6, 1.6]) {
      for (const z of [-1.5, 1.5]) {
        const foot = new THREE.Mesh(new THREE.BoxGeometry(0.42, 0.16, 0.42), frameMat);
        foot.position.set(x, 0.08, z);
        group.add(foot);
      }
    }

    // Twin Portal Columns & Loading Bridge
    for (const z of [-1.45, 1.45]) {
      for (const x of [-0.92, 0.92]) {
        const col = new THREE.Mesh(new THREE.BoxGeometry(0.36, 3.8, 0.52), frameMat);
        col.position.set(x, 2.3, z);
        col.castShadow = true;
        group.add(col);

        const rail = new THREE.Mesh(new THREE.BoxGeometry(0.10, 3.2, 0.06), steelMat);
        rail.position.set(x, 2.3, z + (z < 0 ? 0.27 : -0.27));
        group.add(rail);
      }

      const bridge = new THREE.Mesh(new THREE.BoxGeometry(2.35, 0.52, 0.62), frameMat);
      bridge.position.set(0, 4.25, z);
      group.add(bridge);

      // Hydraulic Rams & Pistons
      const ram = new THREE.Mesh(new THREE.CylinderGeometry(0.22, 0.22, 0.68, 24), ramMat);
      ram.name = `Hydraulic loading ram ${z}`;
      ram.position.set(0, 4.65, z);
      group.add(ram);

      const piston = new THREE.Mesh(new THREE.CylinderGeometry(0.12, 0.12, 0.60, 24), chromeMat);
      piston.name = `Chrome loading piston ${z}`;
      piston.position.set(0, 4.05, z);
      group.add(piston);

      // Bearing Chocks
      for (const [y, chockName] of [[2.5, 'top'], [1.5, 'bottom']] as const) {
        const chock = new THREE.Mesh(new THREE.BoxGeometry(0.84, 0.74, 0.36), chockMat);
        chock.name = `Split bearing chock ${chockName} ${z}`;
        chock.position.set(0, y, z);
        chock.castShadow = true;
        group.add(chock);

        const flange = new THREE.Mesh(new THREE.CylinderGeometry(0.32, 0.32, 0.08, 24), steelMat);
        flange.rotation.x = Math.PI / 2;
        flange.position.set(0, y, z + (z > 0 ? 0.19 : -0.19));
        group.add(flange);

        const seal = new THREE.Mesh(new THREE.CylinderGeometry(0.22, 0.22, 0.05, 24), brassMat);
        seal.rotation.x = Math.PI / 2;
        seal.position.set(0, y, z + (z > 0 ? 0.24 : -0.24));
        group.add(seal);
      }
    }

    // 1. TOP HARDENED CALENDER ROLL
    this.calenderRollTop = new THREE.Mesh(new THREE.CylinderGeometry(0.48, 0.48, 2.35, 48), chromeMat);
    this.calenderRollTop.name = 'Upper precision roll';
    this.calenderRollTop.rotation.x = Math.PI / 2;
    this.calenderRollTop.position.set(0, 2.5, 0);
    this.calenderRollTop.castShadow = true;
    group.add(this.calenderRollTop);

    // 2. BOTTOM HARDENED CALENDER ROLL
    this.calenderRollBottom = new THREE.Mesh(new THREE.CylinderGeometry(0.48, 0.48, 2.35, 48), chromeMat);
    this.calenderRollBottom.name = 'Lower precision roll';
    this.calenderRollBottom.rotation.x = Math.PI / 2;
    this.calenderRollBottom.position.set(0, 1.5, 0);
    this.calenderRollBottom.castShadow = true;
    group.add(this.calenderRollBottom);

    // Web Guide Rollers
    for (const x of [-1.75, 1.75]) {
      const guide = new THREE.Mesh(new THREE.CylinderGeometry(0.14, 0.14, 2.35, 32), chromeMat);
      guide.rotation.x = Math.PI / 2;
      guide.position.set(x, 2.0, 0);
      guide.castShadow = true;
      group.add(guide);
    }

    // Dual Analog Pressure Gauges (Nip Hydraulic Force)
    const gauge1 = this.createGaugeMesh(0.16, 'NIP PRESSURE', '18.5 bar');
    gauge1.position.set(-0.95, 3.2, 1.74);
    group.add(gauge1);

    const gauge2 = this.createGaugeMesh(0.16, 'FORCE LOAD', '120 kN');
    gauge2.position.set(0.95, 3.2, 1.74);
    group.add(gauge2);

    // Optical Gap Micrometer Readouts
    const micrometerDisp = this.createDigitalDisplayMesh(0.65, 0.4, 'OPTICAL GAP MICROMETER', '124.5 µm', 'COMPACTION: 32.4%');
    micrometerDisp.position.set(0, 3.2, 1.74);
    group.add(micrometerDisp);

    // Electrical Drive Cabinet
    const cabinet = new THREE.Mesh(new THREE.BoxGeometry(2.0, 2.2, 0.8), panelMat);
    cabinet.position.set(0, 1.5, -2.4);
    group.add(cabinet);

    const cautionPlate = this.createWarningLabelMesh(0.8, 0.35, 'HIGH VOLTAGE 480V', 'LOCKOUT BEFORE SERVICE');
    cautionPlate.position.set(0, 2.0, -2.81);
    cautionPlate.rotation.y = Math.PI;
    group.add(cautionPlate);

    // Operator Console
    const consoleUnit = new THREE.Mesh(new THREE.BoxGeometry(0.82, 0.62, 0.22), panelMat);
    consoleUnit.position.set(2.2, 1.85, 1.85);
    group.add(consoleUnit);

    const hmiDisp = this.createDigitalDisplayMesh(0.62, 0.38, 'CALENDER HMI', 'ROLL TEMP: 115°C', 'SPEED: 25.0 m/min');
    hmiDisp.position.set(2.2, 1.85, 1.97);
    group.add(hmiDisp);

    const stackLight = this.createStackLightMesh();
    stackLight.position.set(1.9, 2.2, 1.85);
    group.add(stackLight);

    return group;
  }

  private buildCyclerStation(): THREE.Group {
    const group = new THREE.Group();
    const rackMat = new THREE.MeshStandardMaterial({ color: 0x1A202C, roughness: 0.4, metalness: 0.5 });
    const panelMat = new THREE.MeshStandardMaterial({ color: 0xCFD8DC, roughness: 0.25, metalness: 0.85 });
    const steelMat = new THREE.MeshStandardMaterial({ color: 0xDFE7E9, metalness: 0.90, roughness: 0.18 });
    const goldMat = new THREE.MeshStandardMaterial({ color: 0xD4AF37, metalness: 0.95, roughness: 0.15 });
    const chromeMat = new THREE.MeshStandardMaterial({ color: 0xFFFFFF, metalness: 0.98, roughness: 0.05 });

    // 19-Inch 42U Server Rack Cabinet
    const rack = new THREE.Mesh(new THREE.BoxGeometry(2.4, 4.4, 1.8), rackMat);
    rack.name = 'Cycler server rack';
    rack.position.set(0, 2.2, 0);
    rack.castShadow = true;
    group.add(rack);

    // 6 Individual Rack-Mounted Cycler Sub-Chassis Modules
    for (let row = 0; row < 6; row++) {
      const yMod = 0.85 + row * 0.54;
      const chassis = new THREE.Mesh(new THREE.BoxGeometry(2.15, 0.46, 0.06), panelMat);
      chassis.name = `Cycler module ${row}`;
      chassis.position.set(0, yMod, 0.92);
      group.add(chassis);

      // Chrome pull grab handles
      for (const xH of [-0.95, 0.95]) {
        const handle = new THREE.Mesh(new THREE.CylinderGeometry(0.02, 0.02, 0.32, 12), chromeMat);
        handle.position.set(xH, yMod, 0.98);
        group.add(handle);
      }

      // Mini OLED Channel Status Display
      const miniDisp = this.createDigitalDisplayMesh(0.45, 0.18, `CH ${row * 8 + 1}-${row * 8 + 8}`, '4.20 V', '2.50 A');
      miniDisp.position.set(-0.45, yMod, 0.96);
      group.add(miniDisp);

      // 8 Channel Status LEDs & Gold BNC Terminals
      for (let c = 0; c < 8; c++) {
        const colX = 0.05 + (c / 7) * 0.8;
        const ledMat = new THREE.MeshBasicMaterial({ color: (row + c) % 3 === 0 ? 0xF59E42 : 0x10B981 });
        const led = new THREE.Mesh(new THREE.SphereGeometry(0.025, 12, 12), ledMat);
        led.position.set(colX, yMod + 0.08, 0.96);
        group.add(led);

        const bnc = new THREE.Mesh(new THREE.CylinderGeometry(0.025, 0.025, 0.03, 12), goldMat);
        bnc.rotation.x = Math.PI / 2;
        bnc.position.set(colX, yMod - 0.08, 0.96);
        group.add(bnc);
      }
    }

    // Slide-Out Coin-Cell Test Fixture Shelf
    const shelf = new THREE.Mesh(new THREE.BoxGeometry(2.1, 0.10, 0.85), steelMat);
    shelf.position.set(0, 1.4, 1.25);
    group.add(shelf);

    for (let i = 0; i < 6; i++) {
      const xC = -0.8 + (i / 5) * 1.6;
      const coinCell = new THREE.Mesh(new THREE.CylinderGeometry(0.12, 0.12, 0.04, 24), goldMat);
      coinCell.position.set(xC, 1.48, 1.25);
      group.add(coinCell);
    }

    // Cable Trays & Overhead Stack Light
    for (const xTray of [-1.15, 1.15]) {
      const tray = new THREE.Mesh(new THREE.BoxGeometry(0.08, 3.8, 0.25), new THREE.MeshStandardMaterial({ color: 0x087F8C, metalness: 0.5 }));
      tray.position.set(xTray, 2.2, 0.88);
      group.add(tray);
    }

    const stackLight = this.createStackLightMesh();
    stackLight.position.set(1.0, 4.45, 0);
    group.add(stackLight);

    return group;
  }

  private buildElectrodeWeb() {
    // Continuous strip connecting the entire plant from X = -6 to X = 19
    const foilGeo = new THREE.PlaneGeometry(25, 1.8, 64, 1);
    this.foilMaterial = new THREE.MeshStandardMaterial({
      color: 0xD4D8DB, // Aluminum by default for Warwick
      metalness: 0.94,
      roughness: 0.16,
      side: THREE.DoubleSide
    });
    this.foilStrip = new THREE.Mesh(foilGeo, this.foilMaterial);
    this.foilStrip.rotation.x = -Math.PI / 2;
    this.foilStrip.position.set(6.5, 2.0, 0);
    this.group.add(this.foilStrip);

    // Coated Active Material (Top layer, starts after coater at X = -2.5 to X = 19)
    const coatedGeo = new THREE.PlaneGeometry(21.5, 1.7, 64, 1);
    this.coatedMaterial = new THREE.MeshStandardMaterial({
      color: 0x393E46, // NMC622 by default
      roughness: 0.78,
      metalness: 0.15,
      side: THREE.DoubleSide
    });
    this.coatedLayer = new THREE.Mesh(coatedGeo, this.coatedMaterial);
    this.coatedLayer.rotation.x = -Math.PI / 2;
    this.coatedLayer.position.set(8.25, 2.01, 0);
    this.group.add(this.coatedLayer);
  }

  private buildProcessOverlays() {
    const markerMaterial = new THREE.MeshBasicMaterial({ color: 0x38B7C4, transparent: true, opacity: 0.72, side: THREE.DoubleSide });
    for (let i = 0; i < 18; i++) {
      const marker = new THREE.Mesh(new THREE.PlaneGeometry(0.08, 1.62), markerMaterial);
      marker.rotation.x = -Math.PI / 2;
      marker.position.set(-5.5 + i * 1.45, 2.035, 0);
      marker.visible = false;
      this.webMotionMarkers.push(marker);
      this.group.add(marker);
    }

    this.coatingBead = new THREE.Mesh(
      new THREE.BoxGeometry(0.12, 0.055, 2.1),
      new THREE.MeshStandardMaterial({ color: 0x087F8C, emissive: 0x087F8C, emissiveIntensity: 0.5, roughness: 0.35 }),
    );
    this.coatingBead.position.set(-2.6, 2.065, 0);
    this.coatingBead.visible = false;
    this.group.add(this.coatingBead);

    this.nipResponse = new THREE.Mesh(
      new THREE.BoxGeometry(1.7, 0.07, 1.75),
      new THREE.MeshStandardMaterial({ color: 0xF59E42, emissive: 0xF59E42, emissiveIntensity: 0.28, transparent: true, opacity: 0.7 }),
    );
    this.nipResponse.position.set(11.65, 2.045, 0);
    this.nipResponse.visible = false;
    this.group.add(this.nipResponse);

    const chart = new THREE.Group();
    chart.position.set(17.5, 2.4, 1.25);
    chart.add(new THREE.LineSegments(
      new THREE.EdgesGeometry(new THREE.PlaneGeometry(2.1, 1.25)),
      new THREE.LineBasicMaterial({ color: 0xD1E3E7 }),
    ));
    const measurementMarker = new THREE.Mesh(
      new THREE.RingGeometry(.13, .2, 32),
      new THREE.MeshBasicMaterial({ color: 0x087F8C, side: THREE.DoubleSide }),
    );
    measurementMarker.position.z = .01;
    chart.add(measurementMarker);
    chart.visible = false;
    this.characterizationTrace = chart;
    this.group.add(chart);

    const assembly = new THREE.Group();
    assembly.position.set(14.5, 2.05, 0);
    const disc = new THREE.Mesh(
      new THREE.CylinderGeometry(.58, .58, .075, 48),
      new THREE.MeshStandardMaterial({ color: 0xD6DDE0, metalness: .78, roughness: .26 }),
    );
    disc.rotation.x = Math.PI / 2;
    assembly.add(disc);
    const probe = new THREE.Mesh(
      new THREE.CylinderGeometry(.055, .055, 1.35, 18),
      new THREE.MeshStandardMaterial({ color: 0x087F8C, emissive: 0x087F8C, emissiveIntensity: .3 }),
    );
    probe.position.set(0, 1.05, .15);
    probe.userData.restY = probe.position.y;
    assembly.add(probe);
    assembly.visible = false;
    this.assemblyProbe = assembly;
    this.group.add(assembly);
  }

  private addAnnotation(text: string, position: THREE.Vector3, color = '#087F8C') {
    const canvas = document.createElement('canvas');
    canvas.width = 512;
    canvas.height = 112;
    const context = canvas.getContext('2d')!;
    context.fillStyle = 'rgba(20,42,53,.92)';
    context.beginPath();
    context.roundRect(4, 4, 504, 104, 26);
    context.fill();
    context.fillStyle = color;
    context.fillRect(4, 4, 12, 104);
    context.fillStyle = '#ffffff';
    context.font = '600 30px system-ui, sans-serif';
    context.textBaseline = 'middle';
    context.fillText(text, 38, 56, 450);
    const texture = new THREE.CanvasTexture(canvas);
    texture.colorSpace = THREE.SRGBColorSpace;
    const sprite = new THREE.Sprite(new THREE.SpriteMaterial({ map: texture, transparent: true, depthTest: false }));
    sprite.position.copy(position);
    sprite.scale.set(1.65, .36, 1);
    sprite.renderOrder = 20;
    this.annotationGroup.add(sprite);
    const anchor = position.clone().add(new THREE.Vector3(0, -.42, 0));
    this.annotationGroup.add(new THREE.Line(
      new THREE.BufferGeometry().setFromPoints([anchor, anchor.clone().add(new THREE.Vector3(0, -1.05, -.25))]),
      new THREE.LineBasicMaterial({ color: new THREE.Color(color), transparent: true, opacity: .72, depthTest: false }),
    ));
  }

  private clearAnnotations() {
    this.annotationGroup.traverse((object) => {
      if (object instanceof THREE.Sprite) {
        object.material.map?.dispose();
        object.material.dispose();
      } else if (object instanceof THREE.Line) {
        object.geometry.dispose();
        (object.material as THREE.Material).dispose();
      }
    });
    this.annotationGroup.clear();
  }

  private registerStageHitbox(stageId: string, parentGroup: THREE.Group, beaconPos: THREE.Vector3) {
    // 1. Invisible Click Hitbox
    const hitboxGeo = new THREE.BoxGeometry(4.2, 5.2, 4.2);
    const hitboxMat = new THREE.MeshBasicMaterial({ visible: false });
    const hitbox = new THREE.Mesh(hitboxGeo, hitboxMat);
    hitbox.position.set(beaconPos.x, beaconPos.y - 1.0, beaconPos.z);
    hitbox.userData = { stageId };
    this.group.add(hitbox);
    this.stageHitboxes.push(hitbox);

    // 2. Visual Tech-Teal Floating Beacon Ring above the stage
    const beaconGroup = new THREE.Group();
    beaconGroup.position.copy(beaconPos);

    const ringGeo = new THREE.RingGeometry(0.45, 0.55, 32);
    const ringMat = new THREE.MeshBasicMaterial({
      color: 0x087F8C,
      side: THREE.DoubleSide,
      transparent: true,
      opacity: 0.75
    });
    const ring = new THREE.Mesh(ringGeo, ringMat);
    ring.rotation.x = Math.PI / 2;
    beaconGroup.add(ring);

    const markerPinGeo = new THREE.ConeGeometry(0.18, 0.4, 16);
    const markerPinMat = new THREE.MeshStandardMaterial({
      color: 0x087F8C,
      emissive: 0x087F8C,
      emissiveIntensity: 0.4
    });
    const markerPin = new THREE.Mesh(markerPinGeo, markerPinMat);
    markerPin.rotation.x = Math.PI;
    markerPin.position.y = 0.4;
    beaconGroup.add(markerPin);

    this.group.add(beaconGroup);
    this.interactiveBeacons.push(beaconGroup);
  }

  public resolveStageId(hitboxStageId: string): string {
    if (this.activeScenario === 'warwick_nmc622_calendering') {
      if (hitboxStageId === 'formulation' || hitboxStageId === 'mixing') return 'slurry_prep';
      if (hitboxStageId === 'coating') return 'pilot_coating';
      if (hitboxStageId === 'calendering') return 'calendering';
      if (hitboxStageId === 'characterization') return 'rate_characterization';
      return hitboxStageId;
    } else {
      return hitboxStageId;
    }
  }

  public setScenario(scenarioId: ScenarioId) {
    this.activeScenario = scenarioId;
    if (scenarioId === 'drakopoulos_graphite') {
      // Copper substrate & graphite dark coating
      this.foilMaterial.color.setHex(0xC87533);
      this.foilMaterial.roughness = 0.22;
      this.coatedMaterial.color.setHex(0x1C2321);
      this.coatedMaterial.roughness = 0.88;
    } else {
      // Aluminum substrate & NMC622 slate coating
      this.foilMaterial.color.setHex(0xD4D8DB);
      this.foilMaterial.roughness = 0.16;
      this.coatedMaterial.color.setHex(0x393E46);
      this.coatedMaterial.roughness = 0.78;
    }
  }

  public setProcessExperience(stageId: string, phase: ProcessExperiencePhase, decision?: ReplayStep) {
    this.processStageId = stageId;
    this.processPhase = phase;
    this.interactiveBeacons.forEach((beacon) => { beacon.visible = stageId === 'overview'; });
    this.clearAnnotations();
    if (!decision || !['AI_DECISION', 'ILLUSTRATED_PROCESS_RUN', 'RESULT_REVEAL'].includes(phase)) return;

    const control = (key: string) => decision.controlsSummary[key];
    if (stageId === 'coating') {
      this.addAnnotation(`Web speed · ${control('coating_speed_m_per_min')} m/min`, new THREE.Vector3(-3.2, 3.65, 1.7));
      this.addAnnotation(`Coating gap · ${control('coating_gap_um')} µm`, new THREE.Vector3(-1.5, 3.15, 1.7), '#F59E42');
      this.addAnnotation(`Same recipe · dryer ${control('drying_temperature_c')} °C`, new THREE.Vector3(4, 4.45, 1.75));
    } else if (stageId === 'calendering') {
      this.addAnnotation(`Roll surface · ${control('roll_temperature_c')} °C`, new THREE.Vector3(10.2, 4.45, 1.85), '#F59E42');
      this.addAnnotation(`Target density · ${control('target_density_g_cm3')} g/cm³`, new THREE.Vector3(12.2, 3.35, 1.85));
      this.addAnnotation(`Loading · ${control('target_coating_weight_gsm')} g/m²`, new THREE.Vector3(11.2, 5.25, 1.6));
    }
    if (phase === 'RESULT_REVEAL') {
      const decimals = this.activeScenario === 'drakopoulos_graphite' ? 2 : 4;
      const unit = this.activeScenario === 'drakopoulos_graphite' ? 'mAh/g D30' : '5C/0.2C ratio';
      const x = stageId === 'coating' ? 1.8 : 13.1;
      this.addAnnotation(`Recorded · ${decision.revealedTarget.toFixed(decimals)} ${unit}`, new THREE.Vector3(x, 5.15, 1.9), '#F59E42');
    }
  }

  public getHitboxes(): THREE.Mesh[] {
    return this.stageHitboxes;
  }

  private rotateRoll(mesh: THREE.Mesh, amount: number) {
    const axis = mesh.userData.rotationAxis;
    if (axis === 'x') mesh.rotation.x += amount;
    else if (axis === 'z') mesh.rotation.z += amount;
    else mesh.rotation.y += amount;
  }

  public update(delta: number, elapsed: number) {
    // 1. Continuous synchronized roll rotation
    const processRunning = this.processPhase === 'ILLUSTRATED_PROCESS_RUN';
    const mechanismSpeed = processRunning ? 4.4 : 0.35;
    if (this.calenderRollTop && this.calenderRollBottom) {
      this.rotateRoll(this.calenderRollTop, delta * mechanismSpeed);
      this.rotateRoll(this.calenderRollBottom, -delta * mechanismSpeed);
    }
    if (this.backingRoll) {
      this.rotateRoll(this.backingRoll, delta * mechanismSpeed);
    }
    if (this.mixerShaft) {
      this.mixerShaft.rotation.y += delta * (this.processStageId === 'mixing' ? 5.8 : 0.6);
    }

    const cutawayTarget = this.processStageId !== 'overview' && this.processPhase !== 'PROCESS_EXIT' ? 1 : 0;
    this.processBlend = THREE.MathUtils.damp(this.processBlend, cutawayTarget, 4.5, delta);
    this.formulationHoppers.forEach((hopper, index) => {
      const rest = hopper.userData.restPosition as THREE.Vector3;
      hopper.position.copy(rest).add(new THREE.Vector3((index - 1) * .42 * this.processBlend * (['formulation', 'slurry_prep'].includes(this.processStageId) ? 1 : 0), .18 * this.processBlend * (['formulation', 'slurry_prep'].includes(this.processStageId) ? 1 : 0), 0));
    });
    this.coaterCutawayParts.forEach((part) => {
      const rest = part.userData.restPosition as THREE.Vector3;
      const active = ['coating', 'pilot_coating'].includes(this.processStageId) ? this.processBlend : 0;
      part.position.copy(rest);
      if (part.name.includes('Slot die manifold')) part.position.y += .42 * active;
      else part.position.z += Math.sign(rest.z || 1) * .28 * active;
    });
    this.calenderCutawayParts.forEach((part) => {
      const rest = part.userData.restPosition as THREE.Vector3;
      const active = this.processStageId === 'calendering' ? this.processBlend : 0;
      part.position.copy(rest);
      if (part.name.includes('Hydraulic') || part.name.includes('Chrome')) part.position.y += .28 * active;
      else part.position.z += Math.sign(rest.z || 1) * .38 * active;
    });

    if (this.mixerVessel) {
      const material = this.mixerVessel.material as THREE.MeshStandardMaterial;
      const active = this.processStageId === 'mixing' && this.processPhase !== 'PROCESS_EXIT';
      material.transparent = active;
      material.opacity = active ? .24 : 1;
      material.depthWrite = !active;
    }
    if (this.dryerTunnel) {
      const material = this.dryerTunnel.material as THREE.MeshStandardMaterial;
      const active = (this.processStageId === 'drying' || (this.processStageId === 'coating' && processRunning));
      material.transparent = active;
      material.opacity = active ? .2 : 1;
      material.depthWrite = !active;
    }

    this.webMotionMarkers.forEach((marker, index) => {
      marker.visible = processRunning;
      if (processRunning) marker.position.x = -5.5 + ((elapsed * 2.2 + index * 1.45) % 25);
    });
    if (this.coatingBead) {
      this.coatingBead.visible = processRunning && this.processStageId === 'coating';
      this.coatingBead.scale.y = .7 + Math.sin(elapsed * 7) * .18;
    }
    if (this.nipResponse) {
      this.nipResponse.visible = processRunning && this.processStageId === 'calendering';
      this.nipResponse.scale.x = .72 + Math.sin(elapsed * 5) * .08;
    }
    if (this.characterizationTrace) {
      this.characterizationTrace.visible = ['characterization', 'rate_characterization'].includes(this.processStageId) && this.processPhase !== 'PROCESS_ENTERING';
      this.characterizationTrace.scale.setScalar(THREE.MathUtils.lerp(.05, 1, this.processBlend));
    }
    if (this.assemblyProbe) {
      this.assemblyProbe.visible = this.processStageId === 'cell_assembly' && this.processPhase !== 'PROCESS_ENTERING';
      const probe = this.assemblyProbe.children[1];
      probe.position.y = probe.userData.restY + Math.sin(elapsed * 2.4) * .12;
    }

    // 2. Pulse interactive beacons
    this.interactiveBeacons.forEach((beacon, i) => {
      beacon.position.y += Math.sin(elapsed * 2.5 + i) * 0.0015;
      beacon.rotation.y += delta * 0.8;
    });

    // 3. Subtle breathing in IR lamps
    this.irLamps.forEach((lamp, i) => {
      const active = this.processStageId === 'drying' || (this.processStageId === 'coating' && processRunning);
      lamp.intensity = (active ? 3.2 : 1.6) + Math.sin(elapsed * 3.0 + i) * 0.35;
    });
  }
}
