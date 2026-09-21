import * as THREE from 'three';
import { ScenarioId } from '../../data/types';

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

  constructor(onStageClick?: StageInteractionCallback) {
    this.group = new THREE.Group();
    this.onStageClick = onStageClick;
    this.buildCleanroomEnvironment();
    this.buildManufacturingLine();
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
    this.registerStageHitbox('formulation', formulationGroup, new THREE.Vector3(-15.5, 3.5, 0));

    // 1. Stage 2: Planetary High-Shear Vacuum Mixer (X = -11)
    const mixerGroup = this.buildMixerUnit();
    mixerGroup.position.set(-11, 0, 0);
    this.group.add(mixerGroup);
    this.registerStageHitbox('mixing', mixerGroup, new THREE.Vector3(-11, 3.5, 0));

    // 2. Foil Uncoiler Reel (X = -6.5)
    const uncoilerGroup = this.buildUncoilerReel();
    uncoilerGroup.position.set(-6.5, 0, 0);
    this.group.add(uncoilerGroup);

    // 3. Stage 3: Precision Slot-Die Coater (X = -2.5)
    const coaterGroup = this.buildSlotDieCoater();
    coaterGroup.position.set(-2.5, 0, 0);
    this.group.add(coaterGroup);
    this.registerStageHitbox('coating', coaterGroup, new THREE.Vector3(-2.5, 3.5, 0));

    // 4. Stage 4: Convection & IR Drying Tunnel (X = 4)
    const dryerGroup = this.buildDryingTunnel();
    dryerGroup.position.set(4, 0, 0);
    this.group.add(dryerGroup);
    this.registerStageHitbox('drying', dryerGroup, new THREE.Vector3(4, 3.8, 0));

    // 5. Stage 5: Precision Calendering Machine (Signature Asset) (X = 11)
    const calenderGroup = this.buildPrecisionCalender();
    calenderGroup.position.set(11, 0, 0);
    this.group.add(calenderGroup);
    this.registerStageHitbox('calendering', calenderGroup, new THREE.Vector3(11, 4.2, 0));

    // 6. Stage 6: Electrochemical Half-Cell Test Station (X = 17.5)
    const cyclerGroup = this.buildCyclerStation();
    cyclerGroup.position.set(17.5, 0, 0);
    this.group.add(cyclerGroup);
    this.registerStageHitbox('characterization', cyclerGroup, new THREE.Vector3(17.5, 3.5, 0));

    // 7. Continuous Moving Electrode Foil Web
    this.buildElectrodeWeb();
  }

  private buildFormulationUnit(): THREE.Group {
    const group = new THREE.Group();
    const chassisMat = new THREE.MeshStandardMaterial({ color: 0x2C3E50, roughness: 0.35, metalness: 0.45 });
    const stainlessMat = new THREE.MeshStandardMaterial({ color: 0xDFE7E9, metalness: 0.88, roughness: 0.18 });
    const accentMat = new THREE.MeshStandardMaterial({ color: 0x087F8C, metalness: 0.6, roughness: 0.25 });
    const hopperMat = new THREE.MeshStandardMaterial({ color: 0xE8F4F2, metalness: 0.4, roughness: 0.2 });

    // Base pedestal
    const benchGeo = new THREE.BoxGeometry(3.2, 0.4, 2.8);
    const bench = new THREE.Mesh(benchGeo, chassisMat);
    bench.position.y = 0.2;
    bench.receiveShadow = true;
    group.add(bench);

    // Active Material Powder Silo (Tapered Cone Hopper)
    const silo1Geo = new THREE.CylinderGeometry(0.48, 0.22, 1.8, 24);
    const silo1 = new THREE.Mesh(silo1Geo, hopperMat);
    silo1.position.set(-0.85, 1.7, -0.4);
    silo1.castShadow = true;
    group.add(silo1);

    // Conductive Carbon Black Silo
    const silo2Geo = new THREE.CylinderGeometry(0.38, 0.18, 1.6, 24);
    const silo2 = new THREE.Mesh(silo2Geo, hopperMat);
    silo2.position.set(0.0, 1.6, -0.4);
    silo2.castShadow = true;
    group.add(silo2);

    // Polymer Binder Dispenser Meter
    const silo3Geo = new THREE.CylinderGeometry(0.32, 0.15, 1.5, 24);
    const silo3 = new THREE.Mesh(silo3Geo, stainlessMat);
    silo3.position.set(0.8, 1.55, -0.4);
    silo3.castShadow = true;
    group.add(silo3);

    // Digital Gravimetric Load-Cell Weighing Platform
    const scaleGeo = new THREE.BoxGeometry(2.6, 0.16, 1.1);
    const scale = new THREE.Mesh(scaleGeo, accentMat);
    scale.position.set(0, 0.48, 0.5);
    group.add(scale);

    // Digital Control Console Display
    const consoleGeo = new THREE.BoxGeometry(0.6, 0.8, 0.4);
    const consoleUnit = new THREE.Mesh(consoleGeo, chassisMat);
    consoleUnit.position.set(-1.1, 1.2, 0.6);
    group.add(consoleUnit);

    const screenGeo = new THREE.PlaneGeometry(0.45, 0.3);
    const screenMat = new THREE.MeshBasicMaterial({ color: 0x087F8C });
    const screen = new THREE.Mesh(screenGeo, screenMat);
    screen.position.set(-1.1, 1.35, 0.81);
    group.add(screen);

    // Pneumatic Transfer Chute leading into Planetary Mixer
    const chuteCurve = new THREE.CatmullRomCurve3([
      new THREE.Vector3(1.2, 1.1, 0),
      new THREE.Vector3(2.4, 1.6, 0),
      new THREE.Vector3(4.2, 2.1, 0)
    ]);
    const chuteGeo = new THREE.TubeGeometry(chuteCurve, 24, 0.09, 12, false);
    const chute = new THREE.Mesh(chuteGeo, stainlessMat);
    group.add(chute);

    return group;
  }

  private buildMixerUnit(): THREE.Group {
    const group = new THREE.Group();
    const steelMat = new THREE.MeshStandardMaterial({ color: 0xDFE7E9, metalness: 0.90, roughness: 0.16 });
    const chassisMat = new THREE.MeshStandardMaterial({ color: 0x2C3E50, roughness: 0.35, metalness: 0.45 });

    // Machine Base Pedestal
    const baseGeo = new THREE.BoxGeometry(3.2, 0.4, 3.2);
    const base = new THREE.Mesh(baseGeo, chassisMat);
    base.position.y = 0.2;
    base.receiveShadow = true;
    group.add(base);

    // Cylindrical Vacuum Mixing Vessel
    const vesselGeo = new THREE.CylinderGeometry(1.2, 1.0, 2.4, 32);
    const vessel = new THREE.Mesh(vesselGeo, steelMat);
    vessel.position.y = 1.6;
    vessel.castShadow = true;
    group.add(vessel);

    // Domed Top Cover
    const domeGeo = new THREE.SphereGeometry(1.2, 32, 16, 0, Math.PI * 2, 0, Math.PI / 2);
    const dome = new THREE.Mesh(domeGeo, steelMat);
    dome.position.y = 2.8;
    group.add(dome);

    // Overhead Drive Motor
    const motorGeo = new THREE.CylinderGeometry(0.55, 0.55, 1.4, 24);
    const motor = new THREE.Mesh(motorGeo, chassisMat);
    motor.position.y = 3.9;
    group.add(motor);

    // Agitator Drive Shaft (animated)
    const shaftGeo = new THREE.CylinderGeometry(0.12, 0.12, 2.0, 16);
    this.mixerShaft = new THREE.Mesh(shaftGeo, steelMat);
    this.mixerShaft.position.y = 2.0;
    group.add(this.mixerShaft);

    // Slurry Feed Pipe leading toward coater
    const pipeCurve = new THREE.CatmullRomCurve3([
      new THREE.Vector3(1.1, 1.0, 0),
      new THREE.Vector3(2.5, 0.8, 0),
      new THREE.Vector3(4.0, 1.8, 0),
      new THREE.Vector3(7.0, 2.2, 0)
    ]);
    const pipeGeo = new THREE.TubeGeometry(pipeCurve, 32, 0.08, 12, false);
    const pipeMat = new THREE.MeshStandardMaterial({ color: 0x087F8C, metalness: 0.7, roughness: 0.25 });
    const pipe = new THREE.Mesh(pipeGeo, pipeMat);
    group.add(pipe);

    return group;
  }

  private buildUncoilerReel(): THREE.Group {
    const group = new THREE.Group();
    const frameMat = new THREE.MeshStandardMaterial({ color: 0x334E68, roughness: 0.4, metalness: 0.5 });
    const foilRollMat = new THREE.MeshStandardMaterial({ color: 0xD4D8DB, metalness: 0.95, roughness: 0.14 });

    // A-frame stand
    const standGeo = new THREE.CylinderGeometry(0.1, 0.1, 2.2, 12);
    const leg1 = new THREE.Mesh(standGeo, frameMat);
    leg1.position.set(0, 1.1, -1.2);
    group.add(leg1);
    const leg2 = new THREE.Mesh(standGeo, frameMat);
    leg2.position.set(0, 1.1, 1.2);
    group.add(leg2);

    // Master Foil Coil
    const rollGeo = new THREE.CylinderGeometry(0.65, 0.65, 2.0, 32);
    const roll = new THREE.Mesh(rollGeo, foilRollMat);
    roll.rotation.x = Math.PI / 2;
    roll.position.set(0, 1.8, 0);
    group.add(roll);

    return group;
  }

  private buildSlotDieCoater(): THREE.Group {
    const group = new THREE.Group();
    const frameMat = new THREE.MeshStandardMaterial({ color: 0x2C3E50, roughness: 0.35, metalness: 0.45 });
    const steelMat = new THREE.MeshStandardMaterial({ color: 0xDFE7E9, metalness: 0.92, roughness: 0.15 });
    const chromeMat = new THREE.MeshStandardMaterial({ color: 0xFFFFFF, metalness: 0.98, roughness: 0.08 });

    // Machine Base
    const baseGeo = new THREE.BoxGeometry(2.8, 0.5, 3.2);
    const base = new THREE.Mesh(baseGeo, frameMat);
    base.position.y = 0.25;
    group.add(base);

    // Steel Backing Roll
    const backingGeo = new THREE.CylinderGeometry(0.55, 0.55, 2.2, 32);
    this.backingRoll = new THREE.Mesh(backingGeo, chromeMat);
    this.backingRoll.rotation.x = Math.PI / 2;
    this.backingRoll.position.set(0, 1.8, 0);
    this.backingRoll.castShadow = true;
    group.add(this.backingRoll);

    // Slot-Die Coating Head (Positioned above backing roll with precise micrometer gap)
    const dieBodyGeo = new THREE.BoxGeometry(0.7, 0.6, 2.2);
    const dieHead = new THREE.Mesh(dieBodyGeo, steelMat);
    dieHead.position.set(-0.25, 2.5, 0);
    dieHead.castShadow = true;
    group.add(dieHead);

    // Precision Micrometer Gap Adjusters
    for (const z of [-0.8, 0.8]) {
      const knobGeo = new THREE.CylinderGeometry(0.12, 0.12, 0.4, 16);
      const knobMat = new THREE.MeshStandardMaterial({ color: 0xF59E42, metalness: 0.8, roughness: 0.2 });
      const knob = new THREE.Mesh(knobGeo, knobMat);
      knob.position.set(-0.25, 2.9, z);
      group.add(knob);
    }

    // Slurry Lip Indicator
    const beadGeo = new THREE.CylinderGeometry(0.04, 0.04, 2.0, 16);
    const beadMat = new THREE.MeshStandardMaterial({ color: 0x222831, roughness: 0.3 });
    const bead = new THREE.Mesh(beadGeo, beadMat);
    bead.rotation.x = Math.PI / 2;
    bead.position.set(-0.02, 2.36, 0);
    group.add(bead);

    return group;
  }

  private buildDryingTunnel(): THREE.Group {
    const group = new THREE.Group();
    const ovenHousingMat = new THREE.MeshStandardMaterial({ color: 0xE2EBED, roughness: 0.28, metalness: 0.25 });
    const chassisMat = new THREE.MeshStandardMaterial({ color: 0x2C3E50, roughness: 0.35, metalness: 0.45 });
    const windowMat = new THREE.MeshPhysicalMaterial({
      color: 0x243E4C,
      transparent: true,
      opacity: 0.45,
      roughness: 0.1,
      metalness: 0.1
    });

    // Oven Main Insulated Tunnel (7m long)
    const tunnelGeo = new THREE.BoxGeometry(7.0, 2.4, 2.8);
    const tunnel = new THREE.Mesh(tunnelGeo, ovenHousingMat);
    tunnel.position.set(0, 1.9, 0);
    tunnel.castShadow = true;
    tunnel.receiveShadow = true;
    group.add(tunnel);

    // Support Legs
    for (const x of [-3.0, 0, 3.0]) {
      for (const z of [-1.2, 1.2]) {
        const legGeo = new THREE.BoxGeometry(0.25, 0.8, 0.25);
        const leg = new THREE.Mesh(legGeo, chassisMat);
        leg.position.set(x, 0.4, z);
        group.add(leg);
      }
    }

    // Inspection Observation Windows with Orange Glow from IR Lamps inside
    for (const x of [-2.0, 0.0, 2.0]) {
      const winGeo = new THREE.PlaneGeometry(1.2, 0.8);
      const win = new THREE.Mesh(winGeo, windowMat);
      win.position.set(x, 2.0, 1.41);
      group.add(win);

      // IR Radiant Heating PointLight glowing through window
      const irLight = new THREE.PointLight(0xF59E42, 2.0, 6.0);
      irLight.position.set(x, 2.0, 0);
      this.irLamps.push(irLight);
      group.add(irLight);
    }

    // Overhead Exhaust Air Ducts
    for (const x of [-1.8, 1.8]) {
      const ductGeo = new THREE.CylinderGeometry(0.35, 0.35, 1.2, 24);
      const duct = new THREE.Mesh(ductGeo, chassisMat);
      duct.position.set(x, 3.7, 0);
      group.add(duct);
    }

    return group;
  }

  private buildPrecisionCalender(): THREE.Group {
    const group = new THREE.Group();
    // Bright chrome calender rolls with specular shine
    const chromeRollMat = new THREE.MeshStandardMaterial({
      color: 0xFFFFFF,
      metalness: 0.98,
      roughness: 0.08,
    });
    // Two-tone cleanroom machine frame
    const frameMat = new THREE.MeshStandardMaterial({
      color: 0x2C3E50,
      roughness: 0.35,
      metalness: 0.45
    });
    const panelMat = new THREE.MeshStandardMaterial({
      color: 0xE2EBED,
      roughness: 0.25,
      metalness: 0.20
    });
    const brassMat = new THREE.MeshStandardMaterial({
      color: 0xD4AF37,
      metalness: 0.85,
      roughness: 0.25
    });

    // Heavy Machine Bed
    const bedGeo = new THREE.BoxGeometry(3.6, 0.6, 3.4);
    const bed = new THREE.Mesh(bedGeo, frameMat);
    bed.position.y = 0.3;
    bed.receiveShadow = true;
    group.add(bed);

    // Front Machine Fascia Panel (Adds visual relief and contrast)
    const panelGeo = new THREE.BoxGeometry(3.62, 0.4, 0.05);
    const panel = new THREE.Mesh(panelGeo, panelMat);
    panel.position.set(0, 0.3, 1.71);
    group.add(panel);

    // Twin Vertical Structural Upright Columns (Left & Right Bearing Blocks)
    for (const z of [-1.3, 1.3]) {
      const uprightGeo = new THREE.BoxGeometry(1.4, 4.4, 0.6);
      const upright = new THREE.Mesh(uprightGeo, frameMat);
      upright.position.set(0, 2.5, z);
      upright.castShadow = true;
      group.add(upright);

      // Hydraulic Compression Actuators on top
      const cylGeo = new THREE.CylinderGeometry(0.32, 0.32, 0.9, 24);
      const cyl = new THREE.Mesh(cylGeo, frameMat);
      cyl.position.set(0, 5.0, z);
      group.add(cyl);

      // Piston Rod
      const pistonGeo = new THREE.CylinderGeometry(0.18, 0.18, 0.6, 24);
      const piston = new THREE.Mesh(pistonGeo, chromeRollMat);
      piston.position.set(0, 4.4, z);
      group.add(piston);

      // Bearing Housing Sleeves
      const sleeveGeo = new THREE.CylinderGeometry(0.52, 0.52, 0.62, 24);
      const sleeveTop = new THREE.Mesh(sleeveGeo, brassMat);
      sleeveTop.rotation.x = Math.PI / 2;
      sleeveTop.position.set(0, 2.5, z);
      group.add(sleeveTop);

      const sleeveBottom = new THREE.Mesh(sleeveGeo, brassMat);
      sleeveBottom.rotation.x = Math.PI / 2;
      sleeveBottom.position.set(0, 1.5, z);
      group.add(sleeveBottom);
    }

    // 1. TOP HARDENED CALENDER ROLL (Diameter 0.96m, Width 2.2m)
    const rollTopGeo = new THREE.CylinderGeometry(0.48, 0.48, 2.2, 48);
    this.calenderRollTop = new THREE.Mesh(rollTopGeo, chromeRollMat);
    this.calenderRollTop.rotation.x = Math.PI / 2;
    this.calenderRollTop.position.set(0, 2.5, 0);
    this.calenderRollTop.castShadow = true;
    group.add(this.calenderRollTop);

    // 2. BOTTOM HARDENED CALENDER ROLL (Counter-rotating)
    const rollBottomGeo = new THREE.CylinderGeometry(0.48, 0.48, 2.2, 48);
    this.calenderRollBottom = new THREE.Mesh(rollBottomGeo, chromeRollMat);
    this.calenderRollBottom.rotation.x = Math.PI / 2;
    this.calenderRollBottom.position.set(0, 1.5, 0);
    this.calenderRollBottom.castShadow = true;
    group.add(this.calenderRollBottom);

    // Calendering Nip Gap Marker / Tech Laser Line
    const nipIndicatorGeo = new THREE.RingGeometry(0.49, 0.53, 32);
    const nipIndicatorMat = new THREE.MeshBasicMaterial({ color: 0x087F8C, side: THREE.DoubleSide });
    const nipRing = new THREE.Mesh(nipIndicatorGeo, nipIndicatorMat);
    nipRing.position.set(0, 2.0, 1.12);
    group.add(nipRing);

    // Infeed and Outfeed Foil Guide Idle Rollers
    for (const x of [-1.4, 1.4]) {
      const guideGeo = new THREE.CylinderGeometry(0.12, 0.12, 2.1, 24);
      const guide = new THREE.Mesh(guideGeo, chromeRollMat);
      guide.rotation.x = Math.PI / 2;
      guide.position.set(x, 2.0, 0);
      group.add(guide);
    }

    // Digital Calender Console with Gap / Temperature Display
    const consoleGeo = new THREE.BoxGeometry(0.7, 1.1, 0.5);
    const consoleUnit = new THREE.Mesh(consoleGeo, frameMat);
    consoleUnit.position.set(1.5, 1.8, 1.8);
    group.add(consoleUnit);

    const screenGeo = new THREE.PlaneGeometry(0.55, 0.35);
    const screenMat = new THREE.MeshBasicMaterial({ color: 0x087F8C });
    const screen = new THREE.Mesh(screenGeo, screenMat);
    screen.position.set(1.5, 2.1, 2.06);
    group.add(screen);

    return group;
  }

  private buildCyclerStation(): THREE.Group {
    const group = new THREE.Group();
    const rackMat = new THREE.MeshStandardMaterial({ color: 0x2C3E50, roughness: 0.35, metalness: 0.5 });
    const panelMat = new THREE.MeshStandardMaterial({ color: 0x1A202C, roughness: 0.4, metalness: 0.6 });

    // Multi-Channel Cycler Server Rack
    const rackGeo = new THREE.BoxGeometry(2.4, 4.2, 1.8);
    const rack = new THREE.Mesh(rackGeo, rackMat);
    rack.position.set(0, 2.1, 0);
    rack.castShadow = true;
    group.add(rack);

    // Front Instrument Bay Panel
    const bayGeo = new THREE.BoxGeometry(2.2, 3.8, 0.05);
    const bay = new THREE.Mesh(bayGeo, panelMat);
    bay.position.set(0, 2.1, 0.91);
    group.add(bay);

    // Channel LEDs
    for (let row = 0; row < 6; row++) {
      for (let col = -0.8; col <= 0.8; col += 0.4) {
        const ledGeo = new THREE.SphereGeometry(0.04, 12, 12);
        const ledMat = new THREE.MeshBasicMaterial({ color: row % 2 === 0 ? 0x087F8C : 0xF59E42 });
        const led = new THREE.Mesh(ledGeo, ledMat);
        led.position.set(col, 1.2 + row * 0.45, 0.94);
        group.add(led);
      }
    }

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

  public getHitboxes(): THREE.Mesh[] {
    return this.stageHitboxes;
  }

  public update(delta: number, elapsed: number) {
    // 1. Continuous synchronized roll rotation
    if (this.calenderRollTop && this.calenderRollBottom) {
      this.calenderRollTop.rotation.y += delta * 1.8;
      this.calenderRollBottom.rotation.y -= delta * 1.8;
    }
    if (this.backingRoll) {
      this.backingRoll.rotation.y += delta * 1.8;
    }
    if (this.mixerShaft) {
      this.mixerShaft.rotation.y += delta * 3.5;
    }

    // 2. Pulse interactive beacons
    this.interactiveBeacons.forEach((beacon, i) => {
      beacon.position.y += Math.sin(elapsed * 2.5 + i) * 0.0015;
      beacon.rotation.y += delta * 0.8;
    });

    // 3. Subtle breathing in IR lamps
    this.irLamps.forEach((lamp, i) => {
      lamp.intensity = 1.6 + Math.sin(elapsed * 3.0 + i) * 0.35;
    });
  }
}
