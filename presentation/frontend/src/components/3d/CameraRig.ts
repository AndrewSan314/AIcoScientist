import * as THREE from 'three';
import gsap from 'gsap';

export interface CameraWaypoint {
  position: THREE.Vector3;
  target: THREE.Vector3;
  fov?: number;
}

export const SCENE_WAYPOINTS: Record<number, CameraWaypoint> = {
  // Scene 1: Enter the Laboratory — High cinematic wide shot of the cleanroom
  1: {
    position: new THREE.Vector3(0, 14, 28),
    target: new THREE.Vector3(0, 2, 0),
    fov: 42
  },
  // Scene 2: Explore the Manufacturing Process — Closer angle framing the connected line overview
  2: {
    position: new THREE.Vector3(4, 9, 18),
    target: new THREE.Vector3(4, 2.2, 0),
    fov: 44
  },
  // Scene 3: Inside the Electrode — Macro close-up on porous microstructure
  3: {
    position: new THREE.Vector3(0, 4.5, 7.5),
    target: new THREE.Vector3(0, 1.2, 0),
    fov: 38
  },
  // Scene 4: AI Optimization Studio — Elevated technical view over 3D response surface
  4: {
    position: new THREE.Vector3(0, 11, 16),
    target: new THREE.Vector3(0, 2.5, 0),
    fov: 40
  },
  // Scene 5: Scientific Evidence — Balanced perspective framing 3D line & telemetry
  5: {
    position: new THREE.Vector3(-4, 10, 22),
    target: new THREE.Vector3(1, 2.8, 0),
    fov: 44
  }
};

// Stage-specific camera positions for Scene 2 exploration across both Drakopoulos & Warwick
export const STAGE_WAYPOINTS: Record<string, CameraWaypoint> = {
  // Overview of the entire production line
  overview: {
    position: new THREE.Vector3(4, 9, 18),
    target: new THREE.Vector3(4, 2.2, 0),
    fov: 44
  },
  // Drakopoulos: Stage 1 Formulation
  formulation: {
    position: new THREE.Vector3(-15, 6.5, 12),
    target: new THREE.Vector3(-15, 2.2, 0),
    fov: 40
  },
  // Drakopoulos: Stage 2 Mixing
  mixing: {
    position: new THREE.Vector3(-11, 6.5, 12),
    target: new THREE.Vector3(-11, 2.2, 0),
    fov: 40
  },
  // Warwick: Stage 1 Slurry Preparation
  slurry_prep: {
    position: new THREE.Vector3(-12, 6.5, 12),
    target: new THREE.Vector3(-12, 2.2, 0),
    fov: 40
  },
  // Drakopoulos: Stage 3 Slot-Die Coating
  coating: {
    position: new THREE.Vector3(-6.4, 4.7, 8.2),
    target: new THREE.Vector3(-2.3, 1.95, 0),
    fov: 35
  },
  // Warwick: Stage 2 Pilot Roll Coating
  pilot_coating: {
    position: new THREE.Vector3(-6.4, 4.7, 8.2),
    target: new THREE.Vector3(-2.3, 1.95, 0),
    fov: 35
  },
  // Drakopoulos: Stage 4 Drying Tunnel
  drying: {
    position: new THREE.Vector3(4, 7, 13),
    target: new THREE.Vector3(4, 2.5, 0),
    fov: 42
  },
  // Both: Calendering Machine (Signature Asset)
  calendering: {
    position: new THREE.Vector3(6.6, 4.9, 8.5),
    target: new THREE.Vector3(11, 2.1, 0),
    fov: 34
  },
  // Warwick: Stage 4 Half-Cell Assembly
  cell_assembly: {
    position: new THREE.Vector3(14.5, 6.5, 12),
    target: new THREE.Vector3(14.5, 2.2, 0),
    fov: 40
  },
  // Drakopoulos: Stage 6 Characterization
  characterization: {
    position: new THREE.Vector3(17.5, 6.5, 12),
    target: new THREE.Vector3(17.5, 2.2, 0),
    fov: 40
  },
  // Warwick: Stage 5 Rate Characterization
  rate_characterization: {
    position: new THREE.Vector3(17.5, 6.5, 12),
    target: new THREE.Vector3(17.5, 2.2, 0),
    fov: 40
  }
};

const PROCESS_WAYPOINTS: Record<string, { explore: CameraWaypoint; run: CameraWaypoint }> = {
  coating: {
    explore: { position: new THREE.Vector3(-5.4, 3.8, 7.0), target: new THREE.Vector3(-2.45, 2.05, 0), fov: 32 },
    run: { position: new THREE.Vector3(0.8, 5.0, 10.0), target: new THREE.Vector3(1.0, 2.15, 0), fov: 38 },
  },
  pilot_coating: {
    explore: { position: new THREE.Vector3(-5.2, 3.45, 5.4), target: new THREE.Vector3(-2.45, 2.05, 0), fov: 31 },
    run: { position: new THREE.Vector3(-4.5, 3.2, 4.8), target: new THREE.Vector3(-2.3, 2.0, 0), fov: 30 },
  },
  calendering: {
    explore: { position: new THREE.Vector3(8.0, 4.0, 7.5), target: new THREE.Vector3(11, 2.02, 0), fov: 31 },
    run: { position: new THREE.Vector3(9.0, 3.35, 6.2), target: new THREE.Vector3(11.35, 1.98, 0), fov: 29 },
  },
  formulation: {
    explore: { position: new THREE.Vector3(-16.5, 4.7, 7.2), target: new THREE.Vector3(-15.2, 1.8, 0), fov: 35 },
    run: STAGE_WAYPOINTS.formulation,
  },
  slurry_prep: {
    explore: { position: new THREE.Vector3(-15.2, 5.2, 8), target: new THREE.Vector3(-13.5, 2, 0), fov: 36 },
    run: STAGE_WAYPOINTS.slurry_prep,
  },
  mixing: {
    explore: { position: new THREE.Vector3(-12.9, 4.5, 6.1), target: new THREE.Vector3(-11, 2, 0), fov: 33 },
    run: STAGE_WAYPOINTS.mixing,
  },
  drying: {
    explore: { position: new THREE.Vector3(1.0, 4.4, 7.1), target: new THREE.Vector3(4.1, 2.35, 0), fov: 34 },
    run: { position: new THREE.Vector3(4, 3.8, 5.3), target: new THREE.Vector3(5.2, 2.2, 0), fov: 31 },
  },
  characterization: {
    explore: { position: new THREE.Vector3(15.1, 4.5, 6.2), target: new THREE.Vector3(17.5, 2.3, 0), fov: 33 },
    run: STAGE_WAYPOINTS.characterization,
  },
  rate_characterization: {
    explore: { position: new THREE.Vector3(15.1, 4.5, 6.2), target: new THREE.Vector3(17.5, 2.3, 0), fov: 33 },
    run: STAGE_WAYPOINTS.rate_characterization,
  },
};

export interface TourStageData {
  stageId: string;
  order: number;
  label: string;
  category: string;
  action: string;
  contextDescription: string;
  telemetry?: string;
  waypoint: CameraWaypoint;
  duration: number;
}

export interface TourUpdateInfo {
  stageIndex: number;
  totalStages: number;
  stage: TourStageData;
  progress: number;
  isPaused: boolean;
}

export const DRAKOPOULOS_TOUR_STAGES: TourStageData[] = [
  {
    stageId: 'formulation',
    order: 1,
    label: 'Slurry Formulation & Gravimetric Dosing',
    category: 'STAGE 01 · PRECURSOR FEED',
    action: 'Micro-gravimetric delivery of graphite active powder, carbon black, and PVDF binder solution',
    contextDescription: 'Illustrative gravimetric dosing · Active synthetic graphite, Super C65 carbon black & CMC/SBR binder',
    waypoint: { position: new THREE.Vector3(-15.2, 5.2, 8.8), target: new THREE.Vector3(-15.5, 2.0, 0), fov: 36 },
    duration: 5.5,
  },
  {
    stageId: 'mixing',
    order: 2,
    label: 'Planetary High-Shear Vacuum Mixing',
    category: 'STAGE 02 · HYDRODYNAMIC DISPERSION',
    action: 'Planetary dual-shaft shear dispersion to de-agglomerate particles without air entrapment',
    contextDescription: 'Illustrative planetary high-shear mixing · Controlled hydrodynamic shear stress dispersion',
    waypoint: { position: new THREE.Vector3(-11.0, 4.8, 7.8), target: new THREE.Vector3(-11.0, 2.1, 0), fov: 34 },
    duration: 5.5,
  },
  {
    stageId: 'coating',
    order: 3,
    label: 'Precision Slot-Die Web Coating',
    category: 'STAGE 03 · FLUID FILM DEPOSITION',
    action: 'Slot-die extrusion depositing uniform wet slurry ribbon onto continuous copper foil collector',
    contextDescription: 'Verified recipe slot-die web coating · Replay controls: Line Speed (0.1–0.4 m/min), Coating Gap (100–200 µm)',
    waypoint: { position: new THREE.Vector3(-4.8, 3.8, 6.8), target: new THREE.Vector3(-2.5, 2.0, 0), fov: 32 },
    duration: 6.0,
  },
  {
    stageId: 'drying',
    order: 4,
    label: 'Infrared & Convection Drying Tunnel',
    category: 'STAGE 04 · HEAT & MASS TRANSFER',
    action: 'Thermal drying removing NMP solvent while preventing binder surface migration',
    contextDescription: 'Verified convection drying tunnel · Replay control: Temperature (60–100 °C)',
    waypoint: { position: new THREE.Vector3(3.2, 4.8, 8.8), target: new THREE.Vector3(4.0, 2.3, 0), fov: 36 },
    duration: 5.5,
  },
  {
    stageId: 'calendering',
    order: 5,
    label: 'Hardened Precision Roll Calendering',
    category: 'STAGE 05 · COMPACTION & POROSITY',
    action: 'Precision roll calendering pass evaluating compaction vs uncalendered status',
    contextDescription: 'Electrode roll compaction · Replay control: Calendering applied (True/False)',
    waypoint: { position: new THREE.Vector3(8.5, 4.2, 7.2), target: new THREE.Vector3(11.0, 2.1, 0), fov: 30 },
    duration: 6.0,
  },
  {
    stageId: 'characterization',
    order: 6,
    label: 'Electrochemical Half-Cell Metrology',
    category: 'STAGE 06 · QUALITY GATE VALIDATION',
    action: 'Electrochemical cycling protocol measuring delithiation capacity and rate capability',
    contextDescription: 'Electrochemical half-cell cycling · Target: Specific discharge capacity at Cycle 30 (D30 mAh/g)',
    waypoint: { position: new THREE.Vector3(15.2, 4.6, 7.5), target: new THREE.Vector3(17.5, 2.2, 0), fov: 34 },
    duration: 6.0,
  },
  {
    stageId: 'overview',
    order: 7,
    label: 'Connected Pilot Plant Overview',
    category: 'EXHIBITION · DIGITAL TWIN REPLAY',
    action: 'Multi-stage manufacturing line reflecting historical experimental research context',
    contextDescription: 'Drakopoulos et al. (2021) · 12 complete recipes · Seed 11 offline optimization replay · Not a live production line',
    waypoint: { position: new THREE.Vector3(4.0, 10.5, 21.0), target: new THREE.Vector3(3.0, 2.2, 0), fov: 44 },
    duration: 6.5,
  },
];

export const WARWICK_TOUR_STAGES: TourStageData[] = [
  {
    stageId: 'slurry_prep',
    order: 1,
    label: 'NMC622 Slurry Preparation & Homogenization',
    category: 'STAGE 01 · SLURRY SYNTHESIS',
    action: 'Dispersion of nickel-rich NMC622 cathode powder, Super P carbon, and PVDF in NMP',
    contextDescription: 'Illustrative slurry synthesis · NMC622 cathode powder, PVDF binder & NMP solvent',
    waypoint: { position: new THREE.Vector3(-12.5, 5.4, 9.0), target: new THREE.Vector3(-12.0, 2.1, 0), fov: 36 },
    duration: 5.5,
  },
  {
    stageId: 'pilot_coating',
    order: 2,
    label: 'Pilot Roll-to-Roll Doctor Blade Coating',
    category: 'STAGE 02 · CONTINUOUS WEB DEPOSITION',
    action: 'Continuous web application onto battery-grade aluminum collector foil',
    contextDescription: 'Illustrative roll-to-roll web coating · Doctor blade continuous deposition on aluminum foil',
    waypoint: { position: new THREE.Vector3(-4.8, 3.8, 6.8), target: new THREE.Vector3(-2.5, 2.0, 0), fov: 32 },
    duration: 6.0,
  },
  {
    stageId: 'drying',
    order: 3,
    label: 'Convection Flotation Drying Chamber',
    category: 'STAGE 03 · SOLVENT DESICCATION',
    action: 'Air flotation tunnel removing NMP solvent while maintaining film integrity',
    contextDescription: 'Illustrative flotation drying chamber · Multi-zone solvent desiccation',
    waypoint: { position: new THREE.Vector3(3.2, 4.8, 8.8), target: new THREE.Vector3(4.0, 2.3, 0), fov: 36 },
    duration: 5.5,
  },
  {
    stageId: 'calendering',
    order: 4,
    label: 'Heated Roll Calendering & Density Compaction',
    category: 'STAGE 04 · WARWICK SIGNATURE ASSET',
    action: 'Heated mechanical calendering exploring temperature regimes to maximize rate retention',
    contextDescription: 'Warwick signature calendering asset · Replay controls: Roll Temperature (85–145 °C) & Target Density (2.7–3.2 g/cm³)',
    waypoint: { position: new THREE.Vector3(8.5, 4.2, 7.2), target: new THREE.Vector3(11.0, 2.1, 0), fov: 30 },
    duration: 6.0,
  },
  {
    stageId: 'cell_assembly',
    order: 5,
    label: 'Coin Cell Assembly & Electrolyte Infiltration',
    category: 'STAGE 05 · PROTOTYPE PACKAGING',
    action: 'Electrode disc punching, Celgard separator stacking, and LP57 electrolyte wetting',
    contextDescription: 'Illustrative coin-cell prototype packaging · Pre-cycling assembly process',
    waypoint: { position: new THREE.Vector3(13.2, 4.8, 8.2), target: new THREE.Vector3(14.5, 2.1, 0), fov: 34 },
    duration: 5.5,
  },
  {
    stageId: 'rate_characterization',
    order: 6,
    label: 'High-Rate Fast Charging Metrology (5C / 0.2C)',
    category: 'STAGE 06 · EMPIRICAL PERFORMANCE TEST',
    action: 'Galvanostatic cycling verifying fast-charging rate capability',
    contextDescription: 'Electrochemical rate capability metrology · Target: 5C/0.2C capacity retention ratio',
    waypoint: { position: new THREE.Vector3(15.2, 4.6, 7.5), target: new THREE.Vector3(17.5, 2.2, 0), fov: 34 },
    duration: 6.0,
  },
  {
    stageId: 'overview',
    order: 7,
    label: 'Calendering Optimization Pilot Plant',
    category: 'EXHIBITION · DIGITAL TWIN REPLAY',
    action: 'Validated offline process optimization replay connecting calendering controls to rate performance',
    contextDescription: 'Warwick NMC622 calendering dataset · 18 candidates evaluated · Seed 11 offline optimization replay · Verified rate retention target',
    waypoint: { position: new THREE.Vector3(4.0, 10.5, 21.0), target: new THREE.Vector3(3.0, 2.2, 0), fov: 44 },
    duration: 6.5,
  },
];

export class CameraRig {
  private camera: THREE.PerspectiveCamera;
  private currentTarget: THREE.Vector3;
  private isTransitioning: boolean = false;
  private tweenTimeline: gsap.core.Timeline | null = null;
  private idleTime: number = 0;
  private activeScene: number = 1;
  private isTourActiveFlag: boolean = false;
  private isTourPausedFlag: boolean = false;
  private tourStages: TourStageData[] = [];
  private currentTourIndex: number = 0;
  private tourTimer: number = 0;
  private onTourUpdateCallback?: (info: TourUpdateInfo) => void;

  constructor(camera: THREE.PerspectiveCamera) {
    this.camera = camera;
    this.currentTarget = new THREE.Vector3(0, 2, 0);

    // Initialize to Scene 1
    const wp = SCENE_WAYPOINTS[1];
    this.camera.position.copy(wp.position);
    this.currentTarget.copy(wp.target);
    this.camera.lookAt(this.currentTarget);
  }

  public transitionToScene(sceneNumber: number, duration: number = 1.4, onComplete?: () => void) {
    const wp = SCENE_WAYPOINTS[sceneNumber] || SCENE_WAYPOINTS[1];
    this.activeScene = sceneNumber;
    this.animateToWaypoint(wp, duration, onComplete);
  }

  public transitionToStage(stageId: string, duration: number = 1.2, onComplete?: () => void) {
    let wp = STAGE_WAYPOINTS[stageId];
    if (!wp) {
      // Fuzzy stage fallback
      if (stageId.includes('coat')) wp = STAGE_WAYPOINTS['coating'];
      else if (stageId.includes('slurry') || stageId.includes('mix')) wp = STAGE_WAYPOINTS['mixing'];
      else if (stageId.includes('calender')) wp = STAGE_WAYPOINTS['calendering'];
      else if (stageId.includes('dry')) wp = STAGE_WAYPOINTS['drying'];
      else if (stageId.includes('assembl')) wp = STAGE_WAYPOINTS['cell_assembly'];
      else if (stageId.includes('rate') || stageId.includes('character')) wp = STAGE_WAYPOINTS['characterization'];
      else wp = STAGE_WAYPOINTS['overview'];
    }
    this.animateToWaypoint(wp, duration, onComplete);
  }

  public transitionToProcess(stageId: string, phase: string) {
    if (phase === 'PROCESS_EXIT' || stageId === 'overview') {
      this.animateToWaypoint(STAGE_WAYPOINTS.overview, 0.85);
      return;
    }
    const waypoints = PROCESS_WAYPOINTS[stageId];
    if (!waypoints) {
      this.transitionToStage(stageId, 1.1);
      return;
    }
    this.animateToWaypoint(
      phase === 'ILLUSTRATED_PROCESS_RUN' ? waypoints.run : waypoints.explore,
      phase === 'ILLUSTRATED_PROCESS_RUN' ? 3.1 : 1.15,
    );
  }

  public animateToWaypoint(wp: CameraWaypoint, duration: number = 1.4, onComplete?: () => void) {
    if (this.tweenTimeline) {
      this.tweenTimeline.kill();
    }

    this.isTransitioning = true;
    const currentFov = { fov: this.camera.fov };
    const targetFov = wp.fov ?? 42;

    this.tweenTimeline = gsap.timeline({
      onUpdate: () => {
        this.camera.lookAt(this.currentTarget);
      },
      onComplete: () => {
        this.isTransitioning = false;
        if (onComplete) onComplete();
      }
    });

    this.tweenTimeline.to(this.camera.position, {
      x: wp.position.x,
      y: wp.position.y,
      z: wp.position.z,
      duration,
      ease: 'power3.inOut'
    }, 0);

    this.tweenTimeline.to(this.currentTarget, {
      x: wp.target.x,
      y: wp.target.y,
      z: wp.target.z,
      duration,
      ease: 'power3.inOut'
    }, 0);

    if (Math.abs(this.camera.fov - targetFov) > 0.5) {
      this.tweenTimeline.to(currentFov, {
        fov: targetFov,
        duration,
        ease: 'power2.inOut',
        onUpdate: () => {
          this.camera.fov = currentFov.fov;
          this.camera.updateProjectionMatrix();
        }
      }, 0);
    }
  }

  public startCinematicTour(scenarioId: string, onUpdate?: (info: TourUpdateInfo) => void) {
    this.isTourActiveFlag = true;
    this.isTourPausedFlag = false;
    this.currentTourIndex = 0;
    this.tourTimer = 0;
    this.onTourUpdateCallback = onUpdate;
    this.tourStages = scenarioId === 'drakopoulos_graphite' ? DRAKOPOULOS_TOUR_STAGES : WARWICK_TOUR_STAGES;
    this.transitionToTourStage(0);
  }

  public stopCinematicTour() {
    this.isTourActiveFlag = false;
    this.isTourPausedFlag = false;
    this.onTourUpdateCallback = undefined;
    this.animateToWaypoint(STAGE_WAYPOINTS.overview, 1.2);
  }

  public toggleTourPause(): boolean {
    this.isTourPausedFlag = !this.isTourPausedFlag;
    this.emitTourUpdate();
    return this.isTourPausedFlag;
  }

  public stepTour(direction: 1 | -1) {
    if (!this.isTourActiveFlag || this.tourStages.length === 0) return;
    this.tourTimer = 0;
    this.currentTourIndex = (this.currentTourIndex + direction + this.tourStages.length) % this.tourStages.length;
    this.transitionToTourStage(this.currentTourIndex);
  }

  public isTourActive(): boolean {
    return this.isTourActiveFlag;
  }

  public isTourPaused(): boolean {
    return this.isTourPausedFlag;
  }

  public getTourInfo(): TourUpdateInfo | null {
    if (!this.isTourActiveFlag || this.tourStages.length === 0) return null;
    const stage = this.tourStages[this.currentTourIndex];
    return {
      stageIndex: this.currentTourIndex,
      totalStages: this.tourStages.length,
      stage,
      progress: Math.min(1, this.tourTimer / stage.duration),
      isPaused: this.isTourPausedFlag,
    };
  }

  private transitionToTourStage(index: number) {
    const stage = this.tourStages[index];
    if (!stage) return;
    this.animateToWaypoint(stage.waypoint, 2.0);
    this.emitTourUpdate();
  }

  private emitTourUpdate() {
    if (this.onTourUpdateCallback) {
      const info = this.getTourInfo();
      if (info) this.onTourUpdateCallback(info);
    }
  }

  public update(delta: number) {
    if (this.isTourActiveFlag && !this.isTourPausedFlag) {
      this.tourTimer += delta;
      const stage = this.tourStages[this.currentTourIndex];
      if (stage) {
        // Slow cinematic panning drift
        const drift = Math.sin(this.tourTimer * 0.5) * 0.12;
        this.camera.position.x += drift * delta;
        this.camera.lookAt(this.currentTarget);

        this.emitTourUpdate();

        if (this.tourTimer >= stage.duration) {
          this.tourTimer = 0;
          this.currentTourIndex = (this.currentTourIndex + 1) % this.tourStages.length;
          this.transitionToTourStage(this.currentTourIndex);
        }
      }
      return;
    }

    // Subtle breathing/idle motion when not actively transitioning
    if (!this.isTransitioning) {
      this.idleTime += delta * 0.4;
      if (this.activeScene === 1) {
        // Subtle drift in Hero mode
        const driftX = Math.sin(this.idleTime * 0.5) * 0.6;
        const driftY = Math.cos(this.idleTime * 0.3) * 0.25;
        this.camera.position.x = SCENE_WAYPOINTS[1].position.x + driftX;
        this.camera.position.y = SCENE_WAYPOINTS[1].position.y + driftY;
        this.camera.lookAt(this.currentTarget);
      } else {
        this.camera.lookAt(this.currentTarget);
      }
    }
  }

  public resetCurrentScene(duration: number = 1.0) {
    this.transitionToScene(this.activeScene, duration);
  }

  public getTarget(): THREE.Vector3 {
    return this.currentTarget;
  }
}
