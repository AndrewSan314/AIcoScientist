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

export class CameraRig {
  private camera: THREE.PerspectiveCamera;
  private currentTarget: THREE.Vector3;
  private isTransitioning: boolean = false;
  private tweenTimeline: gsap.core.Timeline | null = null;
  private idleTime: number = 0;
  private activeScene: number = 1;

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

  public update(delta: number) {
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
