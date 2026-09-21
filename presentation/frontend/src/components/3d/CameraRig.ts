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
    position: new THREE.Vector3(-2.5, 6.5, 12),
    target: new THREE.Vector3(-2.5, 2.2, 0),
    fov: 40
  },
  // Warwick: Stage 2 Pilot Roll Coating
  pilot_coating: {
    position: new THREE.Vector3(-2.5, 6.5, 12),
    target: new THREE.Vector3(-2.5, 2.2, 0),
    fov: 40
  },
  // Drakopoulos: Stage 4 Drying Tunnel
  drying: {
    position: new THREE.Vector3(4, 7, 13),
    target: new THREE.Vector3(4, 2.5, 0),
    fov: 42
  },
  // Both: Calendering Machine (Signature Asset)
  calendering: {
    position: new THREE.Vector3(11, 6.5, 11),
    target: new THREE.Vector3(11, 2.2, 0),
    fov: 38
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
