import React, { useEffect, useRef } from 'react';
import * as THREE from 'three';
import { CameraRig } from './CameraRig';
import { ManufacturingWorld } from './ManufacturingWorld';
import { ElectrodeMicrostructure } from './ElectrodeMicrostructure';
import { OptimizationVisualization } from './OptimizationVisualization';
import { ExhibitionScenario, ScenarioId } from '../../data/types';

interface PersistentWorldCanvasProps {
  currentScene: number;
  activeScenario: ExhibitionScenario;
  selectedStageId?: string;
  replayStep: number;
  microstructureCompression: number;
  autoMorphMicrostructure: boolean;
  cameraResetTrigger?: number;
  selectedCandidateId?: string;
  onStageSelect?: (stageId: string) => void;
  onCandidateSelect?: (candidateId: string) => void;
  onSceneSelect?: (sceneNumber: number) => void;
}

export const PersistentWorldCanvas: React.FC<PersistentWorldCanvasProps> = ({
  currentScene,
  activeScenario,
  selectedStageId,
  replayStep,
  microstructureCompression,
  autoMorphMicrostructure,
  cameraResetTrigger,
  selectedCandidateId,
  onStageSelect,
  onCandidateSelect,
  onSceneSelect
}) => {
  const mountRef = useRef<HTMLDivElement>(null);
  const rendererRef = useRef<THREE.WebGLRenderer | null>(null);
  const sceneRef = useRef<THREE.Scene | null>(null);
  const cameraRef = useRef<THREE.PerspectiveCamera | null>(null);
  const cameraRigRef = useRef<CameraRig | null>(null);
  const manufacturingWorldRef = useRef<ManufacturingWorld | null>(null);
  const microstructureRef = useRef<ElectrodeMicrostructure | null>(null);
  const optimizationRef = useRef<OptimizationVisualization | null>(null);
  const raycasterRef = useRef<THREE.Raycaster>(new THREE.Raycaster());
  const mouseRef = useRef<THREE.Vector2>(new THREE.Vector2());

  // Mouse drag orbit controls for Microstructure (Scene 3)
  const isDraggingRef = useRef<boolean>(false);
  const prevMousePosRef = useRef<{ x: number; y: number }>({ x: 0, y: 0 });

  useEffect(() => {
    if (!mountRef.current) return;
    const container = mountRef.current;
    const width = container.clientWidth || window.innerWidth;
    const height = container.clientHeight || window.innerHeight;

    // 1. Scene Setup
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0xF4F7F7); // Pearl white background
    scene.fog = new THREE.FogExp2(0xF4F7F7, 0.012);
    sceneRef.current = scene;

    // 2. Camera Setup
    const camera = new THREE.PerspectiveCamera(42, width / height, 0.1, 1000);
    cameraRef.current = camera;
    const cameraRig = new CameraRig(camera);
    cameraRigRef.current = cameraRig;

    // 3. Renderer Setup
    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true, powerPreference: 'high-performance' });
    renderer.setSize(width, height);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.08;
    container.appendChild(renderer.domElement);
    rendererRef.current = renderer;

    // 4. Studio Lighting Rig
    const hemiLight = new THREE.HemisphereLight(0xFFFFFF, 0xDCE8EC, 0.75);
    scene.add(hemiLight);

    // Primary Directional Sun / Studio Key Light (Top-Right Front)
    const dirLight = new THREE.DirectionalLight(0xFFFFFF, 1.35);
    dirLight.position.set(16, 28, 18);
    dirLight.castShadow = true;
    dirLight.shadow.mapSize.width = 2048;
    dirLight.shadow.mapSize.height = 2048;
    dirLight.shadow.camera.near = 0.5;
    dirLight.shadow.camera.far = 120;
    dirLight.shadow.camera.left = -28;
    dirLight.shadow.camera.right = 28;
    dirLight.shadow.camera.top = 28;
    dirLight.shadow.camera.bottom = -28;
    dirLight.shadow.bias = -0.0005;
    scene.add(dirLight);

    // Frontal Cleanroom Fill Light (Directly illuminates front panels, rolls, and particles!)
    const frontFill = new THREE.DirectionalLight(0xF4F7F7, 0.95);
    frontFill.position.set(0, 16, 26);
    scene.add(frontFill);

    // Accent tech teal rim light
    const tealRim = new THREE.DirectionalLight(0x087F8C, 0.85);
    tealRim.position.set(-18, 14, -15);
    scene.add(tealRim);

    // 5. 3D Subsystems
    const manufacturingWorld = new ManufacturingWorld((stageId) => {
      if (onStageSelect) onStageSelect(stageId);
    });
    manufacturingWorldRef.current = manufacturingWorld;
    scene.add(manufacturingWorld.group);

    const microstructure = new ElectrodeMicrostructure(activeScenario.microstructure);
    microstructureRef.current = microstructure;
    microstructure.group.position.set(0, 0, 0);
    microstructure.group.visible = false;
    scene.add(microstructure.group);

    const optimizationVis = new OptimizationVisualization(activeScenario, (candidateId) => {
      if (onCandidateSelect) onCandidateSelect(candidateId);
    });
    optimizationRef.current = optimizationVis;
    optimizationVis.group.position.set(0, 0, 0);
    optimizationVis.group.visible = false;
    scene.add(optimizationVis.group);

    // Initial configuration
    cameraRig.transitionToScene(currentScene, 0.1);

    // 6. Event Handlers
    const handleResize = () => {
      if (!container || !renderer || !camera) return;
      const w = container.clientWidth;
      const h = container.clientHeight;
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
      renderer.setSize(w, h);
    };
    window.addEventListener('resize', handleResize);

    const handlePointerDown = (e: MouseEvent) => {
      isDraggingRef.current = true;
      prevMousePosRef.current = { x: e.clientX, y: e.clientY };
    };

    const handlePointerMove = (e: MouseEvent) => {
      if (!isDraggingRef.current) return;
      const deltaX = e.clientX - prevMousePosRef.current.x;
      const deltaY = e.clientY - prevMousePosRef.current.y;
      prevMousePosRef.current = { x: e.clientX, y: e.clientY };

      // In Scene 3 (Microstructure), user can drag to rotate the sample!
      if (currentScene === 3 && microstructureRef.current) {
        microstructureRef.current.group.rotation.y += deltaX * 0.008;
        microstructureRef.current.group.rotation.x = Math.max(
          -0.5,
          Math.min(0.5, microstructureRef.current.group.rotation.x + deltaY * 0.008)
        );
      }
    };

    const handlePointerUp = () => {
      isDraggingRef.current = false;
    };

    const handleClick = (e: MouseEvent) => {
      if (!container || !camera) return;
      const rect = container.getBoundingClientRect();
      mouseRef.current.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
      mouseRef.current.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;

      raycasterRef.current.setFromCamera(mouseRef.current, camera);

      // Handle Scene 1 or 2 machine clicks
      if ((currentScene === 1 || currentScene === 2) && manufacturingWorldRef.current) {
        const hitboxes = manufacturingWorldRef.current.getHitboxes();
        const intersects = raycasterRef.current.intersectObjects(hitboxes);
        if (intersects.length > 0) {
          const rawStageId = intersects[0].object.userData?.stageId;
          if (rawStageId) {
            const resolvedStageId = manufacturingWorldRef.current.resolveStageId(rawStageId);
            if (currentScene === 1 && onSceneSelect) {
              onSceneSelect(2);
            }
            if (onStageSelect) {
              onStageSelect(resolvedStageId);
            }
          }
        }
      }

      // Handle Scene 4 candidate clicks
      if (currentScene === 4 && optimizationRef.current) {
        const hitboxes = optimizationRef.current.getHitboxes();
        const intersects = raycasterRef.current.intersectObjects(hitboxes);
        if (intersects.length > 0) {
          const candidateId = intersects[0].object.userData?.candidateId;
          if (candidateId && onCandidateSelect) {
            onCandidateSelect(candidateId);
          }
        }
      }
    };

    const domElem = renderer.domElement;
    domElem.addEventListener('mousedown', handlePointerDown);
    window.addEventListener('mousemove', handlePointerMove);
    window.addEventListener('mouseup', handlePointerUp);
    domElem.addEventListener('click', handleClick);

    // 7. Animation Loop
    let animationFrameId: number;
    const clock = new THREE.Clock();

    const animate = () => {
      animationFrameId = requestAnimationFrame(animate);
      const delta = clock.getDelta();
      const elapsed = clock.getElapsedTime();

      if (cameraRigRef.current) {
        cameraRigRef.current.update(delta);
      }
      if (manufacturingWorldRef.current) {
        manufacturingWorldRef.current.update(delta, elapsed);
      }
      if (microstructureRef.current) {
        microstructureRef.current.update(delta);
      }
      if (optimizationRef.current) {
        optimizationRef.current.update(delta, elapsed);
      }

      renderer.render(scene, camera);
    };
    animate();

    return () => {
      cancelAnimationFrame(animationFrameId);
      window.removeEventListener('resize', handleResize);
      domElem.removeEventListener('mousedown', handlePointerDown);
      window.removeEventListener('mousemove', handlePointerMove);
      window.removeEventListener('mouseup', handlePointerUp);
      domElem.removeEventListener('click', handleClick);
      if (container.contains(domElem)) {
        container.removeChild(domElem);
      }
      renderer.dispose();
    };
  }, []);

  // Update scene transitions
  useEffect(() => {
    if (!cameraRigRef.current) return;
    cameraRigRef.current.transitionToScene(currentScene);

    // Update 3D subsystem visibilities
    if (manufacturingWorldRef.current) {
      manufacturingWorldRef.current.group.visible = currentScene === 1 || currentScene === 2 || currentScene === 5;
    }
    if (microstructureRef.current) {
      microstructureRef.current.group.visible = currentScene === 3;
      if (currentScene === 3) {
        microstructureRef.current.resetOrientation();
      }
    }
    if (optimizationRef.current) {
      optimizationRef.current.group.visible = currentScene === 4;
    }
  }, [currentScene]);

  // Update stage camera focus in Scene 2
  useEffect(() => {
    if (currentScene === 2 && selectedStageId && cameraRigRef.current) {
      cameraRigRef.current.transitionToStage(selectedStageId);
    }
  }, [currentScene, selectedStageId]);

  // Update active scenario
  useEffect(() => {
    if (manufacturingWorldRef.current) {
      manufacturingWorldRef.current.setScenario(activeScenario.id);
    }
    if (microstructureRef.current) {
      microstructureRef.current.updateScenarioSpec(activeScenario.microstructure, microstructureCompression);
    }
    if (optimizationRef.current) {
      optimizationRef.current.setScenario(activeScenario);
    }
  }, [activeScenario]);

  // Update replay step
  useEffect(() => {
    if (optimizationRef.current) {
      optimizationRef.current.setStep(replayStep);
    }
  }, [replayStep]);

  // Update microstructure compression progress
  useEffect(() => {
    if (microstructureRef.current) {
      microstructureRef.current.setCompressionProgress(microstructureCompression);
    }
  }, [microstructureCompression]);

  // Update auto morph state
  useEffect(() => {
    if (microstructureRef.current) {
      microstructureRef.current.setAutoMorph(autoMorphMicrostructure);
    }
  }, [autoMorphMicrostructure]);

  // Update selected candidate inspect halo
  useEffect(() => {
    if (optimizationRef.current) {
      optimizationRef.current.setSelectedCandidate(selectedCandidateId);
    }
  }, [selectedCandidateId]);

  // Smooth Camera Reset without canvas teardown
  useEffect(() => {
    if (cameraResetTrigger && cameraResetTrigger > 0) {
      if (cameraRigRef.current) {
        cameraRigRef.current.resetCurrentScene(1.0);
      }
      if (microstructureRef.current) {
        microstructureRef.current.resetOrientation();
      }
    }
  }, [cameraResetTrigger]);

  return (
    <div
      ref={mountRef}
      className="absolute inset-0 w-full h-full pointer-events-auto z-0 overflow-hidden"
      style={{ touchAction: 'none' }}
    />
  );
};
