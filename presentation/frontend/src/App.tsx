import React, { useState, useEffect, useCallback } from 'react';
import { Camera } from 'lucide-react';
import { ScenarioId, ExhibitionScenario } from './data/types';
import { DRAKOPOULOS_SCENARIO, WARWICK_SCENARIO } from './data/exhibitionData';
import { validateAllScenarios } from './data/validation';
import { PersistentWorldCanvas } from './components/3d/PersistentWorldCanvas';
import type { TourUpdateInfo } from './components/3d/CameraRig';
import { Navbar } from './components/ui/Navbar';
import { SceneNavigator } from './components/ui/SceneNavigator';
import { Scene1Hero } from './components/ui/Scene1Hero';
import { Scene2ProcessExplorer } from './components/ui/Scene2ProcessExplorer';
import { Scene3MicrostructureInspector } from './components/ui/Scene3MicrostructureInspector';
import { Scene4OptimizationStudio } from './components/ui/Scene4OptimizationStudio';
import { Scene5ScientificEvidence } from './components/ui/Scene5ScientificEvidence';
import { EvidenceLimitationsDrawer } from './components/ui/EvidenceLimitationsDrawer';
import { pendingDecision, revealNextDecision, type ProcessExperiencePhase } from './data/processExperience';

const SCENARIOS: Record<ScenarioId, ExhibitionScenario> = {
  warwick_nmc622_calendering: WARWICK_SCENARIO,
  drakopoulos_graphite: DRAKOPOULOS_SCENARIO
};

export const App: React.FC = () => {
  // 1. Core State
  const [activeScenarioId, setActiveScenarioId] = useState<ScenarioId>('warwick_nmc622_calendering');
  const [currentScene, setCurrentScene] = useState<number>(1);
  const [selectedStageId, setSelectedStageId] = useState<string>('overview');
  const [replayStep, setReplayStep] = useState<number>(0);
  const [microstructureCompression, setMicrostructureCompression] = useState<number>(0.25);
  const [autoMorphMicrostructure, setAutoMorphMicrostructure] = useState<boolean>(false);
  const [selectedCandidateId, setSelectedCandidateId] = useState<string | undefined>(undefined);
  const [evidenceDrawerOpen, setEvidenceDrawerOpen] = useState<boolean>(false);
  const [cameraResetCounter, setCameraResetCounter] = useState<number>(0);
  const [processPhase, setProcessPhase] = useState<ProcessExperiencePhase>('OVERVIEW');
  const [isCinematicTourActive, setIsCinematicTourActive] = useState<boolean>(false);
  const [tourInfo, setTourInfo] = useState<TourUpdateInfo | null>(null);
  const [tourActionTrigger, setTourActionTrigger] = useState<{ action: 'next' | 'prev' | 'togglePause'; timestamp: number } | null>(null);
  const [isCleanScreenMode, setIsCleanScreenMode] = useState<boolean>(false);
  const [isOptimizationOrbiting, setIsOptimizationOrbiting] = useState<boolean>(false);

  const activeScenario = SCENARIOS[activeScenarioId];

  // 2. Validate Scenarios on Startup (Fail-Closed Scientific Integrity Check)
  useEffect(() => {
    const val = validateAllScenarios(SCENARIOS);
    if (!val.valid) {
      console.error('CRITICAL: Scientific dataset validation failed:', val.errors);
      alert(`Scientific Dataset Integrity Error: ${val.errors[0]}`);
    }
  }, []);

  // 3. Scenario Change Handler
  const handleScenarioChange = useCallback((newScenarioId: ScenarioId) => {
    setActiveScenarioId(newScenarioId);
    setReplayStep(0);
    setSelectedCandidateId(undefined);
    setSelectedStageId('overview');
    setMicrostructureCompression(0.25);
    setAutoMorphMicrostructure(false);
    setProcessPhase('OVERVIEW');
    setIsOptimizationOrbiting(false);
  }, []);

  // 4. Scene Transition Handlers
  const handleSceneSelect = useCallback((sceneNumber: number) => {
    setCurrentScene(sceneNumber);
    if (sceneNumber === 3) {
      // Focus calendering when entering microstructure
      setSelectedStageId('calendering');
    }
    if (sceneNumber !== 2) {
      setIsCinematicTourActive(false);
    }
    if (sceneNumber !== 4) {
      setIsOptimizationOrbiting(false);
    }
  }, []);

  // 5. Stage Exploration Handler
  const handleStageSelect = useCallback((stageId: string) => {
    setSelectedStageId(stageId);
    setProcessPhase(stageId === 'overview' ? 'OVERVIEW' : 'PROCESS_ENTERING');
  }, []);

  useEffect(() => {
    if (processPhase !== 'PROCESS_ENTERING') return;
    const timer = window.setTimeout(() => setProcessPhase('PROCESS_EXPLORE'), 900);
    return () => window.clearTimeout(timer);
  }, [processPhase, selectedStageId]);

  useEffect(() => {
    if (processPhase !== 'ILLUSTRATED_PROCESS_RUN') return;
    const decision = pendingDecision(activeScenario, replayStep);
    if (!decision) return;
    setSelectedCandidateId(decision.selectedCandidateId);
    const timer = window.setTimeout(() => {
      setReplayStep((step) => revealNextDecision(activeScenario, step));
      setProcessPhase('RESULT_REVEAL');
    }, activeScenario.id === 'drakopoulos_graphite' ? 4200 : 3400);
    return () => window.clearTimeout(timer);
  }, [activeScenario, processPhase, replayStep]);

  useEffect(() => {
    if (processPhase !== 'PROCESS_EXIT') return;
    const timer = window.setTimeout(() => {
      setSelectedStageId('overview');
      setProcessPhase('OVERVIEW');
    }, 850);
    return () => window.clearTimeout(timer);
  }, [processPhase]);

  // 6. Keyboard Shortcuts for Exhibition Presentation
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Don't trigger if user is interacting with an input
      if (['INPUT', 'SELECT', 'TEXTAREA'].includes((e.target as HTMLElement)?.tagName)) {
        return;
      }

      if (e.key === '1') handleSceneSelect(1);
      else if (e.key === '2') handleSceneSelect(2);
      else if (e.key === '3') handleSceneSelect(3);
      else if (e.key === '4') handleSceneSelect(4);
      else if (e.key === '5') handleSceneSelect(5);
      else if (e.key === 'ArrowRight') {
        setCurrentScene((prev) => Math.min(5, prev + 1));
      } else if (e.key === 'ArrowLeft') {
        setCurrentScene((prev) => Math.max(1, prev - 1));
      } else if (e.key === 's' || e.key === 'S') {
        handleScenarioChange(
          activeScenarioId === 'warwick_nmc622_calendering' ? 'drakopoulos_graphite' : 'warwick_nmc622_calendering'
        );
      } else if (e.key === 'r' || e.key === 'R') {
        setCameraResetCounter((c) => c + 1);
      } else if (e.key === 'h' || e.key === 'H') {
        setIsCleanScreenMode((prev) => !prev);
      } else if (e.key === 'Escape') {
        setEvidenceDrawerOpen(false);
        setIsCleanScreenMode(false);
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [activeScenarioId, handleSceneSelect, handleScenarioChange]);

  return (
    <div className="relative w-screen h-screen overflow-hidden bg-[#F4F7F7] font-sans select-none text-[#142A35]">
      {/* 1. PERSISTENT 3D WORLD CANVAS (Dominates Entire Screen) */}
      <PersistentWorldCanvas
        currentScene={currentScene}
        activeScenario={activeScenario}
        selectedStageId={selectedStageId}
        replayStep={replayStep}
        processPhase={processPhase}
        microstructureCompression={microstructureCompression}
        autoMorphMicrostructure={autoMorphMicrostructure}
        cameraResetTrigger={cameraResetCounter}
        selectedCandidateId={selectedCandidateId}
        isOptimizationOrbiting={isOptimizationOrbiting}
        isCinematicTourActive={isCinematicTourActive}
        onCinematicTourUpdate={(info) => setTourInfo(info)}
        tourActionTrigger={tourActionTrigger}
        onStageSelect={handleStageSelect}
        onCandidateSelect={(id) => setSelectedCandidateId(id)}
        onSceneSelect={handleSceneSelect}
      />

      {/* Floating Exit Button when Clean Screen Mode is enabled */}
      {isCleanScreenMode && (
        <button
          onClick={() => setIsCleanScreenMode(false)}
          className="pointer-events-auto fixed top-4 right-4 z-50 flex items-center gap-2 rounded-full border border-white/40 bg-[#142A35]/85 px-4 py-2 text-xs font-bold text-white shadow-2xl backdrop-blur-xl hover:bg-[#142A35] transition"
          title="Exit Clean Screen (Hotkey: H)"
        >
          <Camera className="h-4 w-4 text-[#38BDF8]" />
          <span>Exit Clean Screen (H)</span>
        </button>
      )}

      {/* 2. TOP NAVBAR */}
      <div className={`transition-opacity duration-300 ${isCleanScreenMode ? 'opacity-0 pointer-events-none' : 'opacity-100'}`}>
        <Navbar
          scenarios={SCENARIOS}
          activeScenarioId={activeScenarioId}
          currentScene={currentScene}
          onScenarioChange={handleScenarioChange}
          onSceneSelect={handleSceneSelect}
          onResetCamera={() => setCameraResetCounter((c) => c + 1)}
          onToggleEvidenceDrawer={() => setEvidenceDrawerOpen(true)}
        />
      </div>

      {/* 3. SCENE CONTEXTUAL HUD OVERLAYS */}
      <main className={`relative w-full h-full pointer-events-none z-10 transition-opacity duration-300 ${isCleanScreenMode ? 'opacity-0 pointer-events-none' : 'opacity-100'}`}>
        {currentScene === 1 && (
          <Scene1Hero
            scenario={activeScenario}
            onExploreLine={() => handleSceneSelect(2)}
            onExploreMicrostructure={() => handleSceneSelect(3)}
          />
        )}

        {currentScene === 2 && (
          <Scene2ProcessExplorer
            scenario={activeScenario}
            selectedStageId={selectedStageId}
            processPhase={processPhase}
            replayStep={replayStep}
            isCinematicTourActive={isCinematicTourActive}
            tourInfo={tourInfo}
            onSelectStage={handleStageSelect}
            onBeginDecision={() => {
              const decision = pendingDecision(activeScenario, replayStep);
              setSelectedCandidateId(decision?.selectedCandidateId);
              setProcessPhase('AI_DECISION');
            }}
            onRunProcess={() => setProcessPhase('ILLUSTRATED_PROCESS_RUN')}
            onNextDecision={() => {
              setSelectedCandidateId(pendingDecision(activeScenario, replayStep)?.selectedCandidateId);
              setProcessPhase('AI_DECISION');
            }}
            onRepeatDecision={() => {
              const previous = Math.max(0, replayStep - 1);
              setReplayStep(previous);
              setSelectedCandidateId(pendingDecision(activeScenario, previous)?.selectedCandidateId);
              setProcessPhase('AI_DECISION');
            }}
            onReturnOverview={() => setProcessPhase('PROCESS_EXIT')}
            onNavigateMicrostructure={() => handleSceneSelect(3)}
            onNavigateOptimization={() => handleSceneSelect(4)}
            onStartVideoTour={() => setIsCinematicTourActive(true)}
            onExitVideoTour={() => setIsCinematicTourActive(false)}
            onToggleTourPause={() => setTourActionTrigger({ action: 'togglePause', timestamp: Date.now() })}
            onStepTour={(dir) => setTourActionTrigger({ action: dir === 1 ? 'next' : 'prev', timestamp: Date.now() })}
            onToggleCleanScreen={() => setIsCleanScreenMode((prev) => !prev)}
          />
        )}

        {currentScene === 3 && (
          <Scene3MicrostructureInspector
            scenario={activeScenario}
            compression={microstructureCompression}
            isAutoMorphing={autoMorphMicrostructure}
            onCompressionChange={(val) => setMicrostructureCompression(val)}
            onToggleAutoMorph={() => setAutoMorphMicrostructure((prev) => !prev)}
            onResetCompression={() => {
              setMicrostructureCompression(0);
              setAutoMorphMicrostructure(false);
            }}
            onProceedToOptimization={() => handleSceneSelect(4)}
          />
        )}

        {currentScene === 4 && (
          <Scene4OptimizationStudio
            scenario={activeScenario}
            replayStep={replayStep}
            onStepChange={(step) => setReplayStep(step)}
            onSelectCandidate={(id) => setSelectedCandidateId(id)}
            selectedCandidateId={selectedCandidateId}
            onProceedToEvidence={() => handleSceneSelect(5)}
            isOrbiting={isOptimizationOrbiting}
            onToggleOrbit={() => setIsOptimizationOrbiting((prev) => !prev)}
          />
        )}

        {currentScene === 5 && (
          <Scene5ScientificEvidence
            scenario={activeScenario}
            onToggleLimitationsDrawer={() => setEvidenceDrawerOpen(true)}
            onSwitchScenario={handleScenarioChange}
            onRestartJourney={() => handleSceneSelect(1)}
          />
        )}
      </main>

      {/* 4. BOTTOM DOCKED SCENE NAVIGATOR */}
      <div className={`transition-opacity duration-300 ${isCleanScreenMode ? 'opacity-0 pointer-events-none' : 'opacity-100'}`}>
        <SceneNavigator
          currentScene={currentScene}
          onSceneSelect={handleSceneSelect}
        />
      </div>

      {/* 5. SCIENTIFIC EVIDENCE & LIMITATIONS DRAWER */}
      <EvidenceLimitationsDrawer
        isOpen={evidenceDrawerOpen}
        onClose={() => setEvidenceDrawerOpen(false)}
        scenario={activeScenario}
      />
    </div>
  );
};

export default App;
