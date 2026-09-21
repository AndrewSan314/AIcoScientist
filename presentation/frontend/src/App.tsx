import React, { useState, useEffect, useCallback } from 'react';
import { ScenarioId, ExhibitionScenario } from './data/types';
import { DRAKOPOULOS_SCENARIO, WARWICK_SCENARIO } from './data/exhibitionData';
import { validateAllScenarios } from './data/validation';
import { PersistentWorldCanvas } from './components/3d/PersistentWorldCanvas';
import { Navbar } from './components/ui/Navbar';
import { SceneNavigator } from './components/ui/SceneNavigator';
import { Scene1Hero } from './components/ui/Scene1Hero';
import { Scene2ProcessExplorer } from './components/ui/Scene2ProcessExplorer';
import { Scene3MicrostructureInspector } from './components/ui/Scene3MicrostructureInspector';
import { Scene4OptimizationStudio } from './components/ui/Scene4OptimizationStudio';
import { Scene5ScientificEvidence } from './components/ui/Scene5ScientificEvidence';
import { EvidenceLimitationsDrawer } from './components/ui/EvidenceLimitationsDrawer';

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
  }, []);

  // 4. Scene Transition Handlers
  const handleSceneSelect = useCallback((sceneNumber: number) => {
    setCurrentScene(sceneNumber);
    if (sceneNumber === 2) {
      // Show full connected production line overview first
      setSelectedStageId('overview');
    } else if (sceneNumber === 3) {
      // Focus calendering when entering microstructure
      setSelectedStageId('calendering');
    }
  }, []);

  // 5. Stage Exploration Handler
  const handleStageSelect = useCallback((stageId: string) => {
    setSelectedStageId(stageId);
  }, []);

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
      } else if (e.key === 'Escape') {
        setEvidenceDrawerOpen(false);
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
        microstructureCompression={microstructureCompression}
        autoMorphMicrostructure={autoMorphMicrostructure}
        cameraResetTrigger={cameraResetCounter}
        selectedCandidateId={selectedCandidateId}
        onStageSelect={handleStageSelect}
        onCandidateSelect={(id) => setSelectedCandidateId(id)}
        onSceneSelect={handleSceneSelect}
      />

      {/* 2. TOP NAVBAR */}
      <Navbar
        scenarios={SCENARIOS}
        activeScenarioId={activeScenarioId}
        currentScene={currentScene}
        onScenarioChange={handleScenarioChange}
        onSceneSelect={handleSceneSelect}
        onResetCamera={() => setCameraResetCounter((c) => c + 1)}
        onToggleEvidenceDrawer={() => setEvidenceDrawerOpen(true)}
      />

      {/* 3. SCENE CONTEXTUAL HUD OVERLAYS */}
      <main className="relative w-full h-full pointer-events-none z-10">
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
            onSelectStage={handleStageSelect}
            onNavigateMicrostructure={() => handleSceneSelect(3)}
            onNavigateOptimization={() => handleSceneSelect(4)}
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
      <SceneNavigator
        currentScene={currentScene}
        onSceneSelect={handleSceneSelect}
      />

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
