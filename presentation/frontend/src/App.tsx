import React, { useState, useEffect } from 'react';
import { SnapshotData, DataMode, WorkspaceTab, DiscoveryFlowState, DatasetOption, RevealPhase } from './types/mission_control';
import { Header } from './components/Header';
import { PresenterMode, SCENES } from './components/PresenterMode';
import { SpeakerNotesModal } from './components/SpeakerNotesModal';
import { DiscoveryLabWorkspace } from './views/DiscoveryLabWorkspace';
import { EvidenceBenchmarksWorkspace } from './views/EvidenceBenchmarksWorkspace';
import { ResearchSystemWorkspace } from './views/ResearchSystemWorkspace';
import { Loader2, AlertCircle } from 'lucide-react';
import { validateSnapshot } from './utils/snapshotValidation';

export const App: React.FC = () => {
  const [data, setData] = useState<SnapshotData | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  // Active Workspace Navigation (3 primary workspaces)
  const [currentWorkspace, setCurrentWorkspace] = useState<WorkspaceTab>('discovery');
  
  // Discovery Lab Operable Flow State
  const [discoveryFlowState, setDiscoveryFlowState] = useState<DiscoveryFlowState>('setup');
  const [discoveryDataset, setDiscoveryDataset] = useState<DatasetOption>('controlled_synthesis');
  const [cockpitStep, setCockpitStep] = useState<number>(1);
  const [discoveryRevealPhase, setDiscoveryRevealPhase] = useState<RevealPhase>('A_SCORED');

  // Evidence Benchmarks Question State
  const [benchmarkQuestion, setBenchmarkQuestion] = useState<number>(1);

  // How It Works Subtab State
  const [systemSubtab, setSystemSubtab] = useState<'architecture' | 'audit' | 'verification'>('architecture');

  // Presenter Mode State
  const [presenterMode, setPresenterMode] = useState<boolean>(false);
  const [currentScene, setCurrentScene] = useState<number>(0);
  const [notesOpen, setNotesOpen] = useState<boolean>(false);

  // Load deterministic snapshot on mount
  useEffect(() => {
    fetch('/snapshot.json')
      .then((res) => {
        if (!res.ok) {
          throw new Error(`Failed to load snapshot.json (HTTP ${res.status})`);
        }
        return res.json();
      })
      .then((json: SnapshotData) => {
        const valRes = validateSnapshot(json);
        if (!valRes.valid) {
          console.error('Snapshot validation failed:', valRes.errors);
          throw new Error(`Snapshot integrity failure: ${valRes.errors[0]}`);
        }
        setData(json);
        setLoading(false);
      })
      .catch((err) => {
        console.error('Error fetching snapshot:', err);
        setError(err.message || 'Unknown error loading snapshot data');
        setLoading(false);
      });
  }, []);

  // Keyboard shortcut listener
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (['INPUT', 'SELECT', 'TEXTAREA'].includes((e.target as HTMLElement)?.tagName)) {
        return;
      }

      if (e.key === 'p' || e.key === 'P') {
        setPresenterMode((prev) => !prev);
      } else if (e.key === 'n' || e.key === 'N') {
        setNotesOpen((prev) => !prev);
      } else if (!presenterMode) {
        if (e.key === '1') setCurrentWorkspace('discovery');
        if (e.key === '2') setCurrentWorkspace('benchmarks');
        if (e.key === '3') setCurrentWorkspace('system');
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [presenterMode]);

  // Handle Scene Change in Presenter Mode (Synchronizes full workspace state)
  const handleSceneSelect = (sceneIndex: number) => {
    setCurrentScene(sceneIndex);
    const scene = SCENES[sceneIndex];
    if (scene) {
      setCurrentWorkspace(scene.workspace);
      if (scene.flowState) setDiscoveryFlowState(scene.flowState);
      if (scene.datasetOption) setDiscoveryDataset(scene.datasetOption);
      if (scene.stepIndex !== undefined) setCockpitStep(scene.stepIndex);
      if (scene.revealPhase) setDiscoveryRevealPhase(scene.revealPhase);
      if (scene.questionId) setBenchmarkQuestion(scene.questionId);
      if (scene.subtab) setSystemSubtab(scene.subtab);
    }
  };

  const handleNextScene = () => {
    if (currentScene < SCENES.length - 1) {
      handleSceneSelect(currentScene + 1);
    }
  };

  const handlePrevScene = () => {
    if (currentScene > 0) {
      handleSceneSelect(currentScene - 1);
    }
  };

  const handleExitPresenter = () => {
    setPresenterMode(false);
  };

  // Determine active data mode
  const currentMode: DataMode = 
    currentWorkspace === 'benchmarks' && benchmarkQuestion === 4
      ? 'HISTORICAL_REPLAY'
      : currentWorkspace === 'discovery'
      ? 'CONTROLLED_SYNTHETIC'
      : 'LIVE_COMPUTED';

  if (loading) {
    return (
      <div className="min-h-screen bg-[#F4F3EE] flex flex-col items-center justify-center p-6 text-[#17201F]">
        <Loader2 className="w-10 h-10 text-[#DC2626] animate-spin mb-4" />
        <h2 className="text-base font-bold tracking-tight">Initializing AIcoScientist Discovery Console...</h2>
        <p className="text-xs text-[#66706C] mt-1 font-mono">Loading deterministic scientific snapshot</p>
      </div>
    );
  }

  if (error || !data) {
    return (
      <div className="min-h-screen bg-[#F4F3EE] flex flex-col items-center justify-center p-6 text-[#17201F]">
        <div className="max-w-md w-full bg-[#FCFCFA] p-6 rounded-xl border border-[#D9DFDB] shadow-lg text-center">
          <AlertCircle className="w-10 h-10 text-[#B91C1C] mx-auto mb-3" />
          <h2 className="text-base font-bold text-[#17201F]">Snapshot Loading Error</h2>
          <p className="text-xs text-[#66706C] mt-2 font-mono break-all">{error}</p>
          <button
            onClick={() => window.location.reload()}
            className="mt-4 px-4 py-2 bg-[#17201F] text-white rounded-md text-xs font-semibold hover:bg-[#243331] cursor-pointer"
          >
            Retry Loading
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className={`min-h-screen flex flex-col bg-[#F4F3EE] text-[#17201F] ${presenterMode ? 'pt-14' : ''}`}>
      {/* Presenter Mode Top HUD (when active) */}
      {presenterMode && (
        <PresenterMode
          currentScene={currentScene}
          totalScenes={SCENES.length}
          onPrevScene={handlePrevScene}
          onNextScene={handleNextScene}
          onSelectScene={handleSceneSelect}
          onExit={handleExitPresenter}
          onToggleNotes={() => setNotesOpen(true)}
          onJumpToWorkspace={(ws, qId) => {
            setCurrentWorkspace(ws);
            if (qId) setBenchmarkQuestion(qId);
          }}
        />
      )}

      {/* Main Mission Control Header */}
      <Header
        currentWorkspace={currentWorkspace}
        onSelectWorkspace={(ws) => setCurrentWorkspace(ws)}
        currentMode={currentMode}
        onLaunchPresenter={() => {
          setPresenterMode(true);
          handleSceneSelect(0);
        }}
        onToggleNotes={() => setNotesOpen(true)}
        headCommit={data.provenance?.head_commit}
        branch={data.provenance?.branch}
      />

      {/* Main Viewport Container */}
      <main className="flex-1 max-w-[1720px] w-full mx-auto px-4 sm:px-8 lg:px-12 pt-6">
        {currentWorkspace === 'discovery' && (
          <DiscoveryLabWorkspace
            data={data}
            controlledStepIndex={cockpitStep}
            onStepChange={setCockpitStep}
            controlledFlowState={discoveryFlowState}
            onFlowStateChange={setDiscoveryFlowState}
            controlledDataset={discoveryDataset}
            onDatasetChange={setDiscoveryDataset}
            controlledRevealPhase={discoveryRevealPhase}
            onRevealPhaseChange={setDiscoveryRevealPhase}
          />
        )}
        {currentWorkspace === 'benchmarks' && (
          <EvidenceBenchmarksWorkspace
            data={data}
            controlledQuestionId={benchmarkQuestion}
            onQuestionChange={setBenchmarkQuestion}
          />
        )}
        {currentWorkspace === 'system' && (
          <ResearchSystemWorkspace
            data={data}
            initialSubtab={systemSubtab}
          />
        )}
      </main>

      {/* Scientific Editorial Footer */}
      <footer className="border-t border-[#D9DFDB] bg-[#FCFCFA] py-4 mt-auto">
        <div className="max-w-[1720px] mx-auto px-4 sm:px-8 lg:px-12 flex flex-col sm:flex-row items-center justify-between text-xs text-[#66706C] gap-2">
          <div className="flex items-center gap-2.5">
            <span className="font-semibold text-[#17201F]">AIcoScientist Discovery Console</span>
            <span className="text-[#D9DFDB]">|</span>
            <span className="text-2xs text-[#8F9995]">Controlled & Validated Reproducible Snapshot</span>
          </div>
          <div className="flex items-center gap-4 text-2xs font-mono text-[#8F9995]">
            <span>5,333 Audit Events</span>
            <span>•</span>
            <span>180 Trajectories</span>
            <span>•</span>
            <span className="text-[#DC2626] font-semibold">48/50 Gates Passed</span>
          </div>
        </div>
      </footer>

      {/* Speaker Notes Modal */}
      <SpeakerNotesModal
        isOpen={notesOpen}
        onClose={() => setNotesOpen(false)}
        activeSceneIndex={presenterMode ? currentScene : undefined}
      />
    </div>
  );
};

export default App;
