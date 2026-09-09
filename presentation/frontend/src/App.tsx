import React, { useState, useEffect } from 'react';
import { SnapshotData, DataMode } from './types/mission_control';
import { Header, NavTab } from './components/Header';
import { PresenterMode, SCENES } from './components/PresenterMode';
import { SpeakerNotesModal } from './components/SpeakerNotesModal';
import { OverviewView } from './views/OverviewView';
import { DecisionCockpitView } from './views/DecisionCockpitView';
import { ALabAtlasView } from './views/ALabAtlasView';
import { BenchmarkLabView } from './views/BenchmarkLabView';
import { ElectrolyteView } from './views/ElectrolyteView';
import { ArchitectureView } from './views/ArchitectureView';
import { ReadinessView } from './views/ReadinessView';
import { Loader2, AlertCircle } from 'lucide-react';

export const App: React.FC = () => {
  const [data, setData] = useState<SnapshotData | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  const [currentTab, setCurrentTab] = useState<NavTab>('overview');
  const [presenterMode, setPresenterMode] = useState<boolean>(false);
  const [currentScene, setCurrentScene] = useState<number>(0);
  const [notesOpen, setNotesOpen] = useState<boolean>(false);

  // Controlled step for Cockpit (synced with Presenter Mode)
  const [cockpitStep, setCockpitStep] = useState<number>(1);

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
      // Don't intercept if user is typing in an input
      if (['INPUT', 'SELECT', 'TEXTAREA'].includes((e.target as HTMLElement)?.tagName)) {
        return;
      }

      if (e.key === 'p' || e.key === 'P') {
        setPresenterMode((prev) => !prev);
      } else if (e.key === 'n' || e.key === 'N') {
        setNotesOpen((prev) => !prev);
      } else if (!presenterMode) {
        if (e.key === '1') setCurrentTab('overview');
        if (e.key === '2') setCurrentTab('cockpit');
        if (e.key === '3') setCurrentTab('alab');
        if (e.key === '4') setCurrentTab('benchmarks');
        if (e.key === '5') setCurrentTab('electrolyte');
        if (e.key === '6') setCurrentTab('architecture');
        if (e.key === '7') setCurrentTab('readiness');
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [presenterMode]);

  // Handle Scene Change in Presenter Mode
  const handleSceneSelect = (sceneIndex: number) => {
    setCurrentScene(sceneIndex);
    const scene = SCENES[sceneIndex];
    if (scene) {
      setCurrentTab(scene.tab);
      if (sceneIndex === 3) {
        // The Wow moment scene -> Step 2 of controlled campaign
        setCockpitStep(2);
      } else if (sceneIndex === 2) {
        setCockpitStep(1);
      }
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
    currentTab === 'alab' 
      ? 'HISTORICAL_REPLAY' 
      : currentTab === 'cockpit' 
      ? 'CONTROLLED_SYNTHETIC' 
      : currentTab === 'readiness' 
      ? 'CONTROLLED_SYNTHETIC' 
      : currentTab === 'electrolyte' 
      ? 'CONTROLLED_SYNTHETIC' 
      : 'LIVE_COMPUTED';

  if (loading) {
    return (
      <div className="min-h-screen bg-slate-50 flex flex-col items-center justify-center p-6 text-slate-800">
        <Loader2 className="w-10 h-10 text-emerald-700 animate-spin mb-4" />
        <h2 className="text-base font-bold tracking-tight">Initializing AIcoScientist Mission Control...</h2>
        <p className="text-xs text-slate-500 mt-1 font-mono">Loading deterministic snapshot.json</p>
      </div>
    );
  }

  if (error || !data) {
    return (
      <div className="min-h-screen bg-slate-50 flex flex-col items-center justify-center p-6 text-slate-800">
        <div className="max-w-md w-full bg-white p-6 rounded-xl border border-slate-200 shadow-lg text-center">
          <AlertCircle className="w-10 h-10 text-crimson-600 mx-auto mb-3" />
          <h2 className="text-base font-bold text-slate-900">Snapshot Loading Error</h2>
          <p className="text-xs text-slate-600 mt-2 font-mono break-all">{error}</p>
          <button
            onClick={() => window.location.reload()}
            className="mt-4 px-4 py-2 bg-slate-900 text-white rounded-md text-xs font-semibold hover:bg-slate-800 cursor-pointer"
          >
            Retry Loading
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className={`min-h-screen flex flex-col bg-slate-50 text-slate-900 ${presenterMode ? 'pt-14' : ''}`}>
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
          onJumpToTab={(tab) => setCurrentTab(tab)}
        />
      )}

      {/* Main Mission Control Header */}
      <Header
        currentTab={currentTab}
        onSelectTab={(tab) => setCurrentTab(tab)}
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
      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 pt-6">
        {currentTab === 'overview' && <OverviewView data={data} onNavigate={(tab) => setCurrentTab(tab)} />}
        {currentTab === 'cockpit' && (
          <DecisionCockpitView
            data={data}
            controlledStep={presenterMode ? cockpitStep : undefined}
            onStepChange={setCockpitStep}
          />
        )}
        {currentTab === 'alab' && <ALabAtlasView data={data} />}
        {currentTab === 'benchmarks' && <BenchmarkLabView data={data} />}
        {currentTab === 'electrolyte' && <ElectrolyteView data={data} />}
        {currentTab === 'architecture' && <ArchitectureView data={data} />}
        {currentTab === 'readiness' && <ReadinessView data={data} />}
      </main>

      {/* Footer */}
      <footer className="border-t border-slate-200 bg-white py-4 mt-auto">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 flex flex-col sm:flex-row items-center justify-between text-xs text-slate-500 gap-2">
          <div className="flex items-center gap-3">
            <span className="font-semibold text-slate-700">AIcoScientist Discovery Mission Control</span>
            <span className="text-slate-300">|</span>
            <span className="font-mono text-2xs">SHA {data.provenance?.head_commit?.substring(0, 10)}</span>
          </div>
          <div className="flex items-center gap-4 text-2xs font-mono">
            <span>5,333 Audit Events</span>
            <span>180 Trajectories</span>
            <span>48/50 Gates Passed</span>
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
