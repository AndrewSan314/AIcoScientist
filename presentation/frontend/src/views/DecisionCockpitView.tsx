import React, { useState, useEffect } from 'react';
import { SnapshotData, CampaignStep, ScoredActionRecord } from '../types/mission_control';
import { ModeBadge } from '../components/ModeBadge';
import { StarkHologramSphere } from '../components/StarkHologramSphere';
import { 
  Lock, 
  Unlock, 
  ArrowRight, 
  RotateCcw, 
  Play, 
  Layers, 
  FlaskConical, 
  CheckCircle2, 
  Sparkles, 
  ChevronDown, 
  ChevronUp, 
  Info,
  Zap,
  Check
} from 'lucide-react';

interface DecisionCockpitViewProps {
  data: SnapshotData;
  controlledStep?: number;
  onStepChange?: (step: number) => void;
}

type RevealState = 'STATE_A_BEFORE' | 'STATE_B_LOCKED' | 'STATE_C_REVEAL' | 'STATE_D_UPDATED';

export const DecisionCockpitView: React.FC<DecisionCockpitViewProps> = ({
  data,
  controlledStep,
  onStepChange,
}) => {
  const [selectedCampaignType, setSelectedCampaignType] = useState<'controlled' | 'alab_replay'>('controlled');
  const [internalStep, setInternalStep] = useState<number>(1);
  const [revealState, setRevealState] = useState<RevealState>('STATE_D_UPDATED');
  const [selectedCandidateId, setSelectedCandidateId] = useState<string | null>(null);
  const [showWhyNot, setShowWhyNot] = useState<boolean>(false);
  const [candidateViewMode, setCandidateViewMode] = useState<'3D_STARK' | '2D_GRID'>('3D_STARK');
  const [isHologramScanning, setIsHologramScanning] = useState<boolean>(false);

  const currentStepNum = controlledStep !== undefined ? controlledStep : internalStep;

  const campaign = selectedCampaignType === 'controlled' 
    ? data.flagship_campaign 
    : data.alab_replay_campaign;

  const currentStepData: CampaignStep = campaign?.steps?.[currentStepNum - 1] || campaign?.steps?.[0] || {
    step: 1,
    preregistration: null,
    observation: null,
    belief_update: null,
    top_actions: [],
    total_actions_evaluated: 0,
  };

  const prereg = currentStepData.preregistration;
  const obs = currentStepData.observation;
  const action = prereg?.action;

  // Sync selected candidate
  useEffect(() => {
    if (action?.candidate_id) {
      setSelectedCandidateId(action.candidate_id);
    }
  }, [action?.candidate_id, currentStepNum]);

  const handleStepSelect = (s: number) => {
    if (onStepChange) {
      onStepChange(s);
    } else {
      setInternalStep(s);
    }
    setRevealState('STATE_D_UPDATED');
  };

  const handleResetReplay = () => {
    handleStepSelect(1);
    setRevealState('STATE_A_BEFORE');
  };

  const stepThroughReveal = () => {
    if (revealState === 'STATE_A_BEFORE') setRevealState('STATE_B_LOCKED');
    else if (revealState === 'STATE_B_LOCKED') setRevealState('STATE_C_REVEAL');
    else if (revealState === 'STATE_C_REVEAL') setRevealState('STATE_D_UPDATED');
    else if (revealState === 'STATE_D_UPDATED') {
      if (currentStepNum < (campaign?.steps?.length || 4)) {
        handleStepSelect(currentStepNum + 1);
        setRevealState('STATE_A_BEFORE');
        setIsHologramScanning(true);
      }
    }
  };

  // Beliefs
  const initialBeliefs = data.flagship_campaign.initial_beliefs || {
    H1_PHASE_PURITY_LIMITED: 0.3333,
    H2_COMPOSITION_HOMOGENEITY_LIMITED: 0.3333,
    H3_MORPHOLOGY_KINETICS_LIMITED: 0.3333,
  };

  const beliefsBefore = prereg?.beliefs_before || initialBeliefs;
  const beliefsAfter = (revealState === 'STATE_D_UPDATED') 
    ? (obs?.beliefs_after || beliefsBefore) 
    : beliefsBefore;

  const posteriorDelta = obs?.posterior_delta || {
    H1_PHASE_PURITY_LIMITED: 0,
    H2_COMPOSITION_HOMOGENEITY_LIMITED: 0,
    H3_MORPHOLOGY_KINETICS_LIMITED: 0,
  };

  const hypotheses = data.hypotheses || {};

  // Scores
  const higNats = prereg?.expected_hig_nats || 0;
  const discUtil = prereg?.discovery_utility || 0;
  const estCost = action?.estimated_cost || 1;
  const totalScore = prereg?.total_action_score || 0;

  return (
    <div className="space-y-6 pb-16 animate-fadeIn max-w-7xl mx-auto">
      {/* Top Header & Integrated Sequence Bar */}
      <div className="bg-white p-5 rounded-xl border border-slate-200 shadow-xs space-y-4">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
          <div>
            <div className="flex items-center gap-2 mb-1">
              <h1 className="text-xl font-bold text-slate-900 tracking-tight">Autonomous Decision Cockpit</h1>
              <ModeBadge 
                mode={selectedCampaignType === 'controlled' ? 'CONTROLLED_SYNTHETIC' : 'HISTORICAL_REPLAY'} 
                size="sm"
              />
            </div>
            <p className="text-xs text-slate-500">
              Joint Candidate × Modality recommendation with preregistered predictions & Bayesian posterior tracking.
            </p>
          </div>

          {/* Controls: Campaign & Step Selectors */}
          <div className="flex flex-wrap items-center gap-3">
            <div className="flex bg-slate-100 p-1 rounded-lg border border-slate-200 text-xs">
              <button
                onClick={() => { setSelectedCampaignType('controlled'); handleStepSelect(1); }}
                className={`px-3 py-1 rounded-md font-medium transition cursor-pointer ${
                  selectedCampaignType === 'controlled'
                    ? 'bg-white text-emerald-800 shadow-xs font-semibold'
                    : 'text-slate-600 hover:text-slate-900'
                }`}
              >
                Controlled Syn
              </button>
              <button
                onClick={() => { setSelectedCampaignType('alab_replay'); handleStepSelect(1); }}
                className={`px-3 py-1 rounded-md font-medium transition cursor-pointer ${
                  selectedCampaignType === 'alab_replay'
                    ? 'bg-white text-emerald-800 shadow-xs font-semibold'
                    : 'text-slate-600 hover:text-slate-900'
                }`}
              >
                A-Lab Replay (PG)
              </button>
            </div>

            {/* Stepper */}
            <div className="flex items-center bg-slate-100 p-1 rounded-lg border border-slate-200 text-xs">
              <span className="text-slate-400 font-mono text-xs px-2">STEP</span>
              {(campaign?.steps || []).map((s) => (
                <button
                  key={s.step}
                  onClick={() => handleStepSelect(s.step)}
                  className={`w-7 h-7 rounded-md font-mono text-xs font-semibold transition cursor-pointer flex items-center justify-center ${
                    currentStepNum === s.step
                      ? 'bg-emerald-600 text-white shadow-xs'
                      : 'text-slate-600 hover:bg-slate-200'
                  }`}
                >
                  {s.step}
                </button>
              ))}
            </div>

            <button
              onClick={handleResetReplay}
              className="p-1.5 rounded-lg border border-slate-200 bg-white hover:bg-slate-50 text-slate-600 text-xs transition cursor-pointer flex items-center gap-1"
              title="Reset to Step 1"
            >
              <RotateCcw className="w-3.5 h-3.5" />
              <span className="hidden sm:inline">Reset</span>
            </button>
          </div>
        </div>

        {/* Clean Scientific Sequence Stepper (White & Emerald) */}
        <div className="pt-3 border-t border-slate-100 flex flex-col md:flex-row md:items-center justify-between gap-3">
          <div className="flex items-center gap-1 sm:gap-2 text-xs overflow-x-auto pb-1">
            {/* State A */}
            <button
              onClick={() => setRevealState('STATE_A_BEFORE')}
              className={`px-3 py-1.5 rounded-lg border text-left transition cursor-pointer flex items-center gap-2 ${
                revealState === 'STATE_A_BEFORE'
                  ? 'bg-emerald-50 border-emerald-500 text-emerald-900 font-bold shadow-2xs'
                  : 'bg-white border-slate-200 text-slate-500 hover:border-slate-300'
              }`}
            >
              <span className="w-4 h-4 rounded-full bg-emerald-600 text-white flex items-center justify-center text-3xs font-mono">1</span>
              <span>Candidate Scored</span>
            </button>

            <span className="text-slate-300">→</span>

            {/* State B */}
            <button
              onClick={() => setRevealState('STATE_B_LOCKED')}
              className={`px-3 py-1.5 rounded-lg border text-left transition cursor-pointer flex items-center gap-2 ${
                revealState === 'STATE_B_LOCKED'
                  ? 'bg-emerald-50 border-emerald-500 text-emerald-900 font-bold shadow-2xs'
                  : 'bg-white border-slate-200 text-slate-500 hover:border-slate-300'
              }`}
            >
              <Lock className="w-3.5 h-3.5 text-emerald-700" />
              <span>Preregistration Locked</span>
            </button>

            <span className="text-slate-300">→</span>

            {/* State C */}
            <button
              onClick={() => setRevealState('STATE_C_REVEAL')}
              className={`px-3 py-1.5 rounded-lg border text-left transition cursor-pointer flex items-center gap-2 ${
                revealState === 'STATE_C_REVEAL'
                  ? 'bg-emerald-50 border-emerald-500 text-emerald-900 font-bold shadow-2xs'
                  : 'bg-white border-slate-200 text-slate-500 hover:border-slate-300'
              }`}
            >
              <Unlock className="w-3.5 h-3.5 text-emerald-700" />
              <span>Evidence Revealed</span>
            </button>

            <span className="text-slate-300">→</span>

            {/* State D */}
            <button
              onClick={() => setRevealState('STATE_D_UPDATED')}
              className={`px-3 py-1.5 rounded-lg border text-left transition cursor-pointer flex items-center gap-2 ${
                revealState === 'STATE_D_UPDATED'
                  ? 'bg-emerald-50 border-emerald-500 text-emerald-900 font-bold shadow-2xs'
                  : 'bg-white border-slate-200 text-slate-500 hover:border-slate-300'
              }`}
            >
              <CheckCircle2 className="w-3.5 h-3.5 text-emerald-600" />
              <span>Posterior Updated</span>
            </button>
          </div>

          {/* Primary Action Button */}
          <button
            onClick={stepThroughReveal}
            className="px-4 py-2 rounded-lg bg-emerald-600 hover:bg-emerald-700 text-xs font-semibold text-white shadow-xs transition flex items-center justify-center gap-2 cursor-pointer shrink-0"
          >
            {revealState === 'STATE_A_BEFORE' && (
              <>
                <Lock className="w-3.5 h-3.5" />
                <span>Lock Preregistration (State B)</span>
              </>
            )}
            {revealState === 'STATE_B_LOCKED' && (
              <>
                <Unlock className="w-3.5 h-3.5" />
                <span>Reveal Observation (State C)</span>
              </>
            )}
            {revealState === 'STATE_C_REVEAL' && (
              <>
                <Play className="w-3.5 h-3.5" />
                <span>Update Posterior (State D)</span>
              </>
            )}
            {revealState === 'STATE_D_UPDATED' && (
              <>
                <span>Advance to Next Step</span>
                <ArrowRight className="w-3.5 h-3.5" />
              </>
            )}
          </button>
        </div>
      </div>

      {/* Main 2-Column Clean Layout */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
        {/* LEFT COLUMN: 3D Stark Hologram Discovery Space (7 cols) */}
        <div className="lg:col-span-7 space-y-4">
          <div className="bg-white p-5 rounded-xl border border-slate-200 shadow-xs space-y-4">
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 border-b border-slate-100 pb-3">
              <div className="flex items-center gap-2">
                <FlaskConical className="w-4 h-4 text-emerald-600" />
                <h2 className="text-sm font-bold text-slate-900 uppercase tracking-wide">Candidate Discovery Space</h2>
              </div>

              {/* View Switcher: 3D Stark Lattice vs 2D Matrix */}
              <div className="flex items-center bg-slate-100 p-0.5 rounded-lg border border-slate-200 text-xs">
                <button
                  onClick={() => setCandidateViewMode('3D_STARK')}
                  className={`px-3 py-1 rounded-md font-mono text-2xs font-bold flex items-center gap-1.5 transition cursor-pointer ${
                    candidateViewMode === '3D_STARK'
                      ? 'bg-emerald-600 text-white shadow-xs'
                      : 'text-slate-600 hover:text-slate-900'
                  }`}
                >
                  <Zap className="w-3 h-3" />
                  <span>3D Atomic Lattice</span>
                </button>
                <button
                  onClick={() => setCandidateViewMode('2D_GRID')}
                  className={`px-3 py-1 rounded-md font-mono text-2xs font-bold flex items-center gap-1.5 transition cursor-pointer ${
                    candidateViewMode === '2D_GRID'
                      ? 'bg-emerald-600 text-white shadow-xs'
                      : 'text-slate-600 hover:text-slate-900'
                  }`}
                >
                  <span>2D Grid</span>
                </button>
              </div>
            </div>

            {/* View Content */}
            {candidateViewMode === '3D_STARK' ? (
              <div className="rounded-xl overflow-hidden border border-emerald-950/40">
                <StarkHologramSphere
                  candidates={data.flagship_campaign.candidates || []}
                  selectedCandidateId={selectedCandidateId || action?.candidate_id}
                  onSelectCandidate={(id) => setSelectedCandidateId(id)}
                  currentStep={currentStepData}
                  isScanning={isHologramScanning}
                  onScanComplete={(winner) => {
                    setIsHologramScanning(false);
                    setSelectedCandidateId(winner);
                  }}
                />
              </div>
            ) : (
              <div className="grid grid-cols-2 sm:grid-cols-3 gap-2">
                {(data.flagship_campaign.candidates || []).map((cand) => {
                  const isTarget = cand.candidate_id === (selectedCandidateId || action?.candidate_id);
                  return (
                    <button
                      key={cand.candidate_id}
                      onClick={() => setSelectedCandidateId(cand.candidate_id)}
                      className={`p-3 rounded-lg border text-left transition cursor-pointer ${
                        isTarget
                          ? 'border-emerald-500 bg-emerald-50/60 ring-2 ring-emerald-400'
                          : 'border-slate-200 bg-white hover:border-slate-300'
                      }`}
                    >
                      <div className="flex items-center justify-between">
                        <span className="font-mono text-xs font-bold text-slate-900">
                          {cand.candidate_id.replace('controlled-', 'Syn-')}
                        </span>
                        {isTarget && (
                          <span className="w-2 h-2 rounded-full bg-emerald-600 animate-pulse" />
                        )}
                      </div>
                      <div className="text-2xs text-slate-500 mt-1 truncate">
                        {cand.composition_label}
                      </div>
                    </button>
                  );
                })}
              </div>
            )}

            {/* Candidate Metadata Strip */}
            <div className="p-3.5 rounded-lg bg-emerald-50/40 border border-emerald-100 flex flex-wrap items-center justify-between text-xs gap-3">
              <div>
                <span className="text-slate-500">Selected Formulation: </span>
                <span className="font-bold text-slate-900 font-mono">
                  {selectedCandidateId || action?.candidate_id || 'PG_0309'}
                </span>
              </div>
              <div className="flex items-center gap-4 text-2xs text-slate-600 font-mono">
                <span>XRD Cost: <strong>1.0 Unit</strong></span>
                <span>•</span>
                <span>Refinement Cost: <strong>1.0 Unit</strong></span>
                <span>•</span>
                <span className="text-emerald-700 font-semibold">Feasibility: Satisfied</span>
              </div>
            </div>
          </div>
        </div>

        {/* RIGHT COLUMN: Decision Intelligence & Hypotheses (5 cols) */}
        <div className="lg:col-span-5 space-y-4">
          {/* Card 1: Hero Recommendation & VoI Score */}
          <div className="bg-white p-5 rounded-xl border border-slate-200 shadow-xs space-y-4 relative overflow-hidden">
            <div className="absolute top-0 left-0 right-0 h-1 bg-emerald-600" />

            <div className="flex items-center justify-between">
              <span className="text-2xs font-bold uppercase tracking-wider px-2 py-0.5 rounded bg-emerald-100 text-emerald-800 border border-emerald-200">
                Recommended Decision
              </span>
              <span className="text-2xs font-mono text-slate-400">
                Policy: {campaign?.policy || 'HYBRID'}
              </span>
            </div>

            <div>
              <div className="text-2xl font-extrabold text-slate-900 tracking-tight flex items-baseline gap-2">
                <span>{action?.action_type || 'XRD'}</span>
                <span className="text-sm font-semibold text-emerald-700 font-mono">
                  on {action?.candidate_id || 'PG_0309'}
                </span>
              </div>
              <p className="text-xs text-slate-500 mt-1">
                Selected by maximizing expected Hypothesis Information Gain and discovery yield net of cost.
              </p>
            </div>

            {/* Large VoI Net Score Display */}
            <div className="p-4 rounded-xl bg-emerald-50/50 border border-emerald-100 flex items-center justify-between">
              <div>
                <span className="text-2xs font-bold uppercase text-emerald-800 tracking-wider">Net Acquisition Score</span>
                <div className="text-2xl font-extrabold text-emerald-700 font-mono">
                  +{totalScore.toFixed(3)} <span className="text-xs font-normal text-emerald-600">nats</span>
                </div>
              </div>
              <div className="text-right text-2xs text-slate-500 font-mono">
                <div>w_HIG: 0.8 | w_Disc: 0.8</div>
                <div>Cost Penalty: 2.0</div>
              </div>
            </div>

            {/* Score Decomposition Waterfall in White & Emerald */}
            <div className="space-y-2 pt-1 text-xs">
              <span className="text-2xs font-bold uppercase text-slate-400 tracking-wider">Value of Information Decomposition</span>

              <div className="space-y-1.5">
                <div className="flex justify-between text-2xs font-mono">
                  <span className="text-slate-600">Hypothesis Info Gain (HIG):</span>
                  <span className="font-bold text-emerald-700">+{higNats.toFixed(4)} nats</span>
                </div>
                <div className="w-full bg-slate-100 h-2 rounded-full overflow-hidden">
                  <div className="bg-emerald-600 h-full rounded-full" style={{ width: `${Math.min(100, higNats * 150)}%` }} />
                </div>

                <div className="flex justify-between text-2xs font-mono pt-1">
                  <span className="text-slate-600">Discovery Utility (ΔU):</span>
                  <span className="font-bold text-emerald-600">+{discUtil.toFixed(4)}</span>
                </div>
                <div className="w-full bg-slate-100 h-2 rounded-full overflow-hidden">
                  <div className="bg-emerald-400 h-full rounded-full" style={{ width: `${Math.min(100, discUtil * 120)}%` }} />
                </div>

                <div className="flex justify-between text-2xs font-mono pt-1">
                  <span className="text-slate-600">Measurement Cost Penalty:</span>
                  <span className="font-bold text-slate-600">-{(estCost * 0.18).toFixed(4)}</span>
                </div>
                <div className="w-full bg-slate-100 h-2 rounded-full overflow-hidden">
                  <div className="bg-slate-300 h-full rounded-full" style={{ width: `${Math.min(100, estCost * 25)}%` }} />
                </div>
              </div>
            </div>

            {/* Preregistration Ledger Status Pill */}
            <div className="p-3 rounded-lg bg-slate-50 border border-slate-200 text-2xs text-slate-600 flex items-start gap-2">
              <Lock className="w-3.5 h-3.5 text-emerald-600 shrink-0 mt-0.5" />
              <span>
                <strong>Preregistration Guarantee:</strong> Observable distribution committed to evidence ledger before physical data unblinding.
              </span>
            </div>
          </div>

          {/* Card 2: Competing Hypotheses Observatory */}
          <div className="bg-white p-5 rounded-xl border border-slate-200 shadow-xs space-y-3">
            <div className="flex items-center justify-between border-b border-slate-100 pb-2">
              <div className="flex items-center gap-2">
                <Layers className="w-4 h-4 text-emerald-600" />
                <h3 className="text-xs font-bold text-slate-900 uppercase tracking-wide">Hypothesis Observatory</h3>
              </div>
              <span className="text-2xs font-mono px-2 py-0.5 rounded bg-slate-100 text-slate-600">
                3 Competing Models
              </span>
            </div>

            <div className="space-y-2.5">
              {Object.entries(hypotheses).map(([hid, hdef]) => {
                const pBefore = beliefsBefore[hid] || 0.3333;
                const pAfter = beliefsAfter[hid] || pBefore;
                const delta = (revealState === 'STATE_D_UPDATED') ? (posteriorDelta[hid] || 0) : 0;
                const isDominant = pAfter > 0.45;

                return (
                  <div key={hid} className="p-3 rounded-lg border border-slate-200 bg-slate-50/50 space-y-1.5">
                    <div className="flex items-center justify-between text-xs">
                      <span className="font-semibold text-slate-900">{hdef.title}</span>
                      <div className="flex items-center gap-2 font-mono">
                        <span className="font-bold text-emerald-800">{(pAfter * 100).toFixed(1)}%</span>
                        {delta !== 0 && (
                          <span className={`text-2xs font-bold ${delta > 0 ? 'text-emerald-600' : 'text-slate-400'}`}>
                            {delta > 0 ? `+${(delta * 100).toFixed(1)}%` : `${(delta * 100).toFixed(1)}%`}
                          </span>
                        )}
                      </div>
                    </div>

                    {/* Progress bar in clean emerald */}
                    <div className="w-full bg-slate-200 h-2 rounded-full overflow-hidden">
                      <div 
                        className={`h-full rounded-full transition-all duration-500 ${
                          isDominant ? 'bg-emerald-600' : 'bg-emerald-400'
                        }`} 
                        style={{ width: `${pAfter * 100}%` }} 
                      />
                    </div>
                  </div>
                );
              })}
            </div>
          </div>

          {/* Card 3: Revealed Measurement (States C & D) */}
          {(revealState === 'STATE_C_REVEAL' || revealState === 'STATE_D_UPDATED') && obs && (
            <div className="bg-emerald-50/50 p-4 rounded-xl border border-emerald-200 text-xs space-y-2">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-1.5 text-emerald-900 font-bold">
                  <Unlock className="w-3.5 h-3.5 text-emerald-700" />
                  <span>Revealed Observable: {obs.action.action_type}</span>
                </div>
                <span className="font-mono text-2xs text-emerald-700">
                  {obs.timestamp?.substring(11, 19)} UTC
                </span>
              </div>

              <div className="bg-white p-2.5 rounded-lg border border-emerald-100 font-mono text-2xs space-y-1 text-slate-700">
                {obs.observed_measurement?.observable_names?.map((name, i) => {
                  const val = Array.isArray(obs.observed_measurement.value) 
                    ? obs.observed_measurement.value[i] 
                    : obs.observed_measurement.value;
                  return (
                    <div key={name} className="flex justify-between">
                      <span className="text-slate-500">{name}:</span>
                      <span className="font-bold text-slate-900">{typeof val === 'number' ? val.toFixed(4) : String(val)}</span>
                    </div>
                  );
                })}
              </div>

              <div className="flex justify-between text-2xs text-emerald-900 pt-1 font-mono">
                <span>Realized Entropy Reduction:</span>
                <span className="font-bold text-emerald-700">
                  {obs.realized_entropy_reduction_nats?.toFixed(4) || '0.0000'} nats
                </span>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
