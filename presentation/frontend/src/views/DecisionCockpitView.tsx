import React, { useState, useEffect } from 'react';
import { SnapshotData, CampaignStep, ScoredActionRecord } from '../types/mission_control';
import { ModeBadge } from '../components/ModeBadge';
import { 
  Lock, 
  Unlock, 
  ArrowRight, 
  RotateCcw, 
  Play, 
  HelpCircle, 
  Layers, 
  FlaskConical, 
  CheckCircle2, 
  Scale, 
  Sparkles, 
  ChevronDown, 
  ChevronUp, 
  BarChart2, 
  Info,
  Clock,
  Fingerprint
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
  const [counterfactualPolicy, setCounterfactualPolicy] = useState<string>('HYBRID');

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
  const upd = currentStepData.belief_update;
  const action = prereg?.action;

  // Selected candidate defaults to currently recommended candidate
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
      }
    }
  };

  // Hypotheses beliefs computation
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

  // Score components
  const higNats = prereg?.expected_hig_nats || 0;
  const discUtil = prereg?.discovery_utility || 0;
  const estCost = action?.estimated_cost || 1;
  const totalScore = prereg?.total_action_score || 0;

  // Counterfactual score recalculation
  const computeCounterfactualScore = (act: ScoredActionRecord, policy: string) => {
    const hig = act.expected_hig_nats;
    const disc = act.discovery_utility;
    const cost = act.normalized_cost;
    if (policy === 'PURE_HIG') return hig;
    if (policy === 'DISCOVERY_ONLY') return disc;
    if (policy === 'UNCERTAINTY_ONLY') return hig;
    return 0.8 * hig + 0.8 * disc - 2.0 * cost;
  };

  return (
    <div className="space-y-6 pb-16 animate-fadeIn">
      {/* Top Cockpit Control Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 bg-white p-4 rounded-xl border border-slate-200 shadow-xs">
        <div>
          <div className="flex items-center gap-2 mb-1">
            <h1 className="text-xl font-bold text-slate-900 tracking-tight">Autonomous Decision Cockpit</h1>
            <ModeBadge 
              mode={selectedCampaignType === 'controlled' ? 'CONTROLLED_SYNTHETIC' : 'HISTORICAL_REPLAY'} 
              size="sm"
            />
          </div>
          <p className="text-xs text-slate-500">
            Joint Candidate × Modality recommendation loop with pre-registered predictions & Bayesian belief tracking
          </p>
        </div>

        {/* Controls: Campaign switcher & Step Stepper */}
        <div className="flex flex-wrap items-center gap-3">
          {/* Campaign Selector */}
          <div className="flex bg-slate-100 p-1 rounded-lg border border-slate-200 text-xs">
            <button
              onClick={() => { setSelectedCampaignType('controlled'); handleStepSelect(1); }}
              className={`px-2.5 py-1 rounded-md font-medium transition cursor-pointer ${
                selectedCampaignType === 'controlled'
                  ? 'bg-white text-slate-900 shadow-xs font-semibold'
                  : 'text-slate-600 hover:text-slate-900'
              }`}
            >
              Controlled World (Clean H1)
            </button>
            <button
              onClick={() => { setSelectedCampaignType('alab_replay'); handleStepSelect(1); }}
              className={`px-2.5 py-1 rounded-md font-medium transition cursor-pointer ${
                selectedCampaignType === 'alab_replay'
                  ? 'bg-white text-slate-900 shadow-xs font-semibold'
                  : 'text-slate-600 hover:text-slate-900'
              }`}
            >
              A-Lab Sample Replay (PG)
            </button>
          </div>

          {/* Step Selector Buttons */}
          <div className="flex items-center bg-slate-100 p-1 rounded-lg border border-slate-200 text-xs">
            <span className="text-slate-400 font-mono text-xs px-2">STEP</span>
            {(campaign?.steps || []).map((s) => (
              <button
                key={s.step}
                onClick={() => handleStepSelect(s.step)}
                className={`w-7 h-7 rounded-md font-mono text-xs font-semibold transition cursor-pointer flex items-center justify-center ${
                  currentStepNum === s.step
                    ? 'bg-emerald-700 text-white shadow-xs'
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

      {/* Preregister -> Reveal -> Update Interactive Banner */}
      <div className="bg-slate-900 text-white rounded-xl p-4 shadow-md border border-slate-800">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pb-3 border-b border-slate-800">
          <div className="flex items-center gap-2">
            <Sparkles className="w-4 h-4 text-emerald-400" />
            <span className="text-xs font-bold uppercase tracking-wider text-emerald-400">
              Interactive Scientific Sequence
            </span>
            <span className="text-xs text-slate-400 font-mono">Step {currentStepNum} of {campaign?.steps?.length || 4}</span>
          </div>

          <div className="flex items-center gap-2">
            <button
              onClick={stepThroughReveal}
              className="px-3.5 py-1.5 rounded-md bg-emerald-600 hover:bg-emerald-500 text-xs font-semibold text-white shadow-xs transition flex items-center gap-2 cursor-pointer"
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
                  <span>Execute Belief Update (State D)</span>
                </>
              )}
              {revealState === 'STATE_D_UPDATED' && (
                <>
                  <span>Next Scientific Step</span>
                  <ArrowRight className="w-3.5 h-3.5" />
                </>
              )}
            </button>
          </div>
        </div>

        {/* 4 State Progress Indicator */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-2 pt-3 text-xs">
          <div 
            onClick={() => setRevealState('STATE_A_BEFORE')}
            className={`p-2.5 rounded-lg border cursor-pointer transition ${
              revealState === 'STATE_A_BEFORE' 
                ? 'bg-slate-800 border-emerald-500 text-white' 
                : 'bg-slate-900/50 border-slate-800 text-slate-400 hover:text-slate-300'
            }`}
          >
            <div className="font-mono text-2xs uppercase text-slate-500">State A</div>
            <div className="font-semibold text-xs mt-0.5">Candidate Scored</div>
            <div className="text-2xs text-slate-400 mt-1">Predictions active; data hidden</div>
          </div>

          <div 
            onClick={() => setRevealState('STATE_B_LOCKED')}
            className={`p-2.5 rounded-lg border cursor-pointer transition ${
              revealState === 'STATE_B_LOCKED' 
                ? 'bg-slate-800 border-emerald-500 text-white' 
                : 'bg-slate-900/50 border-slate-800 text-slate-400 hover:text-slate-300'
            }`}
          >
            <div className="font-mono text-2xs uppercase text-slate-500">State B</div>
            <div className="font-semibold text-xs mt-0.5 flex items-center gap-1">
              <Lock className="w-3 h-3 text-amber-400" />
              <span>Preregistration Locked</span>
            </div>
            <div className="text-2xs text-slate-400 mt-1">Immutable ledger event logged</div>
          </div>

          <div 
            onClick={() => setRevealState('STATE_C_REVEAL')}
            className={`p-2.5 rounded-lg border cursor-pointer transition ${
              revealState === 'STATE_C_REVEAL' 
                ? 'bg-slate-800 border-emerald-500 text-white' 
                : 'bg-slate-900/50 border-slate-800 text-slate-400 hover:text-slate-300'
            }`}
          >
            <div className="font-mono text-2xs uppercase text-slate-500">State C</div>
            <div className="font-semibold text-xs mt-0.5 flex items-center gap-1">
              <Unlock className="w-3 h-3 text-blue-400" />
              <span>Evidence Revealed</span>
            </div>
            <div className="text-2xs text-slate-400 mt-1">Canonical observables shown</div>
          </div>

          <div 
            onClick={() => setRevealState('STATE_D_UPDATED')}
            className={`p-2.5 rounded-lg border cursor-pointer transition ${
              revealState === 'STATE_D_UPDATED' 
                ? 'bg-slate-800 border-emerald-500 text-white' 
                : 'bg-slate-900/50 border-slate-800 text-slate-400 hover:text-slate-300'
            }`}
          >
            <div className="font-mono text-2xs uppercase text-slate-500">State D</div>
            <div className="font-semibold text-xs mt-0.5 flex items-center gap-1">
              <CheckCircle2 className="w-3 h-3 text-emerald-400" />
              <span>Posterior Updated</span>
            </div>
            <div className="text-2xs text-slate-400 mt-1">Bayes factors & entropy delta</div>
          </div>
        </div>
      </div>

      {/* Main Three-Zone Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-5 items-start">
        {/* Left Zone: Hypothesis Observatory (4 cols) */}
        <div className="lg:col-span-4 space-y-4">
          <div className="sci-card p-4">
            <div className="flex items-center justify-between mb-2">
              <div className="flex items-center gap-2">
                <Layers className="w-4 h-4 text-violet-700" />
                <h2 className="text-sm font-bold text-slate-900 uppercase tracking-wide">Hypothesis Observatory</h2>
              </div>
              <span className="text-2xs font-mono px-2 py-0.5 rounded bg-violet-50 text-violet-800 border border-violet-200">
                3 Competing Models
              </span>
            </div>
            <p className="text-xs text-slate-500 mb-4">
              Relative explanatory model weights among simplified competing mechanistic hypotheses.
            </p>

            {/* Hypothesis Cards */}
            <div className="space-y-3">
              {Object.entries(hypotheses).map(([hid, hdef]) => {
                const pBefore = beliefsBefore[hid] || 0.3333;
                const pAfter = beliefsAfter[hid] || pBefore;
                const delta = (revealState === 'STATE_D_UPDATED') ? (posteriorDelta[hid] || 0) : 0;
                const isWinner = pAfter > 0.5;

                return (
                  <div
                    key={hid}
                    className={`p-3.5 rounded-lg border transition ${
                      isWinner 
                        ? 'border-violet-300 bg-violet-50/30' 
                        : 'border-slate-200 bg-white'
                    }`}
                  >
                    <div className="flex items-start justify-between gap-2 mb-1.5">
                      <div>
                        <div className="text-xs font-bold text-slate-900">{hdef.title}</div>
                        <div className="text-2xs font-mono text-slate-400">{hid}</div>
                      </div>
                      <div className="text-right font-mono-num">
                        <div className="text-sm font-bold text-slate-900">
                          {(pAfter * 100).toFixed(1)}%
                        </div>
                        {revealState === 'STATE_D_UPDATED' && delta !== 0 && (
                          <div className={`text-2xs font-bold ${delta > 0 ? 'text-emerald-700' : 'text-slate-500'}`}>
                            {delta > 0 ? `+${(delta * 100).toFixed(1)}%` : `${(delta * 100).toFixed(1)}%`}
                          </div>
                        )}
                      </div>
                    </div>

                    {/* Belief Progress Bar */}
                    <div className="h-2 w-full bg-slate-100 rounded-full overflow-hidden my-2">
                      <div
                        className="belief-bar h-full bg-violet-600 rounded-full"
                        style={{ width: `${Math.max(1, pAfter * 100)}%` }}
                      />
                    </div>

                    {/* Assumptions & Falsification */}
                    <div className="text-2xs text-slate-600 space-y-1 mt-2 pt-2 border-t border-slate-100">
                      <div>
                        <span className="font-semibold text-slate-700">Physical premise: </span>
                        <span>{hdef.assumptions?.[0] || 'Domain constraint'}</span>
                      </div>
                      <div>
                        <span className="font-semibold text-emerald-700">Supports: </span>
                        <span>{hdef.falsification_signature?.strongly_supporting_patterns?.[0] || 'High phase yield'}</span>
                      </div>
                      <div>
                        <span className="font-semibold text-crimson-700">Falsified if: </span>
                        <span>{hdef.falsification_signature?.strongly_falsifying_patterns?.[0] || 'Impurity phase exceeds target'}</span>
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>

            {/* Scientific Note */}
            <div className="mt-4 p-3 rounded-lg bg-slate-50 border border-slate-200 text-2xs text-slate-500 flex items-start gap-2">
              <Info className="w-3.5 h-3.5 text-slate-400 shrink-0 mt-0.5" />
              <span>
                <strong>Epistemic Note:</strong> Belief distributions reflect normalized relative likelihood across simplified mechanistic representations, not physical certainty that exactly one simplified model is true.
              </span>
            </div>
          </div>
        </div>

        {/* Center Zone: Candidate x Measurement Landscape (4 cols) */}
        <div className="lg:col-span-4 space-y-4">
          <div className="sci-card p-4">
            <div className="flex items-center justify-between mb-2">
              <div className="flex items-center gap-2">
                <FlaskConical className="w-4 h-4 text-emerald-700" />
                <h2 className="text-sm font-bold text-slate-900 uppercase tracking-wide">Candidate Space</h2>
              </div>
              <span className="text-2xs font-mono text-slate-500">
                {selectedCampaignType === 'controlled' ? '12 Controlled Syn' : 'A-Lab Synthesis Library'}
              </span>
            </div>
            <p className="text-xs text-slate-500 mb-3">
              Candidate materials with multi-modal action feasibility and characterization history.
            </p>

            {/* Candidate Grid */}
            <div className="grid grid-cols-3 sm:grid-cols-4 gap-2 mb-4">
              {(data.flagship_campaign.candidates || []).map((cand) => {
                const isCurrentActionCand = cand.candidate_id === action?.candidate_id;
                const isSelected = cand.candidate_id === selectedCandidateId;

                return (
                  <button
                    key={cand.candidate_id}
                    onClick={() => setSelectedCandidateId(cand.candidate_id)}
                    className={`p-2 rounded-lg border text-left transition cursor-pointer flex flex-col justify-between ${
                      isCurrentActionCand
                        ? 'border-emerald-500 bg-emerald-50/50 ring-2 ring-emerald-400'
                        : isSelected
                        ? 'border-slate-400 bg-slate-100'
                        : 'border-slate-200 bg-white hover:border-slate-300'
                    }`}
                  >
                    <div className="flex items-center justify-between">
                      <span className="font-mono text-2xs font-bold text-slate-900">
                        {cand.candidate_id.replace('controlled-', 'Syn-')}
                      </span>
                      {isCurrentActionCand && (
                        <span className="w-2 h-2 rounded-full bg-emerald-600 animate-pulse" title="Next action target" />
                      )}
                    </div>
                    <div className="text-2xs text-slate-500 mt-1">
                      {cand.composition_label}
                    </div>
                  </button>
                );
              })}
            </div>

            {/* Selected Candidate Inspector */}
            {selectedCandidateId && (
              <div className="p-3.5 rounded-lg border border-slate-200 bg-slate-50 text-xs space-y-2">
                <div className="flex items-center justify-between border-b border-slate-200 pb-1.5">
                  <span className="font-bold text-slate-900">Selected: {selectedCandidateId}</span>
                  <span className="font-mono text-2xs px-2 py-0.5 rounded bg-emerald-100 text-emerald-800">
                    {selectedCandidateId === action?.candidate_id ? 'NEXT RECOMMENDED TARGET' : 'FEASIBLE'}
                  </span>
                </div>
                <div className="grid grid-cols-2 gap-2 text-2xs">
                  <div>
                    <span className="text-slate-500">XRD Scan Cost: </span>
                    <span className="font-mono font-semibold text-slate-800">1.0 Cost Unit</span>
                  </div>
                  <div>
                    <span className="text-slate-500">Refinement Cost: </span>
                    <span className="font-mono font-semibold text-slate-800">1.0 Cost Unit</span>
                  </div>
                  <div>
                    <span className="text-slate-500">Synthesis Test Cost: </span>
                    <span className="font-mono font-semibold text-slate-800">2.0 Cost Units</span>
                  </div>
                  <div>
                    <span className="text-slate-500">Prerequisite Status: </span>
                    <span className="font-semibold text-emerald-700">Satisfied</span>
                  </div>
                </div>
              </div>
            )}

            {/* Revealed Observation Card (if State C or D) */}
            {(revealState === 'STATE_C_REVEAL' || revealState === 'STATE_D_UPDATED') && obs && (
              <div className="mt-4 p-3.5 rounded-lg border border-blue-200 bg-blue-50/40 text-xs space-y-2">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-1.5 text-blue-900 font-bold">
                    <Unlock className="w-3.5 h-3.5 text-blue-700" />
                    <span>Revealed Measurement: {obs.action.action_type}</span>
                  </div>
                  <span className="font-mono text-2xs text-blue-800">
                    {obs.timestamp?.substring(11, 19)} UTC
                  </span>
                </div>

                <div className="bg-white p-2.5 rounded border border-blue-200 space-y-1 text-2xs font-mono">
                  {obs.observed_measurement?.observable_names ? (
                    obs.observed_measurement.observable_names.map((name, i) => {
                      const val = Array.isArray(obs.observed_measurement.value) 
                        ? obs.observed_measurement.value[i] 
                        : obs.observed_measurement.value;
                      return (
                        <div key={name} className="flex justify-between">
                          <span className="text-slate-600 truncate max-w-[200px]">{name}:</span>
                          <span className="font-bold text-blue-950">
                            {typeof val === 'number' ? val.toFixed(4) : String(val)}
                          </span>
                        </div>
                      );
                    })
                  ) : (
                    <div className="flex justify-between">
                      <span className="text-slate-600">Measurement Value:</span>
                      <span className="font-bold text-blue-950">{JSON.stringify(obs.observed_measurement?.value)}</span>
                    </div>
                  )}
                </div>

                {/* Likelihood & Realized Entropy */}
                <div className="pt-2 border-t border-blue-200/60 flex justify-between text-2xs text-blue-950">
                  <span>Realized Entropy Reduction:</span>
                  <span className="font-mono font-bold text-emerald-700">
                    {obs.realized_entropy_reduction_nats?.toFixed(4) || '0.0000'} nats
                  </span>
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Right Zone: Next Best Experiment Hero Card (4 cols) */}
        <div className="lg:col-span-4 space-y-4">
          <div className="sci-card p-4 border-2 border-emerald-600/60 bg-linear-to-b from-white to-emerald-50/20">
            <div className="flex items-center justify-between mb-2">
              <span className="text-2xs font-bold uppercase tracking-wider px-2 py-0.5 rounded bg-emerald-100 text-emerald-800 border border-emerald-200">
                Hero Recommendation
              </span>
              <span className="font-mono text-xs text-slate-500 font-semibold">
                Policy: {campaign?.policy || 'HYBRID'}
              </span>
            </div>

            {/* Action Identity */}
            <div className="mb-4">
              <div className="text-2xl font-extrabold text-slate-900 tracking-tight flex items-baseline gap-2">
                <span>{action?.action_type || 'XRD'}</span>
                <span className="text-sm font-semibold text-emerald-800">
                  on {action?.candidate_id || 'Syn-C'}
                </span>
              </div>
              <p className="text-xs text-slate-600 mt-1">
                Selected joint action: Acquires diagnostic characterization to separate H1 Phase Purity from H2 Homogeneity before physical outcome testing.
              </p>
            </div>

            {/* Score Composition Waterfall */}
            <div className="p-3.5 bg-white rounded-lg border border-slate-200 space-y-2 mb-4">
              <div className="text-2xs font-bold uppercase tracking-wider text-slate-500">
                Score Composition Waterfall
              </div>

              {/* HIG */}
              <div>
                <div className="flex justify-between text-xs mb-1">
                  <span className="text-slate-600">+ Information Gain (HIG):</span>
                  <span className="font-mono font-bold text-emerald-700">{higNats.toFixed(4)} nats</span>
                </div>
                <div className="h-1.5 w-full bg-slate-100 rounded-full overflow-hidden">
                  <div className="h-full bg-emerald-600" style={{ width: `${Math.min(100, (higNats / 0.7) * 100)}%` }} />
                </div>
              </div>

              {/* Discovery */}
              <div>
                <div className="flex justify-between text-xs mb-1">
                  <span className="text-slate-600">+ Discovery Utility:</span>
                  <span className="font-mono font-bold text-amber-700">{discUtil.toFixed(4)}</span>
                </div>
                <div className="h-1.5 w-full bg-slate-100 rounded-full overflow-hidden">
                  <div className="h-full bg-amber-500" style={{ width: `${Math.min(100, discUtil * 100)}%` }} />
                </div>
              </div>

              {/* Cost */}
              <div>
                <div className="flex justify-between text-xs mb-1">
                  <span className="text-slate-600">− Normalized Cost:</span>
                  <span className="font-mono font-bold text-crimson-700">{estCost.toFixed(1)}</span>
                </div>
                <div className="h-1.5 w-full bg-slate-100 rounded-full overflow-hidden">
                  <div className="h-full bg-crimson-500" style={{ width: `${Math.min(100, (estCost / 2.0) * 100)}%` }} />
                </div>
              </div>

              {/* Net Score */}
              <div className="pt-2 border-t border-slate-100 flex justify-between items-baseline font-mono">
                <span className="text-xs font-bold text-slate-800">= Total Action Score:</span>
                <span className="text-base font-extrabold text-slate-900">{totalScore.toFixed(4)}</span>
              </div>
            </div>

            {/* Falsification Criterion & Diagnostics */}
            <div className="p-3 rounded-lg bg-emerald-50/50 border border-emerald-200/80 text-2xs space-y-1.5 mb-4">
              <div className="flex items-center gap-1 font-semibold text-emerald-900">
                <CheckCircle2 className="w-3.5 h-3.5 text-emerald-700" />
                <span>Preregistered Scientific Criterion:</span>
              </div>
              <p className="text-emerald-950">
                Distinguishes whether structural purity correlates with reaction completion before initiating irreversible thermal processing.
              </p>
            </div>

            {/* Why Not Comparison Toggle */}
            <div className="border-t border-slate-200 pt-3">
              <button
                onClick={() => setShowWhyNot(!showWhyNot)}
                className="w-full flex items-center justify-between text-xs font-semibold text-slate-700 hover:text-slate-900 cursor-pointer"
              >
                <span>Why not another experiment? (Alternative Actions)</span>
                {showWhyNot ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
              </button>

              {showWhyNot && (
                <div className="mt-3 space-y-2 pt-2 border-t border-slate-100">
                  <div className="flex items-center justify-between text-2xs text-slate-500 pb-1">
                    <span>Counterfactual Policy:</span>
                    <select
                      value={counterfactualPolicy}
                      onChange={(e) => setCounterfactualPolicy(e.target.value)}
                      className="bg-slate-50 border border-slate-200 rounded px-1.5 py-0.5 text-2xs"
                    >
                      <option value="HYBRID">HYBRID (Default)</option>
                      <option value="PURE_HIG">PURE_HIG (Information Only)</option>
                      <option value="DISCOVERY_ONLY">DISCOVERY_ONLY (Property)</option>
                    </select>
                  </div>

                  <div className="overflow-x-auto">
                    <table className="sci-table text-2xs font-mono">
                      <thead>
                        <tr>
                          <th>Action</th>
                          <th>HIG</th>
                          <th>Cost</th>
                          <th>Score</th>
                          <th>Gap</th>
                        </tr>
                      </thead>
                      <tbody>
                        {(currentStepData.top_actions || []).slice(0, 5).map((act, idx) => {
                          const cfScore = computeCounterfactualScore(act, counterfactualPolicy);
                          const isTop = idx === 0;
                          return (
                            <tr key={act.action?.action_id || idx} className={isTop ? 'bg-emerald-50/40 font-bold' : ''}>
                              <td>{act.action?.action_type}_{act.action?.candidate_id?.replace('controlled-', '')}</td>
                              <td>{act.expected_hig_nats?.toFixed(3)}</td>
                              <td>{act.action?.estimated_cost}</td>
                              <td>{cfScore.toFixed(3)}</td>
                              <td className={isTop ? 'text-emerald-700' : 'text-slate-400'}>
                                {isTop ? 'Top' : `-${(totalScore - cfScore).toFixed(3)}`}
                              </td>
                            </tr>
                          );
                        })}
                      </tbody>
                    </table>
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
