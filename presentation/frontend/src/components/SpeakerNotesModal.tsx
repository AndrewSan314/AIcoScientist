import React from 'react';
import { X, BookOpen, Clock, Target, CheckCircle2 } from 'lucide-react';

interface SpeakerNotesModalProps {
  isOpen: boolean;
  onClose: () => void;
  activeSceneIndex?: number;
}

export const SpeakerNotesModal: React.FC<SpeakerNotesModalProps> = ({
  isOpen,
  onClose,
  activeSceneIndex = 0,
}) => {
  if (!isOpen) return null;

  const scriptScenes = [
    {
      scene: 1,
      title: '1. Research Question & Dataset Selection (1:00)',
      lead: '“Traditional materials optimization asks which material performs best. We ask which experiment should be performed next, and why.”',
      talkingPoints: [
        'Materials discovery is not just curve fitting or surrogate property maximization; it is an active information acquisition loop under costly, multi-modal characterization.',
        'Existing Bayesian optimization algorithms blindly optimize one scalar property. In real labs, diagnostic characterization (XRD, spectroscopy, electron microscopy) resolves mechanisms.',
        'We expose three distinct evidence modes: controlled synthetic inference, A-Lab historical replay, and electrolyte surrogate optimization.',
      ],
    },
    {
      scene: 2,
      title: '2. Configure & Run Autonomous Discovery (1:00)',
      lead: '“Closed-loop Bayesian discovery under cost-penalized hypothesis information gain (HIG).”',
      talkingPoints: [
        'Notice the policy options: each control resolves to a recorded source policy when that policy/run exists.',
        'The console keeps source action scores, preregistration, observations, and posterior updates separate.',
        'The progress ticker is presentation playback; it does not execute a new experiment.',
      ],
    },
    {
      scene: 3,
      title: '3. Recorded Action & Score Decomposition (1:00)',
      lead: '“The console exposes source score fields without reconstructing absent weighting components.”',
      talkingPoints: [
        'Look at the Recorded Action Card: the selected candidate and modality come from the recorded run.',
        'The Score Decomposition Waterfall stays unavailable when the source did not persist weights and normalized components.',
        'Use the source manifest and action payload for provenance; do not infer a cost or mechanism claim from a missing field.',
      ],
    },
    {
      scene: 4,
      title: '4. Scientific Wow Moment: Preregister → Reveal → Update (1:00)',
      lead: '“Preregistration before reveal strictly firewalls observations, preventing hindsight bias with immutable audit logging.”',
      talkingPoints: [
        'State 1 (Scored): All candidate actions are objectively evaluated using predictive distributions.',
        'State 2 (Preregistered): The source-recorded plan is shown before the observation reveal.',
        'State 3 (Observed): the source-linked observation is revealed for the preregistered action.',
        'State 4 (Belief Updated): the recorded posterior is displayed without calling it physical confirmation.',
      ],
    },
    {
      scene: 5,
      title: '5. Policy Efficiency & Robustness Evidence (1:00)',
      lead: '“The benchmark artifact records policy trajectories, sensitivity, and calibration metrics with their limits.”',
      talkingPoints: [
        'When the advisor asks: "Does your custom policy actually beat standard Bayesian optimization?" — here is the answer.',
        'The benchmark matrix contains source trajectory records across its documented policies, worlds, and seeds.',
        'The workspace shows the recorded policy summaries and avoids turning them into universal causal claims.',
      ],
    },
    {
      scene: 6,
      title: '6. A-Lab Physical Synthesis Replay & Calibration (1:00)',
      lead: '“A-Lab replay uses source-linked samples and discloses missing modality linkage.”',
      talkingPoints: [
        'Tested against the A-Lab Precursor Genome: 1,035 real physical synthesis experiments conducted by robotic laboratories.',
        'Calibration metrics are shown from the source artifact, not treated as a universal physical accuracy claim.',
        'SEM and EDS are disclosed as unavailable to candidate replay when sample-level linkage is absent.',
      ],
    },
    {
      scene: 7,
      title: '7. Combinatorial Scaling & 50-Gate Verification (1:00)',
      lead: '“Virtual-pool screening and surrogate trajectories are shown with stage-specific timing and model limits.”',
      talkingPoints: [
        'The source diagnostic records the virtual candidate pool, working set, timing stage, and latent gap.',
        'The manifest reports the validation gate count and the workspace keeps partial readiness visible.',
        'The surrogate artifact explicitly limits interpretation to its documented in-silico model scope.',
      ],
    },
  ];

  const currentScript = scriptScenes[activeSceneIndex] || scriptScenes[0];

  return (
    <div className="fixed inset-0 z-50 bg-[#17201F]/60 backdrop-blur-xs flex items-center justify-center p-4">
      <div className="bg-[#FCFCFA] rounded-2xl max-w-2xl w-full border border-[#D9DFDB] shadow-2xl flex flex-col max-h-[85vh] overflow-hidden">
        {/* Modal Header */}
        <div className="px-6 py-4 border-b border-[#D9DFDB] flex items-center justify-between bg-[#F4F3EE]">
          <div className="flex items-center gap-2">
            <BookOpen className="w-5 h-5 text-[#DC2626]" />
            <h2 className="text-base font-bold text-[#17201F]">Advisor Presentation Speaker Notes</h2>
          </div>
          <button
            onClick={onClose}
            className="p-1 text-[#66706C] hover:text-[#17201F] rounded transition cursor-pointer"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Scene Navigation Tabs */}
        <div className="px-6 py-2.5 bg-white border-b border-[#D9DFDB] flex gap-1.5 overflow-x-auto">
          {scriptScenes.map((s, idx) => (
            <div
              key={s.scene}
              className={`px-3 py-1 text-2xs font-mono rounded-lg transition ${
                idx === activeSceneIndex
                  ? 'bg-[#FEF2F2] text-[#991B1B] font-bold border border-[#FECACA]'
                  : 'text-[#66706C]'
              }`}
            >
              Scene {s.scene}
            </div>
          ))}
        </div>

        {/* Modal Body */}
        <div className="p-6 overflow-y-auto space-y-4 flex-1">
          <div>
            <div className="flex items-center gap-2 text-2xs font-mono text-[#DC2626] uppercase font-bold tracking-wider">
              <Clock className="w-3.5 h-3.5" />
              <span>Target Duration: 1:00 min</span>
            </div>
            <h3 className="text-lg font-bold text-[#17201F] mt-1">{currentScript.title}</h3>
          </div>

          <div className="p-4 rounded-xl bg-[#FEF2F2] border border-[#FECACA] text-xs text-[#991B1B] italic leading-relaxed">
            {currentScript.lead}
          </div>

          <div className="space-y-2">
            <h4 className="text-xs font-bold text-[#17201F] uppercase font-mono tracking-wider">
              Key Talking Points
            </h4>
            <ul className="space-y-2.5">
              {currentScript.talkingPoints.map((point, i) => (
                <li key={i} className="flex items-start gap-2.5 text-xs text-[#66706C] leading-relaxed">
                  <span className="w-1.5 h-1.5 rounded-full bg-[#B91C1C] mt-1.5 shrink-0" />
                  <span>{point}</span>
                </li>
              ))}
            </ul>
          </div>
        </div>

        {/* Modal Footer */}
        <div className="px-6 py-3 border-t border-[#D9DFDB] bg-[#F4F3EE] flex items-center justify-between text-2xs text-[#66706C]">
          <span>Total presentation duration: ~7 minutes</span>
          <button
            onClick={onClose}
            className="px-4 py-1.5 bg-[#B91C1C] hover:bg-[#991B1B] text-white rounded-lg font-semibold transition cursor-pointer"
          >
            Close Notes
          </button>
        </div>
      </div>
    </div>
  );
};
