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
      title: '1. The Research Question & Contribution (1:00)',
      lead: '“Traditional materials optimization asks which material performs best. We ask which experiment should be performed next, and why.”',
      talkingPoints: [
        'Materials discovery is not just curve fitting or surrogate optimization; it is an active information acquisition loop under costly, multi-modal characterization.',
        'Existing Bayesian optimization routines blindly pull the property lever. Real laboratories have diagnostic characterization tools (XRD, Rietveld refinement, spectroscopy) that can confirm or refute mechanisms.',
        'AIcoScientist treats hypothesis testing and characterization selection as first-class decision actions alongside property measurements.',
      ],
    },
    {
      scene: 2,
      title: '2. Discovery Lab: Competing Hypotheses & Predictions (1:00)',
      lead: '“Maintaining three formal competing hypotheses with preregistered Gaussian predictive distributions.”',
      talkingPoints: [
        'Look at the Left Panel: The Hypothesis Belief Trajectory plots posterior probability shifts over sequential steps.',
        'Toggle to Predictive Distributions: We see Gaussian probability densities for each hypothesis over observable descriptors before any data is revealed.',
        'Notice our explicit disclaimer: Belief weights represent relative explanatory model likelihoods among simplified competing models, not a physical claim that one theory is absolute truth.',
      ],
    },
    {
      scene: 3,
      title: '3. Next Experiment: Candidate × Modality Trade-off (1:30)',
      lead: '“Jointly selecting candidate and modality using exact score decomposition S(a) = w_H·HIG + w_D·D - w_C·C.”',
      talkingPoints: [
        'Look at the Candidate × Modality Matrix: Rows represent feasible modalities (XRD, Refinement) and columns represent candidate materials.',
        'Look at the Waterfall Decomposition: The total score is a signed dimensionless composite scalar. Only raw HIG is measured in nats.',
        'The engine chose XRD on controlled-3 because it yields 0.506 nats of information gain at half the cost of an outcome test, maximizing scientific insight per dollar spent.',
      ],
    },
    {
      scene: 4,
      title: '4. The Scientific Wow Moment (Preregister → Reveal → Update) (1:30)',
      lead: '“Preregistration before reveal strictly firewalls observations, preventing hindsight bias with immutable audit logging.”',
      talkingPoints: [
        'State A: Action is selected, predictive distributions are simulated, but ground truth is strictly firewalled.',
        'State B: Preregistration record is locked into the immutable evidence ledger with timestamp and event sequence.',
        'State C: Evidence is revealed (real canonical descriptors or refinement phase fractions).',
        'State D: Bayesian update executed. Watch belief bars shift toward H1 as target phase fraction exceeds 0.94. Log Bayes factor confirms evidence diagnostic power.',
      ],
    },
    {
      scene: 5,
      title: '5. Policy Benchmarks: Clean vs Stress Worlds (1:00)',
      lead: '“180 controlled trajectories prove HYBRID achieves 100% MAP hypothesis recovery with bounded experimental expenditure.”',
      talkingPoints: [
        'A common advisor question: "Does this actually outperform standard policies?"',
        'We evaluated 180 full closed-loop trajectories across 6 policies, 6 worlds, and 5 random seeds.',
        'Pure HIG recovers the true hypothesis fastest (mean 1.2 steps) but ignores utility; Discovery Only acts like standard BO and fails to distinguish mechanisms.',
        'HYBRID achieves a proven compromise: 100% MAP hypothesis recovery with bounded experimental expenditure.',
      ],
    },
    {
      scene: 6,
      title: '6. Real A-Lab Replay, Electrolyte Scaling & Governance (1:00)',
      lead: '“1,035 real physical samples and 333k electrolyte screening with honest disclosure of empirical boundaries.”',
      talkingPoints: [
        'Validated on the A-Lab Precursor Genome: 1,035 real physical synthesis records with search and landmark buttons (PG_0309: Co3B3H9O13).',
        'Calibration coverage is marked A_LAB_CALIBRATION_PARTIAL because the 50% interval covers 95.2% of points (conservative over-dispersion rather than overconfidence).',
        'At scale: Stage-1 screening filtered 333,333 virtual electrolyte candidates down to 200 in 2.535 seconds with zero latent loss.',
        'Our readiness screen reports 48 out of 50 Boolean validation gates passed with complete cryptographic manifest verification.',
      ],
    },
  ];

  const currentScript = scriptScenes[activeSceneIndex] || scriptScenes[0];

  return (
    <div className="fixed inset-0 z-50 bg-slate-900/60 backdrop-blur-xs flex items-center justify-center p-4">
      <div className="bg-white rounded-2xl max-w-2xl w-full border border-slate-200 shadow-2xl flex flex-col max-h-[85vh] overflow-hidden">
        {/* Modal Header */}
        <div className="px-6 py-4 border-b border-slate-100 flex items-center justify-between bg-slate-50">
          <div className="flex items-center gap-2">
            <BookOpen className="w-5 h-5 text-emerald-700" />
            <h2 className="text-base font-bold text-slate-900">Advisor Presentation Speaker Notes</h2>
          </div>
          <button
            onClick={onClose}
            className="p-1 text-slate-400 hover:text-slate-600 rounded transition cursor-pointer"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Modal Content */}
        <div className="p-6 overflow-y-auto space-y-6">
          {/* Active Scene Card */}
          <div className="p-4 rounded-xl bg-emerald-50/80 border border-emerald-200">
            <div className="flex items-center justify-between mb-2">
              <span className="text-xs font-mono font-bold uppercase tracking-wider text-emerald-800">
                Active Scene Script
              </span>
              <div className="flex items-center gap-1 text-xs text-emerald-700 font-medium">
                <Clock className="w-3.5 h-3.5" />
                <span>6–8 Min Total Talk Track</span>
              </div>
            </div>
            <h3 className="text-base font-extrabold text-emerald-950 mb-1">
              {currentScript.title}
            </h3>
            <p className="text-xs italic text-emerald-900 font-serif leading-relaxed">
              {currentScript.lead}
            </p>
          </div>

          {/* Key Talking Points */}
          <div>
            <h4 className="text-xs font-mono font-bold uppercase tracking-wider text-slate-500 mb-3 flex items-center gap-1.5">
              <Target className="w-4 h-4 text-emerald-700" />
              <span>Key Defense Talking Points</span>
            </h4>
            <div className="space-y-2.5">
              {currentScript.talkingPoints.map((point, idx) => (
                <div
                  key={idx}
                  className="flex items-start gap-2.5 p-3 rounded-lg bg-slate-50 border border-slate-100 text-xs text-slate-800"
                >
                  <CheckCircle2 className="w-4 h-4 text-emerald-600 shrink-0 mt-0.5" />
                  <span className="leading-relaxed">{point}</span>
                </div>
              ))}
            </div>
          </div>

          {/* All Scenes Quick Reference */}
          <div className="pt-4 border-t border-slate-100">
            <h4 className="text-xs font-mono font-bold uppercase tracking-wider text-slate-500 mb-2">
              All Scenes Quick Reference
            </h4>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 text-xs">
              {scriptScenes.map((s, idx) => (
                <div
                  key={s.scene}
                  className={`p-2 rounded border font-mono ${
                    activeSceneIndex === idx
                      ? 'bg-emerald-50 border-emerald-300 font-bold text-emerald-900'
                      : 'bg-white border-slate-200 text-slate-600'
                  }`}
                >
                  Scene {s.scene}: {s.title.split('(')[0]}
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Modal Footer */}
        <div className="px-6 py-3 border-t border-slate-100 bg-slate-50 flex items-center justify-between text-xs text-slate-500">
          <span>Press <strong>N</strong> to toggle notes at any time during presentation</span>
          <button
            onClick={onClose}
            className="px-4 py-1.5 bg-slate-900 text-white rounded-md text-xs font-semibold hover:bg-slate-800 cursor-pointer"
          >
            Close Notes
          </button>
        </div>
      </div>
    </div>
  );
};
