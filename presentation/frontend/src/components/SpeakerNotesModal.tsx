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
      title: 'The Research Question (1:00)',
      lead: '“Traditional materials optimization asks which material performs best. We ask which experiment should be performed next, and why.”',
      talkingPoints: [
        'Materials discovery is not just curve fitting or surrogate optimization; it is an active information acquisition loop under costly, multi-modal characterization.',
        'Existing Bayesian optimization routines blindly pull the property lever. Real laboratories have diagnostic characterization tools (XRD, Rietveld refinement, microscopy) that can confirm or refute mechanisms.',
        'AIcoScientist treats hypothesis testing and characterization selection as first-class decision actions alongside property measurements.',
      ],
    },
    {
      scene: 2,
      title: 'Universal Scientific Decision Loop (1:00)',
      lead: '“One reusable mathematical abstraction across solid-state synthesis, battery electrolytes, and electrocatalysis.”',
      talkingPoints: [
        'Notice the central engine architecture: MaterialDomainAdapter cleanly decouples candidate chemistry schemas from the inference engine.',
        'At each decision step, candidate materials and eligible characterization modalities are jointly enumerated.',
        'Expected Hypothesis Information Gain (HIG) is calculated in nats via Monte Carlo sampling before allocating experimental budget.',
      ],
    },
    {
      scene: 3,
      title: 'Decision Cockpit & Competing Hypotheses (1:30)',
      lead: '“Maintaining competing mechanistic hypotheses rather than a single black-box regression.”',
      talkingPoints: [
        'Look at the Left Zone: Three formal hypotheses (Phase Purity, Composition Homogeneity, Morphology Kinetics) with prior belief distribution.',
        'Note the explicit disclaimer: These represent relative explanatory model weights among simplified competing models, not ontological truth.',
        'Look at the Center Zone: Candidate space with real cost and prerequisite constraints. Refinement requires prior XRD; outcome tests consume physical sample.',
        'Look at the Right Zone: The hero card chooses both candidate AND modality. The waterfall chart breaks down Net Score = w_hig · HIG + w_disc · Discovery - w_cost · Cost.',
      ],
    },
    {
      scene: 4,
      title: 'The Preregister → Reveal → Update Loop (1:30)',
      lead: '“Preregistration before reveal prevents scientific hindsight bias and guarantees auditability.”',
      talkingPoints: [
        'State A: Action is selected, predictive distributions are simulated, but ground truth is strictly firewalled.',
        'State B: Preregistration record is locked into the immutable evidence ledger with timestamp and event sequence.',
        'State C: Evidence is revealed (real canonical descriptors or refinement phase fractions).',
        'State D: Bayesian update in log space. Watch the posterior shift toward H1 as target phase fraction exceeds 0.94. Log Bayes factor confirms evidence diagnostic power.',
      ],
    },
    {
      scene: 5,
      title: 'Evaluation Breadth: 180 Controlled Trajectories (1:00)',
      lead: '“Systematic benchmark across clean and stress worlds proves policy trade-offs.”',
      talkingPoints: [
        'We evaluated 180 full closed-loop trajectories across 6 policies, 6 worlds, and 5 seeds.',
        'Pure HIG achieves near-instant MAP hypothesis recovery (mean 1.2 steps) but ignores discovery value.',
        'Discovery-Only finds high-utility materials but achieves weaker hypothesis separation.',
        'HYBRID balances information gain against discovery and experimental cost, achieving 100% recovery with bounded expenditure.',
        'HIG sensitivity analysis shows MC32 provides rank correlation above 0.85, resolving Monte Carlo noise.',
      ],
    },
    {
      scene: 6,
      title: 'A-Lab Evidence Atlas & Scientific Rigor (1:00)',
      lead: '“Evaluating on real inorganic synthesis data while honestly reporting boundaries.”',
      talkingPoints: [
        'We mapped 1,035 real solid-state synthesis samples from the A-Lab Precursor Genome (CC BY 4.0).',
        'Canonical XRD and Rietveld refinement are linked for 1,030 samples. Notice our honesty: SEM and EDS archives exist at precursor level but lack candidate sample ID linkage, so they are explicitly labeled NOT AVAILABLE.',
        'Calibration evaluation reveals REFINEMENT coverage is partial (over-dispersed at 50% interval). We report this limitation as scientific rigor.',
        'Chemistry-family generalization is evaluated across elemental holdouts and explicitly flagged as NOT ESTABLISHED.',
      ],
    },
    {
      scene: 7,
      title: 'Electrolyte Discovery Scale (0:45)',
      lead: '“Screening a 333,333-candidate virtual space with bounded closed-loop execution.”',
      talkingPoints: [
        'Demonstrates our earlier battery electrolyte optimization work at massive scale.',
        'Stage-1 multi-objective screening downsizes 333,333 virtual formulations into a 200-candidate working set in 2.5 seconds with zero latent optimum loss.',
        'Honest negative result: In frozen ExtraTrees surrogate simulation, BoTorch EI achieves lower pure-property latent regret (0.0257 vs 0.0788), while Hybrid delivers superior cumulative information gain (1.58 nats vs 0.56 nats).',
      ],
    },
    {
      scene: 8,
      title: 'Scientific Contributions & Readiness (0:45)',
      lead: '“48 out of 50 boolean validation gates passed with complete provenance.”',
      talkingPoints: [
        'Every single metric shown tonight is backed by the 5,333 audit events recorded in the ledger.',
        'The two failing gates represent authentic research boundaries: conservative refinement calibration and out-of-family generalization.',
        'Summary: We have built a production-grade, mathematically grounded scientific decision framework ready for prospective physical laboratory trials.',
      ],
    },
  ];

  return (
    <div className="fixed inset-0 z-50 overflow-y-auto bg-slate-900/60 backdrop-blur-xs flex justify-end animate-fadeIn">
      <div className="w-full max-w-2xl bg-white h-full shadow-2xl flex flex-col border-l border-slate-200">
        {/* Header */}
        <div className="p-5 border-b border-slate-200 flex items-center justify-between bg-slate-50">
          <div className="flex items-center gap-2">
            <BookOpen className="w-5 h-5 text-emerald-700" />
            <div>
              <h2 className="text-base font-bold text-slate-900">Advisor Presentation Script & Talk Track</h2>
              <p className="text-xs text-slate-500">6–8 Minute Structured Demonstration Guide</p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="p-1.5 rounded-lg text-slate-400 hover:text-slate-700 hover:bg-slate-200 transition"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Content */}
        <div className="p-6 overflow-y-auto flex-1 space-y-6">
          <div className="bg-emerald-50 border border-emerald-200 rounded-lg p-3 text-xs text-emerald-900 flex items-start gap-2">
            <Clock className="w-4 h-4 text-emerald-700 mt-0.5 shrink-0" />
            <div>
              <strong>Presentation Tip:</strong> Deliver with calm scientific precision. Emphasize that every number is traceable to audited artifacts, and explain that our limitations (e.g. failing generalization gate) demonstrate rigorous scientific integrity.
            </div>
          </div>

          <div className="space-y-4">
            {scriptScenes.map((item, idx) => (
              <div
                key={item.scene}
                className={`p-4 rounded-lg border transition ${
                  activeSceneIndex === idx
                    ? 'border-emerald-500 bg-emerald-50/20 shadow-xs ring-1 ring-emerald-400'
                    : 'border-slate-200 bg-white hover:border-slate-300'
                }`}
              >
                <div className="flex items-center justify-between mb-1.5">
                  <span className="text-xs font-bold uppercase tracking-wider text-emerald-700">
                    Scene {item.scene}
                  </span>
                  <span className="text-xs font-mono text-slate-400">{item.title.split('(')[1]?.replace(')', '') || ''}</span>
                </div>
                <h3 className="text-sm font-bold text-slate-900 mb-2">{item.title.split('(')[0]}</h3>
                <blockquote className="border-l-2 border-emerald-600 pl-3 py-1 my-2 bg-slate-50 text-xs italic text-slate-700 font-medium">
                  {item.lead}
                </blockquote>
                <ul className="mt-2 space-y-1.5 text-xs text-slate-600">
                  {item.talkingPoints.map((tp, i) => (
                    <li key={i} className="flex items-start gap-2">
                      <CheckCircle2 className="w-3.5 h-3.5 text-slate-400 mt-0.5 shrink-0" />
                      <span>{tp}</span>
                    </li>
                  ))}
                </ul>
              </div>
            ))}
          </div>
        </div>

        {/* Footer */}
        <div className="p-4 border-t border-slate-200 bg-slate-50 flex justify-between items-center text-xs text-slate-500 font-mono">
          <span>AIcoScientist Advisor Mission Control</span>
          <button
            onClick={onClose}
            className="px-4 py-2 bg-slate-900 text-white rounded-md hover:bg-slate-800 transition font-sans font-medium"
          >
            Close Notes
          </button>
        </div>
      </div>
    </div>
  );
};
