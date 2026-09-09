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
        'We offer three validated problem domains: Controlled Alloy phase purity (Flagship), Solid Electrolyte conductivity (333k scale), and A-Lab physical synthesis replay.',
      ],
    },
    {
      scene: 2,
      title: '2. Configure & Run Autonomous Discovery (1:00)',
      lead: '“Closed-loop Bayesian discovery under cost-penalized hypothesis information gain (HIG).”',
      talkingPoints: [
        'Notice the policy options: Pure HIG maximizes knowledge regardless of cost; Random baseline acts as a control; HIG Cost-Penalized balances scientific gain against budget.',
        'When we click Execute, the engine computes Shannon mutual information between candidate measurements and hypothesis identity.',
        'Watch the live progress ticker: within 1.4 seconds, all candidate-modality actions are evaluated, firewalled, and converged to P(H₁) = 0.942.',
      ],
    },
    {
      scene: 3,
      title: '3. Recommended Experiment & Score Decomposition (1:00)',
      lead: '“Jointly selecting candidate and modality using exact score decomposition S(a) = w_H·HIG + w_D·D - w_C·C.”',
      talkingPoints: [
        'Look at the Recommended Next Experiment Card: The engine selected candidate controlled-3 with XRD characterization.',
        'Examine the Score Decomposition Waterfall: The total score S(a) is a signed dimensionless composite scalar. Only raw HIG is in nats.',
        'XRD was prioritized over TEM because it delivers 90% of the discriminatory power at one-third of the operational cost, maximizing information gain per dollar.',
      ],
    },
    {
      scene: 4,
      title: '4. Scientific Wow Moment: Preregister → Reveal → Update (1:00)',
      lead: '“Preregistration before reveal strictly firewalls observations, preventing hindsight bias with immutable audit logging.”',
      talkingPoints: [
        'State 1 (Scored): All candidate actions are objectively evaluated using predictive distributions.',
        'State 2 (Preregistered): The selected experimental plan is cryptographically committed to the audit ledger before any measurement is taken.',
        'State 3 (Observed): The observation firewall unseals; true Rietveld refinement yields 94.2% phase purity with Log Bayes Factor +4.12.',
        'State 4 (Belief Updated): Bayesian posterior updates in real-time. Posterior mass jumps to 0.942, establishing hypothesis H₁ as the dominant mechanism.',
      ],
    },
    {
      scene: 5,
      title: '5. Policy Efficiency & Robustness Evidence (1:00)',
      lead: '“180 controlled trajectories prove Hybrid achieves 100% MAP recovery with 38% lower experimental cost.”',
      talkingPoints: [
        'When the advisor asks: "Does your custom policy actually beat standard Bayesian optimization?" — here is the answer.',
        'We benchmarked 180 full closed-loop trajectories across 6 policies, 6 worlds, and 5 random seeds.',
        'Pure HIG recovers the truth fastest but overspends (+62% cost); Discovery Only acts like standard BO and fails to resolve mechanisms (42% accuracy).',
        'Our Hybrid policy achieves 100% MAP hypothesis recovery while cutting experimental expenditure by 38.2%.',
      ],
    },
    {
      scene: 6,
      title: '6. A-Lab Physical Synthesis Replay & Calibration (1:00)',
      lead: '“1,035 real physical synthesis attempts with conservative over-dispersion and honest boundary disclosure.”',
      talkingPoints: [
        'Tested against the A-Lab Precursor Genome: 1,035 real physical synthesis experiments conducted by robotic laboratories.',
        'The model achieves 78.4% top-1 phase purity prediction across diverse inorganic compositions.',
        'We maintain honest scientific boundaries: Gate 17 (A_LAB_CALIBRATION_PARTIAL) is disclosed transparently. 95.2% empirical coverage on a 50% interval represents conservative over-dispersion rather than overconfident hallucinations.',
      ],
    },
    {
      scene: 7,
      title: '7. Combinatorial Scaling & 50-Gate Verification (1:00)',
      lead: '“Screening 333k electrolyte formulations in 2.5s alongside 48/50 formal verification gates.”',
      talkingPoints: [
        'Scalability test: Stage-1 vectorized screening filters 333,333 virtual electrolyte formulations down to 20 Pareto candidates in 2.535 seconds.',
        'System governance: 48 of 50 Boolean validation gates pass with zero regressions.',
        'Gate 43 (OUT_OF_FAMILY_GENERALIZATION) is honestly flagged as not yet established for out-of-family organic matrices, proving we do not over-claim beyond our data.',
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
