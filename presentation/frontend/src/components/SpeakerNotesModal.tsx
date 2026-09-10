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
      title: '1. What problem does AIcoScientist solve? (0:30)',
      lead: '“AIcoScientist does not only rank materials. It decides which candidate × measurement experiment is most valuable to inspect next.”',
      talkingPoints: [
        'Keep the distinction explicit: Next Experiment is not Best Material Found.',
        'The project has three evidence regimes: controlled inference, A-Lab historical replay, and electrolyte surrogate optimization.',
      ],
    },
    {
      scene: 2,
      title: '2. What should we test next? (0:40)',
      lead: '“This card is a recorded candidate × measurement action under the selected source policy.”',
      talkingPoints: [
        'Read candidate, measurement, recorded score, information gain, and cost first.',
        'It means highest-priority recorded experiment—not that the candidate is the best material.',
      ],
    },
    {
      scene: 3,
      title: '3. Why this experiment? (0:45)',
      lead: '“Each decision-matrix cell is one possible experiment; the highlighted cell is the recorded selected action.”',
      talkingPoints: [
        'The policy balances scientific information, discovery value, and recorded measurement cost.',
        'Information gain means expected reduction in uncertainty about which model best explains the system.',
      ],
    },
    {
      scene: 4,
      title: '4. Lock prediction → reveal evidence → update model support (0:50)',
      lead: '“The prediction is recorded before the observation is revealed, so the update can be inspected without hindsight.”',
      talkingPoints: [
        'Different models predict different outcomes; that disagreement makes a measurement informative.',
        'Posterior model weight is model support from the evidence, not proof of a physical mechanism.',
      ],
    },
    {
      scene: 5,
      title: '5. Best candidate found in an optimization task (0:40)',
      lead: '“Optimization has a separate Best Found result. It is not the same thing as the next-experiment recommendation.”',
      talkingPoints: [
        'The electrolyte result is a frozen in-silico surrogate trajectory, not a new physical battery measurement.',
        'The card exposes candidate ID, query index, best selected latent value, and regret when recorded.',
      ],
    },
    {
      scene: 6,
      title: '6. Controlled benchmark evidence (0:40)',
      lead: '“Controlled worlds answer whether the decision strategy and belief updates work when benchmark ground truth is known.”',
      talkingPoints: [
        'Policy comparison, calibration, and sensitivity are supporting evidence after the core loop is understood.',
        'Controlled success does not independently establish prospective physical-world validity.',
      ],
    },
    {
      scene: 7,
      title: '7. A-Lab historical physical-data replay (0:40)',
      lead: '“A-Lab replays real historical synthesis and characterization records; it is not live autonomous laboratory execution.”',
      talkingPoints: [
        'Source sample metadata stays prominent and unlinked modalities remain unavailable to the replay.',
        'Calibration is a source artifact and its scope remains explicitly bounded.',
      ],
    },
    {
      scene: 8,
      title: '8. Large-space surrogate optimization (0:40)',
      lead: '“333,333 virtual formulations are screened to a 200-candidate working set before sequential recorded surrogate queries.”',
      talkingPoints: [
        'Use this evidence to discuss scalability, not physical validation or causal mechanism discovery.',
        'Attia and FeCoNi remain additional simulator/dataset benchmark evidence outside this primary scientific loop.',
      ],
    },
    {
      scene: 9,
      title: '9. Evidence boundaries and next research step (0:30)',
      lead: '“The current evidence is controlled validation, historical replay, and simulation—not a prospectively validated autonomous physical laboratory.”',
      talkingPoints: [
        'End by naming the evidence mode for every claim.',
        'The next research step is prospective physical validation with pre-specified success criteria.',
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
          <span>Total presentation duration: ~6 minutes</span>
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
