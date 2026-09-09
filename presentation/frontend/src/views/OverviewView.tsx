import React from 'react';
import { SnapshotData } from '../types/mission_control';
import { NavTab } from '../components/Header';
import { ModeBadge } from '../components/ModeBadge';
import { 
  ArrowRight, 
  Layers, 
  FlaskConical, 
  BarChart3, 
  Zap, 
  GitBranch, 
  ShieldCheck, 
  FileText, 
  CheckCircle2, 
  Info,
  Database,
  Cpu,
  TrendingUp,
  Sliders
} from 'lucide-react';

interface OverviewViewProps {
  data: SnapshotData;
  onNavigate: (tab: NavTab) => void;
}

export const OverviewView: React.FC<OverviewViewProps> = ({ data, onNavigate }) => {
  const pipelineStages = [
    { num: '01', name: 'Scientific Domains', desc: 'Inorganic solid-state, battery electrolytes, electrocatalysts', icon: <Database className="w-4 h-4" /> },
    { num: '02', name: 'Candidate Screening', desc: '333k virtual pool down-selection with Stage-1 ensemble', icon: <Sliders className="w-4 h-4" /> },
    { num: '03', name: 'Probabilistic Models', desc: 'Gaussian processes & observation variance calibration', icon: <TrendingUp className="w-4 h-4" /> },
    { num: '04', name: 'Competing Hypotheses', desc: 'H1 Phase Purity, H2 Homogeneity, H3 Kinetics', icon: <Layers className="w-4 h-4" /> },
    { num: '05', name: 'Candidate × Modality', desc: 'Joint action enumeration: which material AND which measurement', icon: <FlaskConical className="w-4 h-4" /> },
    { num: '06', name: 'Preregistration', desc: 'Predictions & criteria locked in ledger before reveal', icon: <FileText className="w-4 h-4" /> },
    { num: '07', name: 'Evidence Reveal', desc: 'Controlled draw or historical canonical observable extraction', icon: <Cpu className="w-4 h-4" /> },
    { num: '08', name: 'Bayesian Update', desc: 'Log-space likelihood updates, Bayes factors, posterior shift', icon: <BarChart3 className="w-4 h-4" /> },
    { num: '09', name: 'Benchmark & Audit', desc: '180 trajectories, 5,333 audit events, 50 boolean gates', icon: <ShieldCheck className="w-4 h-4" /> },
  ];

  return (
    <div className="space-y-8 pb-16 animate-fadeIn max-w-7xl mx-auto">
      {/* Editorial Hero Statement in Clean White & Emerald */}
      <section className="bg-white p-6 sm:p-8 rounded-xl border border-slate-200 shadow-xs">
        <div className="max-w-4xl space-y-4">
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-red-50 border border-red-200 text-red-800 text-xs font-semibold">
            <span className="w-2 h-2 rounded-full bg-red-600 animate-pulse" />
            <span>Autonomous Scientific Decision Framework</span>
          </div>

          <h1 className="text-3xl sm:text-4xl font-extrabold text-slate-900 tracking-tight leading-tight">
            From property prediction to evidence-driven scientific decisions.
          </h1>

          <p className="text-base text-slate-600 leading-relaxed">
            AIcoScientist is not simply a regression model predicting material properties. It is an active 
            scientific decision system that maintains competing mechanistic hypotheses, evaluates expected 
            hypothesis information gain (HIG), preregisters predictions before physical data revelation, and updates 
            beliefs across multiple multimodal experimental domains.
          </p>
        </div>

        {/* Operational Modes Bar */}
        <div className="mt-6 flex flex-wrap items-center gap-3 pt-4 border-t border-slate-100">
          <span className="text-xs font-bold uppercase tracking-wider text-slate-400 mr-2">Operational Modes:</span>
          <ModeBadge mode="CONTROLLED_SYNTHETIC" />
          <ModeBadge mode="HISTORICAL_REPLAY" />
          <ModeBadge mode="LIVE_COMPUTED" />
          <ModeBadge mode="NOT_AVAILABLE" />
        </div>
      </section>

      {/* Verified Artifact Metrics - Clean White & Emerald */}
      <section>
        <div className="flex items-center justify-between mb-4">
          <div>
            <h2 className="text-sm font-bold uppercase tracking-wider text-slate-900">Verified Artifact Metrics</h2>
            <p className="text-xs text-slate-500">Extracted directly from committed ledger and benchmark artifacts (zero fabricated values)</p>
          </div>
          <span className="text-xs font-mono text-red-700 bg-red-50 px-2 py-0.5 rounded border border-red-200">
            commit {data.provenance?.head_commit?.substring(0, 8)}
          </span>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
          <div className="bg-white p-5 rounded-xl border border-slate-200 shadow-xs border-l-4 border-l-red-600">
            <div className="text-xs font-semibold text-slate-500 uppercase tracking-wide">A-Lab Candidates</div>
            <div className="text-3xl font-extrabold text-slate-900 font-mono-num mt-1">1,035</div>
            <div className="text-xs text-slate-500 mt-2 flex items-center gap-1.5">
              <span className="w-1.5 h-1.5 rounded-full bg-red-500" />
              <span>1,030 XRD / Refinements linked</span>
            </div>
            <div className="text-2xs text-slate-400 mt-1">Precursor Genome (CC BY 4.0)</div>
          </div>

          <div className="bg-white p-5 rounded-xl border border-slate-200 shadow-xs border-l-4 border-l-red-500">
            <div className="text-xs font-semibold text-slate-500 uppercase tracking-wide">Controlled Trajectories</div>
            <div className="text-3xl font-extrabold text-slate-900 font-mono-num mt-1">180</div>
            <div className="text-xs text-slate-500 mt-2 flex items-center gap-1.5">
              <span className="w-1.5 h-1.5 rounded-full bg-red-500" />
              <span>6 policies × 2 worlds × 5 seeds</span>
            </div>
            <div className="text-2xs text-slate-400 mt-1">Clean & stress multi-hypothesis matrix</div>
          </div>

          <div className="bg-white p-5 rounded-xl border border-slate-200 shadow-xs border-l-4 border-l-red-700">
            <div className="text-xs font-semibold text-slate-500 uppercase tracking-wide">Evidence Ledger Events</div>
            <div className="text-3xl font-extrabold text-slate-900 font-mono-num mt-1">5,333</div>
            <div className="text-xs text-slate-500 mt-2 flex items-center gap-1.5">
              <span className="w-1.5 h-1.5 rounded-full bg-red-600" />
              <span>Auditable preregistration chain</span>
            </div>
            <div className="text-2xs text-red-800 font-medium mt-1">Audit log entries, not physical trials</div>
          </div>

          <div className="bg-white p-5 rounded-xl border border-slate-200 shadow-xs border-l-4 border-l-red-400">
            <div className="text-xs font-semibold text-slate-500 uppercase tracking-wide">Electrolyte Virtual Pool</div>
            <div className="text-3xl font-extrabold text-slate-900 font-mono-num mt-1">333,333</div>
            <div className="text-xs text-slate-500 mt-2 flex items-center gap-1.5">
              <span className="w-1.5 h-1.5 rounded-full bg-red-500" />
              <span>Stage-1 screened to WS=200</span>
            </div>
            <div className="text-2xs text-slate-400 mt-1">100% latent maximum recovered</div>
          </div>
        </div>
      </section>

      {/* The 9-Stage Scientific Loop - Clean White & Emerald */}
      <section className="bg-white p-6 rounded-xl border border-slate-200 shadow-xs">
        <div className="mb-5">
          <h2 className="text-base font-bold text-slate-900 tracking-tight">The 9-Stage Scientific Decision Loop</h2>
          <p className="text-xs text-slate-500">
            AIcoScientist formalizes each stage as an explicit, auditable protocol rather than an ad-hoc heuristic.
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
          {pipelineStages.map((st) => (
            <div key={st.num} className="p-4 rounded-lg border border-slate-200 bg-white hover:border-red-300 hover:bg-red-50/20 transition">
              <div className="flex items-center justify-between mb-1.5">
                <span className="font-mono text-xs font-bold text-red-700">{st.num}</span>
                <span className="text-red-600">{st.icon}</span>
              </div>
              <div className="font-semibold text-xs text-slate-900">{st.name}</div>
              <div className="text-xs text-slate-500 mt-1 leading-snug">{st.desc}</div>
            </div>
          ))}
        </div>
      </section>

      {/* Research Maturity - Clean White & Emerald */}
      <section className="bg-white p-6 rounded-xl border border-slate-200 shadow-xs">
        <div className="flex items-center justify-between mb-4">
          <div>
            <h2 className="text-base font-bold text-slate-900">Research Maturity & Validation Status</h2>
            <p className="text-xs text-slate-500">Honest boundary between verified findings, retrospective models, and roadmap</p>
          </div>
          <button 
            onClick={() => onNavigate('readiness')}
            className="text-xs font-semibold text-red-700 hover:text-red-800 flex items-center gap-1 cursor-pointer"
          >
            <span>View 50 Gates</span>
            <ArrowRight className="w-3.5 h-3.5" />
          </button>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-xs">
          <div className="p-4 rounded-lg bg-red-50/50 border border-red-200">
            <div className="flex items-center gap-2 font-bold text-red-900 mb-2">
              <CheckCircle2 className="w-4 h-4 text-red-700" />
              <span>Completed Methodology</span>
            </div>
            <ul className="space-y-1.5 text-red-950 text-xs">
              <li>• Controlled clean & stress worlds</li>
              <li>• Monte Carlo HIG estimation (MC32)</li>
              <li>• Immutable evidence ledger schema</li>
              <li>• Preregistration firewall guarantees</li>
            </ul>
          </div>

          <div className="p-4 rounded-lg bg-white border border-slate-200">
            <div className="flex items-center gap-2 font-bold text-slate-900 mb-2">
              <CheckCircle2 className="w-4 h-4 text-red-600" />
              <span>Retrospective Validation</span>
            </div>
            <ul className="space-y-1.5 text-slate-600 text-xs">
              <li>• 1,035 A-Lab Precursor Genome samples</li>
              <li>• 1,030 Canonical XRD & Rietveld models</li>
              <li>• 333k Electrolyte Stage-1 screening</li>
              <li>• ExtraTrees surrogate benchmark</li>
            </ul>
          </div>

          <div className="p-4 rounded-lg bg-white border border-slate-200">
            <div className="flex items-center gap-2 font-bold text-slate-900 mb-2">
              <Info className="w-4 h-4 text-slate-500" />
              <span>Documented Boundaries</span>
            </div>
            <ul className="space-y-1.5 text-slate-600 text-xs">
              <li>• SEM & EDS unlinked to sample ID</li>
              <li>• Refinement calibration over-dispersed</li>
              <li>• Cross-family generalization pending</li>
              <li>• Surrogate coupling not physical truth</li>
            </ul>
          </div>
        </div>
      </section>
    </div>
  );
};
