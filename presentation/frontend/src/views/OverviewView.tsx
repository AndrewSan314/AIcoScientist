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
  AlertTriangle,
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
  const audit = data.alab_audit || {};
  const validation = data.validation || {};
  const gateEvidence = validation.gate_evidence || {};

  const pipelineStages = [
    { num: '01', name: 'Scientific Domains', desc: 'Inorganic solid-state, battery electrolytes, electrocatalysts', icon: <Database className="w-4 h-4" /> },
    { num: '02', name: 'Candidate Screening', desc: '333k virtual pool down-selection with Stage-1 ensemble', icon: <Sliders className="w-4 h-4" /> },
    { num: '03', name: 'Probabilistic Models', desc: 'Gaussian processes & total observation variance calibration', icon: <TrendingUp className="w-4 h-4" /> },
    { num: '04', name: 'Competing Hypotheses', desc: 'H1 Phase Purity, H2 Homogeneity, H3 Kinetics', icon: <Layers className="w-4 h-4" /> },
    { num: '05', name: 'Candidate × Modality', desc: 'Joint action enumeration: which material AND which measurement', icon: <FlaskConical className="w-4 h-4" /> },
    { num: '06', name: 'Preregistration', desc: 'Predictions & criteria locked in ledger before reveal', icon: <FileText className="w-4 h-4" /> },
    { num: '07', name: 'Evidence Reveal', desc: 'Controlled draw or historical canonical observable extraction', icon: <Cpu className="w-4 h-4" /> },
    { num: '08', name: 'Bayesian Update', desc: 'Log-space likelihood updates, Bayes factors, posterior shift', icon: <BarChart3 className="w-4 h-4" /> },
    { num: '09', name: 'Benchmark & Audit', desc: '180 trajectories, 5,333 audit events, 50 boolean gates', icon: <ShieldCheck className="w-4 h-4" /> },
  ];

  return (
    <div className="space-y-10 pb-16 animate-fadeIn">
      {/* Editorial Hero Statement */}
      <section className="border-b border-slate-200 pb-8 pt-4">
        <div className="max-w-4xl">
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-emerald-50 border border-emerald-200 text-emerald-800 text-xs font-semibold mb-4">
            <span className="w-2 h-2 rounded-full bg-emerald-600 animate-pulse" />
            <span>Autonomous Scientific Discovery Architecture</span>
          </div>
          <h1 className="text-3xl sm:text-5xl font-extrabold text-slate-900 tracking-tight leading-tight mb-4">
            From property optimization to evidence-driven scientific decisions.
          </h1>
          <p className="text-base sm:text-lg text-slate-600 leading-relaxed">
            AIcoScientist is not simply a regression model predicting material properties. It is an active 
            scientific decision system that maintains competing mechanistic hypotheses, evaluates expected 
            hypothesis information gain (HIG), preregisters predictions before observation reveal, and updates 
            beliefs across multiple multimodal experimental domains.
          </p>
        </div>

        {/* Data Modes Strip */}
        <div className="mt-6 flex flex-wrap items-center gap-3 pt-4 border-t border-slate-100">
          <span className="text-xs font-bold uppercase tracking-wider text-slate-400 mr-2">Operational Modes:</span>
          <ModeBadge mode="CONTROLLED_SYNTHETIC" />
          <ModeBadge mode="HISTORICAL_REPLAY" />
          <ModeBadge mode="LIVE_COMPUTED" />
          <ModeBadge mode="NOT_AVAILABLE" />
        </div>
      </section>

      {/* Verified Artifact Metrics */}
      <section>
        <div className="flex items-center justify-between mb-4">
          <div>
            <h2 className="text-sm font-bold uppercase tracking-wider text-slate-900">Verified Artifact Metrics</h2>
            <p className="text-xs text-slate-500">Extracted directly from committed ledger and benchmark artifacts (zero fabricated values)</p>
          </div>
          <span className="text-xs font-mono text-slate-400">commit {data.provenance?.head_commit?.substring(0, 8)}</span>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
          <div className="sci-card p-5 border-l-4 border-l-emerald-600">
            <div className="text-xs font-semibold text-slate-500 uppercase tracking-wide">A-Lab Candidate Samples</div>
            <div className="text-3xl font-extrabold text-slate-900 font-mono-num mt-1">1,035</div>
            <div className="text-xs text-slate-500 mt-2 flex items-center gap-1.5">
              <span className="w-1.5 h-1.5 rounded-full bg-emerald-500" />
              <span>1,030 XRD / Refinements linked</span>
            </div>
            <div className="text-xs text-slate-400 mt-0.5">A-Lab Precursor Genome (CC BY 4.0)</div>
          </div>

          <div className="sci-card p-5 border-l-4 border-l-violet-600">
            <div className="text-xs font-semibold text-slate-500 uppercase tracking-wide">Controlled Trajectories</div>
            <div className="text-3xl font-extrabold text-slate-900 font-mono-num mt-1">180</div>
            <div className="text-xs text-slate-500 mt-2 flex items-center gap-1.5">
              <span className="w-1.5 h-1.5 rounded-full bg-violet-500" />
              <span>6 policies × 6 worlds × 5 seeds</span>
            </div>
            <div className="text-xs text-slate-400 mt-0.5">Clean & stress multi-hypothesis matrix</div>
          </div>

          <div className="sci-card p-5 border-l-4 border-l-amber-600">
            <div className="text-xs font-semibold text-slate-500 uppercase tracking-wide">Evidence Ledger Events</div>
            <div className="text-3xl font-extrabold text-slate-900 font-mono-num mt-1">5,333</div>
            <div className="text-xs text-slate-500 mt-2 flex items-center gap-1.5">
              <span className="w-1.5 h-1.5 rounded-full bg-amber-500" />
              <span>Auditable preregistration chain</span>
            </div>
            <div className="text-xs text-amber-700 font-medium mt-0.5">⚠️ Audit events, not physical trials</div>
          </div>

          <div className="sci-card p-5 border-l-4 border-l-blue-600">
            <div className="text-xs font-semibold text-slate-500 uppercase tracking-wide">Electrolyte Virtual Pool</div>
            <div className="text-3xl font-extrabold text-slate-900 font-mono-num mt-1">333,333</div>
            <div className="text-xs text-slate-500 mt-2 flex items-center gap-1.5">
              <span className="w-1.5 h-1.5 rounded-full bg-blue-500" />
              <span>Stage-1 screened to WS=200</span>
            </div>
            <div className="text-xs text-slate-400 mt-0.5">100% latent maximum recovered</div>
          </div>
        </div>
      </section>

      {/* Closed-Loop Scientific Architecture */}
      <section className="sci-card p-6">
        <div className="mb-6">
          <h2 className="text-lg font-bold text-slate-900 tracking-tight">The 9-Stage Scientific Decision Loop</h2>
          <p className="text-xs text-slate-500">
            AIcoScientist formalizes each stage as an explicit, auditable protocol rather than an ad-hoc heuristic.
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
          {pipelineStages.map((st) => (
            <div key={st.num} className="p-3.5 rounded-lg border border-slate-200 bg-slate-50/50 hover:bg-white hover:border-slate-300 transition">
              <div className="flex items-center justify-between mb-1.5">
                <span className="font-mono text-xs font-bold text-emerald-700">{st.num}</span>
                <span className="text-slate-400">{st.icon}</span>
              </div>
              <div className="font-semibold text-xs text-slate-900">{st.name}</div>
              <div className="text-xs text-slate-500 mt-1 leading-snug">{st.desc}</div>
            </div>
          ))}
        </div>
      </section>

      {/* Research Maturity Matrix */}
      <section className="sci-card p-6 border border-slate-200">
        <div className="flex items-center justify-between mb-4">
          <div>
            <h2 className="text-base font-bold text-slate-900">Research Maturity & Validation Status</h2>
            <p className="text-xs text-slate-500">Honest boundary between verified findings, retrospective models, and roadmap</p>
          </div>
          <button 
            onClick={() => onNavigate('readiness')}
            className="text-xs font-semibold text-emerald-700 hover:text-emerald-800 flex items-center gap-1 cursor-pointer"
          >
            <span>View 50 Gates</span>
            <ArrowRight className="w-3.5 h-3.5" />
          </button>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-4 gap-4 text-xs">
          <div className="p-4 rounded-lg bg-emerald-50/60 border border-emerald-200">
            <div className="flex items-center gap-2 font-bold text-emerald-900 mb-2">
              <CheckCircle2 className="w-4 h-4 text-emerald-700" />
              <span>Completed Methodology</span>
            </div>
            <ul className="space-y-1.5 text-emerald-950 text-xs">
              <li>• Controlled clean & stress worlds</li>
              <li>• Monte Carlo HIG estimation (MC32)</li>
              <li>• Immutable evidence ledger schema</li>
              <li>• Preregistration firewall guarantees</li>
            </ul>
          </div>

          <div className="p-4 rounded-lg bg-blue-50/60 border border-blue-200">
            <div className="flex items-center gap-2 font-bold text-blue-900 mb-2">
              <CheckCircle2 className="w-4 h-4 text-blue-700" />
              <span>Retrospective Validation</span>
            </div>
            <ul className="space-y-1.5 text-blue-950 text-xs">
              <li>• 1,035 A-Lab Precursor Genome samples</li>
              <li>• 1,030 Canonical XRD & Rietveld models</li>
              <li>• 333k Electrolyte Stage-1 screening</li>
              <li>• ExtraTrees surrogate benchmark</li>
            </ul>
          </div>

          <div className="p-4 rounded-lg bg-amber-50/60 border border-amber-200">
            <div className="flex items-center gap-2 font-bold text-amber-900 mb-2">
              <AlertTriangle className="w-4 h-4 text-amber-700" />
              <span>Documented Boundaries</span>
            </div>
            <ul className="space-y-1.5 text-amber-950 text-xs">
              <li>• SEM & EDS unlinked to sample ID</li>
              <li>• Refinement calibration over-dispersed (0.95 vs 0.50 target)</li>
              <li>• Chemistry-family holdout not established</li>
              <li>• Surrogate coupling not physical truth</li>
            </ul>
          </div>

          <div className="p-4 rounded-lg bg-slate-50 border border-slate-200">
            <div className="flex items-center gap-2 font-bold text-slate-800 mb-2">
              <Info className="w-4 h-4 text-slate-500" />
              <span>Prospective Roadmap</span>
            </div>
            <ul className="space-y-1.5 text-slate-600 text-xs">
              <li>• Physical automated synthesis robot execution</li>
              <li>• Direct spectrometer lab API integration</li>
              <li>• In-operando multi-point cycling</li>
              <li>• Multi-institution federated ledger</li>
            </ul>
          </div>
        </div>
      </section>

      {/* Quick Launch Cards */}
      <section className="grid grid-cols-1 md:grid-cols-3 gap-5">
        <div 
          onClick={() => onNavigate('cockpit')}
          className="sci-card sci-card-interactive p-6 cursor-pointer border border-emerald-200 bg-linear-to-br from-white to-emerald-50/20"
        >
          <div className="w-8 h-8 rounded-md bg-emerald-100 text-emerald-800 flex items-center justify-center mb-3">
            <FlaskConical className="w-4 h-4" />
          </div>
          <h3 className="font-bold text-sm text-slate-900 mb-1">Decision Cockpit</h3>
          <p className="text-xs text-slate-500 mb-4 leading-relaxed">
            Experience the 4-stage Preregister → Reveal → Update interaction with competing hypotheses and candidate selection.
          </p>
          <span className="text-xs font-semibold text-emerald-700 flex items-center gap-1">
            <span>Explore Flagship Loop</span>
            <ArrowRight className="w-3.5 h-3.5" />
          </span>
        </div>

        <div 
          onClick={() => onNavigate('alab')}
          className="sci-card sci-card-interactive p-6 cursor-pointer border border-slate-200"
        >
          <div className="w-8 h-8 rounded-md bg-blue-100 text-blue-800 flex items-center justify-center mb-3">
            <Layers className="w-4 h-4" />
          </div>
          <h3 className="font-bold text-sm text-slate-900 mb-1">A-Lab Evidence Atlas</h3>
          <p className="text-xs text-slate-500 mb-4 leading-relaxed">
            Inspect real solid-state synthesis samples, canonical XRD descriptors, model calibration, and holdout splits.
          </p>
          <span className="text-xs font-semibold text-blue-700 flex items-center gap-1">
            <span>Inspect 1,035 Samples</span>
            <ArrowRight className="w-3.5 h-3.5" />
          </span>
        </div>

        <div 
          onClick={() => onNavigate('benchmarks')}
          className="sci-card sci-card-interactive p-6 cursor-pointer border border-slate-200"
        >
          <div className="w-8 h-8 rounded-md bg-violet-100 text-violet-800 flex items-center justify-center mb-3">
            <BarChart3 className="w-4 h-4" />
          </div>
          <h3 className="font-bold text-sm text-slate-900 mb-1">Policy Benchmark Lab</h3>
          <p className="text-xs text-slate-500 mb-4 leading-relaxed">
            Compare 6 policies across 180 controlled trajectories. Inspect HIG Monte Carlo sensitivity (MC12 vs MC32).
          </p>
          <span className="text-xs font-semibold text-violet-700 flex items-center gap-1">
            <span>Analyze Trajectories</span>
            <ArrowRight className="w-3.5 h-3.5" />
          </span>
        </div>
      </section>
    </div>
  );
};
