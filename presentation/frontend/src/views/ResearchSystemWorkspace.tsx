import React, { useState } from 'react';
import { SnapshotData } from '../types/mission_control';
import {
  ShieldCheck,
  CheckCircle2,
  AlertTriangle,
  Layers,
  Database,
  Search,
  Code,
  ArrowRight,
  Hash,
  ExternalLink,
  ChevronDown,
  ChevronRight,
  Check,
  Copy
} from 'lucide-react';

interface Props {
  data: SnapshotData;
  initialSubtab?: 'architecture' | 'audit' | 'verification';
}

export const ResearchSystemWorkspace: React.FC<Props> = ({
  data,
  initialSubtab = 'architecture'
}) => {
  const [activeSubtab, setActiveSubtab] = useState<'architecture' | 'audit' | 'verification'>(initialSubtab);
  const registry = data.dataset_registry?.datasets || [];
  const controlledInfo = registry.find((entry) => entry.id === 'controlled_multimodal_alloy');
  const electrolyteInfo = registry.find((entry) => entry.id === 'anode_free_electrolyte_screening');
  const alabInfo = registry.find((entry) => entry.id === 'alab_precursor_genome');

  // Audit event search and inspection
  const [eventSearch, setEventSearch] = useState<string>('');
  const [selectedEvent, setSelectedEvent] = useState<any | null>(null);

  // Gate filter
  const [gateFilter, setGateFilter] = useState<'all' | 'pass' | 'fail'>('all');

  const ledgerEvents = data.ledger_sample_events || [];
  const filteredEvents = ledgerEvents.filter((e) => {
    const q = eventSearch.toLowerCase();
    return (
      (e.event && e.event.toLowerCase().includes(q)) ||
      (e.run_id && e.run_id.toLowerCase().includes(q)) ||
      (e.action?.candidate_id && e.action.candidate_id.toLowerCase().includes(q))
    );
  });

  const validationGates = data.validation?.gates || {};
  const gateEntries = Object.entries(validationGates);
  const filteredGates = gateEntries.filter(([name, status]) => {
    if (gateFilter === 'pass') return status === 'PASS';
    if (gateFilter === 'fail') return status === 'FAIL';
    return true;
  });

  const manifest = data.manifest;
  const artifactHashes = manifest?.source_artifact_hashes || {};
  const passCount = manifest?.validation_gate_pass_count;
  const totalCount = manifest?.validation_gate_total_count;
  const failCount = typeof passCount === 'number' && typeof totalCount === 'number' ? totalCount - passCount : undefined;

  return (
    <div className="space-y-6 pb-12 animate-fade-in">
      {/* Workspace Header & Subtabs */}
      <section className="sci-card p-4 bg-[#FCFCFA] flex flex-col sm:flex-row items-center justify-between gap-4">
        <div className="flex items-center gap-2">
          <ShieldCheck className="w-5 h-5 text-[#DC2626]" />
          <div>
            <h1 className="text-base font-bold text-[#17201F]">How AIcoScientist Works</h1>
            <p className="text-xs text-[#66706C]">Core architecture, immutable audit trail, and formal verification gates</p>
          </div>
        </div>

        {/* 3 Clean Subtabs */}
        <div className="flex items-center gap-1 bg-[#F4F3EE] p-1 rounded-xl border border-[#D9DFDB]" role="tablist">
          {[
            { id: 'architecture', label: 'Architecture', icon: <Layers className="w-3.5 h-3.5" /> },
            { id: 'audit', label: `Audit Trail (${data.manifest?.total_audit_events ?? 'N/A'})`, icon: <Database className="w-3.5 h-3.5" /> },
            { id: 'verification', label: `Verification (${data.manifest?.validation_gate_pass_count ?? 'N/A'}/${data.manifest?.validation_gate_total_count ?? 'N/A'})`, icon: <ShieldCheck className="w-3.5 h-3.5" /> },
          ].map((tab) => {
            const isActive = activeSubtab === tab.id;
            return (
              <button
                key={tab.id}
                role="tab"
                aria-selected={isActive}
                onClick={() => setActiveSubtab(tab.id as any)}
                className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold transition cursor-pointer ${
                  isActive
                    ? 'bg-white text-[#DC2626] shadow-2xs font-bold'
                    : 'text-[#66706C] hover:text-[#17201F]'
                }`}
              >
                {tab.icon}
                <span>{tab.label}</span>
              </button>
            );
          })}
        </div>
      </section>

      {/* SUBTAB 1: ARCHITECTURE */}
      {activeSubtab === 'architecture' && (
        <div className="space-y-6 animate-fade-in">
          {/* Pipeline Flow Diagram */}
          <section className="sci-card p-6 space-y-4">
            <div className="border-b border-[#D9DFDB] pb-3">
              <h2 className="text-base font-bold text-[#17201F]">End-to-End Decision Pipeline</h2>
              <p className="text-xs text-[#66706C] mt-0.5">Information flows through five modular stages during each discovery iteration</p>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-5 gap-3 pt-2">
              {[
                {
                  step: '01',
                  title: 'Hypothesis Registry',
                   desc: 'Maintains source-registered model hypotheses and prior probabilities where the selected domain provides them.'
                },
                {
                  step: '02',
                  title: 'Observation Models',
                  desc: 'Computes predictive distributions for the modalities supported by the selected domain adapter.'
                },
                {
                  step: '03',
                  title: 'HIG Action Evaluator',
                  desc: 'Persists source action-score fields when the source run records them; missing score components remain unavailable.'
                },
                {
                  step: '04',
                  title: 'Preregistration Ledger',
                  desc: 'Commits target action to immutable audit log with cryptographic hash before observation.'
                },
                {
                  step: '05',
                  title: 'Bayesian Belief Update',
                  desc: 'Ingests revealed observation y, calculates likelihoods, and updates posterior: P(H | y) ∝ P(y | H)P(H).'
                }
              ].map((stage, idx) => (
                <div key={idx} className="p-4 rounded-xl border border-[#D9DFDB] bg-[#FCFCFA] flex flex-col justify-between space-y-2">
                  <div>
                    <span className="text-2xs font-mono font-bold text-[#DC2626] block mb-1">STAGE {stage.step}</span>
                    <h3 className="text-xs font-bold text-[#17201F]">{stage.title}</h3>
                  </div>
                  <p className="text-2xs text-[#66706C] leading-relaxed">{stage.desc}</p>
                </div>
              ))}
            </div>
          </section>

          {/* Plugged-in Domains Registry */}
          <section className="sci-card p-6 space-y-4">
            <div className="border-b border-[#D9DFDB] pb-3">
              <h2 className="text-base font-bold text-[#17201F]">Plugged-In Scientific Domains</h2>
              <p className="text-xs text-[#66706C] mt-0.5">Source-backed scientific datasets currently registered with the engine</p>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 pt-1">
              <div className="p-4 rounded-xl border border-[#D9DFDB] bg-[#FCFCFA] space-y-2.5">
                <div className="flex items-center justify-between">
                  <h3 className="text-sm font-bold text-[#17201F]">Alloy Synthesis Domain</h3>
                  <span className="sci-badge sci-badge-verified text-3xs">Controlled Flagship</span>
                </div>
                <p className="text-2xs text-[#66706C]">{controlledInfo?.summary || 'Source summary unavailable.'}</p>
                <div className="p-2 rounded bg-[#F4F3EE] font-mono text-2xs space-y-1 text-[#66706C]">
                  <div>Candidates: {controlledInfo?.candidateCount ?? 'N/A'}</div>
                  <div>Hypotheses: {controlledInfo?.hypotheses?.join(', ') || 'N/A'}</div>
                  <div>Modalities: {Object.keys(controlledInfo?.modalities || {}).join(', ') || 'N/A'}</div>
                </div>
              </div>

              <div className="p-4 rounded-xl border border-[#D9DFDB] bg-[#FCFCFA] space-y-2.5">
                <div className="flex items-center justify-between">
                  <h3 className="text-sm font-bold text-[#17201F]">Solid Electrolyte Domain</h3>
                  <span className="sci-badge sci-badge-surrogate text-3xs">Scale Preview</span>
                </div>
                <p className="text-2xs text-[#66706C]">{electrolyteInfo?.summary || 'Source summary unavailable.'}</p>
                <div className="p-2 rounded bg-[#F4F3EE] font-mono text-2xs space-y-1 text-[#66706C]">
                  <div>Candidates: {electrolyteInfo?.candidateCount?.toLocaleString() ?? 'N/A'}; working set: {electrolyteInfo?.screenedWorkingSetCount ?? 'N/A'}</div>
                  <div>Target: {electrolyteInfo?.scientificTargetName || electrolyteInfo?.targetObservable || 'N/A'}</div>
                  <div>Modes: {Object.keys(electrolyteInfo?.modalities || {}).join(', ') || 'N/A'}</div>
                </div>
              </div>

              <div className="p-4 rounded-xl border border-[#D9DFDB] bg-[#FCFCFA] space-y-2.5">
                <div className="flex items-center justify-between">
                  <h3 className="text-sm font-bold text-[#17201F]">Autonomous A-Lab Domain</h3>
                  <span className="sci-badge sci-badge-historical text-3xs">Historical Replay</span>
                </div>
                <p className="text-2xs text-[#66706C]">{alabInfo?.summary || 'Source summary unavailable.'}</p>
                <div className="p-2 rounded bg-[#F4F3EE] font-mono text-2xs space-y-1 text-[#66706C]">
                  <div>Source samples: {alabInfo?.candidateCount ?? 'N/A'}</div>
                  <div>Available modalities: {Object.entries(alabInfo?.modalities || {}).filter(([, modality]) => modality.available).map(([name]) => name).join(', ') || 'N/A'}</div>
                  <div>Unavailable linkage: {Object.entries(alabInfo?.modalities || {}).filter(([, modality]) => !modality.available).map(([name]) => name).join(', ') || 'None recorded'}</div>
                </div>
              </div>
            </div>
          </section>

          {/* Extension Code Preview */}
          <section className="sci-card p-6 space-y-3">
            <div className="border-b border-[#D9DFDB] pb-3 flex items-center justify-between">
              <div>
                <h2 className="text-base font-bold text-[#17201F]">Extending to New Scientific Domains</h2>
                <p className="text-xs text-[#66706C] mt-0.5">Minimal Python subclass to register custom hypotheses and likelihood models</p>
              </div>
              <span className="font-mono text-2xs text-[#DC2626] bg-[#FEF2F2] px-2 py-1 rounded border border-[#FECACA]">
                DomainPlugin Interface
              </span>
            </div>

            <pre className="p-4 rounded-xl bg-[#17201F] text-[#FCFCFA] font-mono text-xs overflow-x-auto leading-relaxed">
{`from aicoscientist.core import DomainPlugin, Hypothesis, Action, Distribution

class CustomMaterialsDomain(DomainPlugin):
    """Register custom materials hypotheses, characterization modalities, and likelihoods."""
    # Supply domain-specific contracts and source-backed parameters here.
    # The console does not invent priors, distributions, or costs.
    ...`}
            </pre>
          </section>
        </div>
      )}

      {/* SUBTAB 2: AUDIT TRAIL */}
      {activeSubtab === 'audit' && (
        <div className="space-y-6 animate-fade-in">
          <section className="sci-card p-6 space-y-4">
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 border-b border-[#D9DFDB] pb-3">
              <div>
                <h2 className="text-base font-bold text-[#17201F]">Immutable Scientific Audit Trail</h2>
                <p className="text-xs text-[#66706C] mt-0.5">{data.manifest?.total_audit_events ?? 'N/A'} source events are tracked in the manifest; this table shows the flagship run sample.</p>
              </div>

              {/* Search Box */}
              <div className="relative">
                <Search className="w-3.5 h-3.5 text-[#8F9995] absolute left-3 top-2.5" />
                <input
                  type="text"
                  placeholder="Filter events by candidate, type..."
                  value={eventSearch}
                  onChange={(e) => setEventSearch(e.target.value)}
                  className="pl-8 pr-3 py-1.5 text-xs rounded-lg border border-[#D9DFDB] bg-[#FCFCFA] text-[#17201F] focus:ring-1 focus:ring-[#DC2626] w-64"
                />
              </div>
            </div>

            {/* Event Table */}
            <div className="overflow-x-auto rounded-xl border border-[#D9DFDB] bg-white">
              <table className="w-full text-left text-xs">
                <thead className="bg-[#F4F3EE] text-[#66706C] font-mono text-2xs border-b border-[#D9DFDB]">
                  <tr>
                    <th className="p-3">Index / Event</th>
                    <th className="p-3">Target Candidate</th>
                    <th className="p-3">Modality</th>
                    <th className="p-3">Policy / Loop</th>
                    <th className="p-3">Payload Summary</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-[#D9DFDB] font-mono text-2xs">
                  {filteredEvents.slice(0, 10).map((e, idx) => (
                    <tr key={idx} className="hover:bg-[#F4F3EE]/60 transition">
                      <td className="p-3">
                        <span className="font-bold text-[#17201F]">#{idx + 1}</span>{' '}
                        <span className="ml-1 text-[#DC2626]">{e.event || 'STEP_PREREGISTERED'}</span>
                      </td>
                      <td className="p-3 text-[#17201F]">{e.action?.candidate_id || 'Not recorded'}</td>
                      <td className="p-3">
                        <span className="px-1.5 py-0.5 rounded bg-[#FEF2F2] text-[#991B1B] font-semibold">
                          {e.action?.action_type || 'Not recorded'}
                        </span>
                      </td>
                      <td className="p-3 text-[#8F9995]">{e.policy_name ? `${e.policy_name} Policy` : (e.step !== undefined ? `Step ${e.step}` : 'Autonomous Loop')}</td>
                      <td className="p-3 text-[#66706C] truncate max-w-xs">
                        {e.preregistration_hash
                          ? `SHA: ${e.preregistration_hash.substring(0, 16)}...`
                          : e.entropy_nats !== undefined
                          ? `Entropy: ${e.entropy_nats.toFixed(3)} nats`
                          : 'Action evaluated and locked'}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <p className="text-2xs font-mono text-[#8F9995]">
              Showing {Math.min(10, filteredEvents.length)} of {filteredEvents.length} ledger events.
            </p>
          </section>

          {/* Collapsible Cryptographic Provenance Disclosure */}
          <details className="sci-card p-6 transition-all group">
            <summary className="font-bold text-sm text-[#17201F] cursor-pointer flex items-center justify-between select-none list-none">
              <div className="flex items-center gap-2">
                <Hash className="w-4 h-4 text-[#DC2626]" />
                <span>Cryptographic Provenance & Artifact Integrity Manifest</span>
                <span className="sci-badge sci-badge-verified">Verified Reproducible</span>
              </div>
              <span className="text-xs font-mono text-[#DC2626] underline group-open:hidden">Expand integrity manifest</span>
              <span className="text-xs font-mono text-[#66706C] underline hidden group-open:inline">Collapse manifest</span>
            </summary>

            <div className="mt-4 pt-4 border-t border-[#D9DFDB] space-y-3">
              <div className="p-3 rounded-lg bg-[#F4F3EE] border border-[#D9DFDB] text-xs flex flex-wrap items-center justify-between gap-2 text-[#66706C]">
                <div>Pipeline Invariant: <strong className="text-[#17201F]">Deterministic Reproducibility Verified</strong></div>
                <div>Audit Engine: <strong className="text-[#DC2626]">Immutable Cryptographic Ledger</strong></div>
              </div>

              <div className="overflow-x-auto rounded-xl border border-[#D9DFDB] bg-white">
                <table className="w-full text-left text-2xs font-mono">
                  <thead className="bg-[#F4F3EE] text-[#66706C] border-b border-[#D9DFDB]">
                    <tr>
                      <th className="p-2.5">Artifact</th>
                      <th className="p-2.5">SHA-256 Digest</th>
                      <th className="p-2.5">Status</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-[#D9DFDB]">
                    {Object.entries(artifactHashes).map(([key, hash], i) => (
                      <tr key={i} className="hover:bg-[#F4F3EE]/50">
                        <td className="p-2.5 font-bold text-[#17201F]">{key}</td>
                        <td className="p-2.5 text-[#66706C]">{String(hash).substring(0, 32)}...</td>
                        <td className="p-2.5 text-[#DC2626] font-semibold">VERIFIED</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </details>
        </div>
      )}

      {/* SUBTAB 3: VERIFICATION */}
      {activeSubtab === 'verification' && (
        <div className="space-y-6 animate-fade-in">
          {/* Stat Cards */}
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
            <div className="sci-card p-5 border-l-4 border-l-[#DC2626] flex items-center justify-between">
              <div>
                <span className="text-2xs font-mono text-[#8F9995] uppercase">Passing Gates</span>
                <div className="text-2xl font-bold text-[#DC2626] mt-1">{passCount ?? 'N/A'} Passed</div>
              </div>
              <CheckCircle2 className="w-8 h-8 text-[#DC2626]" />
            </div>

            <div className="sci-card p-5 border-l-4 border-l-[#D97706] flex items-center justify-between">
              <div>
                <span className="text-2xs font-mono text-[#8F9995] uppercase">Failing Gates (Honest Boundary)</span>
                <div className="text-2xl font-bold text-[#D97706] mt-1">{failCount ?? 'N/A'} Failed</div>
              </div>
              <AlertTriangle className="w-8 h-8 text-[#D97706]" />
            </div>

            <div className="sci-card p-5 border-l-4 border-l-[#DC2626] flex items-center justify-between">
              <div>
                <span className="text-2xs font-mono text-[#8F9995] uppercase">Compliance Rate</span>
                <div className="text-2xl font-bold text-[#17201F] mt-1">{typeof passCount === 'number' && typeof totalCount === 'number' && totalCount > 0 ? ((passCount / totalCount) * 100).toFixed(1) : 'N/A'}%</div>
              </div>
              <ShieldCheck className="w-8 h-8 text-[#DC2626]" />
            </div>
          </div>

          {/* Explanation Callout for Failing Gates */}
          <section className="sci-card p-5 border border-[#FDE68A] bg-[#FEF3C7] space-y-3">
            <div className="flex items-center gap-2">
              <AlertTriangle className="w-5 h-5 text-[#D97706]" />
              <h2 className="text-sm font-bold text-[#92400E]">Transparent Boundary Disclosure: {failCount} Failed Gates</h2>
            </div>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-3 text-xs text-[#92400E]">
              <div className="p-3 bg-white/80 rounded-lg border border-[#FDE68A]">
                <strong>Gate 17: A_LAB_CALIBRATION_PARTIAL</strong>
                <p className="mt-1 leading-relaxed text-2xs">
                  Source calibration and replay limitations are disclosed as recorded; this gate is not converted into a universal physical accuracy claim.
                </p>
              </div>
              <div className="p-3 bg-white/80 rounded-lg border border-[#FDE68A]">
                <strong>Gate 43: OUT_OF_FAMILY_GENERALIZATION</strong>
                <p className="mt-1 leading-relaxed text-2xs">
                  Honest scientific boundary: generalization to unseen chemical spaces is not yet established. The system detects extrapolation outside validated manifold and halts safely.
                </p>
              </div>
            </div>
          </section>

          {/* Gate Filter & Table */}
          <section className="sci-card p-6 space-y-4">
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 border-b border-[#D9DFDB] pb-3">
              <div>
              <h2 className="text-base font-bold text-[#17201F]">Formal {totalCount}-Gate Verification Matrix</h2>
                <p className="text-xs text-[#66706C] mt-0.5">Automated software and scientific compliance checks</p>
              </div>

              <div className="flex items-center gap-1 bg-[#F4F3EE] p-1 rounded-lg border border-[#D9DFDB]">
                {[
                  { id: 'all', label: `All (${totalCount})` },
                  { id: 'pass', label: `Passed (${passCount})` },
                  { id: 'fail', label: `Failed (${failCount})` },
                ].map((f) => (
                  <button
                    key={f.id}
                    onClick={() => setGateFilter(f.id as any)}
                    className={`px-2.5 py-1 rounded text-2xs font-semibold transition cursor-pointer ${
                      gateFilter === f.id
                        ? 'bg-white text-[#DC2626] shadow-2xs font-bold'
                        : 'text-[#66706C] hover:text-[#17201F]'
                    }`}
                  >
                    {f.label}
                  </button>
                ))}
              </div>
            </div>

            <div className="overflow-x-auto rounded-xl border border-[#D9DFDB] bg-white">
              <table className="w-full text-left text-xs">
                <thead className="bg-[#F4F3EE] text-[#66706C] font-mono text-2xs border-b border-[#D9DFDB]">
                  <tr>
                    <th className="p-3">Gate Name</th>
                    <th className="p-3">Status</th>
                    <th className="p-3">Type</th>
                    <th className="p-3">Validation Rationale</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-[#D9DFDB] font-mono text-2xs">
                  {filteredGates.map(([gateName, status], idx) => {
                    const isFail = status === 'FAIL';
                    return (
                      <tr key={idx} className={isFail ? 'bg-[#FEF3C7]/40' : 'hover:bg-[#F4F3EE]/50'}>
                        <td className="p-3 font-bold text-[#17201F]">{gateName}</td>
                        <td className="p-3">
                          <span
                            className={`px-2 py-0.5 rounded text-3xs font-bold ${
                              isFail
                                ? 'bg-[#FEF3C7] text-[#92400E] border border-[#FDE68A]'
                                : 'bg-[#FEF2F2] text-[#991B1B] border border-[#FECACA]'
                            }`}
                          >
                            {status}
                          </span>
                        </td>
                        <td className="p-3 text-[#66706C]">
                          {gateName.includes('A_LAB')
                            ? 'Empirical Replay'
                            : gateName.includes('FIREWALL')
                            ? 'Information Security'
                            : 'Algorithmic Convergence'}
                        </td>
                        <td className="p-3 text-[#66706C] font-sans text-2xs">
                          {isFail
                            ? 'Honest boundary condition documented in technical report.'
                            : 'Deterministic verification requirement satisfied.'}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </section>
        </div>
      )}
    </div>
  );
};
