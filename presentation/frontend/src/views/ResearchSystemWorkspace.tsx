import React, { useState } from 'react';
import { SnapshotData } from '../types/mission_control';
import {
  ShieldCheck,
  GitBranch,
  FileCode,
  CheckCircle2,
  XCircle,
  Hash,
  Database,
  Layers,
  Search,
  ExternalLink,
  Copy,
  Check
} from 'lucide-react';

interface Props {
  data: SnapshotData;
}

export const ResearchSystemWorkspace: React.FC<Props> = ({ data }) => {
  const [activeTab, setActiveTab] = useState<'architecture' | 'ledger' | 'readiness' | 'manifest'>('architecture');

  // Ledger state
  const [eventSearch, setEventSearch] = useState<string>('');
  const [selectedEvent, setSelectedEvent] = useState<any | null>(null);
  const [copiedHash, setCopiedHash] = useState<string | null>(null);

  // Validation gates filter
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
  const filteredGates = gateEntries.filter(([_, status]) => {
    if (gateFilter === 'pass') return status === 'PASS';
    if (gateFilter === 'fail') return status === 'FAIL';
    return true;
  });

  const manifest = data.manifest;
  const artifactHashes = manifest?.source_artifact_hashes || {};

  const handleCopy = (text: string, key: string) => {
    navigator.clipboard.writeText(text);
    setCopiedHash(key);
    setTimeout(() => setCopiedHash(null), 2000);
  };

  return (
    <div className="space-y-6 pb-12">
      {/* Workspace Header */}
      <section className="rounded-3xl border border-slate-200 bg-white p-5 sm:p-6 shadow-xs">
        <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-4">
          <div>
            <div className="flex items-center gap-2 mb-1.5">
              <span className="inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full text-2xs font-mono font-bold bg-emerald-50 text-emerald-800 border border-emerald-200 uppercase tracking-wider">
                <ShieldCheck className="w-3 h-3 text-emerald-600" />
                Research System & Governance
              </span>
              <span className="text-2xs font-mono text-slate-400">|</span>
              <span className="text-2xs font-mono text-slate-500">
                Reusable Architecture • 5,333 Ledger Records • 48/50 Gates Passed
              </span>
            </div>
            <h1 className="text-xl sm:text-2xl font-extrabold text-slate-900 tracking-tight">
              System Architecture, Audit Ledger & Research Readiness
            </h1>
            <p className="text-xs sm:text-sm text-slate-600 mt-1 max-w-4xl">
              Inspect the core object-oriented scientific abstractions, immutable audit event stream, software governance gates, and cryptographic data manifest.
            </p>
          </div>

          <div className="flex items-center gap-2 font-mono text-xs">
            <span className="px-3 py-1.5 bg-emerald-50 text-emerald-800 font-bold border border-emerald-200 rounded-lg">
              48/50 Gates Pass
            </span>
            <span className="px-3 py-1.5 bg-slate-100 text-slate-700 font-bold border border-slate-200 rounded-lg">
              5,333 Audit Events
            </span>
          </div>
        </div>

        {/* System Sub-Navigation */}
        <div className="mt-5 pt-4 border-t border-slate-100 flex flex-wrap gap-2">
          {[
            { id: 'architecture', label: 'Reusable Core Architecture', icon: <Layers className="w-4 h-4" /> },
            { id: 'ledger', label: '5,333-Event Evidence Ledger', icon: <Database className="w-4 h-4" /> },
            { id: 'readiness', label: '50-Gate Verification Matrix', icon: <ShieldCheck className="w-4 h-4" /> },
            { id: 'manifest', label: 'Cryptographic Provenance Manifest', icon: <Hash className="w-4 h-4" /> }
          ].map((tab) => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id as any)}
              className={`flex items-center gap-2 px-4 py-2 rounded-xl text-xs font-mono font-bold transition cursor-pointer ${
                activeTab === tab.id
                  ? 'bg-emerald-600 text-white shadow-2xs'
                  : 'bg-slate-100 text-slate-600 hover:bg-slate-200'
              }`}
            >
              {tab.icon}
              <span>{tab.label}</span>
            </button>
          ))}
        </div>
      </section>

      {/* Tab 1: Reusable Core Architecture */}
      {activeTab === 'architecture' && (
        <section className="space-y-6">
          <div className="rounded-3xl border border-slate-200 bg-white p-6 shadow-xs">
            <div className="flex items-center gap-2 mb-2">
              <span className="flex h-6 w-6 items-center justify-center rounded-md bg-emerald-600 text-xs font-black text-white">
                Ψ
              </span>
              <h2 className="text-base font-bold text-slate-900">
                Domain-Agnostic Scientific Abstractions
              </h2>
            </div>
            <p className="text-xs text-slate-600 mb-6">
              AIcoScientist separates the physical material domain representation from the active inference decision loop via standard interfaces.
            </p>

            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 font-mono text-xs">
              {data.architecture?.core_abstractions?.map((abs) => (
                <div key={abs.name} className="p-4 rounded-2xl border border-slate-200 bg-slate-50/60 flex flex-col justify-between">
                  <div>
                    <strong className="text-emerald-800 text-sm block font-bold">{abs.name}</strong>
                    <span className="text-3xs text-slate-400 block mt-0.5">{abs.file}</span>
                    <p className="text-xs text-slate-600 font-sans mt-2">{abs.role}</p>
                  </div>
                  <div className="mt-3 pt-2 border-t border-slate-200/60 flex justify-between items-center text-3xs text-slate-400">
                    <span>Protocol: Standard Interface</span>
                    <span className="text-emerald-700 font-semibold">Ready</span>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Domain Decoupling Showcase */}
          <div className="rounded-3xl border border-slate-200 bg-white p-6 shadow-xs">
            <h3 className="text-sm font-bold text-slate-900 mb-3">
              Multi-Domain Implementation Registry
            </h3>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 font-mono text-xs">
              {data.architecture?.domains?.map((dom) => (
                <div key={dom.domain_id} className="p-4 rounded-2xl border border-slate-200 bg-white flex flex-col justify-between">
                  <div>
                    <div className="flex items-center justify-between mb-1">
                      <strong className="text-slate-900 text-sm">{dom.name}</strong>
                      <span className="px-2 py-0.5 rounded text-3xs font-bold bg-emerald-50 text-emerald-800 border border-emerald-200">
                        {dom.status}
                      </span>
                    </div>
                    <p className="text-xs text-slate-600 font-sans mt-1.5">{dom.purpose}</p>
                  </div>
                  <div className="mt-4 pt-2 border-t border-slate-100 text-3xs text-slate-500 space-y-1">
                    <div>Modalities: {dom.modalities.join(', ')}</div>
                    <div>Working Candidates: {dom.candidates_count}</div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </section>
      )}

      {/* Tab 2: Evidence Ledger */}
      {activeTab === 'ledger' && (
        <section className="rounded-3xl border border-slate-200 bg-white p-6 shadow-xs space-y-4">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
            <div>
              <h2 className="text-base font-bold text-slate-900">
                Immutable Evidence Ledger Event Stream (5,333 Events)
              </h2>
              <p className="text-xs text-slate-500">
                Chronological record of action scoring, pre-reveal predictions, observation reveals, and Bayesian updates
              </p>
            </div>

            <div className="relative">
              <Search className="w-3.5 h-3.5 text-slate-400 absolute left-3 top-2.5" />
              <input
                type="text"
                placeholder="Filter events (e.g. PREREGISTERED...)"
                value={eventSearch}
                onChange={(e) => setEventSearch(e.target.value)}
                className="pl-8 pr-3 py-1.5 text-xs font-mono bg-white border border-slate-200 rounded-lg text-slate-800 focus:outline-emerald-600 w-64"
              />
            </div>
          </div>

          <div className="overflow-x-auto border border-slate-200 rounded-xl bg-white shadow-2xs max-h-[460px]">
            <table className="w-full text-left border-collapse text-xs font-mono">
              <thead className="sticky top-0 bg-slate-50 border-b border-slate-200 text-2xs uppercase text-slate-500 z-10">
                <tr>
                  <th className="py-2.5 px-3">Event Seq</th>
                  <th className="py-2.5 px-3">Event Type</th>
                  <th className="py-2.5 px-3">Candidate / Action</th>
                  <th className="py-2.5 px-3">Timestamp</th>
                  <th className="py-2.5 px-3 text-right">Inspect</th>
                </tr>
              </thead>
              <tbody>
                {filteredEvents.slice(0, 100).map((ev, idx) => (
                  <tr
                    key={idx}
                    className="border-b border-slate-100 last:border-b-0 hover:bg-slate-50/60"
                  >
                    <td className="py-2 px-3 text-slate-500 font-bold">
                      #{ev.event_sequence ?? ev.global_event_sequence ?? idx}
                    </td>
                    <td className="py-2 px-3">
                      <span className={`px-2 py-0.5 rounded text-2xs font-bold ${
                        ev.event === 'PREREGISTERED_SELECTED_ACTION'
                          ? 'bg-amber-100 text-amber-800'
                          : ev.event === 'MEASUREMENT_REVEALED'
                          ? 'bg-blue-100 text-blue-800'
                          : ev.event === 'BELIEF_UPDATE'
                          ? 'bg-violet-100 text-violet-800'
                          : 'bg-emerald-100 text-emerald-800'
                      }`}>
                        {ev.event}
                      </span>
                    </td>
                    <td className="py-2 px-3 text-slate-800 font-semibold">
                      {ev.action?.candidate_id || ev.action?.action_type || 'System'}
                    </td>
                    <td className="py-2 px-3 text-slate-400 text-2xs">
                      {ev.timestamp ? ev.timestamp.substring(11, 19) : 'N/A'}
                    </td>
                    <td className="py-2 px-3 text-right">
                      <button
                        onClick={() => setSelectedEvent(ev)}
                        className="px-2 py-1 rounded bg-slate-100 hover:bg-slate-200 text-slate-700 text-2xs cursor-pointer font-bold"
                      >
                        JSON
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {/* Event JSON Modal */}
          {selectedEvent && (
            <div className="p-4 bg-slate-900 rounded-2xl text-slate-100 border border-slate-800 space-y-2">
              <div className="flex items-center justify-between">
                <span className="text-xs font-bold font-mono text-emerald-400">
                  Event: {selectedEvent.event} (Seq #{selectedEvent.event_sequence ?? 0})
                </span>
                <button
                  onClick={() => setSelectedEvent(null)}
                  className="text-xs text-slate-400 hover:text-white font-mono"
                >
                  [Close]
                </button>
              </div>
              <pre className="text-2xs font-mono overflow-x-auto max-h-60 p-3 bg-slate-950 rounded-xl text-slate-300">
                {JSON.stringify(selectedEvent, null, 2)}
              </pre>
            </div>
          )}
        </section>
      )}

      {/* Tab 3: Readiness 50-Gate Matrix */}
      {activeTab === 'readiness' && (
        <section className="rounded-3xl border border-slate-200 bg-white p-6 shadow-xs space-y-4">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
            <div>
              <h2 className="text-base font-bold text-slate-900">
                Boolean Validation Gate Matrix (48 / 50 Passed)
              </h2>
              <p className="text-xs text-slate-500">
                Software, schema, Monte Carlo bound compliance, and statistical invariant gates
              </p>
            </div>

            <div className="flex items-center gap-1 bg-slate-100 p-0.5 rounded-lg border border-slate-200 text-xs font-mono">
              {(['all', 'pass', 'fail'] as const).map((f) => (
                <button
                  key={f}
                  onClick={() => setGateFilter(f)}
                  className={`px-3 py-1 rounded-md transition cursor-pointer uppercase ${
                    gateFilter === f
                      ? 'bg-white text-emerald-800 font-bold shadow-2xs'
                      : 'text-slate-600 hover:text-slate-900'
                  }`}
                >
                  {f === 'all' ? 'All (50)' : f === 'pass' ? 'Pass (48)' : 'Fail (2)'}
                </button>
              ))}
            </div>
          </div>

          <div className="p-3 bg-slate-50 border border-slate-200 rounded-xl text-xs text-slate-600 font-mono">
            <strong>Advisor Guardrail:</strong> 48/50 is a software invariant verification metric, NOT a model accuracy score. The 2 failing gates represent documented empirical scientific boundaries.
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-3 font-mono text-xs">
            {filteredGates.map(([gateName, status]) => {
              const isPass = status === 'PASS';
              return (
                <div
                  key={gateName}
                  className={`p-3 rounded-xl border flex items-center justify-between ${
                    isPass
                      ? 'bg-white border-slate-200'
                      : 'bg-amber-50/70 border-amber-300'
                  }`}
                >
                  <div className="flex items-center gap-2">
                    {isPass ? (
                      <CheckCircle2 className="w-4 h-4 text-emerald-600 shrink-0" />
                    ) : (
                      <XCircle className="w-4 h-4 text-amber-600 shrink-0" />
                    )}
                    <span className={`text-xs ${isPass ? 'text-slate-800' : 'text-amber-950 font-bold'}`}>
                      {gateName}
                    </span>
                  </div>
                  <span className={`px-2 py-0.5 rounded text-2xs font-bold ${
                    isPass ? 'bg-emerald-50 text-emerald-800' : 'bg-amber-200 text-amber-900'
                  }`}>
                    {status}
                  </span>
                </div>
              );
            })}
          </div>
        </section>
      )}

      {/* Tab 4: Cryptographic Provenance Manifest */}
      {activeTab === 'manifest' && (
        <section className="rounded-3xl border border-slate-200 bg-white p-6 shadow-xs space-y-4">
          <div>
            <h2 className="text-base font-bold text-slate-900">
              Cryptographic Data Manifest & Source Provenance
            </h2>
            <p className="text-xs text-slate-500">
              SHA-256 integrity checksums for all 11 source artifacts powering this presentation
            </p>
          </div>

          <div className="space-y-2 font-mono text-xs">
            {Object.entries(artifactHashes).map(([path, hash]) => (
              <div
                key={path}
                className="p-3 bg-slate-50 border border-slate-200 rounded-xl flex flex-col sm:flex-row sm:items-center justify-between gap-2"
              >
                <div className="truncate">
                  <span className="text-slate-800 font-semibold">{path}</span>
                </div>
                <div className="flex items-center gap-2 shrink-0">
                  <code className="text-3xs bg-white px-2 py-1 rounded border border-slate-200 text-slate-600 font-mono">
                    {hash.substring(0, 16)}...
                  </code>
                  <button
                    onClick={() => handleCopy(hash, path)}
                    className="p-1 text-slate-400 hover:text-slate-700 cursor-pointer"
                    title="Copy full SHA-256 hash"
                  >
                    {copiedHash === path ? (
                      <Check className="w-3.5 h-3.5 text-emerald-600" />
                    ) : (
                      <Copy className="w-3.5 h-3.5" />
                    )}
                  </button>
                </div>
              </div>
            ))}
          </div>

          <div className="mt-4 pt-3 border-t border-slate-100 flex flex-col sm:flex-row items-center justify-between text-2xs font-mono text-slate-500 gap-2">
            <span>Branch: {manifest?.source_branch || 'integration/multimodal-scientific-engine'}</span>
            <span>Presentation Commit: {manifest?.presentation_build_commit?.substring(0, 10)}</span>
            <a
              href="https://doi.org/10.5281/zenodo.21285546"
              target="_blank"
              rel="noreferrer"
              className="text-emerald-700 hover:underline inline-flex items-center gap-1"
            >
              <span>Zenodo Dataset DOI: 10.5281/zenodo.21285546</span>
              <ExternalLink className="w-3 h-3" />
            </a>
          </div>
        </section>
      )}
    </div>
  );
};
