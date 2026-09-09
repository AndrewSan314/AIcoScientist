import React, { useState } from 'react';
import { SnapshotData } from '../types/mission_control';
import { ModeBadge } from '../components/ModeBadge';
import { 
  GitBranch, 
  Layers, 
  Database, 
  Copy, 
  Check, 
  Code,
  ShieldCheck,
  Cpu
} from 'lucide-react';

interface ArchitectureViewProps {
  data: SnapshotData;
}

export const ArchitectureView: React.FC<ArchitectureViewProps> = ({ data }) => {
  const [activeSubTab, setActiveSubTab] = useState<'architecture' | 'ledger'>('architecture');
  const [selectedEventIndex, setSelectedEventIndex] = useState<number>(0);
  const [copied, setCopied] = useState<boolean>(false);
  const [eventTypeFilter, setEventTypeFilter] = useState<string>('ALL');

  const arch = data.architecture || { core_abstractions: [], domains: [] };
  const ledgerEvents = data.ledger_sample_events || [];

  const filteredEvents = eventTypeFilter === 'ALL' 
    ? ledgerEvents 
    : ledgerEvents.filter(e => e.event === eventTypeFilter);

  const activeEvent = filteredEvents[selectedEventIndex] || filteredEvents[0];

  const handleCopyJson = () => {
    if (activeEvent) {
      navigator.clipboard.writeText(JSON.stringify(activeEvent, null, 2));
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    }
  };

  return (
    <div className="space-y-8 pb-16 animate-fadeIn max-w-7xl mx-auto">
      {/* GlowBal Header */}
      <div className="rounded-3xl border border-slate-200 bg-white p-6 sm:p-8 shadow-xs">
        <div className="flex flex-col gap-2">
          <p className="text-xs font-bold uppercase tracking-[0.18em] text-emerald-600">
            System Abstractions & Auditability
          </p>
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
            <h1 className="font-extrabold text-2xl sm:text-3xl tracking-tight text-slate-900">
              Cross-Domain Architecture & Evidence Ledger
            </h1>
            <div className="flex items-center gap-2">
              <ModeBadge mode="LIVE_COMPUTED" size="sm" />
              <div className="flex bg-slate-100 p-1 rounded-lg border border-slate-200 text-xs">
                <button
                  onClick={() => setActiveSubTab('architecture')}
                  className={`px-3 py-1 rounded-md font-semibold transition cursor-pointer ${
                    activeSubTab === 'architecture'
                      ? 'bg-white text-emerald-800 shadow-xs font-bold'
                      : 'text-slate-600 hover:text-slate-900'
                  }`}
                >
                  System Design
                </button>
                <button
                  onClick={() => setActiveSubTab('ledger')}
                  className={`px-3 py-1 rounded-md font-semibold transition cursor-pointer ${
                    activeSubTab === 'ledger'
                      ? 'bg-white text-emerald-800 shadow-xs font-bold'
                      : 'text-slate-600 hover:text-slate-900'
                  }`}
                >
                  Evidence Ledger (5,333 Events)
                </button>
              </div>
            </div>
          </div>
          <p className="max-w-3xl text-sm leading-relaxed text-slate-600 mt-1">
            Universal scientific decision abstractions cleanly decoupled from experimental hardware, 
            backed by an immutable, cryptographically verifiable evidence ledger.
          </p>
        </div>
      </div>

      {activeSubTab === 'architecture' ? (
        /* Architecture Tab */
        <div className="space-y-8">
          {/* SECTION 1: Core Abstractions */}
          <section className="space-y-4">
            <div className="flex items-center justify-between border-b border-slate-200 pb-3">
              <div className="flex items-center gap-2.5">
                <span className="flex h-7 w-7 items-center justify-center rounded-lg bg-emerald-600 text-xs font-black text-white shadow-xs">
                  01
                </span>
                <div>
                  <h2 className="text-lg font-bold text-slate-900 tracking-tight">Core Scientific Abstractions</h2>
                  <p className="text-xs text-slate-500">Reusable domain-agnostic protocol definitions</p>
                </div>
              </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {arch.core_abstractions.map((item, idx) => (
                <div key={idx} className="rounded-2xl border border-slate-200 bg-white p-5 shadow-xs space-y-2 hover:border-emerald-300 transition">
                  <div className="flex items-center justify-between">
                    <span className="font-mono text-xs font-bold text-slate-900">{item.name}</span>
                    <span className="text-2xs font-mono px-2 py-0.5 rounded bg-emerald-50 text-emerald-800 border border-emerald-200 truncate max-w-[180px]" title={item.file}>
                      {item.file}
                    </span>
                  </div>
                  <p className="text-xs text-slate-600 leading-relaxed">{item.role}</p>
                  <div className="pt-2 border-t border-slate-100 flex items-center gap-1.5 text-2xs font-mono text-slate-400">
                    <Code className="w-3 h-3 text-emerald-600" />
                    <span>Contract Interface</span>
                  </div>
                </div>
              ))}
            </div>
          </section>

          {/* SECTION 2: Domain Instantiations */}
          <section className="space-y-4">
            <div className="flex items-center justify-between border-b border-slate-200 pb-3">
              <div className="flex items-center gap-2.5">
                <span className="flex h-7 w-7 items-center justify-center rounded-lg bg-emerald-600 text-xs font-black text-white shadow-xs">
                  02
                </span>
                <div>
                  <h2 className="text-lg font-bold text-slate-900 tracking-tight">Concrete Experimental Domain Implementations</h2>
                  <p className="text-xs text-slate-500">Zero coupling between domain physics and decision reasoning</p>
                </div>
              </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {arch.domains.map((dom, idx) => (
                <div key={idx} className="rounded-2xl border border-slate-200 bg-white p-6 shadow-xs space-y-3">
                  <div className="flex items-center justify-between border-b border-slate-100 pb-2">
                    <span className="font-bold text-sm text-slate-900">{dom.name}</span>
                    <span className="text-2xs font-mono px-2 py-0.5 rounded bg-slate-100 text-slate-600">
                      {dom.status}
                    </span>
                  </div>
                  <div className="space-y-2 text-xs">
                    <div>
                      <span className="text-slate-500">Purpose: </span>
                      <span className="font-semibold text-slate-800">{dom.purpose}</span>
                    </div>
                    <div>
                      <span className="text-slate-500">Candidate Pool: </span>
                      <span className="font-mono font-bold text-emerald-800">{dom.candidates_count.toLocaleString()}</span>
                    </div>
                    <div>
                      <span className="text-slate-500">Modalities: </span>
                      <span className="font-mono text-slate-700">{dom.modalities.join(' • ')}</span>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </section>
        </div>
      ) : (
        /* Ledger Inspector Tab */
        <section className="space-y-4">
          <div className="flex items-center justify-between border-b border-slate-200 pb-3">
            <div className="flex items-center gap-2.5">
              <span className="flex h-7 w-7 items-center justify-center rounded-lg bg-emerald-600 text-xs font-black text-white shadow-xs">
                01
              </span>
              <div>
                <h2 className="text-lg font-bold text-slate-900 tracking-tight">Evidence Ledger Event Stream</h2>
                <p className="text-xs text-slate-500">Cryptographically verifiable immutable audit log</p>
              </div>
            </div>

            <div className="flex items-center gap-2 text-xs">
              <span className="text-slate-500">Filter Event:</span>
              <select
                value={eventTypeFilter}
                onChange={(e) => setEventTypeFilter(e.target.value)}
                className="bg-white border border-slate-200 rounded-md px-2.5 py-1 text-xs font-semibold text-slate-800 shadow-2xs"
              >
                <option value="ALL">All Events (5,333)</option>
                <option value="CANDIDATE_REGISTERED">CANDIDATE_REGISTERED</option>
                <option value="HYPOTHESIS_REGISTERED">HYPOTHESIS_REGISTERED</option>
                <option value="ACTION_PREREGISTERED">ACTION_PREREGISTERED</option>
                <option value="MEASUREMENT_REVEALED">MEASUREMENT_REVEALED</option>
                <option value="BELIEF_UPDATED">BELIEF_UPDATED</option>
              </select>
            </div>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
            {/* Event List (5 cols) */}
            <div className="lg:col-span-5 rounded-2xl border border-slate-200 bg-white p-3 shadow-xs max-h-[520px] overflow-y-auto space-y-1.5">
              {filteredEvents.map((evt, idx) => (
                <button
                  key={idx}
                  onClick={() => setSelectedEventIndex(idx)}
                  className={`w-full p-3 rounded-xl border text-left transition cursor-pointer text-xs space-y-1 ${
                    selectedEventIndex === idx
                      ? 'border-emerald-500 bg-emerald-50/50 shadow-2xs'
                      : 'border-slate-100 bg-white hover:border-slate-200'
                  }`}
                >
                  <div className="flex items-center justify-between">
                    <span className="font-mono text-2xs font-bold text-slate-900">{evt.event}</span>
                    <span className="text-3xs font-mono text-slate-400">#{evt.event_sequence || idx}</span>
                  </div>
                  <div className="text-2xs font-mono text-slate-500 truncate">
                    {evt.candidate_id || evt.hypothesis_id || evt.run_id || 'System Event'}
                  </div>
                </button>
              ))}
            </div>

            {/* Event Detail & JSON Inspector (7 cols) */}
            <div className="lg:col-span-7 rounded-2xl border border-slate-200 bg-white p-5 shadow-xs space-y-4">
              <div className="flex items-center justify-between border-b border-slate-100 pb-2">
                <div>
                  <span className="font-mono text-xs font-bold text-slate-900">{activeEvent?.event}</span>
                  <div className="text-2xs font-mono text-slate-400">{activeEvent?.timestamp}</div>
                </div>

                <button
                  onClick={handleCopyJson}
                  className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg border border-slate-200 bg-white hover:bg-slate-50 text-xs font-semibold text-slate-700 shadow-2xs transition cursor-pointer"
                >
                  {copied ? <Check className="w-3.5 h-3.5 text-emerald-600" /> : <Copy className="w-3.5 h-3.5 text-slate-400" />}
                  <span>{copied ? 'Copied JSON!' : 'Copy JSON'}</span>
                </button>
              </div>

              <pre className="p-4 rounded-xl bg-slate-50 border border-slate-200 text-2xs font-mono text-slate-800 overflow-x-auto max-h-[420px] leading-relaxed">
                {JSON.stringify(activeEvent, null, 2)}
              </pre>
            </div>
          </div>
        </section>
      )}
    </div>
  );
};
