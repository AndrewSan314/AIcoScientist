import React, { useState } from 'react';
import { SnapshotData } from '../types/mission_control';
import { ModeBadge } from '../components/ModeBadge';
import { 
  GitBranch, 
  Layers, 
  Database, 
  Code, 
  Search, 
  Copy, 
  Check, 
  X, 
  FileCode, 
  Clock, 
  ShieldCheck,
  Cpu,
  Lock,
  Unlock,
  CheckCircle2
} from 'lucide-react';

interface ArchitectureViewProps {
  data: SnapshotData;
}

export const ArchitectureView: React.FC<ArchitectureViewProps> = ({ data }) => {
  const [activeSubTab, setActiveSubTab] = useState<'architecture' | 'ledger'>('architecture');
  const [selectedEventIndex, setSelectedEventIndex] = useState<number>(0);
  const [jsonModalOpen, setJsonModalOpen] = useState<boolean>(false);
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
    <div className="space-y-8 pb-16 animate-fadeIn">
      {/* Top Banner */}
      <div className="border-b border-slate-200 pb-6 pt-2">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
          <div>
            <div className="flex items-center gap-2 mb-1">
              <h1 className="text-2xl font-bold text-slate-900 tracking-tight">Cross-Domain Architecture & Evidence Ledger</h1>
              <ModeBadge mode="LIVE_COMPUTED" size="sm" />
            </div>
            <p className="text-xs text-slate-500">
              Universal scientific decision abstractions & cryptographically traceable evidence ledger
            </p>
          </div>

          <div className="flex items-center bg-slate-100 p-1 rounded-lg border border-slate-200 text-xs">
            <button
              onClick={() => setActiveSubTab('architecture')}
              className={`px-3 py-1.5 rounded-md font-semibold transition cursor-pointer ${
                activeSubTab === 'architecture'
                  ? 'bg-white text-slate-900 shadow-xs'
                  : 'text-slate-600 hover:text-slate-900'
              }`}
            >
              System Architecture
            </button>
            <button
              onClick={() => setActiveSubTab('ledger')}
              className={`px-3 py-1.5 rounded-md font-semibold transition cursor-pointer ${
                activeSubTab === 'ledger'
                  ? 'bg-white text-slate-900 shadow-xs'
                  : 'text-slate-600 hover:text-slate-900'
              }`}
            >
              Evidence Ledger Inspector (5,333 Events)
            </button>
          </div>
        </div>
      </div>

      {activeSubTab === 'architecture' ? (
        <>
          {/* Universal Abstractions */}
          <section className="sci-card p-6">
            <div className="mb-6">
              <h2 className="text-sm font-bold uppercase tracking-wider text-slate-900">Reusable Scientific Core Abstractions</h2>
              <p className="text-xs text-slate-500">
                AIcoScientist cleanly separates material domain representations from multi-hypothesis Bayesian decision logic
              </p>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {arch.core_abstractions.map((item) => (
                <div key={item.name} className="p-4 rounded-lg border border-slate-200 bg-slate-50/60 hover:bg-white hover:border-slate-300 transition">
                  <div className="flex items-center justify-between mb-1.5">
                    <span className="font-mono text-xs font-bold text-emerald-800 bg-emerald-50 px-2 py-0.5 rounded border border-emerald-200">
                      {item.name}
                    </span>
                    <span className="font-mono text-2xs text-slate-400">{item.file}</span>
                  </div>
                  <p className="text-xs text-slate-600 mt-2 leading-relaxed">{item.role}</p>
                </div>
              ))}
            </div>
          </section>

          {/* Integrated Domains */}
          <section className="sci-card p-6">
            <div className="mb-6">
              <h2 className="text-sm font-bold uppercase tracking-wider text-slate-900">Evaluated Scientific Domains</h2>
              <p className="text-xs text-slate-500">
                The identical decision engine drives closed-loop experiments across distinct physical systems
              </p>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
              {arch.domains.map((dom) => (
                <div key={dom.domain_id} className="p-5 rounded-lg border border-slate-200 bg-white space-y-3">
                  <div className="flex items-start justify-between">
                    <div>
                      <h3 className="font-bold text-sm text-slate-900">{dom.name}</h3>
                      <div className="text-2xs font-mono text-slate-400 mt-0.5">ID: {dom.domain_id}</div>
                    </div>
                    <span className="text-2xs font-mono px-2 py-0.5 rounded font-bold bg-slate-100 text-slate-700">
                      {dom.status}
                    </span>
                  </div>

                  <div className="text-xs text-slate-600 leading-relaxed">
                    {dom.purpose}
                  </div>

                  <div className="pt-2 border-t border-slate-100 flex flex-wrap gap-1.5 text-2xs font-mono">
                    <span className="text-slate-400 font-sans">Supported Modalities:</span>
                    {dom.modalities.map((m) => (
                      <span key={m} className="px-1.5 py-0.5 rounded bg-slate-100 text-slate-700">
                        {m}
                      </span>
                    ))}
                  </div>
                </div>
              ))}
            </div>
          </section>
        </>
      ) : (
        /* Evidence Ledger Inspector Sub-tab */
        <section className="sci-card p-6 space-y-6">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pb-4 border-b border-slate-200">
            <div>
              <h2 className="text-base font-bold text-slate-900">Immutable Evidence Ledger Inspector</h2>
              <p className="text-xs text-slate-500">
                Auditable chronological event stream guaranteeing preregistration occurs strictly prior to observation reveal
              </p>
            </div>

            <div className="flex items-center gap-3">
              <span className="text-xs font-semibold text-slate-600">Event Type:</span>
              <select
                value={eventTypeFilter}
                onChange={(e) => { setEventTypeFilter(e.target.value); setSelectedEventIndex(0); }}
                className="bg-slate-50 border border-slate-300 rounded px-2.5 py-1 text-xs font-mono"
              >
                <option value="ALL">ALL EVENTS ({ledgerEvents.length})</option>
                <option value="ACTION_SCORE_RECORD">ACTION_SCORE_RECORD</option>
                <option value="PREREGISTERED_SELECTED_ACTION">PREREGISTERED_SELECTED_ACTION</option>
                <option value="MEASUREMENT_REVEALED">MEASUREMENT_REVEALED</option>
                <option value="BELIEF_UPDATE">BELIEF_UPDATE</option>
              </select>
            </div>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
            {/* Event List (5 cols) */}
            <div className="lg:col-span-5 border border-slate-200 rounded-lg overflow-hidden flex flex-col h-[500px]">
              <div className="p-3 bg-slate-50 border-b border-slate-200 text-xs font-bold text-slate-700 flex justify-between">
                <span>Event Stream ({filteredEvents.length})</span>
                <span className="font-mono text-2xs text-slate-400">Chronological</span>
              </div>
              <div className="overflow-y-auto flex-1 divide-y divide-slate-100 text-xs">
                {filteredEvents.map((ev, idx) => {
                  const isSelected = idx === selectedEventIndex;
                  const isPrereg = ev.event === 'PREREGISTERED_SELECTED_ACTION';
                  const isReveal = ev.event === 'MEASUREMENT_REVEALED';
                  const isScore = ev.event === 'ACTION_SCORE_RECORD';

                  return (
                    <div
                      key={ev.event_sequence || idx}
                      onClick={() => setSelectedEventIndex(idx)}
                      className={`p-3 cursor-pointer transition ${
                        isSelected 
                          ? 'bg-emerald-50/70 border-l-4 border-l-emerald-600' 
                          : 'hover:bg-slate-50'
                      }`}
                    >
                      <div className="flex items-center justify-between mb-1">
                        <span className={`font-mono text-2xs font-bold ${
                          isPrereg ? 'text-amber-700' : isReveal ? 'text-blue-700' : isScore ? 'text-slate-600' : 'text-violet-700'
                        }`}>
                          #{ev.event_sequence || idx + 1} · {ev.event}
                        </span>
                        <span className="font-mono text-2xs text-slate-400">Step {ev.step}</span>
                      </div>
                      <div className="text-2xs text-slate-600 font-mono truncate">
                        Action: {ev.action?.action_type} on {ev.action?.candidate_id}
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>

            {/* Event Inspector (7 cols) */}
            <div className="lg:col-span-7 space-y-4">
              {activeEvent && (
                <div className="p-5 rounded-lg border border-slate-200 bg-slate-50/50 space-y-4">
                  <div className="flex items-center justify-between border-b border-slate-200 pb-3">
                    <div>
                      <div className="font-bold text-sm text-slate-900 flex items-center gap-2">
                        <span>{activeEvent.event}</span>
                        <span className="font-mono text-xs px-2 py-0.5 rounded bg-slate-200 text-slate-700">
                          Seq #{activeEvent.event_sequence}
                        </span>
                      </div>
                      <div className="text-2xs font-mono text-slate-400 mt-0.5 flex items-center gap-1">
                        <Clock className="w-3 h-3" />
                        <span>{activeEvent.timestamp}</span>
                      </div>
                    </div>

                    <button
                      onClick={handleCopyJson}
                      className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded bg-white border border-slate-200 hover:bg-slate-100 text-xs font-medium text-slate-700 shadow-xs transition cursor-pointer"
                    >
                      {copied ? <Check className="w-3.5 h-3.5 text-emerald-600" /> : <Copy className="w-3.5 h-3.5" />}
                      <span>{copied ? 'Copied' : 'Copy JSON'}</span>
                    </button>
                  </div>

                  {/* Summary Attributes */}
                  <div className="grid grid-cols-2 gap-3 text-xs">
                    <div className="p-3 rounded bg-white border border-slate-200">
                      <div className="text-2xs text-slate-400 uppercase font-mono">Action ID</div>
                      <div className="font-mono font-bold text-slate-900 mt-1 truncate">
                        {activeEvent.action?.action_id || 'N/A'}
                      </div>
                    </div>
                    <div className="p-3 rounded bg-white border border-slate-200">
                      <div className="text-2xs text-slate-400 uppercase font-mono">Candidate × Modality</div>
                      <div className="font-mono font-bold text-slate-900 mt-1">
                        {activeEvent.action?.action_type} on {activeEvent.action?.candidate_id}
                      </div>
                    </div>
                    {activeEvent.expected_hig_nats !== undefined && (
                      <div className="p-3 rounded bg-white border border-slate-200">
                        <div className="text-2xs text-slate-400 uppercase font-mono">Expected HIG</div>
                        <div className="font-mono font-bold text-emerald-800 mt-1">
                          {activeEvent.expected_hig_nats.toFixed(4)} nats
                        </div>
                      </div>
                    )}
                    {activeEvent.total_action_score !== undefined && (
                      <div className="p-3 rounded bg-white border border-slate-200">
                        <div className="text-2xs text-slate-400 uppercase font-mono">Total Action Score</div>
                        <div className="font-mono font-bold text-slate-900 mt-1">
                          {activeEvent.total_action_score.toFixed(4)}
                        </div>
                      </div>
                    )}
                  </div>

                  {/* Raw JSON viewer */}
                  <div className="mt-4">
                    <div className="text-2xs font-mono font-bold text-slate-500 uppercase mb-1">
                      Raw Ledger Record (Audited Invariant)
                    </div>
                    <pre className="p-3 rounded bg-slate-900 text-slate-100 font-mono text-2xs overflow-x-auto max-h-64">
                      {JSON.stringify(activeEvent, null, 2)}
                    </pre>
                  </div>
                </div>
              )}
            </div>
          </div>
        </section>
      )}
    </div>
  );
};
