import React from 'react';
import { X, ExternalLink, ShieldCheck, AlertTriangle, CheckCircle, Database } from 'lucide-react';
import { ExhibitionScenario } from '../../data/types';

interface EvidenceLimitationsDrawerProps {
  isOpen: boolean;
  onClose: () => void;
  scenario: ExhibitionScenario;
}

export const EvidenceLimitationsDrawer: React.FC<EvidenceLimitationsDrawerProps> = ({
  isOpen,
  onClose,
  scenario
}) => {
  if (!isOpen) return null;

  const prov = scenario.provenance;
  const bm = scenario.benchmark;

  return (
    <div className="fixed inset-0 z-50 pointer-events-auto bg-black/40 backdrop-blur-xs flex justify-end animate-in fade-in duration-200">
      <div className="w-full max-w-xl bg-white h-full shadow-2xl p-8 overflow-y-auto space-y-6 flex flex-col justify-between animate-in slide-in-from-right duration-300">
        <div className="space-y-6">
          {/* Header */}
          <div className="flex items-center justify-between pb-4 border-b border-slate-100">
            <div className="flex items-center space-x-2.5">
              <div className="p-2 rounded-xl bg-[#E8F4F2] text-[#087F8C]">
                <ShieldCheck className="w-5 h-5" />
              </div>
              <div>
                <h3 className="text-lg font-bold text-[#142A35]">Scientific Provenance & Audit</h3>
                <p className="text-xs text-slate-500">Dataset Verification & Capability Boundaries</p>
              </div>
            </div>

            <button
              onClick={onClose}
              className="p-2 text-slate-400 hover:text-slate-700 hover:bg-slate-100 rounded-xl transition-all cursor-pointer"
            >
              <X className="w-5 h-5" />
            </button>
          </div>

          {/* Provenance Card */}
          <div className="p-5 rounded-2xl bg-slate-50 border border-slate-100 space-y-3">
            <div className="flex items-center space-x-2 text-xs font-bold text-[#142A35]">
              <Database className="w-4 h-4 text-[#087F8C]" />
              <span>Source Dataset Provenance</span>
            </div>

            <div className="space-y-2 text-xs">
              <div>
                <span className="text-slate-400 block text-[10px] uppercase font-bold">Paper Title</span>
                <span className="font-semibold text-slate-800 leading-snug">{prov.title}</span>
              </div>

              <div className="grid grid-cols-2 gap-3 pt-1">
                <div>
                  <span className="text-slate-400 block text-[10px] uppercase font-bold">Authors</span>
                  <span className="text-slate-700 font-medium">{prov.authors}</span>
                </div>
                <div>
                  <span className="text-slate-400 block text-[10px] uppercase font-bold">Journal & Year</span>
                  <span className="text-slate-700 font-medium">{prov.journal} ({prov.year})</span>
                </div>
              </div>

              <div className="grid grid-cols-2 gap-3 pt-1">
                <div>
                  <span className="text-slate-400 block text-[10px] uppercase font-bold">Data License</span>
                  <span className="text-slate-700 font-medium">{prov.license}</span>
                </div>
                <div>
                  <span className="text-slate-400 block text-[10px] uppercase font-bold">Evidence Kind</span>
                  <span className="font-mono text-[#087F8C] font-semibold">{scenario.evidenceKind}</span>
                </div>
              </div>

              <div className="pt-2 border-t border-slate-200">
                <a
                  href={`https://doi.org/${prov.doi}`}
                  target="_blank"
                  rel="noreferrer"
                  className="inline-flex items-center space-x-1.5 text-xs text-[#087F8C] font-semibold hover:underline"
                >
                  <span>DOI: {prov.doi}</span>
                  <ExternalLink className="w-3.5 h-3.5" />
                </a>
              </div>
            </div>
          </div>

          {/* Replicate & Sample Integrity */}
          <div className="p-5 rounded-2xl bg-slate-50 border border-slate-100 space-y-2 text-xs">
            <div className="font-bold text-[#142A35] flex items-center space-x-2">
              <CheckCircle className="w-4 h-4 text-emerald-600" />
              <span>Experimental Sample & Replicate Integrity</span>
            </div>
            <p className="text-slate-600 leading-relaxed">
              {prov.replicateSummary}
            </p>
          </div>

          {/* Strict Known Limitations (Mandatory Scientific Honesty) */}
          <div className="p-5 rounded-2xl bg-amber-50/70 border border-amber-200 space-y-3">
            <div className="flex items-center space-x-2 text-xs font-bold text-amber-900">
              <AlertTriangle className="w-4 h-4 text-amber-600 shrink-0" />
              <span>Audited Scientific Limitations & Boundaries</span>
            </div>

            <ul className="space-y-2 text-xs text-amber-950/90 leading-relaxed list-disc list-inside">
              {bm.knownLimitations.map((lim, idx) => (
                <li key={idx} className="pl-1 text-[11px] leading-relaxed">
                  {lim}
                </li>
              ))}
            </ul>
          </div>

          {/* Permitted Scientific Wording Guidance */}
          <div className="p-4 rounded-2xl bg-sky-50/70 border border-sky-100 text-xs text-sky-950 space-y-1">
            <div className="font-bold text-sky-900">Approved Exhibition Statement</div>
            <p className="text-[11px] leading-relaxed text-sky-800 italic">
              "{bm.allowedWording}"
            </p>
          </div>
        </div>

        {/* Dismiss Button */}
        <div className="pt-4 border-t border-slate-100">
          <button
            onClick={onClose}
            className="w-full py-3 bg-[#142A35] hover:bg-[#1f3b49] text-white rounded-xl text-xs font-semibold shadow-xs transition-all cursor-pointer"
          >
            Close Provenance Drawer
          </button>
        </div>
      </div>
    </div>
  );
};
