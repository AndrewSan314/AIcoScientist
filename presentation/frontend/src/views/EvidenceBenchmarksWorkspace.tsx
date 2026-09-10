import React, { useState, useEffect } from 'react';
import { SnapshotData, SampleItem } from '../types/mission_control';
import { PolicyTrajectoryChart } from '../components/charts/PolicyTrajectoryChart';
import { CalibrationCoverageChart } from '../components/charts/CalibrationCoverageChart';
import { SensitivityRankAgreementChart } from '../components/charts/SensitivityRankAgreementChart';
import { ElectrolyteOptimizationChart } from '../components/charts/ElectrolyteOptimizationChart';
import {
  ShieldCheck,
  CheckCircle2,
  AlertTriangle,
  Layers,
  Database,
  Search,
  Zap,
  Info,
  Award
} from 'lucide-react';

interface Props {
  data: SnapshotData;
  initialQuestionId?: number;
  controlledQuestionId?: number;
  onQuestionChange?: (questionId: number) => void;
}

export const EvidenceBenchmarksWorkspace: React.FC<Props> = ({
  data,
  initialQuestionId = 1,
  controlledQuestionId,
  onQuestionChange
}) => {
  const [internalQuestionId, setInternalQuestionId] = useState<number>(initialQuestionId);
  const activeQuestion = controlledQuestionId ?? internalQuestionId;

  const handleSelectQuestion = (id: number) => {
    setInternalQuestionId(id);
    onQuestionChange?.(id);
  };

  useEffect(() => {
    if (controlledQuestionId && controlledQuestionId !== activeQuestion) {
      setInternalQuestionId(controlledQuestionId);
    }
  }, [controlledQuestionId]);

  // Sample catalog state for Q4
  const [sampleSearch, setSampleSearch] = useState<string>('');
  const [selectedSampleId, setSelectedSampleId] = useState<string>('PG_0309');

  const samples = data.samples || [];
  const selectedSample: SampleItem | undefined =
    samples.find((s) => s.sample_id === selectedSampleId) || samples[0];

  const summaryByWorldPolicy = data.benchmarks?.summary_by_world_policy || {};
  const cleanH1 = summaryByWorldPolicy.CLEAN_WORLD_H1_PHASE_PURITY || {};
  const cleanH2 = summaryByWorldPolicy.CLEAN_WORLD_H2_COMPOSITION_HOMOGENEITY || {};
  const sensitivityValues = Object.values<any>(data.sensitivity?.aggregate_by_world_policy || {})
    .map((record) => record.HIG_rank_correlation)
    .filter((value) => typeof value === 'number');
  const xrdCalibration = data.calibration?.XRD?.['XRD.normalized_intensity_std_proxy'] || {};
  const electrolyteRegistry = data.dataset_registry?.datasets.find((dataset) => dataset.id === 'anode_free_electrolyte_screening');
  const electrolyteDiagnostic = data.electrolyte_screening?.working_set_trials?.['200'] || {};
  const cleanH1PureCost = cleanH1.PURE_HIG?.mean_measurement_cost;
  const cleanH1HybridCost = cleanH1.HYBRID?.mean_measurement_cost;
  const cleanH1CostReduction = typeof cleanH1PureCost === 'number' && cleanH1PureCost
    ? ((cleanH1PureCost - cleanH1HybridCost) / cleanH1PureCost) * 100
    : undefined;
  const hybridSimulation = data.electrolyte_simulation?.simulation_policies?.HYBRID_DEFAULT || {};

  const questions = [
    {
      id: 1,
      title: 'Q1: Controlled Inference',
      badge: typeof cleanH1.HYBRID?.recovery_rate_MAP === 'number' ? `${(cleanH1.HYBRID.recovery_rate_MAP * 100).toFixed(0)}% (Clean H1)` : 'MAP recovery not recorded',
      badgeClass: 'sci-badge-verified',
      whatThisProves: typeof cleanH1.HYBRID?.recovery_rate_MAP === 'number' && typeof cleanH2.HYBRID?.recovery_rate_MAP === 'number'
        ? `The source benchmark reports ${(cleanH1.HYBRID.recovery_rate_MAP * 100).toFixed(1)}% MAP recovery in Clean H1 and ${(cleanH2.HYBRID.recovery_rate_MAP * 100).toFixed(1)}% in Clean H2.`
        : 'Source MAP-recovery values are unavailable for the requested benchmark.',
    },
    {
      id: 2,
      title: 'Q2: Stress Robustness',
      badge: sensitivityValues.length ? `ρ = ${Math.min(...sensitivityValues).toFixed(2)}–${Math.max(...sensitivityValues).toFixed(2)}` : 'ρ not recorded',
      badgeClass: 'sci-badge-verified',
      whatThisProves: sensitivityValues.length ? `Source rank correlation ranges from ${Math.min(...sensitivityValues).toFixed(3)} to ${Math.max(...sensitivityValues).toFixed(3)} across the recorded sensitivity matrix.` : 'Source rank-correlation summary is unavailable.'
    },
    {
      id: 3,
      title: 'Q3: Policy Efficiency',
      badge: cleanH1CostReduction !== undefined ? `${cleanH1CostReduction.toFixed(1)}% H1 cost change` : 'Cost not recorded',
      badgeClass: 'sci-badge-verified',
      whatThisProves: cleanH1CostReduction !== undefined ? `In the source Clean H1 summary, HYBRID and PURE_HIG are compared at ${cleanH1CostReduction.toFixed(1)}% relative measurement-cost change; MAP recovery is shown separately.` : 'Source policy-cost summary is unavailable.'
    },
    {
      id: 4,
      title: 'Q4: Physical A-Lab Replay',
      badge: `${samples.length.toLocaleString()} Samples`,
      badgeClass: 'sci-badge-historical',
      whatThisProves: `The source-linked A-Lab catalog contains ${samples.length.toLocaleString()} samples; the displayed interval coverage comes from the calibration artifact.`
    },
    {
      id: 5,
      title: 'Q5: Combinatorial Scale',
      badge: `${electrolyteRegistry?.candidateCount?.toLocaleString() ?? 'N/A'} Formulations`,
      badgeClass: 'sci-badge-surrogate',
      whatThisProves: `The source screening diagnostic reports ${electrolyteRegistry?.candidateCount?.toLocaleString() ?? 'not recorded'} candidates, a ${electrolyteDiagnostic.working_set_size ?? electrolyteRegistry?.screenedWorkingSetCount ?? 'not recorded'}-candidate working set, and latent gap ${electrolyteDiagnostic.screening_latent_gap ?? 'not recorded'}.`
    }
  ];

  const currentQ = questions.find((q) => q.id === activeQuestion) || questions[0];

  return (
    <div className="space-y-6 pb-12 animate-fade-in">
      <section className="sci-card p-5 border-l-4 border-l-[#DC2626]">
        <div className="text-2xs font-mono uppercase tracking-wider text-[#DC2626] font-bold">STEP 5 — DOES THE STRATEGY WORK?</div>
        <h1 className="mt-1 text-lg font-bold text-[#17201F]">Evidence comes after the scientific loop</h1>
        <p className="mt-1 text-xs text-[#66706C] max-w-3xl">These views test different claims: controlled inference, robustness, policy efficiency, historical replay, and large-space surrogate optimization. They are not interchangeable evidence modes.</p>
      </section>

      {/* Compact Horizontal Question Rail (Q1 - Q5) */}
      <section className="sci-card p-2 bg-[#FCFCFA]">
        <div className="grid grid-cols-1 sm:grid-cols-5 gap-2" role="tablist">
          {questions.map((q) => {
            const isSelected = q.id === activeQuestion;
            return (
              <button
                key={q.id}
                role="tab"
                aria-selected={isSelected}
                onClick={() => handleSelectQuestion(q.id)}
                className={`p-3 rounded-xl border text-left transition cursor-pointer flex flex-col justify-between ${
                  isSelected
                    ? 'border-[#B91C1C] bg-[#FEF2F2] shadow-2xs'
                    : 'border-transparent bg-[#FCFCFA] hover:bg-[#F4F3EE]'
                }`}
              >
                <div className="flex items-center justify-between gap-1 mb-1">
                  <span className={`text-xs font-bold ${isSelected ? 'text-[#991B1B]' : 'text-[#17201F]'}`}>
                    {q.title}
                  </span>
                </div>
                <div>
                  <span className={`sci-badge ${q.badgeClass} text-3xs`}>
                    {q.badge}
                  </span>
                </div>
              </button>
            );
          })}
        </div>
      </section>

      {/* "What this proves" Lead Banner */}
      <section className="sci-card p-4 border-l-4 border-l-[#DC2626] bg-[#FCFCFA]">
        <div className="flex items-start gap-3">
          <div className="mt-0.5 w-6 h-6 rounded-lg bg-[#FEF2F2] border border-[#FECACA] flex items-center justify-center text-[#DC2626] shrink-0">
            <ShieldCheck className="w-3.5 h-3.5" />
          </div>
          <div>
            <div className="text-2xs font-mono uppercase tracking-wider text-[#DC2626] font-bold">
              What this evidence answers
            </div>
            <p className="text-sm font-semibold text-[#17201F] mt-0.5">
              {currentQ.whatThisProves}
            </p>
          </div>
        </div>
      </section>

      {/* Lead Chart (60%) + Side Evidence Panel (40%) */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Lead Chart Column */}
        <section className="lg:col-span-7 sci-card p-6 flex flex-col justify-between space-y-4">
          <div className="border-b border-[#D9DFDB] pb-3 flex items-center justify-between">
            <h2 className="text-base font-bold text-[#17201F]">
              {activeQuestion === 1 && 'Controlled Inference: Posterior Convergence'}
              {activeQuestion === 2 && 'Sensitivity: MC12 vs MC32 Rank Agreement'}
              {activeQuestion === 3 && `Policy Benchmark: Information Gain vs Cost (${data.benchmarks?.trajectory_count ?? 'N/A'} Runs)`}
              {activeQuestion === 4 && `A-Lab Empirical Calibration Coverage (${samples.length.toLocaleString()} Source Samples)`}
              {activeQuestion === 5 && `Electrolyte Surrogate Trajectories (${electrolyteRegistry?.candidateCount?.toLocaleString() ?? 'N/A'} Candidate Pool)`}
            </h2>
            <span className="text-2xs font-mono text-[#8F9995]">
              {activeQuestion === 4 ? `${samples.length.toLocaleString()} samples` : activeQuestion === 5 ? `${electrolyteRegistry?.candidateCount?.toLocaleString() ?? 'N/A'} candidates` : 'Controlled benchmark'}
            </span>
          </div>

          <div className="min-h-[340px] flex items-center justify-center">
            {activeQuestion === 1 && (
              <div className="w-full">
                <PolicyTrajectoryChart benchmarks={data.benchmarks} />
                <p className="text-xs text-[#8F9995] text-center mt-3 font-medium">
              What it shows: model support across controlled trajectories. Why it matters: controlled ground truth lets this benchmark test the decision-and-update loop; it does not prove a physical mechanism.
                </p>
              </div>
            )}

            {activeQuestion === 2 && (
              <div className="w-full">
                <SensitivityRankAgreementChart sensitivity={data.sensitivity} />
                <p className="text-xs text-[#8F9995] text-center mt-3 font-medium">
                  What it shows: whether approximate information-gain rankings stay stable when Monte Carlo sample count changes (ρ = {sensitivityValues.length ? `${Math.min(...sensitivityValues).toFixed(3)} – ${Math.max(...sensitivityValues).toFixed(3)}` : 'not recorded'}).
                </p>
              </div>
            )}

            {activeQuestion === 3 && (
              <div className="w-full">
                <PolicyTrajectoryChart benchmarks={data.benchmarks} />
                <p className="text-xs text-[#8F9995] text-center mt-3 font-medium">
                  What it shows: how decision strategies perform under the same controlled benchmark ({data.benchmarks?.trajectory_count ?? 'N/A'} recorded trajectories).
                </p>
              </div>
            )}

            {activeQuestion === 4 && (
              <div className="w-full">
                <CalibrationCoverageChart calibration={data.calibration} />
                <p className="text-xs text-[#8F9995] text-center mt-3 font-medium">
                  What it shows: whether stated uncertainty intervals match observed historical outcomes. This is calibration evidence for the source-linked A-Lab replay.
                </p>
              </div>
            )}

            {activeQuestion === 5 && (
              <div className="w-full">
                <ElectrolyteOptimizationChart simulationData={data.electrolyte_simulation} />
                <p className="text-xs text-[#8F9995] text-center mt-3 font-medium">
                  What it shows: best-found value over recorded surrogate queries after virtual-pool screening. It is a computational benchmark, not physical battery-cycling evidence.
                </p>
              </div>
            )}
          </div>
        </section>

        {/* Side Panel Column: Statistical Metrics & Scientific Provenance */}
        <section className="lg:col-span-5 sci-card p-6 flex flex-col justify-between space-y-5">
          <div className="space-y-4">
            <div className="border-b border-[#D9DFDB] pb-3 flex items-center justify-between">
              <h3 className="text-sm font-bold text-[#17201F]">Source details</h3>
              <span className="sci-badge sci-badge-verified text-3xs">Advanced evidence</span>
            </div>

            {/* Q1 Provenance */}
            {activeQuestion === 1 && (
              <div className="space-y-3 text-xs text-[#66706C]">
                <div className="p-3 rounded-lg bg-[#F4F3EE] border border-[#D9DFDB] space-y-2 font-mono text-2xs">
                  <div className="flex justify-between">
                    <span className="text-[#8F9995]">Sample size:</span>
                    <strong className="text-[#17201F]">{data.benchmarks?.trajectory_count ?? 'N/A'} source trajectories</strong>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-[#8F9995]">Recovery rate:</span>
                    <strong className="text-[#DC2626]">{cleanH1.HYBRID?.recovery_rate_MAP !== undefined ? `${(cleanH1.HYBRID.recovery_rate_MAP * 100).toFixed(1)}% MAP recovery` : 'Not recorded'}</strong>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-[#8F9995]">Mean steps to converge:</span>
                    <strong className="text-[#17201F]">{cleanH1.HYBRID?.['mean_steps_to_posterior_gt_0.8'] ?? 'Not recorded'} steps (&gt; 0.8)</strong>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-[#8F9995]">Ground truth firewall:</span>
                    <strong className="text-[#DC2626]">Source artifact boundary</strong>
                  </div>
                </div>

                <p className="leading-relaxed">
                    This is an inference-method benchmark over controlled source worlds; its results do not establish physical-world validity by themselves.
                </p>
              </div>
            )}

            {/* Q2 Provenance */}
            {activeQuestion === 2 && (
              <div className="space-y-3 text-xs text-[#66706C]">
                <div className="p-3 rounded-lg bg-[#F4F3EE] border border-[#D9DFDB] space-y-2 font-mono text-2xs">
                  <div className="flex justify-between">
                    <span className="text-[#8F9995]">Spearman correlation:</span>
                    <strong className="text-[#DC2626]">ρ ∈ [{sensitivityValues.length ? Math.min(...sensitivityValues).toFixed(3) : 'N/A'}, {sensitivityValues.length ? Math.max(...sensitivityValues).toFixed(3) : 'N/A'}]</strong>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-[#8F9995]">Monte Carlo samples:</span>
                    <strong className="text-[#17201F]">{data.sensitivity?.design?.low_samples ?? 'N/A'} vs {data.sensitivity?.design?.high_samples ?? 'N/A'} samples</strong>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-[#8F9995]">Stability decision:</span>
                    <strong className="text-[#17201F]">{data.sensitivity?.design?.high_samples ? `USE_${data.sensitivity.design.high_samples}_FOR_FULL_MATRIX` : 'Not recorded'}</strong>
                  </div>
                </div>

                <div className="p-3 rounded-lg bg-[#FEF3C7] border border-[#FDE68A] text-2xs text-[#92400E] space-y-1">
                  <div className="font-bold flex items-center gap-1">
                    <AlertTriangle className="w-3.5 h-3.5 text-[#D97706]" />
                    <span>Boundary Disclosure</span>
                  </div>
                  <p>
                    Rankings and the selected Monte Carlo sample counts are reported by the source sensitivity artifact; this panel does not infer a stronger guarantee.
                  </p>
                </div>
              </div>
            )}

            {/* Q3 Provenance */}
            {activeQuestion === 3 && (
              <div className="space-y-3 text-xs text-[#66706C]">
                <div className="overflow-x-auto rounded-lg border border-[#D9DFDB] bg-white">
                  <table className="w-full text-left text-2xs">
                    <thead className="bg-[#F4F3EE] text-[#66706C] font-mono border-b border-[#D9DFDB]">
                      <tr>
                        <th className="p-2">Policy</th>
                        <th className="p-2">Clean H1 Rec</th>
                        <th className="p-2">Mean Cost (H1)</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-[#D9DFDB] font-mono">
                      {Object.entries<any>(cleanH1).map(([policy, stats]) => (
                        <tr key={policy} className={policy === 'HYBRID' ? 'bg-[#FEF2F2] font-bold text-[#991B1B]' : ''}>
                          <td className="p-2">{policy}</td>
                          <td className="p-2">{typeof stats.recovery_rate_MAP === 'number' ? `${(stats.recovery_rate_MAP * 100).toFixed(1)}%` : 'N/A'}</td>
                          <td className="p-2">{typeof stats.mean_measurement_cost === 'number' ? stats.mean_measurement_cost.toFixed(3) : 'N/A'}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>

                <div className="p-3 rounded-lg bg-[#FEF2F2] border border-[#FECACA] text-2xs text-[#991B1B]">
                  <strong>Source comparison:</strong> The table reports the source Clean H1 policy summaries. Cost and recovery are separate recorded metrics; no weighting or causal trade-off is inferred here.
                </div>
              </div>
            )}

            {/* Q4 Provenance (A-Lab) */}
            {activeQuestion === 4 && (
              <div className="space-y-3 text-xs text-[#66706C]">
                <div className="p-3 rounded-lg bg-[#F4F3EE] border border-[#D9DFDB] space-y-2 font-mono text-2xs">
                  <div className="flex justify-between">
                    <span className="text-[#8F9995]">Dataset:</span>
                    <strong className="text-[#17201F]">A-Lab Precursor Genome</strong>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-[#8F9995]">Total physical trials:</span>
                    <strong className="text-[#17201F]">{samples.length.toLocaleString()} source records</strong>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-[#8F9995]">XRD Coverage (50% / 90%):</span>
                    <strong className="text-[#DC2626]">{xrdCalibration.coverage50 !== undefined ? `${(xrdCalibration.coverage50 * 100).toFixed(1)}%` : 'N/A'} / {xrdCalibration.coverage90 !== undefined ? `${(xrdCalibration.coverage90 * 100).toFixed(1)}%` : 'N/A'}</strong>
                  </div>
                </div>

                {/* Honest Boundary Disclaimer */}
                <div className="p-3 rounded-lg bg-[#FEF3C7] border border-[#FDE68A] text-2xs text-[#92400E] space-y-1">
                  <div className="font-bold flex items-center gap-1">
                    <AlertTriangle className="w-3.5 h-3.5 text-[#D97706]" />
                    <span>A_LAB_CALIBRATION_PARTIAL</span>
                  </div>
                  <p>
                    Empirical coverage is {xrdCalibration.coverage50 !== undefined ? `${(xrdCalibration.coverage50 * 100).toFixed(1)}%` : 'not recorded'} for nominal 50% intervals and {xrdCalibration.coverage90 !== undefined ? `${(xrdCalibration.coverage90 * 100).toFixed(1)}%` : 'not recorded'} for nominal 90% intervals. Interpretation remains bounded by the source calibration artifact.
                  </p>
                </div>

                {/* Landmark Sample Inspector */}
                <div className="space-y-2">
                  <div className="flex items-center justify-between text-2xs font-bold text-[#17201F]">
                    <span>Landmark Sample:</span>
                    <span className="font-mono text-[#DC2626]">{selectedSample?.sample_id || 'Not recorded'}</span>
                  </div>
                  <div className="p-2.5 rounded bg-[#FCFCFA] border border-[#D9DFDB] font-mono text-2xs space-y-1">
                    <div>Target: <strong className="text-[#17201F]">{selectedSample?.target_formula || 'Not recorded'}</strong></div>
                    <div>Precursors: {selectedSample?.precursors?.join(', ') || 'Not recorded'}</div>
                    <div>Reaction category: <strong className="text-[#DC2626]">{selectedSample?.reaction_category || 'Not recorded'}</strong></div>
                  </div>
                </div>
              </div>
            )}

            {/* Q5 Provenance (Electrolyte Scale) */}
            {activeQuestion === 5 && (
              <div className="space-y-3 text-xs text-[#66706C]">
                <div className="p-3 rounded-lg bg-[#F4F3EE] border border-[#D9DFDB] space-y-2 font-mono text-2xs">
                  <div className="flex justify-between">
                    <span className="text-[#8F9995]">Virtual candidate space:</span>
                    <strong className="text-[#17201F]">{electrolyteRegistry?.candidateCount?.toLocaleString() ?? 'N/A'} formulations</strong>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-[#8F9995]">Screening latency:</span>
                    <strong className="text-[#DC2626]">{electrolyteDiagnostic.screening_time_sec ?? 'Not recorded'} seconds</strong>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-[#8F9995]">Downstream candidates:</span>
                    <strong className="text-[#17201F]">{electrolyteRegistry?.screenedWorkingSetCount ?? 'Not recorded'} working-set candidates</strong>
                  </div>
                </div>

                {/* Honest Boundary Disclaimer */}
                <div className="p-3 rounded-lg bg-[#FEF3C7] border border-[#FDE68A] text-2xs text-[#92400E] space-y-1">
                  <div className="font-bold flex items-center gap-1">
                    <AlertTriangle className="w-3.5 h-3.5 text-[#D97706]" />
                    <span>OUT_OF_FAMILY_GENERALIZATION: NOT ESTABLISHED</span>
                  </div>
                  <p>
                    The source artifact describes a frozen surrogate approximation; physical generalization beyond its documented scope is not established.
                  </p>
                </div>
              </div>
            )}
          </div>
        </section>
      </div>
    </div>
  );
};
