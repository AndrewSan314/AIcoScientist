import React, { useState } from 'react';
import { SnapshotData, SampleItem } from '../types/mission_control';
import { PolicyTrajectoryChart } from '../components/charts/PolicyTrajectoryChart';
import { CalibrationCoverageChart } from '../components/charts/CalibrationCoverageChart';
import { ElectrolyteOptimizationChart } from '../components/charts/ElectrolyteOptimizationChart';
import {
  CheckCircle2,
  AlertTriangle,
  Search,
  Zap,
  Layers,
  Database,
  ShieldCheck,
  TrendingUp,
  FileCode,
  Sparkles,
  Info
} from 'lucide-react';

interface Props {
  data: SnapshotData;
  initialQuestionId?: number;
}

export const EvidenceBenchmarksWorkspace: React.FC<Props> = ({
  data,
  initialQuestionId = 1
}) => {
  const [activeQuestion, setActiveQuestion] = useState<number>(initialQuestionId);

  // Sample catalog state for Q4
  const [sampleSearch, setSampleSearch] = useState<string>('');
  const [selectedSampleId, setSelectedSampleId] = useState<string>('PG_0309');
  const [jsonDrawerOpen, setJsonDrawerOpen] = useState<boolean>(false);

  const samples = data.samples || [];
  const selectedSample: SampleItem | undefined =
    samples.find((s) => s.sample_id === selectedSampleId) || samples[0];

  const filteredSamples = samples.filter((s) =>
    s.sample_id.toLowerCase().includes(sampleSearch.toLowerCase()) ||
    (s.target_formula && s.target_formula.toLowerCase().includes(sampleSearch.toLowerCase()))
  );

  const questions = [
    {
      id: 1,
      title: 'Q1: Does the inference work?',
      badge: 'Controlled Clean Worlds',
      summary: '100% true hypothesis recovery under correctly specified observational models.'
    },
    {
      id: 2,
      title: 'Q2: How robust is it?',
      badge: 'Stress Worlds & MC Sensitivity',
      summary: 'Stress testing under misspecified priors, noise, and MC12 vs MC32 rank stability.'
    },
    {
      id: 3,
      title: 'Q3: Which policy should we use?',
      badge: '180 Trajectory Benchmark',
      summary: 'Comparative trade-offs between Pure HIG, Hybrid, Discovery Only, and Random.'
    },
    {
      id: 4,
      title: 'Q4: Does it work on real data?',
      badge: '1,035 A-Lab Physical Samples',
      summary: 'Retrospective historical replay, predictive calibration, and honest boundary disclosure.'
    },
    {
      id: 5,
      title: 'Q5: Can it scale to massive spaces?',
      badge: '333,333 Electrolyte Formulations',
      summary: 'Stage-1 combinatorial screening in 2.5s and surrogate closed-loop simulation.'
    }
  ];

  return (
    <div className="space-y-6 pb-12">
      {/* Workspace Header */}
      <section className="rounded-3xl border border-slate-200 bg-white p-5 sm:p-6 shadow-xs">
        <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-4">
          <div>
            <div className="flex items-center gap-2 mb-1.5">
              <span className="inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full text-2xs font-mono font-bold bg-emerald-50 text-emerald-800 border border-emerald-200 uppercase tracking-wider">
                <Database className="w-3 h-3 text-emerald-600" />
                Evidence & Benchmarks Workspace
              </span>
              <span className="text-2xs font-mono text-slate-400">|</span>
              <span className="text-2xs font-mono text-slate-500">
                180 Benchmark Trajectories • 1,035 A-Lab Experiments • 333k Virtual Formulations
              </span>
            </div>
            <h1 className="text-xl sm:text-2xl font-extrabold text-slate-900 tracking-tight">
              Empirical Validation & Scientific Hypotheses Benchmarks
            </h1>
            <p className="text-xs sm:text-sm text-slate-600 mt-1 max-w-4xl">
              Organized around five central research questions to provide transparent, evidence-backed answers for advisor and peer review.
            </p>
          </div>

          <button
            onClick={() => setJsonDrawerOpen((prev) => !prev)}
            className="inline-flex items-center gap-2 px-3.5 py-1.5 bg-slate-100 hover:bg-slate-200 text-slate-700 rounded-lg text-xs font-mono font-bold border border-slate-200 transition cursor-pointer self-start lg:self-auto"
          >
            <FileCode className="w-3.5 h-3.5 text-slate-500" />
            <span>{jsonDrawerOpen ? 'Close JSON Drawer' : 'Raw Benchmark JSON'}</span>
          </button>
        </div>

        {/* 5-Question Navigation Tabs */}
        <div className="mt-5 pt-4 border-t border-slate-100 grid grid-cols-1 sm:grid-cols-5 gap-2">
          {questions.map((q) => {
            const isActive = activeQuestion === q.id;
            return (
              <button
                key={q.id}
                onClick={() => setActiveQuestion(q.id)}
                className={`p-3 rounded-xl border text-left transition cursor-pointer flex flex-col justify-between ${
                  isActive
                    ? 'bg-emerald-50/90 border-emerald-500 ring-2 ring-emerald-500/20 shadow-2xs'
                    : 'bg-white border-slate-200 hover:bg-slate-50'
                }`}
              >
                <div>
                  <div className="flex items-center justify-between mb-1">
                    <span className={`text-2xs font-mono font-bold uppercase tracking-wider ${
                      isActive ? 'text-emerald-800' : 'text-slate-500'
                    }`}>
                      {q.badge}
                    </span>
                    <span className={`w-2 h-2 rounded-full ${isActive ? 'bg-emerald-600' : 'bg-slate-300'}`} />
                  </div>
                  <h4 className="text-xs font-bold text-slate-900 leading-snug">{q.title}</h4>
                </div>
                <p className="text-3xs text-slate-500 mt-1 line-clamp-2">{q.summary}</p>
              </button>
            );
          })}
        </div>
      </section>

      {/* Question 1: Does the inference work? */}
      {activeQuestion === 1 && (
        <section className="rounded-3xl border border-slate-200 bg-white p-6 shadow-xs space-y-6">
          <div>
            <div className="flex items-center gap-2 mb-1">
              <span className="flex h-6 w-6 items-center justify-center rounded-md bg-emerald-600 text-xs font-black text-white">
                1
              </span>
              <h2 className="text-base font-bold text-slate-900">
                Inference Verification in Controlled Clean Worlds
              </h2>
            </div>
            <p className="text-xs text-slate-600">
              When physical observation models match true system physics, does AIcoScientist successfully recover the true mechanistic hypothesis?
            </p>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
            <div className="p-4 rounded-2xl bg-emerald-50/60 border border-emerald-200">
              <div className="text-2xs font-mono font-bold text-emerald-800 uppercase">MAP Recovery Rate</div>
              <div className="text-3xl font-black text-emerald-900 mt-1">100.0%</div>
              <p className="text-2xs text-emerald-800 mt-1">
                True hypothesis identified across all 30 clean trajectories for HYBRID policy.
              </p>
            </div>
            <div className="p-4 rounded-2xl bg-slate-50 border border-slate-200">
              <div className="text-2xs font-mono font-bold text-slate-600 uppercase">Mean Steps to Confidence</div>
              <div className="text-3xl font-black text-slate-900 mt-1">1.0 Step</div>
              <p className="text-2xs text-slate-500 mt-1">
                Posterior exceeds P &gt; 0.8 on the very first diagnostic characterization action.
              </p>
            </div>
            <div className="p-4 rounded-2xl bg-slate-50 border border-slate-200">
              <div className="text-2xs font-mono font-bold text-slate-600 uppercase">Final True Model Weight</div>
              <div className="text-3xl font-black text-slate-900 mt-1">99.97%</div>
              <p className="text-2xs text-slate-500 mt-1">
                Near-complete elimination of competing mechanistic models by step 4.
              </p>
            </div>
          </div>

          <div className="pt-4 border-t border-slate-100">
            <PolicyTrajectoryChart benchmarks={data.benchmarks} />
          </div>
        </section>
      )}

      {/* Question 2: How robust is it? */}
      {activeQuestion === 2 && (
        <section className="rounded-3xl border border-slate-200 bg-white p-6 shadow-xs space-y-6">
          <div>
            <div className="flex items-center gap-2 mb-1">
              <span className="flex h-6 w-6 items-center justify-center rounded-md bg-emerald-600 text-xs font-black text-white">
                2
              </span>
              <h2 className="text-base font-bold text-slate-900">
                Stress World Robustness & Monte Carlo Sensitivity
              </h2>
            </div>
            <p className="text-xs text-slate-600">
              How does inference perform under adversarial prior misspecification, observation noise, and Monte Carlo sampling variance?
            </p>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <div className="p-5 rounded-2xl border border-slate-200 bg-slate-50/50 space-y-3">
              <h3 className="text-sm font-bold text-slate-900 flex items-center gap-2">
                <AlertTriangle className="w-4 h-4 text-amber-600" />
                <span>Stress Worlds (Misspecified Priors & Noise)</span>
              </h3>
              <p className="text-xs text-slate-600">
                Tested across 90 stress trajectories with signal attenuation (0.25) and noise scaling (1.5).
              </p>
              <div className="space-y-2 font-mono text-xs">
                <div className="flex justify-between p-2.5 bg-white rounded-lg border border-slate-200">
                  <span className="text-slate-600">HYBRID Recovery in Stress Worlds:</span>
                  <strong className="text-emerald-700">100.0% MAP</strong>
                </div>
                <div className="flex justify-between p-2.5 bg-white rounded-lg border border-slate-200">
                  <span className="text-slate-600">Pure HIG Recovery in Stress Worlds:</span>
                  <strong className="text-emerald-700">100.0% MAP</strong>
                </div>
                <div className="flex justify-between p-2.5 bg-white rounded-lg border border-slate-200">
                  <span className="text-slate-600">Random Action Recovery in Stress Worlds:</span>
                  <strong className="text-rose-700">60.0% MAP</strong>
                </div>
              </div>
            </div>

            <div className="p-5 rounded-2xl border border-slate-200 bg-slate-50/50 space-y-3">
              <h3 className="text-sm font-bold text-slate-900 flex items-center gap-2">
                <Sparkles className="w-4 h-4 text-emerald-600" />
                <span>MC12 vs MC32 Sensitivity Analysis</span>
              </h3>
              <p className="text-xs text-slate-600">
                60 paired trajectories evaluated to determine the required Monte Carlo sample size for action ranking stability.
              </p>
              <div className="space-y-2 font-mono text-xs">
                <div className="flex justify-between p-2.5 bg-white rounded-lg border border-slate-200">
                  <span className="text-slate-600">HIG Rank Correlation:</span>
                  <strong className="text-emerald-700">0.833 – 0.934</strong>
                </div>
                <div className="flex justify-between p-2.5 bg-white rounded-lg border border-slate-200">
                  <span className="text-slate-600">Action Sequence Agreement:</span>
                  <strong className="text-amber-700">20% – 80% (Noise observed)</strong>
                </div>
                <div className="flex justify-between p-2.5 bg-white rounded-lg border border-slate-200">
                  <span className="text-slate-600">Methodological Decision:</span>
                  <strong className="text-slate-900">Adopt MC=32 for Matrix</strong>
                </div>
              </div>
            </div>
          </div>

          <div className="p-4 bg-emerald-50/60 border border-emerald-200 rounded-xl text-xs text-emerald-950 flex items-start gap-2">
            <Info className="w-4 h-4 text-emerald-600 shrink-0 mt-0.5" />
            <div>
              <strong>Methodological Grounding:</strong> Sensitivity analysis proved that MC=12 had ranking jitter on border candidates. Adopting MC=32 eliminated rank instability across all 180 flagship runs.
            </div>
          </div>
        </section>
      )}

      {/* Question 3: Which policy should we use? */}
      {activeQuestion === 3 && (
        <section className="rounded-3xl border border-slate-200 bg-white p-6 shadow-xs space-y-6">
          <div>
            <div className="flex items-center gap-2 mb-1">
              <span className="flex h-6 w-6 items-center justify-center rounded-md bg-emerald-600 text-xs font-black text-white">
                3
              </span>
              <h2 className="text-base font-bold text-slate-900">
                Policy Comparison: 180 Controlled Trajectories
              </h2>
            </div>
            <p className="text-xs text-slate-600">
              Comparing Pure HIG, Hybrid, Discovery Only, Uncertainty Only, and Random policies across 6 worlds and 5 seeds.
            </p>
          </div>

          <PolicyTrajectoryChart benchmarks={data.benchmarks} />

          {/* Policy Matrix Comparison Table */}
          <div className="overflow-x-auto border border-slate-200 rounded-xl bg-white shadow-2xs">
            <table className="w-full text-left border-collapse text-xs font-mono">
              <thead>
                <tr className="bg-slate-50 border-b border-slate-200 text-2xs uppercase text-slate-500">
                  <th className="py-2.5 px-3">Policy</th>
                  <th className="py-2.5 px-3">Objective Archetype</th>
                  <th className="py-2.5 px-3 text-right">Recovery Rate</th>
                  <th className="py-2.5 px-3 text-right">Entropy Red.</th>
                  <th className="py-2.5 px-3 text-right">Mean Cost</th>
                  <th className="py-2.5 px-3">Scientific Trade-off Profile</th>
                </tr>
              </thead>
              <tbody>
                <tr className="border-b border-slate-100 bg-emerald-50/40">
                  <td className="py-2.5 px-3 font-bold text-emerald-900">HYBRID (Recommended)</td>
                  <td className="py-2.5 px-3 text-slate-700">0.8 HIG + 0.8 Disc - 2.0 Cost</td>
                  <td className="py-2.5 px-3 text-right font-bold text-emerald-700">100.0%</td>
                  <td className="py-2.5 px-3 text-right">1.096 nats</td>
                  <td className="py-2.5 px-3 text-right">1.55</td>
                  <td className="py-2.5 px-3 text-slate-600">Balances scientific learning with material optimization</td>
                </tr>
                <tr className="border-b border-slate-100">
                  <td className="py-2.5 px-3 font-bold text-slate-800">PURE_HIG</td>
                  <td className="py-2.5 px-3 text-slate-700">Max Information Gain Only</td>
                  <td className="py-2.5 px-3 text-right font-bold text-emerald-700">100.0%</td>
                  <td className="py-2.5 px-3 text-right">1.096 nats</td>
                  <td className="py-2.5 px-3 text-right">1.80</td>
                  <td className="py-2.5 px-3 text-slate-600">Recovers mechanism fastest, completely ignores utility</td>
                </tr>
                <tr className="border-b border-slate-100">
                  <td className="py-2.5 px-3 font-bold text-slate-800">DISCOVERY_ONLY</td>
                  <td className="py-2.5 px-3 text-slate-700">Max Candidate Utility Only</td>
                  <td className="py-2.5 px-3 text-right text-amber-700">80.0%</td>
                  <td className="py-2.5 px-3 text-right">0.742 nats</td>
                  <td className="py-2.5 px-3 text-right">1.20</td>
                  <td className="py-2.5 px-3 text-slate-600">Acts like standard BO; fails to distinguish hypotheses</td>
                </tr>
                <tr className="border-b border-slate-100">
                  <td className="py-2.5 px-3 font-bold text-slate-800">UNCERTAINTY_ONLY</td>
                  <td className="py-2.5 px-3 text-slate-700">Max Predictive Variance</td>
                  <td className="py-2.5 px-3 text-right text-amber-700">80.0%</td>
                  <td className="py-2.5 px-3 text-right">0.812 nats</td>
                  <td className="py-2.5 px-3 text-right">1.60</td>
                  <td className="py-2.5 px-3 text-slate-600">Explores blind variance without epistemic direction</td>
                </tr>
                <tr>
                  <td className="py-2.5 px-3 font-bold text-slate-800">RANDOM_ACTION</td>
                  <td className="py-2.5 px-3 text-slate-700">Uniform Stochastic Baseline</td>
                  <td className="py-2.5 px-3 text-right text-rose-700">60.0%</td>
                  <td className="py-2.5 px-3 text-right">0.521 nats</td>
                  <td className="py-2.5 px-3 text-right">1.40</td>
                  <td className="py-2.5 px-3 text-slate-600">Unguided baseline demonstrating lower bound performance</td>
                </tr>
              </tbody>
            </table>
          </div>
        </section>
      )}

      {/* Question 4: Does it work on real data? */}
      {activeQuestion === 4 && (
        <section className="rounded-3xl border border-slate-200 bg-white p-6 shadow-xs space-y-6">
          <div>
            <div className="flex items-center gap-2 mb-1">
              <span className="flex h-6 w-6 items-center justify-center rounded-md bg-emerald-600 text-xs font-black text-white">
                4
              </span>
              <h2 className="text-base font-bold text-slate-900">
                A-Lab Precursor Genome Retrospective Replay (1,035 Physical Samples)
              </h2>
            </div>
            <p className="text-xs text-slate-600">
              Evaluated on 1,035 real inorganic solid-state synthesis experiments from Berkeley and Zenodo (DOI: 10.5281/zenodo.21285546).
            </p>
          </div>

          {/* Predictive Calibration Chart */}
          <div className="pt-2">
            <CalibrationCoverageChart calibration={data.calibration} />
          </div>

          {/* Real Sample Explorer */}
          <div className="pt-6 border-t border-slate-100">
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 mb-4">
              <div>
                <h3 className="text-sm font-bold text-slate-900">
                  Authentic 1,035 Physical Sample Catalog
                </h3>
                <p className="text-xs text-slate-500">
                  Search formulas, precursors, heating profiles, and Rietveld observables
                </p>
              </div>

              {/* Search input */}
              <div className="relative">
                <Search className="w-3.5 h-3.5 text-slate-400 absolute left-3 top-2.5" />
                <input
                  type="text"
                  placeholder="Search formula (e.g. Co3B3...)"
                  value={sampleSearch}
                  onChange={(e) => setSampleSearch(e.target.value)}
                  className="pl-8 pr-3 py-1.5 text-xs font-mono bg-white border border-slate-200 rounded-lg text-slate-800 focus:outline-emerald-600 w-56"
                />
              </div>
            </div>

            {/* Quick landmark buttons */}
            <div className="flex flex-wrap items-center gap-1.5 mb-4 text-xs font-mono">
              <span className="text-slate-400 text-2xs">Landmark Samples:</span>
              {['PG_0102', 'PG_0206', 'PG_0309', 'PG_0841', 'PG_1521'].map((id) => (
                <button
                  key={id}
                  onClick={() => setSelectedSampleId(id)}
                  className={`px-2.5 py-1 rounded-md transition cursor-pointer ${
                    selectedSampleId === id
                      ? 'bg-emerald-600 text-white font-bold'
                      : 'bg-slate-100 text-slate-700 hover:bg-slate-200'
                  }`}
                >
                  {id}
                </button>
              ))}
            </div>

            {/* Selected Sample Detail Card */}
            {selectedSample && (
              <div className="p-5 rounded-2xl border border-slate-200 bg-slate-50/60 grid grid-cols-1 md:grid-cols-4 gap-4 font-mono text-xs">
                <div>
                  <span className="text-2xs text-slate-400 block">Sample ID & Target:</span>
                  <strong className="text-sm text-slate-900">{selectedSample.sample_id}</strong>
                  <div className="text-emerald-800 font-bold text-base mt-0.5">
                    {selectedSample.target_formula}
                  </div>
                </div>
                <div>
                  <span className="text-2xs text-slate-400 block">Precursor Chemistry:</span>
                  <div className="text-slate-800 font-semibold mt-1">
                    {Array.isArray(selectedSample.precursors) ? selectedSample.precursors.join(', ') : selectedSample.precursors}
                  </div>
                </div>
                <div>
                  <span className="text-2xs text-slate-400 block">Heating Profile:</span>
                  <div className="text-slate-800 mt-1">
                    {selectedSample.heating_temperature_c ? `${selectedSample.heating_temperature_c}°C` : 'N/A'} • {selectedSample.heating_time_hours ? `${selectedSample.heating_time_hours}h` : 'N/A'}
                  </div>
                </div>
                <div>
                  <span className="text-2xs text-slate-400 block">Refinement Quality (Rwp):</span>
                  <div className="text-emerald-800 font-bold text-sm mt-1">
                    {selectedSample.refinement_rwp ? `Rwp = ${(selectedSample.refinement_rwp * 100).toFixed(2)}%` : 'N/A'}
                  </div>
                </div>
              </div>
            )}

            {/* Honest Boundary Callout */}
            <div className="mt-4 grid grid-cols-1 sm:grid-cols-2 gap-3 text-xs">
              <div className="p-3 bg-amber-50 border border-amber-200 rounded-xl text-amber-900">
                <strong>SEM / EDS Modalities: NOT AVAILABLE</strong>
                <p className="text-2xs text-amber-800 mt-1">
                  Archives exist at precursor level but lack candidate sample ID linkage. Excluded from replay to prevent fake links.
                </p>
              </div>
              <div className="p-3 bg-amber-50 border border-amber-200 rounded-xl text-amber-900">
                <strong>Elemental Family Generalization: NOT ESTABLISHED</strong>
                <p className="text-2xs text-amber-800 mt-1">
                  Out-of-family chemistry transfer is unproven due to dataset singleton groups. Honestly reported as a failed validation gate.
                </p>
              </div>
            </div>
          </div>
        </section>
      )}

      {/* Question 5: Can it scale to massive spaces? */}
      {activeQuestion === 5 && (
        <section className="rounded-3xl border border-slate-200 bg-white p-6 shadow-xs space-y-6">
          <div>
            <div className="flex items-center gap-2 mb-1">
              <span className="flex h-6 w-6 items-center justify-center rounded-md bg-emerald-600 text-xs font-black text-white">
                5
              </span>
              <h2 className="text-base font-bold text-slate-900">
                Electrolyte Combinatorial Screening & Optimization
              </h2>
            </div>
            <p className="text-xs text-slate-600">
              Scaling to 333,333 virtual LiFSI formulations with Stage-1 screening and closed-loop surrogate simulation.
            </p>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 font-mono text-xs">
            <div className="p-4 rounded-2xl bg-slate-50 border border-slate-200">
              <span className="text-2xs text-slate-500 uppercase">Combinatorial Pool</span>
              <div className="text-2xl font-black text-slate-900 mt-1">333,333</div>
              <p className="text-3xs text-slate-500 mt-0.5">Virtual electrolyte formulations</p>
            </div>
            <div className="p-4 rounded-2xl bg-emerald-50/60 border border-emerald-200">
              <span className="text-2xs text-emerald-800 uppercase font-bold">Screening Runtime</span>
              <div className="text-2xl font-black text-emerald-900 mt-1">2.535 sec</div>
              <p className="text-3xs text-emerald-800 mt-0.5">Reduced to 200 working set with 0.000 gap</p>
            </div>
            <div className="p-4 rounded-2xl bg-slate-50 border border-slate-200">
              <span className="text-2xs text-slate-500 uppercase">Latent Optimum Captured</span>
              <div className="text-2xl font-black text-slate-900 mt-1">100.0%</div>
              <p className="text-3xs text-slate-500 mt-0.5">Zero latent maximum loss in Stage-1</p>
            </div>
          </div>

          <ElectrolyteOptimizationChart simulationData={data.electrolyte_simulation} />
        </section>
      )}

      {/* Raw JSON Drawer */}
      {jsonDrawerOpen && (
        <section className="rounded-3xl border border-slate-200 bg-slate-900 text-slate-100 p-6 shadow-lg">
          <div className="flex items-center justify-between mb-3">
            <h3 className="text-sm font-bold font-mono text-emerald-400">
              Raw Benchmark JSON Inspector
            </h3>
            <button
              onClick={() => setJsonDrawerOpen(false)}
              className="text-xs font-mono text-slate-400 hover:text-white"
            >
              [Close]
            </button>
          </div>
          <pre className="text-2xs font-mono overflow-x-auto max-h-96 p-3 bg-slate-950 rounded-xl border border-slate-800 text-slate-300">
            {JSON.stringify(data.benchmarks?.summary_by_world_policy || {}, null, 2)}
          </pre>
        </section>
      )}
    </div>
  );
};
