import ast, csv, json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).with_name('exhibitionSource.json')

def rows(path):
    return list(csv.DictReader(path.open(encoding='utf-8-sig')))

result = {}
for name, directory, trajectory, policy in [
    ('drakopoulos', 'drakopoulos_rediscovery_v4', 'trajectories/AICOSCIENTIST_PROCESS_SURROGATE_trajectories.json', 'AICOSCIENTIST_PROCESS_SURROGATE'),
    ('warwick', 'warwick_nmc622_calendering', 'trajectories/aicoscientist_full_process_engine_seed_11.json', 'AICOSCIENTIST_FULL_PROCESS_ENGINE'),
]:
    base = ROOT / 'outputs' / directory
    raw = json.loads((base / trajectory).read_text())
    replay = next(x for x in raw if x['seed'] == 11) if isinstance(raw, list) else raw
    warwick = name == 'warwick'
    table = rows(base / ('experiment_table.csv' if warwick else 'eligible_recipe_table.csv'))
    candidates = []
    for row in table:
        cid = row['experiment_id' if warwick else 'recipe_id']
        controls = {k: float(row[k]) for k in (['roll_temperature_c', 'target_density_g_cm3', 'target_coating_weight_gsm'] if warwick else ['coating_speed_m_per_min', 'coating_gap_um', 'drying_temperature_c'])}
        controls.update({'loading_regime': row['loading_regime']} if warwick else {'calendering_applied': row['calendering_applied'] == 'True'})
        candidate = dict(id=cid, displayId=cid if warwick else 'Recipe ' + cid.removeprefix('protocol-')[:6], controls=controls,
            revealedTarget=dict(name='5C/0.2C rate ratio' if warwick else 'D30 Specific Capacity', value=float(row['rate_performance_5c_over_0_2c_mean' if warwick else 'mean_d30_specific_capacity_mah_g']), unit='dimensionless_ratio' if warwick else 'mAh/g'),
            replicateCount=3 if warwick else int(row['valid_d30_replicates']), cellIds=ast.literal_eval(row['replicate_cells']) if warwick else row['cell_ids'].split(';'))
        candidates.append(candidate)
    # Canonical identifiers, never historical outcome rank, determine list order.
    candidates.sort(key=lambda c: c['id'])
    by_id = {c['id']: c for c in candidates}
    steps = []
    for step in replay['steps']:
        c = by_id[step['selected_id']]
        steps.append(dict(step=step['step'], selectedCandidateId=c['id'], selectedCandidateDisplay=c['displayId'], controlsSummary={k:str(v) for k,v in c['controls'].items()}, predictedMean=step['predicted_mean'], predictedStd=step['predicted_std'], acquisitionValue=step['acquisition_value'], revealedTarget=step['revealed_target'], bestSoFar=step['best_so_far'], simpleRegret=step['simple_regret'], isOptimal=step['is_hidden_best'], explanation='Recorded seed 11 selection using expected improvement. Prediction and posterior standard deviation were saved before historical outcome reveal.'))
    summary = next(r for r in rows(base / 'policy_summary.csv') if r['policy'] == policy)
    result[name] = dict(candidates=candidates, replayInitialIds=replay['initial_candidate_ids'], replaySteps=steps, sourceArtifact=f'outputs/{directory}/{trajectory}', hitAt1Pct=100*float(summary['hit_at_1' if warwick else 'hit_rate_step_1']), hitAt3Pct=100*float(summary['hit_at_3' if warwick else 'hit_rate_step_3']))
OUT.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
