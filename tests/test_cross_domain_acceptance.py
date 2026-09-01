import json
import os
import numpy as np
import pytest

from src.domains.alab.adapter import ALabDomainAdapter
from src.domains.auirh.adapter import AuIrRhDomainAdapter
from src.domains.toy_material.adapter import ToyMaterialDomainAdapter
from src.science.decision_engine import ScientificDecisionEngine


def test_same_engine_runs_auirh_toy_and_alab(tmp_path):
    """CROSS-DOMAIN ACCEPTANCE TEST:
    Verifies that the EXACT same ScientificDecisionEngine class runs three distinct material domains:
    1. Au-Ir-Rh thin-film catalyst discovery (AuIrRhDomainAdapter)
    2. Synthetic battery cathode testbed (ToyMaterialDomainAdapter)
    3. Real A-Lab Precursor Genome solid-state synthesis (ALabDomainAdapter)
    with different schemas, hypotheses, objectives, and actions.
    """
    # 1. AuIrRh Engine
    adapter_auirh = AuIrRhDomainAdapter()
    engine_auirh = ScientificDecisionEngine(domain=adapter_auirh, seed=42)
    init_auirh = adapter_auirh.get_default_initial_actions(n_property=3, n_characterization=3, seed=42)
    engine_auirh.initialize(init_auirh)
    rec_auirh = engine_auirh.propose_next_experiment()
    out_auirh = engine_auirh.execute_recommendation(rec_auirh)
    assert out_auirh is not None
    assert engine_auirh.domain_id == "auirh"

    # 2. Toy Material Engine
    adapter_toy = ToyMaterialDomainAdapter()
    engine_toy = ScientificDecisionEngine(domain=adapter_toy, seed=42)
    init_toy = adapter_toy.get_default_initial_actions(n_cap=3, n_sem=3, seed=42)
    engine_toy.initialize(init_toy)
    rec_toy = engine_toy.propose_next_experiment()
    out_toy = engine_toy.execute_recommendation(rec_toy)
    assert out_toy is not None
    assert engine_toy.domain_id == "toy_material"

    # 3. A-Lab Precursor Genome Engine (using lightweight fixture)
    fixture_dir = "tests/fixtures/alab"
    samples_file = os.path.join(fixture_dir, "samples.json")
    with open(samples_file, "r", encoding="utf-8") as f:
        samples = json.load(f)["samples"]

    cache_dir = str(tmp_path / "alab_cross_cache")
    adapter_alab = ALabDomainAdapter(
        data_dir=fixture_dir,
        cache_dir=cache_dir,
        samples=samples,
        min_pca_samples=2,
    )
    engine_alab = ScientificDecisionEngine(domain=adapter_alab, seed=42)
    init_alab = adapter_alab.get_default_initial_actions(n_candidates=2, seed=42)
    engine_alab.initialize(init_alab)
    rec_alab = engine_alab.propose_next_experiment()
    out_alab = engine_alab.execute_recommendation(rec_alab)
    assert out_alab is not None
    assert engine_alab.domain_id == "alab_precursor_genome"

    # Verify diversity of domain schemas and hypotheses across engines
    assert engine_auirh.objectives[0].name == "k0"
    assert engine_toy.objectives[0].name == "capacity"
    assert engine_alab.objectives[0].name == "reaction_conversion"

    assert list(engine_auirh.ensemble.hypotheses.keys()) == ["H1", "H2", "H3"]
    assert list(engine_toy.ensemble.hypotheses.keys()) == [
        "composition_only",
        "temperature_mediated",
        "microstructure_informed",
    ]
    assert list(engine_alab.ensemble.hypotheses.keys()) == [
        "precursor_thermodynamics",
        "process_kinetics",
        "structure_phase_informed",
    ]
