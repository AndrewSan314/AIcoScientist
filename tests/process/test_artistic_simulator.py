from pathlib import Path

import pytest

from src.process.simulators.artistic import ArtisticRecipe, ArtisticRunConfig, ArtisticSimulator, SlurryRecipe


def _slurry() -> SlurryRecipe:
    return SlurryRecipe(1, (5.0,) * 10, (1.0,) + (0.0,) * 9, 0.1, 0.5, 1.0, 0.9, 0.1, 0, 0.5)


def test_artistic_renderer_applies_only_verified_workspace_patch(tmp_path: Path) -> None:
    source = tmp_path / "source" / "NMC" / "Updated version" / "Slurry"
    source.mkdir(parents=True)
    (source / "user_inputs.txt").write_text("variable nAM_part equal @nAM_part@\n", encoding="utf-8")
    source_init = "\n".join(f"variable n_AM{i} equal round(v_n_AM*v_p_AM6)" for i in range(7, 11))
    (source / "init_structure.txt").write_text(source_init, encoding="utf-8")
    simulator = ArtisticSimulator(ArtisticRunConfig(source_root=tmp_path / "source", output_root=tmp_path / "runs"))

    run = simulator.prepare(ArtisticRecipe(slurry=_slurry()), run_id="render")

    rendered = (run / "workspace" / "init_structure.txt").read_text(encoding="utf-8")
    assert "v_p_AM7" in rendered and "v_p_AM10" in rendered
    assert "v_p_AM6" in (source / "init_structure.txt").read_text(encoding="utf-8")


def test_artistic_rejects_non_source_particle_distribution() -> None:
    with pytest.raises(ValueError, match="sum to one"):
        SlurryRecipe(2, (5.0,) * 10, (0.4, 0.4) + (0.0,) * 8, 0.1, 0.5, 1.0, 0.9, 0.1, 0, 0.5)
