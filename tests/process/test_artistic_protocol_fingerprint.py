import pytest

from src.process.simulators.artistic.config import ArtisticRunConfig, FidelityMode, ShortHorizonProtocol, physics_config_fingerprint


def test_compressed_ramp_is_explicit_and_changes_the_physics_fingerprint():
    short_500k = ArtisticRunConfig(fidelity_mode=FidelityMode.SHORT_HORIZON, slurry_steps=500_000)
    short_1m = ArtisticRunConfig(fidelity_mode=FidelityMode.SHORT_HORIZON, slurry_steps=1_000_000)
    assert short_500k.effective_short_horizon_protocol == ShortHorizonProtocol.COMPRESSED_RAMP
    assert short_500k.protocol_schedule["ramp_steps"] == 500_000
    assert physics_config_fingerprint(recipe_fingerprint="recipe", source_commit="commit", source_tree_hash="tree", protocol_schedule=short_500k.protocol_schedule) != physics_config_fingerprint(recipe_fingerprint="recipe", source_commit="commit", source_tree_hash="tree", protocol_schedule=short_1m.protocol_schedule)


def test_reference_schedule_truncation_fails_closed_until_the_pinned_runner_can_represent_it():
    with pytest.raises(ValueError, match="not implementable"):
        ArtisticRunConfig(fidelity_mode=FidelityMode.SHORT_HORIZON, slurry_steps=500_000, short_horizon_protocol=ShortHorizonProtocol.REFERENCE_SCHEDULE_TRUNCATION)
