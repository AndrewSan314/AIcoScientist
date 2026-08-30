from __future__ import annotations

import math
from typing import Any, Mapping

import numpy as np

from src.datasets.base import DatasetSpec
from src.science.records import ExperimentStage, ScientificExperimentRecord


class InformationHorizonError(ValueError):
    """Raised when an experiment record violates information horizon boundaries."""


def _check_finite_numerical_values(data_dict: Mapping[str, Any], field_name: str) -> None:
    """Checks that numerical values are finite (not NaN, null, or Inf)."""
    for k, v in data_dict.items():
        if isinstance(v, (float, int, np.number)):
            if math.isnan(float(v)) or math.isinf(float(v)):
                raise InformationHorizonError(
                    f"Field {field_name!r} contains non-finite numerical value for {k!r}: {v}"
                )
        elif v is None:
            raise InformationHorizonError(
                f"Field {field_name!r} contains None/null value for {k!r}"
            )


def validate_record_against_spec(
    record: ScientificExperimentRecord,
    spec: DatasetSpec,
    strict_dataset_name: bool = True,
) -> None:
    """Validates a ScientificExperimentRecord against DatasetSpec information horizon boundaries."""
    if strict_dataset_name and record.dataset_name != spec.name:
        raise InformationHorizonError(
            f"Record dataset_name {record.dataset_name!r} does not match DatasetSpec name {spec.name!r}"
        )

    # 1. Oracle column firewall: Oracle columns must NEVER be visible in any user-facing field
    oracle_set = set(spec.oracle_columns)
    if oracle_set:
        for field_name, d in [
            ("pre_experiment_features", record.pre_experiment_features),
            ("candidate_variables", record.candidate_variables),
            ("characterization", record.characterization),
            ("performance", record.performance),
        ]:
            overlap = set(d.keys()) & oracle_set
            if overlap:
                raise InformationHorizonError(
                    f"Oracle leakage detected! Field {field_name!r} contains hidden oracle columns: {sorted(overlap)}"
                )

    # 2. Candidate variables must be controllable pre-experiment variables
    cand_vars = set(record.candidate_variables.keys())
    allowed_cand_vars = set(spec.candidate_variables) or set(spec.candidate_columns) or set(spec.pre_experiment_features)
    if not cand_vars.issubset(allowed_cand_vars):
        invalid_vars = cand_vars - allowed_cand_vars
        raise InformationHorizonError(
            f"Candidate variables contain non-controllable features: {sorted(invalid_vars)}. "
            f"Allowed candidate variables: {sorted(allowed_cand_vars)}"
        )

    # Validate subset relationship between candidate variables and pre-experiment features
    if spec.candidate_variables and spec.pre_experiment_features:
        if not set(spec.candidate_variables).issubset(set(spec.pre_experiment_features)):
            invalid_spec_cand = set(spec.candidate_variables) - set(spec.pre_experiment_features)
            raise InformationHorizonError(
                f"DatasetSpec violation: candidate_variables must be a subset of pre_experiment_features. "
                f"Invalid: {sorted(invalid_spec_cand)}"
            )

    # 3. Pre-experiment features check
    pre_feats = set(record.pre_experiment_features.keys())
    allowed_pre_feats = set(spec.pre_experiment_features) or set(spec.feature_columns)
    if not pre_feats.issubset(allowed_pre_feats):
        invalid_pre = pre_feats - allowed_pre_feats
        raise InformationHorizonError(
            f"Pre-experiment features contain unexpected columns: {sorted(invalid_pre)}. "
            f"Allowed: {sorted(allowed_pre_feats)}"
        )
    _check_finite_numerical_values(record.pre_experiment_features, "pre_experiment_features")
    _check_finite_numerical_values(record.candidate_variables, "candidate_variables")

    # 4. Proposal-stage Horizon Check: Characterization and Performance MUST NOT be measured at proposal time
    if record.stage in {ExperimentStage.PROPOSED, ExperimentStage.SCHEDULED}:
        if record.characterization:
            raise InformationHorizonError(
                f"Characterization measurements cannot be present at stage {record.stage.value}. "
                f"Post-experiment characterization is only available after physical execution."
            )
        if record.performance:
            raise InformationHorizonError(
                f"Performance outcomes cannot be present at stage {record.stage.value}. "
                f"Performance targets are only available after experimental measurement."
            )

    # 5. Characterization Horizon Check
    if record.characterization:
        char_keys = set(record.characterization.keys())
        allowed_chars = set(spec.post_experiment_characterization) or set(spec.optional_columns) or set(spec.feature_columns)
        if not char_keys.issubset(allowed_chars):
            invalid_chars = char_keys - allowed_chars
            raise InformationHorizonError(
                f"Characterization contains unknown channels: {sorted(invalid_chars)}. "
                f"Allowed characterization channels: {sorted(allowed_chars)}"
            )
        _check_finite_numerical_values(record.characterization, "characterization")

    # 6. Performance Horizon Check
    if record.performance:
        perf_keys = set(record.performance.keys())
        allowed_perfs = set(spec.targets) or {spec.target_column}
        if not perf_keys.issubset(allowed_perfs):
            invalid_perfs = perf_keys - allowed_perfs
            raise InformationHorizonError(
                f"Performance contains unknown targets: {sorted(invalid_perfs)}. "
                f"Allowed targets: {sorted(allowed_perfs)}"
            )
        _check_finite_numerical_values(record.performance, "performance")

    # 7. Completed Stage Check: Primary target and all required characterization channels must be present
    if record.stage == ExperimentStage.COMPLETED:
        primary_target = spec.target_column
        if primary_target not in record.performance:
            raise InformationHorizonError(
                f"Cannot mark experiment COMPLETED: primary target {primary_target!r} is missing from performance measurements."
            )
        required_chars = set(spec.post_experiment_characterization)
        if required_chars and not required_chars.issubset(set(record.characterization.keys())):
            missing_chars = required_chars - set(record.characterization.keys())
            raise InformationHorizonError(
                f"Cannot mark experiment COMPLETED: required characterization channels are missing: {sorted(missing_chars)}"
            )


def validate_transition_before_append(
    current_record: ScientificExperimentRecord,
    new_stage: ExperimentStage | str,
    delta_payload: Mapping[str, Any],
    spec: DatasetSpec | None = None,
) -> ScientificExperimentRecord:
    """Creates a prospective in-memory copy of record, validates transition and spec boundaries before any ledger commit."""
    target_stage = ExperimentStage(new_stage) if isinstance(new_stage, str) else new_stage
    prospective_record = current_record.copy()
    prospective_record.transition_to(
        new_stage=target_stage,
        characterization=delta_payload.get("characterization"),
        performance=delta_payload.get("performance"),
        measurement_uncertainty=delta_payload.get("measurement_uncertainty"),
        quality_flags=delta_payload.get("quality_flags"),
        failure_reason=delta_payload.get("failure_reason"),
        allow_measurement_revision=bool(delta_payload.get("allow_measurement_revision", False)),
    )

    if spec is not None:
        validate_record_against_spec(prospective_record, spec)

    return prospective_record
