from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from src.datasets.base import DatasetSpec
from src.science.ledger import ExperimentLedger
from src.science.records import ExperimentStage, ScientificExperimentRecord
from src.science.validation import InformationHorizonError


def test_ledger_append_and_reconstruction(tmp_path: Path) -> None:
    db_file = tmp_path / "test_ledger.db"
    ledger = ExperimentLedger(db_file)

    rec = ScientificExperimentRecord(
        experiment_id="EXP_001",
        candidate_id="CAND_001",
        dataset_name="battery_test",
        stage=ExperimentStage.PROPOSED,
        pre_experiment_features={"temp": 350.0, "time": 60.0},
    )

    # Record proposal
    ledger.record_proposal(rec)
    fetched = ledger.get_record("EXP_001")
    assert fetched is not None
    assert fetched.stage == ExperimentStage.PROPOSED
    assert fetched.pre_experiment_features["temp"] == 350.0

    # Transition to EXECUTED
    ledger.append_transition("EXP_001", ExperimentStage.EXECUTED, "EXECUTED", {})
    fetched = ledger.get_record("EXP_001")
    assert fetched.stage == ExperimentStage.EXECUTED

    # Transition to CHARACTERIZED
    ledger.append_transition(
        "EXP_001",
        ExperimentStage.CHARACTERIZED,
        "CHARACTERIZED",
        {"characterization": {"sem_porosity": 0.19}},
    )
    fetched = ledger.get_record("EXP_001")
    assert fetched.stage == ExperimentStage.CHARACTERIZED
    assert fetched.characterization["sem_porosity"] == 0.19

    # Transition to COMPLETED
    ledger.append_transition(
        "EXP_001",
        ExperimentStage.COMPLETED,
        "COMPLETED",
        {"performance": {"cycle_life": 1050.0}},
    )
    fetched = ledger.get_record("EXP_001")
    assert fetched.stage == ExperimentStage.COMPLETED
    assert fetched.performance["cycle_life"] == 1050.0

    # Verify cryptographic hash chain integrity
    valid, errors = ledger.verify_integrity()
    assert valid
    assert len(errors) == 0

    ledger.close()


def test_ledger_tamper_detection(tmp_path: Path) -> None:
    db_file = tmp_path / "tamper_ledger.db"
    ledger = ExperimentLedger(db_file)

    rec1 = ScientificExperimentRecord(
        experiment_id="EXP_T01",
        candidate_id="CAND_T01",
        dataset_name="battery_test",
        pre_experiment_features={"temp": 300.0},
    )
    rec2 = ScientificExperimentRecord(
        experiment_id="EXP_T02",
        candidate_id="CAND_T02",
        dataset_name="battery_test",
        pre_experiment_features={"temp": 400.0},
    )

    ledger.record_proposal(rec1)
    ledger.record_proposal(rec2)

    valid, errors = ledger.verify_integrity()
    assert valid

    ledger.close()

    # Tamper with event 1 in database directly via SQLite
    conn = sqlite3.connect(db_file)
    with conn:
        conn.execute(
            "UPDATE experiment_events SET payload_json = '{\"tampered\": true}' WHERE event_id = 1"
        )
    conn.close()

    # Reopen ledger and verify integrity fails
    tampered_ledger = ExperimentLedger(db_file)
    valid, errors = tampered_ledger.verify_integrity()
    assert not valid
    assert len(errors) > 0
    assert any("tamper detected" in err.lower() or "event_hash mismatch" in err.lower() for err in errors)
    tampered_ledger.close()


def test_ledger_monotonic_proposal_sequence(tmp_path: Path) -> None:
    db_file = tmp_path / "mono_ledger.db"
    ledger = ExperimentLedger(db_file)

    assert ledger.next_proposal_sequence() == 1

    rec1 = ScientificExperimentRecord(
        experiment_id="EXP_001",
        candidate_id="CAND_001",
        dataset_name="test_ds",
        pre_experiment_features={"x": 1.0},
    )
    ledger.record_proposal(rec1)
    assert ledger.next_proposal_sequence() == 2

    # Mark failed
    ledger.append_transition("EXP_001", ExperimentStage.FAILED, "FAILED", {"failure_reason": "synthesis error"})
    # Sequence must still be 2 (next proposal will be 2, never reusing 1)
    assert ledger.next_proposal_sequence() == 2

    rec2 = ScientificExperimentRecord(
        experiment_id="EXP_002",
        candidate_id="CAND_002",
        dataset_name="test_ds",
        pre_experiment_features={"x": 2.0},
    )
    ledger.record_proposal(rec2)
    assert ledger.next_proposal_sequence() == 3

    ledger.close()


def test_ledger_prospective_rejection_leaves_state_untouched(tmp_path: Path) -> None:
    db_file = tmp_path / "reject_ledger.db"
    ledger = ExperimentLedger(db_file)

    spec = DatasetSpec(
        name="test_ds",
        id_column="experiment_id",
        candidate_id_column="candidate_id",
        feature_columns=["x", "z"],
        target_column="y",
        objective="maximize",
        pre_experiment_features=["x"],
        candidate_variables=["x"],
        post_experiment_characterization=["z"],
        targets=["y"],
    )

    rec = ScientificExperimentRecord(
        experiment_id="EXP_001",
        candidate_id="CAND_001",
        dataset_name="test_ds",
        pre_experiment_features={"x": 1.0},
        candidate_variables={"x": 1.0},
    )
    ledger.record_proposal(rec, spec=spec)
    ledger.append_transition("EXP_001", ExperimentStage.EXECUTED, "EXECUTED", {}, spec=spec)

    cursor = ledger._conn.execute("SELECT COUNT(*) as cnt FROM experiment_events")
    initial_event_count = cursor.fetchone()["cnt"]

    # Attempt invalid transition: recording non-finite / NaN characterization
    with pytest.raises(InformationHorizonError):
        ledger.append_transition(
            "EXP_001",
            ExperimentStage.CHARACTERIZED,
            "BAD_CHAR",
            {"characterization": {"z": float("nan")}},
            spec=spec,
        )

    # Verify event count and stage are untouched
    cursor = ledger._conn.execute("SELECT COUNT(*) as cnt FROM experiment_events")
    assert cursor.fetchone()["cnt"] == initial_event_count

    rec_after = ledger.get_record("EXP_001")
    assert rec_after.stage == ExperimentStage.EXECUTED
    assert not rec_after.characterization

    ledger.close()


def test_ledger_projection_verification(tmp_path: Path) -> None:
    db_file = tmp_path / "projection_ledger.db"
    ledger = ExperimentLedger(db_file)

    rec = ScientificExperimentRecord(
        experiment_id="EXP_P01",
        candidate_id="CAND_P01",
        dataset_name="test_ds",
        pre_experiment_features={"x": 1.0},
    )
    ledger.record_proposal(rec)
    ledger.close()

    # Alter experiments table directly
    conn = sqlite3.connect(db_file)
    with conn:
        conn.execute("UPDATE experiments SET current_stage = 'COMPLETED' WHERE experiment_id = 'EXP_P01'")
    conn.close()

    tampered_ledger = ExperimentLedger(db_file)
    valid, errors = tampered_ledger.verify_integrity()
    assert not valid
    assert any("Projection mismatch" in err for err in errors)
    tampered_ledger.close()


def test_ledger_head_and_event_count_tail_truncation_detection(tmp_path: Path) -> None:
    db_file = tmp_path / "tail_trunc_ledger.db"
    ledger = ExperimentLedger(db_file)

    rec1 = ScientificExperimentRecord(
        experiment_id="EXP_001",
        candidate_id="CAND_001",
        dataset_name="test_ds",
        pre_experiment_features={"x": 1.0},
    )
    rec2 = ScientificExperimentRecord(
        experiment_id="EXP_002",
        candidate_id="CAND_002",
        dataset_name="test_ds",
        pre_experiment_features={"x": 2.0},
    )
    ledger.record_proposal(rec1)
    ledger.record_proposal(rec2)
    ledger.append_transition("EXP_001", ExperimentStage.EXECUTED, "EXEC", {})

    valid, errors = ledger.verify_integrity()
    assert valid
    ledger.close()

    # Tamper: delete the last event (event_id 3) via direct SQLite without updating ledger_metadata
    conn = sqlite3.connect(db_file)
    with conn:
        conn.execute("DELETE FROM experiment_events WHERE event_id = 3")
    conn.close()

    tampered_ledger = ExperimentLedger(db_file)
    valid, errors = tampered_ledger.verify_integrity()
    assert not valid
    assert any("tail" in err.lower() or "mismatch" in err.lower() for err in errors)
    tampered_ledger.close()


def test_ledger_optimizer_snapshot_hash_chaining_and_tamper_detection(tmp_path: Path) -> None:
    db_file = tmp_path / "snapshot_ledger.db"
    ledger = ExperimentLedger(db_file)

    rec = ScientificExperimentRecord(
        experiment_id="EXP_001",
        candidate_id="CAND_001",
        dataset_name="test_ds",
        pre_experiment_features={"x": 1.0},
    )
    ledger.record_proposal(rec)
    ledger.save_optimizer_snapshot({"step": 1, "current_best": 150.0}, experiment_id="EXP_001")

    valid, errors = ledger.verify_integrity()
    assert valid
    assert ledger.get_latest_optimizer_snapshot()["current_best"] == 150.0
    ledger.close()

    # Tamper with the snapshot event payload directly
    conn = sqlite3.connect(db_file)
    with conn:
        conn.execute(
            "UPDATE experiment_events SET payload_json = '{\"event_type\": \"OPTIMIZER_STATE_SNAPSHOT\", \"step\": 1, \"snapshot\": {\"current_best\": 9999.0}}' WHERE event_type = 'OPTIMIZER_STATE_SNAPSHOT'"
        )
    conn.close()

    tampered_ledger = ExperimentLedger(db_file)
    valid, errors = tampered_ledger.verify_integrity()
    assert not valid
    assert any("tamper detected" in err.lower() or "event_hash mismatch" in err.lower() for err in errors)
    tampered_ledger.close()


def test_atomic_proposal_sequence_survives_reopen(tmp_path: Path) -> None:
    db_file = tmp_path / "atomic_seq_ledger.db"
    ledger = ExperimentLedger(db_file)

    seq1 = ledger.allocate_proposal_sequence("battery_ds")
    assert seq1 == 1

    seq2 = ledger.allocate_proposal_sequence("battery_ds")
    assert seq2 == 2

    ledger.close()

    # Reopen ledger
    reopened = ExperimentLedger(db_file)
    assert reopened.next_proposal_sequence("battery_ds") == 3

    seq3 = reopened.allocate_proposal_sequence("battery_ds")
    assert seq3 == 3
    assert reopened.next_proposal_sequence("battery_ds") == 4
    reopened.close()


def test_commit_proposal_transaction_atomic_rollback(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    db_file = tmp_path / "rollback_ledger.db"
    ledger = ExperimentLedger(db_file)

    # Simulate failure in the middle of commit_proposal_transaction (e.g. invalid spec validation)
    bad_spec = DatasetSpec(
        name="battery_ds",
        id_column="exp_id",
        candidate_id_column="cand_id",
        feature_columns=["x1"],
        target_column="y",
        pre_experiment_features=["x1"],
        candidate_variables=["x1"],
        targets=["y"],
    )

    with pytest.raises(InformationHorizonError):
        ledger.commit_proposal_transaction(
            dataset_name="battery_ds",
            candidate_id="CAND_001",
            pre_experiment_features={"x1": 1.0, "unexpected_feat": 99.0},  # Fails spec validation
            candidate_variables={"x1": 1.0},
            proposal_metadata_builder=lambda eid, seq: {"step": 1},
            optimizer_snapshot={"step": 1, "current_best": 100.0},
            spec=bad_spec,
        )

    # Assert complete rollback: sequence was NOT consumed, no experiments, no events, no snapshots
    assert ledger.next_proposal_sequence("battery_ds") == 1
    cursor = ledger._conn.execute("SELECT COUNT(*) as cnt FROM experiments WHERE experiment_id != 'SYSTEM_OPTIMIZER'")
    assert cursor.fetchone()["cnt"] == 0
    cursor = ledger._conn.execute("SELECT COUNT(*) as cnt FROM experiment_events")
    assert cursor.fetchone()["cnt"] == 0
    cursor = ledger._conn.execute("SELECT COUNT(*) as cnt FROM optimizer_snapshots")
    assert cursor.fetchone()["cnt"] == 0

    valid, errors = ledger.verify_integrity()
    assert valid

    # Now make valid proposal transaction
    rec = ledger.commit_proposal_transaction(
        dataset_name="battery_ds",
        candidate_id="CAND_001",
        pre_experiment_features={"x1": 1.0},
        candidate_variables={"x1": 1.0},
        proposal_metadata_builder=lambda eid, seq: {"step": 1, "proposal_sequence": seq},
        optimizer_snapshot={"step": 1, "current_best": 100.0},
        spec=bad_spec,
    )
    assert rec.experiment_id == "EXP_BATTER_001"
    assert ledger.next_proposal_sequence("battery_ds") == 2
    valid, errors = ledger.verify_integrity()
    assert valid
    assert len(errors) == 0

    ledger.close()


def test_cache_snapshot_tamper_detected_in_verify_integrity(tmp_path: Path) -> None:
    db_file = tmp_path / "cache_tamper.db"
    ledger = ExperimentLedger(db_file)

    rec = ledger.commit_proposal_transaction(
        dataset_name="synth_ds",
        candidate_id="CAND_001",
        pre_experiment_features={"x": 5.0},
        candidate_variables={"x": 5.0},
        proposal_metadata_builder=lambda eid, seq: {},
        optimizer_snapshot={"step": 1, "current_best": 42.0},
    )
    valid, errors = ledger.verify_integrity()
    assert valid
    ledger.close()

    # Tamper ONLY the optimizer_snapshots cache table
    conn = sqlite3.connect(db_file)
    with conn:
        conn.execute("UPDATE optimizer_snapshots SET snapshot_json = '{\"step\": 1, \"current_best\": 99999.0}'")
    conn.close()

    tampered_ledger = ExperimentLedger(db_file)
    valid, errors = tampered_ledger.verify_integrity()
    assert not valid
    assert any("projection mismatch" in err.lower() or "tamper detected" in err.lower() for err in errors)

    # Authoritative snapshot from hash-chained events remains untampered
    authoritative_snap = tampered_ledger.get_latest_verified_optimizer_snapshot()
    assert authoritative_snap["current_best"] == 42.0
    tampered_ledger.close()

