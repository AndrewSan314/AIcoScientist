from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from src.science.ledger import ExperimentLedger
from src.science.records import ExperimentStage, ScientificExperimentRecord


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
