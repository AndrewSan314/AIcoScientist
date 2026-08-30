from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from src.science.records import ExperimentStage, ScientificExperimentRecord


def _canonical_json(payload: Mapping[str, Any]) -> str:
    """Produces a deterministic, sorted, compact JSON string for cryptographic hash chaining."""
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


class ExperimentLedger:
    """Generic append-only SQLite Experiment Ledger with tamper-evident SHA-256 event hash chaining."""

    def __init__(self, db_path: Path | str = ":memory:") -> None:
        self.db_path = str(db_path)
        if self.db_path != ":memory:":
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.db_path)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        with self._conn:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS experiments (
                    experiment_id TEXT PRIMARY KEY,
                    candidate_id TEXT NOT NULL,
                    dataset_name TEXT NOT NULL,
                    current_stage TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS experiment_events (
                    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    experiment_id TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    previous_event_hash TEXT NOT NULL,
                    event_hash TEXT NOT NULL,
                    FOREIGN KEY (experiment_id) REFERENCES experiments (experiment_id)
                )
                """
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_events_exp_id ON experiment_events (experiment_id)"
            )

    def _get_latest_event_hash(self) -> str:
        cursor = self._conn.execute(
            "SELECT event_hash FROM experiment_events ORDER BY event_id DESC LIMIT 1"
        )
        row = cursor.fetchone()
        return str(row["event_hash"]) if row else ("0" * 64)

    def record_proposal(self, record: ScientificExperimentRecord) -> ScientificExperimentRecord:
        """Records a new experiment proposal event in the ledger."""
        prev_hash = self._get_latest_event_hash()
        payload = record.to_dict()
        canon_payload = _canonical_json(payload)

        hasher = hashlib.sha256()
        hasher.update(prev_hash.encode("utf-8"))
        hasher.update(canon_payload.encode("utf-8"))
        event_hash = hasher.hexdigest()

        now_iso = datetime.now(timezone.utc).isoformat()

        with self._conn:
            self._conn.execute(
                """
                INSERT INTO experiments (experiment_id, candidate_id, dataset_name, current_stage, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    record.experiment_id,
                    record.candidate_id,
                    record.dataset_name,
                    record.stage.value,
                    record.created_at,
                    now_iso,
                ),
            )
            self._conn.execute(
                """
                INSERT INTO experiment_events (experiment_id, event_type, created_at, payload_json, previous_event_hash, event_hash)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    record.experiment_id,
                    "PROPOSAL_CREATED",
                    now_iso,
                    canon_payload,
                    prev_hash,
                    event_hash,
                ),
            )
        return record

    def append_transition(
        self,
        experiment_id: str,
        new_stage: ExperimentStage | str,
        event_type: str,
        delta_payload: Mapping[str, Any],
    ) -> ScientificExperimentRecord:
        """Appends a validated lifecycle event to the ledger and updates the experiment state."""
        current_record = self.get_record(experiment_id)
        if current_record is None:
            raise KeyError(f"Experiment {experiment_id!r} not found in ledger.")

        # Apply transition in memory to validate
        target_stage = ExperimentStage(new_stage) if isinstance(new_stage, str) else new_stage
        current_record.transition_to(
            new_stage=target_stage,
            characterization=delta_payload.get("characterization"),
            performance=delta_payload.get("performance"),
            measurement_uncertainty=delta_payload.get("measurement_uncertainty"),
            quality_flags=delta_payload.get("quality_flags"),
            failure_reason=delta_payload.get("failure_reason"),
        )

        prev_hash = self._get_latest_event_hash()
        full_record_dict = current_record.to_dict()
        event_body = {
            "event_type": event_type,
            "target_stage": target_stage.value,
            "delta": dict(delta_payload),
            "record_snapshot": full_record_dict,
        }
        canon_payload = _canonical_json(event_body)

        hasher = hashlib.sha256()
        hasher.update(prev_hash.encode("utf-8"))
        hasher.update(canon_payload.encode("utf-8"))
        event_hash = hasher.hexdigest()

        now_iso = datetime.now(timezone.utc).isoformat()

        with self._conn:
            self._conn.execute(
                """
                UPDATE experiments
                SET current_stage = ?, updated_at = ?
                WHERE experiment_id = ?
                """,
                (target_stage.value, now_iso, experiment_id),
            )
            self._conn.execute(
                """
                INSERT INTO experiment_events (experiment_id, event_type, created_at, payload_json, previous_event_hash, event_hash)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    experiment_id,
                    event_type,
                    now_iso,
                    canon_payload,
                    prev_hash,
                    event_hash,
                ),
            )

        return current_record

    def get_record(self, experiment_id: str) -> ScientificExperimentRecord | None:
        """Reconstructs the current state of an experiment from its event history."""
        cursor = self._conn.execute(
            """
            SELECT payload_json, event_type
            FROM experiment_events
            WHERE experiment_id = ?
            ORDER BY event_id ASC
            """,
            (experiment_id,),
        )
        rows = cursor.fetchall()
        if not rows:
            return None

        # Initial proposal payload
        first_row = rows[0]
        initial_data = json.loads(first_row["payload_json"])
        record = ScientificExperimentRecord.from_dict(initial_data)

        # Apply subsequent delta events
        for r in rows[1:]:
            event_body = json.loads(r["payload_json"])
            snapshot = event_body.get("record_snapshot")
            if snapshot:
                record = ScientificExperimentRecord.from_dict(snapshot)

        return record

    def list_records(
        self,
        stage: ExperimentStage | str | None = None,
        dataset_name: str | None = None,
    ) -> list[ScientificExperimentRecord]:
        """Lists all experiment records matching optional stage and dataset filters."""
        query = "SELECT experiment_id FROM experiments WHERE 1=1"
        params: list[Any] = []
        if stage is not None:
            stage_val = stage.value if isinstance(stage, ExperimentStage) else str(stage)
            query += " AND current_stage = ?"
            params.append(stage_val)
        if dataset_name is not None:
            query += " AND dataset_name = ?"
            params.append(dataset_name)

        query += " ORDER BY created_at ASC"
        cursor = self._conn.execute(query, params)
        exp_ids = [row["experiment_id"] for row in cursor.fetchall()]
        return [self.get_record(eid) for eid in exp_ids if eid is not None]  # type: ignore

    def list_pending_records(self) -> list[ScientificExperimentRecord]:
        """Returns non-terminal records (in-flight proposals, scheduled, or executed)."""
        terminal_stages = ("COMPLETED", "FAILED", "CANCELLED")
        cursor = self._conn.execute(
            f"SELECT experiment_id FROM experiments WHERE current_stage NOT IN ({','.join(['?']*len(terminal_stages))}) ORDER BY created_at ASC",
            terminal_stages,
        )
        return [self.get_record(row["experiment_id"]) for row in cursor.fetchall()]  # type: ignore

    def list_completed_records(self) -> list[ScientificExperimentRecord]:
        return self.list_records(stage=ExperimentStage.COMPLETED)

    def verify_integrity(self) -> tuple[bool, list[str]]:
        """Verifies cryptographic SHA-256 hash chaining across all events from genesis."""
        cursor = self._conn.execute(
            "SELECT event_id, experiment_id, payload_json, previous_event_hash, event_hash FROM experiment_events ORDER BY event_id ASC"
        )
        rows = cursor.fetchall()
        if not rows:
            return True, []

        errors: list[str] = []
        expected_prev_hash = "0" * 64

        for row in rows:
            eid = row["event_id"]
            prev_h = row["previous_event_hash"]
            curr_h = row["event_hash"]
            raw_payload = row["payload_json"]

            # Canonicalize payload to verify
            try:
                parsed = json.loads(raw_payload)
                canon = _canonical_json(parsed)
            except Exception as exc:
                errors.append(f"Event {eid}: invalid payload JSON: {exc}")
                continue

            if prev_h != expected_prev_hash:
                errors.append(
                    f"Event {eid}: previous_event_hash mismatch. Expected {expected_prev_hash}, got {prev_h}"
                )

            hasher = hashlib.sha256()
            hasher.update(prev_h.encode("utf-8"))
            hasher.update(canon.encode("utf-8"))
            recomputed = hasher.hexdigest()

            if recomputed != curr_h:
                errors.append(
                    f"Event {eid}: event_hash mismatch / tamper detected. Recomputed {recomputed}, recorded {curr_h}"
                )

            expected_prev_hash = curr_h

        is_valid = len(errors) == 0
        return is_valid, errors

    def to_dataframe(self) -> pd.DataFrame:
        """Flattens completed and in-flight records into a standard tabular DataFrame."""
        records = self.list_records()
        rows: list[dict[str, Any]] = []
        for r in records:
            flat: dict[str, Any] = {
                "experiment_id": r.experiment_id,
                "candidate_id": r.candidate_id,
                "dataset_name": r.dataset_name,
                "stage": r.stage.value,
                "created_at": r.created_at,
                "updated_at": r.updated_at,
                "batch_id": r.batch_id,
                "replicate_id": r.replicate_id,
                "failure_reason": r.failure_reason,
            }
            # Flatten pre-experiment features
            for k, v in r.pre_experiment_features.items():
                flat[k] = v
            # Flatten characterization
            for k, v in r.characterization.items():
                flat[k] = v
            # Flatten performance
            for k, v in r.performance.items():
                flat[k] = v
            rows.append(flat)

        return pd.DataFrame(rows)

    def close(self) -> None:
        self._conn.close()
