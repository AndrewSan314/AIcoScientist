from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from src.datasets.base import DatasetSpec
from src.science.records import ExperimentStage, ScientificExperimentRecord
from src.science.validation import validate_record_against_spec, validate_transition_before_append


def _canonical_json(payload: Any) -> str:
    """Produces a deterministic, sorted, compact JSON string for cryptographic hash chaining."""
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


class ExperimentLedger:
    """Append-only SQLite Experiment Ledger with tamper-evident SHA-256 event hash chaining.

    Provides tamper-evident event auditing that detects modification or deletion of hashed historical
    events while the expected chain head/event count remains available.
    """

    def __init__(self, db_path: Path | str = ":memory:") -> None:
        self.db_path = str(db_path)
        if self.db_path != ":memory:":
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.db_path)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._init_schema()

    def _init_schema(self) -> None:
        with self._conn:
            self._conn.execute("PRAGMA foreign_keys = ON")
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
                """
                CREATE TABLE IF NOT EXISTS optimizer_snapshots (
                    snapshot_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    step INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    snapshot_json TEXT NOT NULL
                )
                """
            )
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS ledger_metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
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

    def _update_head_metadata(self, event_hash: str) -> None:
        cursor = self._conn.execute("SELECT COUNT(*) as cnt FROM experiment_events")
        row = cursor.fetchone()
        cnt = int(row["cnt"]) if row else 0
        self._conn.execute(
            "INSERT OR REPLACE INTO ledger_metadata (key, value) VALUES ('head_hash', ?)",
            (event_hash,),
        )
        self._conn.execute(
            "INSERT OR REPLACE INTO ledger_metadata (key, value) VALUES ('event_count', ?)",
            (str(cnt),),
        )

    def next_proposal_sequence(self, dataset_name: str | None = None) -> int:
        """Returns a strictly monotonic proposal sequence integer (1-indexed) derived from the total historical proposals."""
        query = "SELECT COUNT(*) as cnt FROM experiments"
        params: list[Any] = []
        if dataset_name is not None:
            query += " WHERE dataset_name = ?"
            params.append(dataset_name)
        cursor = self._conn.execute(query, params)
        row = cursor.fetchone()
        return int(row["cnt"]) + 1 if row else 1

    def record_proposal(
        self,
        record: ScientificExperimentRecord,
        spec: DatasetSpec | None = None,
    ) -> ScientificExperimentRecord:
        """Validates and records a new experiment proposal event in the ledger transactionally."""
        # 1. Validate record against spec BEFORE database mutation
        if spec is not None:
            validate_record_against_spec(record, spec)

        prev_hash = self._get_latest_event_hash()
        payload = record.to_dict()
        now_iso = datetime.now(timezone.utc).isoformat()

        # Build full scientific event envelope for hashing
        event_envelope = {
            "experiment_id": record.experiment_id,
            "event_type": "PROPOSAL_CREATED",
            "created_at": now_iso,
            "payload": payload,
        }
        canon_envelope = _canonical_json(event_envelope)

        hasher = hashlib.sha256()
        hasher.update(prev_hash.encode("utf-8"))
        hasher.update(canon_envelope.encode("utf-8"))
        event_hash = hasher.hexdigest()

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
                    _canonical_json(payload),
                    prev_hash,
                    event_hash,
                ),
            )
            self._update_head_metadata(event_hash)

        return record

    def append_transition(
        self,
        experiment_id: str,
        new_stage: ExperimentStage | str,
        event_type: str,
        delta_payload: Mapping[str, Any],
        spec: DatasetSpec | None = None,
    ) -> ScientificExperimentRecord:
        """Validates prospective transition BEFORE committing event to the ledger."""
        current_record = self.get_record(experiment_id)
        if current_record is None:
            raise KeyError(f"Experiment {experiment_id!r} not found in ledger.")

        # 1. Validate prospective transition and spec boundaries BEFORE touching SQL
        target_stage = ExperimentStage(new_stage) if isinstance(new_stage, str) else new_stage
        validated_record = validate_transition_before_append(
            current_record=current_record,
            new_stage=target_stage,
            delta_payload=delta_payload,
            spec=spec,
        )

        prev_hash = self._get_latest_event_hash()
        now_iso = datetime.now(timezone.utc).isoformat()

        event_payload = {
            "event_type": event_type,
            "target_stage": target_stage.value,
            "delta": dict(delta_payload),
            "record_snapshot": validated_record.to_dict(),
        }

        # Build full scientific event envelope for hashing
        event_envelope = {
            "experiment_id": experiment_id,
            "event_type": event_type,
            "created_at": now_iso,
            "payload": event_payload,
        }
        canon_envelope = _canonical_json(event_envelope)

        hasher = hashlib.sha256()
        hasher.update(prev_hash.encode("utf-8"))
        hasher.update(canon_envelope.encode("utf-8"))
        event_hash = hasher.hexdigest()

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
                    _canonical_json(event_payload),
                    prev_hash,
                    event_hash,
                ),
            )
            self._update_head_metadata(event_hash)

        return validated_record

    def save_optimizer_snapshot(self, snapshot: Mapping[str, Any]) -> None:
        """Persists a deterministic optimizer state snapshot after completed experimental observations."""
        now_iso = datetime.now(timezone.utc).isoformat()
        step = int(snapshot.get("step", 0))
        canon_json = _canonical_json(snapshot)
        with self._conn:
            self._conn.execute(
                """
                INSERT INTO optimizer_snapshots (step, created_at, snapshot_json)
                VALUES (?, ?, ?)
                """,
                (step, now_iso, canon_json),
            )

    def get_latest_optimizer_snapshot(self) -> dict[str, Any] | None:
        """Retrieves the latest optimizer state snapshot for exact resume."""
        cursor = self._conn.execute(
            "SELECT snapshot_json FROM optimizer_snapshots ORDER BY snapshot_id DESC LIMIT 1"
        )
        row = cursor.fetchone()
        if not row:
            return None
        return json.loads(row["snapshot_json"])

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
        """Verifies cryptographic SHA-256 hash chaining across all events from genesis and checks projection consistency."""
        cursor = self._conn.execute(
            "SELECT event_id, experiment_id, event_type, created_at, payload_json, previous_event_hash, event_hash FROM experiment_events ORDER BY event_id ASC"
        )
        rows = cursor.fetchall()
        errors: list[str] = []

        if not rows:
            return True, []

        expected_prev_hash = "0" * 64

        for row in rows:
            eid = row["event_id"]
            exp_id = row["experiment_id"]
            ev_type = row["event_type"]
            ev_created = row["created_at"]
            prev_h = row["previous_event_hash"]
            curr_h = row["event_hash"]
            raw_payload = row["payload_json"]

            try:
                parsed_payload = json.loads(raw_payload)
            except Exception as exc:
                errors.append(f"Event {eid}: invalid payload JSON: {exc}")
                continue

            if prev_h != expected_prev_hash:
                errors.append(
                    f"Event {eid}: previous_event_hash mismatch. Expected {expected_prev_hash}, got {prev_h}"
                )

            # Reconstruct full envelope
            envelope = {
                "experiment_id": exp_id,
                "event_type": ev_type,
                "created_at": ev_created,
                "payload": parsed_payload,
            }
            canon_env = _canonical_json(envelope)

            hasher = hashlib.sha256()
            hasher.update(prev_h.encode("utf-8"))
            hasher.update(canon_env.encode("utf-8"))
            recomputed = hasher.hexdigest()

            if recomputed != curr_h:
                errors.append(
                    f"Event {eid}: event_hash mismatch / tamper detected. Recomputed {recomputed}, recorded {curr_h}"
                )

            expected_prev_hash = curr_h

        # Verify summary projection table consistency
        exp_cursor = self._conn.execute("SELECT experiment_id, current_stage FROM experiments")
        for exp_row in exp_cursor.fetchall():
            exp_id = exp_row["experiment_id"]
            proj_stage = exp_row["current_stage"]
            try:
                rec = self.get_record(exp_id)
                if rec is None:
                    errors.append(f"Projection mismatch: experiment {exp_id} exists in summary table but has no events.")
                elif rec.stage.value != proj_stage:
                    errors.append(
                        f"Projection mismatch: experiment {exp_id} summary table stage {proj_stage!r} "
                        f"does not match reconstructed event stage {rec.stage.value!r}."
                    )
            except Exception as exc:
                errors.append(f"Projection error: failed to reconstruct experiment {exp_id}: {exc}")

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
            for k, v in r.pre_experiment_features.items():
                flat[k] = v
            for k, v in r.characterization.items():
                flat[k] = v
            for k, v in r.performance.items():
                flat[k] = v
            rows.append(flat)

        return pd.DataFrame(rows)

    def close(self) -> None:
        self._conn.close()
