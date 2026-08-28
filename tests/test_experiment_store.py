import hashlib
import shutil
import sqlite3

import pandas as pd
import pytest

from src.experiment_store import ingest_csvs, initialize_database, load_source_tables
from src.utils import EDX_FILE, ELECTROCHEM_FILE, PROCESS_FILE, SEM_FILE


SOURCE_FILES = {
    "process_data": PROCESS_FILE,
    "sem_features": SEM_FILE,
    "edx_data": EDX_FILE,
    "electrochem_data": ELECTROCHEM_FILE,
}


def _copy_sources(directory):
    sources = {}
    for name, source in SOURCE_FILES.items():
        destination = directory / source.name
        shutil.copy2(source, destination)
        sources[name] = destination
    return sources


def _create_v1_database(database, sources):
    with sqlite3.connect(database) as connection:
        connection.executescript(
            """
            CREATE TABLE schema_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE ingestion_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                started_at TEXT NOT NULL,
                completed_at TEXT,
                status TEXT NOT NULL,
                schema_version INTEGER NOT NULL,
                row_count INTEGER,
                error TEXT
            );
            CREATE TABLE source_files (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ingestion_run_id INTEGER NOT NULL REFERENCES ingestion_runs(id),
                dataset_name TEXT NOT NULL,
                path TEXT NOT NULL,
                sha256 TEXT NOT NULL,
                row_count INTEGER NOT NULL
            );
            INSERT INTO schema_meta(key, value) VALUES ('schema_version', '1');
            INSERT INTO ingestion_runs(
                id, started_at, completed_at, status, schema_version, row_count
            ) VALUES (1, '2026-07-15T00:00:00Z', '2026-07-15T00:00:01Z',
                      'succeeded', 1, 50);
            """
        )
        for name, path in sources.items():
            table = pd.read_csv(path)
            table.to_sql(name, connection, index=False)
            connection.execute(
                "INSERT INTO source_files(ingestion_run_id, dataset_name, path, sha256, row_count) "
                "VALUES (1, ?, ?, ?, ?)",
                (
                    name,
                    str(path.resolve()),
                    hashlib.sha256(path.read_bytes()).hexdigest(),
                    len(table),
                ),
            )


def test_v1_migration_preserves_data_and_lineage(tmp_path):
    sources = _copy_sources(tmp_path)
    database = tmp_path / "experiments.db"
    _create_v1_database(database, sources)

    initialize_database(database)

    with sqlite3.connect(database) as connection:
        assert connection.execute(
            "SELECT value FROM schema_meta WHERE key = 'schema_version'"
        ).fetchone()[0] == "2"
        assert connection.execute(
            "SELECT COUNT(*) FROM experiments WHERE is_active = 1"
        ).fetchone()[0] == 1
        assert connection.execute("SELECT COUNT(*) FROM ingestion_runs").fetchone()[0] == 1
        assert connection.execute("SELECT COUNT(*) FROM source_files").fetchone()[0] == 4
        assert all(
            connection.execute(f'SELECT COUNT(*) FROM "{name}"').fetchone()[0] == 50
            for name in SOURCE_FILES
        )


def test_ingest_stores_validated_tables_and_source_hashes(tmp_path):
    sources = _copy_sources(tmp_path)
    database = tmp_path / "experiments.db"

    run_id = ingest_csvs(sources, database)

    with sqlite3.connect(database) as connection:
        status, row_count = connection.execute(
            "SELECT status, row_count FROM ingestion_runs WHERE id = ?", (run_id,)
        ).fetchone()
        stored_hashes = dict(
            connection.execute(
                "SELECT dataset_name, sha256 FROM source_files WHERE ingestion_run_id = ?",
                (run_id,),
            )
        )
    assert status == "succeeded"
    assert row_count == 50
    assert stored_hashes == {
        name: hashlib.sha256(path.read_bytes()).hexdigest()
        for name, path in sources.items()
    }


def test_invalid_ingest_preserves_last_valid_snapshot_and_records_failure(tmp_path):
    sources = _copy_sources(tmp_path)
    database = tmp_path / "experiments.db"
    ingest_csvs(sources, database, batch_id="batch-1", operator="scientist-a")
    expected = load_source_tables(database)["sem_features"]
    invalid = pd.read_csv(sources["sem_features"])
    invalid.loc[0, "sample_id"] = "NOT_IN_PROCESS_DATA"
    invalid.to_csv(sources["sem_features"], index=False)

    with pytest.raises(ValueError, match="sample_id"):
        ingest_csvs(sources, database, batch_id="batch-2", operator="scientist-a")

    pd.testing.assert_frame_equal(
        load_source_tables(database)["sem_features"], expected
    )
    with sqlite3.connect(database) as connection:
        statuses = [
            row[0]
            for row in connection.execute(
                "SELECT status FROM ingestion_runs ORDER BY id"
            )
        ]
    assert statuses == ["succeeded", "failed"]


def test_two_batches_coexist_and_latest_is_active(tmp_path):
    sources = _copy_sources(tmp_path)
    database = tmp_path / "experiments.db"
    ingest_csvs(sources, database, batch_id="batch-1", operator="scientist-a")
    with sqlite3.connect(database) as connection:
        first_experiment = connection.execute(
            "SELECT id FROM experiments WHERE batch_id = 'batch-1'"
        ).fetchone()[0]

    ingest_csvs(sources, database, batch_id="batch-2", operator="scientist-b")

    with sqlite3.connect(database) as connection:
        experiments = connection.execute(
            "SELECT batch_id, operator, is_active FROM experiments ORDER BY batch_id"
        ).fetchall()
    assert experiments == [
        ("batch-1", "scientist-a", 0),
        ("batch-2", "scientist-b", 1),
    ]
    pd.testing.assert_frame_equal(
        load_source_tables(database, experiment_id=first_experiment)["process_data"],
        load_source_tables(database)["process_data"],
    )


def test_reimport_same_batch_and_sources_is_idempotent(tmp_path):
    sources = _copy_sources(tmp_path)
    database = tmp_path / "experiments.db"

    ingest_csvs(sources, database, batch_id="batch-1", operator="scientist-a")
    ingest_csvs(sources, database, batch_id="batch-1", operator="scientist-a")

    with sqlite3.connect(database) as connection:
        assert connection.execute("SELECT COUNT(*) FROM experiments").fetchone()[0] == 1
        assert connection.execute("SELECT COUNT(*) FROM process_data").fetchone()[0] == 50
        assert connection.execute("SELECT COUNT(*) FROM ingestion_runs").fetchone()[0] == 2
        assert connection.execute("SELECT COUNT(*) FROM source_files").fetchone()[0] == 8


def test_load_returns_all_sources_with_matching_sample_ids(tmp_path):
    sources = _copy_sources(tmp_path)
    database = tmp_path / "experiments.db"
    ingest_csvs(sources, database)

    tables = load_source_tables(database)

    assert set(tables) == set(SOURCE_FILES)
    sample_ids = tables["process_data"]["sample_id"].tolist()
    assert all(table["sample_id"].tolist() == sample_ids for table in tables.values())
