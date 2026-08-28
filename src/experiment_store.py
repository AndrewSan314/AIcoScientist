from contextlib import contextmanager
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
import sqlite3

import pandas as pd

from src.utils import (
    DATABASE_FILE,
    EDX_FILE,
    ELECTROCHEM_FILE,
    PROCESS_FILE,
    SCHEMA_VERSION,
    SEM_FILE,
)


TABLE_COLUMNS = {
    "process_data": [
        "sample_id",
        "si_content",
        "mxene_content",
        "alginate_content",
        "carbon_content",
        "mixing_time",
        "drying_temp",
        "pressing_pressure",
    ],
    "sem_features": [
        "sample_id",
        "particle_size_mean",
        "porosity_score",
        "agglomeration_index",
        "crack_density",
        "surface_uniformity",
    ],
    "edx_data": [
        "sample_id",
        "si_percent",
        "ti_percent",
        "c_percent",
        "o_percent",
        "impurity_percent",
    ],
    "electrochem_data": [
        "sample_id",
        "initial_capacity",
        "capacity_50",
        "capacity_100",
        "retention_100",
        "coulombic_efficiency",
        "rct",
    ],
}

DEFAULT_SOURCE_FILES = {
    "process_data": PROCESS_FILE,
    "sem_features": SEM_FILE,
    "edx_data": EDX_FILE,
    "electrochem_data": ELECTROCHEM_FILE,
}

CORE_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS schema_meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS ingestion_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at TEXT NOT NULL,
    completed_at TEXT,
    status TEXT NOT NULL CHECK (status IN ('running', 'succeeded', 'failed')),
    schema_version INTEGER NOT NULL,
    row_count INTEGER,
    error TEXT
);
CREATE TABLE IF NOT EXISTS source_files (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ingestion_run_id INTEGER NOT NULL REFERENCES ingestion_runs(id),
    dataset_name TEXT NOT NULL,
    path TEXT NOT NULL,
    sha256 TEXT NOT NULL CHECK (length(sha256) = 64),
    row_count INTEGER NOT NULL CHECK (row_count >= 0),
    UNIQUE (ingestion_run_id, dataset_name)
);
"""

EXPERIMENTS_DDL = """
CREATE TABLE IF NOT EXISTS experiments (
    id TEXT PRIMARY KEY,
    batch_id TEXT NOT NULL,
    operator TEXT NOT NULL,
    created_at TEXT NOT NULL,
    source_fingerprint TEXT NOT NULL UNIQUE,
    ingestion_run_id INTEGER REFERENCES ingestion_runs(id),
    is_active INTEGER NOT NULL DEFAULT 0 CHECK (is_active IN (0, 1))
)
"""

TABLE_DDL = {
    "process_data": """
CREATE TABLE IF NOT EXISTS process_data (
    experiment_id TEXT NOT NULL REFERENCES experiments(id),
    sample_id TEXT NOT NULL,
    si_content REAL NOT NULL CHECK (si_content BETWEEN 0 AND 100),
    mxene_content REAL NOT NULL CHECK (mxene_content BETWEEN 0 AND 100),
    alginate_content REAL NOT NULL CHECK (alginate_content BETWEEN 0 AND 100),
    carbon_content REAL NOT NULL CHECK (carbon_content BETWEEN 0 AND 100),
    mixing_time REAL NOT NULL CHECK (mixing_time > 0),
    drying_temp REAL NOT NULL CHECK (drying_temp > 0),
    pressing_pressure REAL NOT NULL CHECK (pressing_pressure > 0),
    PRIMARY KEY (experiment_id, sample_id)
)
""",
    "sem_features": """
CREATE TABLE IF NOT EXISTS sem_features (
    experiment_id TEXT NOT NULL REFERENCES experiments(id),
    sample_id TEXT NOT NULL,
    particle_size_mean REAL NOT NULL CHECK (particle_size_mean > 0),
    porosity_score REAL NOT NULL CHECK (porosity_score BETWEEN 0 AND 1),
    agglomeration_index REAL NOT NULL CHECK (agglomeration_index BETWEEN 0 AND 1),
    crack_density REAL NOT NULL CHECK (crack_density BETWEEN 0 AND 1),
    surface_uniformity REAL NOT NULL CHECK (surface_uniformity BETWEEN 0 AND 1),
    PRIMARY KEY (experiment_id, sample_id)
)
""",
    "edx_data": """
CREATE TABLE IF NOT EXISTS edx_data (
    experiment_id TEXT NOT NULL REFERENCES experiments(id),
    sample_id TEXT NOT NULL,
    si_percent REAL NOT NULL CHECK (si_percent BETWEEN 0 AND 100),
    ti_percent REAL NOT NULL CHECK (ti_percent BETWEEN 0 AND 100),
    c_percent REAL NOT NULL CHECK (c_percent BETWEEN 0 AND 100),
    o_percent REAL NOT NULL CHECK (o_percent BETWEEN 0 AND 100),
    impurity_percent REAL NOT NULL CHECK (impurity_percent BETWEEN 0 AND 100),
    PRIMARY KEY (experiment_id, sample_id)
)
""",
    "electrochem_data": """
CREATE TABLE IF NOT EXISTS electrochem_data (
    experiment_id TEXT NOT NULL REFERENCES experiments(id),
    sample_id TEXT NOT NULL,
    initial_capacity REAL NOT NULL CHECK (initial_capacity > 0),
    capacity_50 REAL NOT NULL CHECK (capacity_50 >= 0),
    capacity_100 REAL NOT NULL CHECK (capacity_100 >= 0),
    retention_100 REAL NOT NULL CHECK (retention_100 BETWEEN 0 AND 100),
    coulombic_efficiency REAL NOT NULL CHECK (coulombic_efficiency BETWEEN 0 AND 100),
    rct REAL NOT NULL CHECK (rct >= 0),
    PRIMARY KEY (experiment_id, sample_id)
)
""",
}


def _now():
    return datetime.now(timezone.utc).isoformat()


@contextmanager
def _connection(database):
    connection = sqlite3.connect(database)
    connection.execute("PRAGMA foreign_keys = ON")
    try:
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def _source_identity(batch_id, hashes):
    source_key = "|".join(f"{name}:{hashes[name]}" for name in sorted(hashes))
    source_digest = sha256(source_key.encode()).hexdigest()
    batch_id = batch_id or f"batch-{source_digest[:12]}"
    fingerprint = sha256(f"{batch_id}|{source_key}".encode()).hexdigest()
    return batch_id, fingerprint, f"exp-{fingerprint[:24]}"


def _create_v2_tables(connection):
    connection.execute(EXPERIMENTS_DDL)
    connection.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS one_active_experiment "
        "ON experiments(is_active) WHERE is_active = 1"
    )
    for ddl in TABLE_DDL.values():
        connection.execute(ddl)


def _migrate_v1_to_v2(connection):
    latest = connection.execute(
        "SELECT id, started_at FROM ingestion_runs "
        "WHERE status = 'succeeded' ORDER BY id DESC LIMIT 1"
    ).fetchone()
    if latest is None:
        raise RuntimeError("Cannot migrate schema v1 without a successful ingestion run")
    run_id, created_at = latest
    hashes = dict(
        connection.execute(
            "SELECT dataset_name, sha256 FROM source_files WHERE ingestion_run_id = ?",
            (run_id,),
        )
    )
    if set(hashes) != set(TABLE_COLUMNS):
        raise RuntimeError("Cannot migrate schema v1 without four source-file hashes")
    batch_id, fingerprint, experiment_id = _source_identity(None, hashes)

    connection.execute("BEGIN IMMEDIATE")
    connection.execute(EXPERIMENTS_DDL)
    connection.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS one_active_experiment "
        "ON experiments(is_active) WHERE is_active = 1"
    )
    for name in TABLE_COLUMNS:
        connection.execute(f'ALTER TABLE "{name}" RENAME TO "{name}_v1"')
    for ddl in TABLE_DDL.values():
        connection.execute(ddl)
    connection.execute(
        "INSERT INTO experiments(id, batch_id, operator, created_at, "
        "source_fingerprint, ingestion_run_id, is_active) "
        "VALUES (?, ?, 'migration', ?, ?, ?, 1)",
        (experiment_id, batch_id, created_at, fingerprint, run_id),
    )
    for name, columns in TABLE_COLUMNS.items():
        column_sql = ", ".join(f'"{column}"' for column in columns)
        connection.execute(
            f'INSERT INTO "{name}" (experiment_id, {column_sql}) '
            f'SELECT ?, {column_sql} FROM "{name}_v1"',
            (experiment_id,),
        )
        connection.execute(f'DROP TABLE "{name}_v1"')
    connection.execute(
        "UPDATE schema_meta SET value = ? WHERE key = 'schema_version'",
        (str(SCHEMA_VERSION),),
    )


def _initialize_schema(connection):
    connection.executescript(CORE_SCHEMA_SQL)
    row = connection.execute(
        "SELECT value FROM schema_meta WHERE key = 'schema_version'"
    ).fetchone()
    if row is None:
        _create_v2_tables(connection)
        connection.execute(
            "INSERT INTO schema_meta(key, value) VALUES ('schema_version', ?)",
            (str(SCHEMA_VERSION),),
        )
    elif int(row[0]) == 1:
        _migrate_v1_to_v2(connection)
    elif int(row[0]) == SCHEMA_VERSION:
        _create_v2_tables(connection)
    else:
        raise RuntimeError(
            f"Unsupported database schema version {row[0]}; expected {SCHEMA_VERSION}"
        )


def initialize_database(database=DATABASE_FILE):
    database = Path(database)
    database.parent.mkdir(parents=True, exist_ok=True)
    with _connection(database) as connection:
        _initialize_schema(connection)
    return database


def _read_sources(source_files):
    if set(source_files) != set(TABLE_COLUMNS):
        missing = sorted(set(TABLE_COLUMNS) - set(source_files))
        unexpected = sorted(set(source_files) - set(TABLE_COLUMNS))
        raise ValueError(f"Invalid source set; missing={missing}, unexpected={unexpected}")

    tables = {}
    for name, columns in TABLE_COLUMNS.items():
        path = Path(source_files[name])
        if not path.is_file():
            raise FileNotFoundError(f"Missing source file: {path}")
        table = pd.read_csv(path)
        missing = sorted(set(columns) - set(table.columns))
        unexpected = sorted(set(table.columns) - set(columns))
        if missing or unexpected:
            raise ValueError(
                f"{name} schema mismatch; missing={missing}, unexpected={unexpected}"
            )
        if table["sample_id"].isna().any():
            raise ValueError(f"{name} contains null sample_id values")
        table["sample_id"] = table["sample_id"].astype(str).str.strip()
        if table["sample_id"].eq("").any() or table["sample_id"].duplicated().any():
            raise ValueError(f"{name} contains invalid or duplicate sample_id values")
        tables[name] = table

    electrochem = tables["electrochem_data"]
    missing_retention = electrochem["retention_100"].isna()
    if missing_retention.any():
        initial = pd.to_numeric(electrochem["initial_capacity"], errors="raise")
        capacity = pd.to_numeric(electrochem["capacity_100"], errors="raise")
        if initial[missing_retention].le(0).any():
            raise ValueError("initial_capacity must be positive to derive retention_100")
        electrochem.loc[missing_retention, "retention_100"] = (
            capacity[missing_retention] / initial[missing_retention] * 100
        )

    for name, table in tables.items():
        numeric_columns = TABLE_COLUMNS[name][1:]
        table[numeric_columns] = table[numeric_columns].apply(
            pd.to_numeric, errors="raise"
        )
        if table[numeric_columns].isna().any().any():
            raise ValueError(f"{name} contains missing numeric values")
        if table[numeric_columns].isin([float("inf"), -float("inf")]).any().any():
            raise ValueError(f"{name} contains non-finite numeric values")

    expected_ids = set(tables["process_data"]["sample_id"])
    for name, table in tables.items():
        if set(table["sample_id"]) != expected_ids:
            raise ValueError(f"{name} sample_id values do not match process_data")

    process = tables["process_data"]
    composition = [
        "si_content",
        "mxene_content",
        "alginate_content",
        "carbon_content",
    ]
    if not process[composition].sum(axis=1).sub(100).abs().le(1e-6).all():
        raise ValueError("Process composition must total 100 wt%")

    domains = {
        "process_data": {
            **{column: (0, 100) for column in composition},
            "mixing_time": (0, None),
            "drying_temp": (0, None),
            "pressing_pressure": (0, None),
        },
        "sem_features": {
            "particle_size_mean": (0, None),
            "porosity_score": (0, 1),
            "agglomeration_index": (0, 1),
            "crack_density": (0, 1),
            "surface_uniformity": (0, 1),
        },
        "edx_data": {
            column: (0, 100) for column in TABLE_COLUMNS["edx_data"][1:]
        },
        "electrochem_data": {
            "initial_capacity": (0, None),
            "capacity_50": (0, None),
            "capacity_100": (0, None),
            "retention_100": (0, 100),
            "coulombic_efficiency": (0, 100),
            "rct": (0, None),
        },
    }
    for name, rules in domains.items():
        for column, (minimum, maximum) in rules.items():
            values = tables[name][column]
            if values.lt(minimum).any() or (maximum is not None and values.gt(maximum).any()):
                raise ValueError(f"{name}.{column} is outside its valid range")
            if minimum == 0 and maximum is None and values.eq(0).any() and column in {
                "mixing_time",
                "drying_temp",
                "pressing_pressure",
                "particle_size_mean",
                "initial_capacity",
            }:
                raise ValueError(f"{name}.{column} must be positive")
    return tables


def _file_hash(path):
    digest = sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _records(table, columns):
    for row in table[columns].itertuples(index=False, name=None):
        yield tuple(value.item() if hasattr(value, "item") else value for value in row)


def ingest_csvs(
    source_files=None,
    database=DATABASE_FILE,
    *,
    batch_id=None,
    operator="system",
):
    source_files = {
        name: Path(path) for name, path in (source_files or DEFAULT_SOURCE_FILES).items()
    }
    database = Path(database)
    database.parent.mkdir(parents=True, exist_ok=True)

    with _connection(database) as connection:
        _initialize_schema(connection)
        run_id = connection.execute(
            "INSERT INTO ingestion_runs(started_at, status, schema_version) "
            "VALUES (?, 'running', ?)",
            (_now(), SCHEMA_VERSION),
        ).lastrowid

    try:
        tables = _read_sources(source_files)
        hashes = {name: _file_hash(path) for name, path in source_files.items()}
        operator = str(operator).strip()
        if not operator:
            raise ValueError("operator must not be empty")
        if batch_id is not None:
            batch_id = str(batch_id).strip()
            if not batch_id:
                raise ValueError("batch_id must not be empty")
        batch_id, fingerprint, experiment_id = _source_identity(batch_id, hashes)
        row_count = len(tables["process_data"])
        with _connection(database) as connection:
            connection.execute("BEGIN IMMEDIATE")
            existing = connection.execute(
                "SELECT id FROM experiments WHERE source_fingerprint = ?",
                (fingerprint,),
            ).fetchone()
            connection.execute("UPDATE experiments SET is_active = 0 WHERE is_active = 1")
            if existing is None:
                connection.execute(
                    "INSERT INTO experiments(id, batch_id, operator, created_at, "
                    "source_fingerprint, ingestion_run_id, is_active) "
                    "VALUES (?, ?, ?, ?, ?, ?, 1)",
                    (experiment_id, batch_id, operator, _now(), fingerprint, run_id),
                )
                for name, columns in TABLE_COLUMNS.items():
                    placeholders = ", ".join("?" for _ in range(len(columns) + 1))
                    column_sql = ", ".join(f'"{column}"' for column in columns)
                    connection.executemany(
                        f'INSERT INTO "{name}" (experiment_id, {column_sql}) '
                        f'VALUES ({placeholders})',
                        (
                            (experiment_id, *row)
                            for row in _records(tables[name], columns)
                        ),
                    )
            else:
                experiment_id = existing[0]
                connection.execute(
                    "UPDATE experiments SET is_active = 1 WHERE id = ?",
                    (experiment_id,),
                )
            for name in TABLE_COLUMNS:
                connection.execute(
                    "INSERT INTO source_files(ingestion_run_id, dataset_name, path, sha256, row_count) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (run_id, name, str(source_files[name].resolve()), hashes[name], len(tables[name])),
                )
            connection.execute(
                "UPDATE ingestion_runs SET completed_at = ?, status = 'succeeded', row_count = ? "
                "WHERE id = ?",
                (_now(), row_count, run_id),
            )
    except Exception as error:
        with _connection(database) as connection:
            connection.execute(
                "UPDATE ingestion_runs SET completed_at = ?, status = 'failed', error = ? "
                "WHERE id = ?",
                (_now(), str(error), run_id),
            )
        raise
    return run_id


def load_source_tables(database=DATABASE_FILE, *, experiment_id=None):
    database = Path(database)
    if not database.is_file():
        raise FileNotFoundError(f"Experiment database does not exist: {database}")
    with _connection(database) as connection:
        version = connection.execute(
            "SELECT value FROM schema_meta WHERE key = 'schema_version'"
        ).fetchone()
        if version is None or int(version[0]) != SCHEMA_VERSION:
            raise RuntimeError("Experiment database schema version is unsupported")
        if experiment_id is None:
            active = connection.execute(
                "SELECT id FROM experiments WHERE is_active = 1"
            ).fetchone()
            if active is None:
                raise RuntimeError("Experiment database has no active experiment")
            experiment_id = active[0]
        elif connection.execute(
            "SELECT 1 FROM experiments WHERE id = ?", (experiment_id,)
        ).fetchone() is None:
            raise ValueError(f"Unknown experiment_id: {experiment_id}")
        return {
            name: pd.read_sql_query(
                f'SELECT {", ".join(columns)} FROM "{name}" '
                "WHERE experiment_id = ? ORDER BY sample_id",
                connection,
                params=(experiment_id,),
            )
            for name, columns in TABLE_COLUMNS.items()
        }
