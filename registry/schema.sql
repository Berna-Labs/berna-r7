-- R7 Registry Schema (R5-compatible, simplified)
-- Aligns with R5 federation spec: UUID, parent_id, domain_key, signature

CREATE TABLE IF NOT EXISTS mother (
    uuid            TEXT PRIMARY KEY,
    version         TEXT NOT NULL,
    path            TEXT NOT NULL,
    created_at      TEXT NOT NULL,
    param_count     INTEGER,
    config_hash     TEXT
);

CREATE TABLE IF NOT EXISTS cells (
    uuid            TEXT PRIMARY KEY,
    parent_uuid     TEXT NOT NULL,      -- → mother.uuid
    name            TEXT NOT NULL,
    domain          TEXT NOT NULL,      -- "math" / "code" / "ar" / ...
    domain_key      TEXT NOT NULL,      -- SHA256(domain|version|param_hash)
    path            TEXT NOT NULL,
    status          TEXT NOT NULL,      -- active / dormant / frozen / dead
    created_at      TEXT NOT NULL,
    param_count     INTEGER,
    config_hash     TEXT,
    manifest_hash   TEXT,
    FOREIGN KEY (parent_uuid) REFERENCES mother(uuid)
);

CREATE TABLE IF NOT EXISTS bindings (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    mother_uuid     TEXT NOT NULL,
    cell_uuid       TEXT NOT NULL,
    binding_token   TEXT NOT NULL,      -- identity issued at discovery
    created_at      TEXT NOT NULL,
    FOREIGN KEY (mother_uuid) REFERENCES mother(uuid),
    FOREIGN KEY (cell_uuid) REFERENCES cells(uuid)
);

CREATE TABLE IF NOT EXISTS splits (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    parent_cell     TEXT,
    child_cell      TEXT NOT NULL,
    reason          TEXT,               -- saturation / manual / ...
    saturation_S    REAL,
    created_at      TEXT NOT NULL,
    FOREIGN KEY (child_cell) REFERENCES cells(uuid)
);

CREATE INDEX IF NOT EXISTS idx_cells_domain ON cells(domain);
CREATE INDEX IF NOT EXISTS idx_cells_parent ON cells(parent_uuid);
CREATE INDEX IF NOT EXISTS idx_bindings_cell ON bindings(cell_uuid);
