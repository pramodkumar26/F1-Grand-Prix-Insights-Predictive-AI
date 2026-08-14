import sqlite3
import pandas as pd

DB_PATH = 'f1.db'

COMPOUND_COLUMNS = [
    'degradation_rate_hard',
    'degradation_rate_hypersoft',
    'degradation_rate_intermediate',
    'degradation_rate_medium',
    'degradation_rate_soft',
    'degradation_rate_supersoft',
    'degradation_rate_ultrasoft',
    'degradation_rate_wet'
]


def load_pairs(conn):
    # pulls teammate relationships, drops solo entries with no teammate to compare against
    pairs = pd.read_sql("SELECT race_id, driver_id, teammate_driver_id FROM teammate_pairs", conn)
    return pairs.dropna(subset=['teammate_driver_id'])


def load_degradation_wide(conn):
    # pulls the per driver per race degradation numbers built earlier
    return pd.read_sql("SELECT * FROM degradation_wide", conn)


def compute_tyre_wear_delta(pairs, degradation_wide):
    # joins each driver's own degradation numbers, then their teammate's, and takes the difference
    merged = pairs.merge(degradation_wide, on=['race_id', 'driver_id'], how='left')

    teammate_values = degradation_wide.rename(columns={'driver_id': 'teammate_driver_id'})
    teammate_values = teammate_values.rename(columns={col: f"{col}_teammate" for col in COMPOUND_COLUMNS})
    merged = merged.merge(teammate_values, on=['race_id', 'teammate_driver_id'], how='left')

    for col in COMPOUND_COLUMNS:
        delta_col = col.replace('degradation_rate_', 'tyre_wear_delta_')
        merged[delta_col] = merged[col] - merged[f"{col}_teammate"]

    keep_cols = ['race_id', 'driver_id', 'teammate_driver_id'] + [
        col.replace('degradation_rate_', 'tyre_wear_delta_') for col in COMPOUND_COLUMNS
    ]
    return merged[keep_cols]


def build_tyre_wear_delta_table():
    conn = sqlite3.connect(DB_PATH)
    pairs = load_pairs(conn)
    degradation_wide = load_degradation_wide(conn)
    delta = compute_tyre_wear_delta(pairs, degradation_wide)

    delta.to_sql('tyre_wear_delta', conn, if_exists='replace', index=False)
    conn.commit()
    conn.close()

    return delta


if __name__ == '__main__':
    delta = build_tyre_wear_delta_table()
    print(f"tyre_wear_delta table built, {len(delta)} rows")
    delta_cols = [c for c in delta.columns if c.startswith('tyre_wear_delta_')]
    print(delta[delta_cols].describe())