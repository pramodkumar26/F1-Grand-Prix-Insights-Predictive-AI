import sqlite3
import pandas as pd

DB_PATH = 'f1.db'


def load_clean_laps(conn):
    # pulls every clean lap along with race and driver ids
    return pd.read_sql("SELECT race_id, driver_id, lap_time FROM clean_laps", conn)


def compute_pace_consistency(laps):
    # standard deviation of clean lap times per driver per race, lower means steadier pace
    grouped = laps.groupby(['race_id', 'driver_id'])['lap_time']
    consistency = grouped.std().reset_index()
    consistency = consistency.rename(columns={'lap_time': 'pace_consistency'})
    consistency['lap_count'] = grouped.count().values
    return consistency


def build_pace_consistency_table():
    conn = sqlite3.connect(DB_PATH)
    laps = load_clean_laps(conn)
    consistency = compute_pace_consistency(laps)

    consistency.to_sql('pace_consistency', conn, if_exists='replace', index=False)
    conn.commit()
    conn.close()

    return consistency


if __name__ == '__main__':
    consistency = build_pace_consistency_table()
    print(f"pace_consistency table built, {len(consistency)} rows")
    print(consistency['pace_consistency'].describe())