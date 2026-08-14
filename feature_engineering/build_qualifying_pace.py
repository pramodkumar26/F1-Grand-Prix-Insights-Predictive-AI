import sqlite3
import pandas as pd

DB_PATH = 'f1.db'
MAX_PLAUSIBLE_GAP = 3.0


def load_best_laps(conn):
    # excludes deleted laps (track limits etc) and laps with no valid time, then takes
    # each driver's single fastest remaining lap for that race's qualifying session
    laps = pd.read_sql("""
        SELECT race_id, driver_id, lap_time
        FROM fact_qualifying_laps
        WHERE deleted = 0 AND lap_time IS NOT NULL
    """, conn)
    idx = laps.groupby(['race_id', 'driver_id'])['lap_time'].idxmin()
    return laps.loc[idx, ['race_id', 'driver_id', 'lap_time']].rename(columns={'lap_time': 'best_quali_lap'})


def load_pairs(conn):
    # pulls teammate relationships, drops solo entries with no teammate to compare against
    pairs = pd.read_sql("SELECT race_id, driver_id, teammate_driver_id FROM teammate_pairs", conn)
    return pairs.dropna(subset=['teammate_driver_id'])


def compute_qualifying_pace_delta(pairs, best_laps):
    # same teammate relative pattern as build_tyre_wear_delta.py, own lap minus teammate's,
    # negative means the driver out qualified their teammate, same car and session that weekend
    merged = pairs.merge(best_laps, on=['race_id', 'driver_id'], how='inner')

    teammate_laps = best_laps.rename(columns={'driver_id': 'teammate_driver_id', 'best_quali_lap': 'teammate_best_quali_lap'})
    merged = merged.merge(teammate_laps, on=['race_id', 'teammate_driver_id'], how='inner')

    merged['qualifying_pace_delta'] = merged['best_quali_lap'] - merged['teammate_best_quali_lap']

    # gaps this large between same car teammates under green flag conditions don't happen,
    # they're a session ending incident on one side (crash, red flag, rain cutting the
    # session short) being compared against a normal lap. Same category of bug as the
    # red flag inflated pit stops caught in build_pit_stop_delta.py. Distribution has a
    # clean elbow around 2-3s then a long tail to 43s, confirming these are incidents,
    # not real pace. Set to null rather than drop the row, so the rest of the driver's
    # race features stay usable.
    implausible = merged['qualifying_pace_delta'].abs() > MAX_PLAUSIBLE_GAP
    merged.loc[implausible, 'qualifying_pace_delta'] = pd.NA

    return merged[['race_id', 'driver_id', 'teammate_driver_id', 'best_quali_lap', 'teammate_best_quali_lap', 'qualifying_pace_delta']]


def build_qualifying_pace_table():
    conn = sqlite3.connect(DB_PATH)
    best_laps = load_best_laps(conn)
    pairs = load_pairs(conn)
    delta = compute_qualifying_pace_delta(pairs, best_laps)

    delta.to_sql('qualifying_pace_delta', conn, if_exists='replace', index=False)
    conn.commit()
    conn.close()

    return delta


if __name__ == '__main__':
    delta = build_qualifying_pace_table()
    print(f"qualifying_pace_delta table built, {len(delta)} rows")
    print(delta['qualifying_pace_delta'].describe())
