import sqlite3
import pandas as pd

DB_PATH = 'f1.db'


def load_clean_laps(conn):
    return pd.read_sql("SELECT race_id, driver_id, lap_number, track_position FROM clean_laps", conn)


def compute_overtakes(laps):
    # a genuine on-track pass: track_position improves lap over lap. clean_laps already
    # excludes pit in/out laps and flagged laps (see build_clean_laps.py), so this isn't
    # picking up pit-cycle position swings, which are much larger and mean something
    # different, being ahead because a rival is in the pits isn't a pass.
    laps = laps.sort_values(['race_id', 'driver_id', 'lap_number'])
    laps['prev_position'] = laps.groupby(['race_id', 'driver_id'])['track_position'].shift()
    laps = laps.dropna(subset=['prev_position'])
    laps['overtake'] = (laps['prev_position'] - laps['track_position']).clip(lower=0)
    return laps.groupby(['race_id', 'driver_id'])['overtake'].sum().reset_index(name='overtakes_made')


def build_overtakes_made_table():
    conn = sqlite3.connect(DB_PATH)
    laps = load_clean_laps(conn)
    overtakes = compute_overtakes(laps)

    overtakes.to_sql('overtakes_made', conn, if_exists='replace', index=False)
    conn.commit()
    conn.close()
    return overtakes


if __name__ == '__main__':
    overtakes = build_overtakes_made_table()
    print(f"overtakes_made table built, {len(overtakes)} driver-races")
    print(overtakes['overtakes_made'].describe().to_string())
