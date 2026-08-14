import sqlite3
import pandas as pd

DB_PATH = 'f1.db'


def load_lap_flags(conn):
    # pulls every lap flag row so we can measure how much of each race ran under a non green flag
    return pd.read_sql("SELECT race_id, track_status FROM lap_flags", conn)


def compute_disruption_ratio(lap_flags):
    # fraction of all driver laps in a race that were run under yellow, safety car, or red flag
    total_laps = lap_flags.groupby('race_id').size()
    disrupted_laps = lap_flags[lap_flags['track_status'] != '1'].groupby('race_id').size()
    ratio = (disrupted_laps / total_laps).fillna(0).reset_index()
    ratio.columns = ['race_id', 'disruption_ratio']
    return ratio


def load_pace_consistency(conn):
    # average pace consistency across the field per race, higher means the whole field struggled
    consistency = pd.read_sql("SELECT race_id, pace_consistency FROM pace_consistency", conn)
    return consistency.groupby('race_id')['pace_consistency'].mean().reset_index()


def zscore(series):
    # standardizes a column so two different scales can be combined into one index
    return (series - series.mean()) / series.std()


def build_track_difficulty_table():
    conn = sqlite3.connect(DB_PATH)
    lap_flags = load_lap_flags(conn)
    disruption = compute_disruption_ratio(lap_flags)
    consistency = load_pace_consistency(conn)

    difficulty = disruption.merge(consistency, on='race_id', how='outer')
    difficulty['disruption_z'] = zscore(difficulty['disruption_ratio'])
    difficulty['consistency_z'] = zscore(difficulty['pace_consistency'])
    difficulty['track_difficulty_index'] = difficulty[['disruption_z', 'consistency_z']].mean(axis=1)

    difficulty.to_sql('track_difficulty_index', conn, if_exists='replace', index=False)
    conn.commit()
    conn.close()

    return difficulty


if __name__ == '__main__':
    difficulty = build_track_difficulty_table()
    print(f"track_difficulty_index table built, {len(difficulty)} rows")
    print(difficulty['track_difficulty_index'].describe())