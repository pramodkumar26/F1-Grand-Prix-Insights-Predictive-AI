import sqlite3
import pandas as pd

DB_PATH = 'f1.db'
WET_THRESHOLD = 0.05


def load_weather(conn):
    # pulls every weather sample, race_id repeats since these are time sampled readings
    return pd.read_sql("SELECT race_id, rainfall FROM fact_weather", conn)


def classify_wet_races(weather):
    # same threshold and logic as build_wet_weather_delta.py, a race counts as wet if rain
    # showed up in more than a small fraction of its samples
    wet_fraction = weather.groupby('race_id')['rainfall'].mean().reset_index()
    wet_fraction['is_wet_race'] = (wet_fraction['rainfall'] > WET_THRESHOLD).astype(int)
    return wet_fraction[['race_id', 'is_wet_race']]


def build_race_wet_flag_table():
    conn = sqlite3.connect(DB_PATH)
    weather = load_weather(conn)
    flags = classify_wet_races(weather)

    flags.to_sql('race_wet_flag', conn, if_exists='replace', index=False)
    conn.commit()
    conn.close()

    return flags


if __name__ == '__main__':
    flags = build_race_wet_flag_table()
    print(f"race_wet_flag table built, {len(flags)} rows")
    print(f"wet races: {flags['is_wet_race'].sum()} / {len(flags)}")
