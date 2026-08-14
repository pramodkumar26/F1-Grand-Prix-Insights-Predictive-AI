import sqlite3
import pandas as pd

DB_PATH = 'f1.db'
WET_THRESHOLD = 0.05


def load_weather(conn):
    # pulls every weather sample, race_id repeats since these are time sampled readings
    return pd.read_sql("SELECT race_id, rainfall FROM fact_weather", conn)


def classify_wet_races(weather):
    # a race counts as wet if rain showed up in more than a small fraction of its samples
    wet_fraction = weather.groupby('race_id')['rainfall'].mean()
    return wet_fraction[wet_fraction > WET_THRESHOLD].index.tolist()


def load_results(conn):
    # pulls grid and finish position so we can compute positions gained
    return pd.read_sql("SELECT race_id, driver_id, grid_position, finish_position FROM fact_race_results", conn)


def load_pairs(conn):
    # pulls teammate relationships, drops solo entries with no teammate to compare against
    pairs = pd.read_sql("SELECT race_id, driver_id, teammate_driver_id FROM teammate_pairs", conn)
    return pairs.dropna(subset=['teammate_driver_id'])


def compute_wet_weather_delta(pairs, results, wet_races):
    # restricts to wet races, computes positions gained, then compares driver against teammate
    results = results[results['race_id'].isin(wet_races)].copy()
    results = results.dropna(subset=['grid_position', 'finish_position'])
    results['positions_gained'] = results['grid_position'] - results['finish_position']

    merged = pairs.merge(results[['race_id', 'driver_id', 'positions_gained']], on=['race_id', 'driver_id'], how='inner')

    teammate_values = results[['race_id', 'driver_id', 'positions_gained']].rename(
        columns={'driver_id': 'teammate_driver_id', 'positions_gained': 'teammate_positions_gained'}
    )
    merged = merged.merge(teammate_values, on=['race_id', 'teammate_driver_id'], how='inner')

    merged['wet_weather_delta'] = merged['positions_gained'] - merged['teammate_positions_gained']
    return merged[['race_id', 'driver_id', 'teammate_driver_id', 'positions_gained', 'teammate_positions_gained', 'wet_weather_delta']]


def build_wet_weather_delta_table():
    conn = sqlite3.connect(DB_PATH)
    weather = load_weather(conn)
    wet_races = classify_wet_races(weather)
    results = load_results(conn)
    pairs = load_pairs(conn)

    delta = compute_wet_weather_delta(pairs, results, wet_races)

    delta.to_sql('wet_weather_delta', conn, if_exists='replace', index=False)
    conn.commit()
    conn.close()

    return delta, wet_races


if __name__ == '__main__':
    delta, wet_races = build_wet_weather_delta_table()
    print(f"wet races identified: {len(wet_races)}")
    print(f"wet_weather_delta table built, {len(delta)} rows")
    print(delta['wet_weather_delta'].describe())