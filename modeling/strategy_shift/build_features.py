import sqlite3
import pandas as pd

DB_PATH = 'f1.db'


def load_results(conn):
    # pulls grid position, finish position, status, and year, classified finishers only
    query = """
        SELECT r.race_id, r.driver_id, r.team_id, r.grid_position, r.finish_position, r.status, d.year
        FROM fact_race_results r
        JOIN dim_race d ON r.race_id = d.race_id
    """
    results = pd.read_sql(query, conn)
    # grid_position=0 pit lane starts are fixed at the source, see collecting_data/fix_grid_position.py
    finished = results['status'].eq('Finished') | results['status'].str.startswith('+', na=False)
    results = results[finished].copy()
    results = results.dropna(subset=['grid_position', 'finish_position'])
    results['positions_gained'] = results['grid_position'] - results['finish_position']
    return results


def load_primary_degradation(conn):
    # for each driver's race, picks the stint with the most laps as their primary compound trajectory
    stints = pd.read_sql("SELECT race_id, driver_id, degradation_rate, lap_count FROM tyre_degradation_stints", conn)
    idx = stints.groupby(['race_id', 'driver_id'])['lap_count'].idxmax()
    primary = stints.loc[idx, ['race_id', 'driver_id', 'degradation_rate']]
    return primary.rename(columns={'degradation_rate': 'primary_degradation_rate'})


def build_strategy_shift_table():
    conn = sqlite3.connect(DB_PATH)

    results = load_results(conn)
    aggressiveness = pd.read_sql("SELECT race_id, driver_id, strategic_aggressiveness FROM strategic_aggressiveness", conn)
    difficulty = pd.read_sql("SELECT race_id, track_difficulty_index FROM track_difficulty_index", conn)
    pit_delta = pd.read_sql("SELECT race_id, driver_id, pit_stop_delta FROM pit_stop_delta", conn)
    degradation = load_primary_degradation(conn)
    # team_strength has one row per driver with an identical value per teammate,
    # dedupe to one row per team per race before merging, or the join fans out 2x
    strength = pd.read_sql("SELECT DISTINCT race_id, team_id, team_strength FROM team_strength", conn)
    form = pd.read_sql("SELECT race_id, driver_id, driver_form FROM driver_form", conn)
    quali = pd.read_sql("SELECT race_id, driver_id, qualifying_pace_delta FROM qualifying_pace_delta", conn)
    wet = pd.read_sql("SELECT race_id, is_wet_race FROM race_wet_flag", conn)

    table = results.merge(aggressiveness, on=['race_id', 'driver_id'], how='inner')
    table = table.merge(difficulty, on='race_id', how='left')
    table = table.merge(pit_delta, on=['race_id', 'driver_id'], how='left')
    table = table.merge(degradation, on=['race_id', 'driver_id'], how='left')
    table = table.merge(strength, on=['race_id', 'team_id'], how='left')
    table = table.merge(form, on=['race_id', 'driver_id'], how='left')
    table = table.merge(quali, on=['race_id', 'driver_id'], how='left')
    table = table.merge(wet, on='race_id', how='left')

    table = table[[
        'race_id', 'driver_id', 'team_id', 'year',
        'positions_gained', 'strategic_aggressiveness',
        'track_difficulty_index', 'pit_stop_delta',
        'primary_degradation_rate', 'grid_position', 'team_strength', 'driver_form',
        'qualifying_pace_delta', 'is_wet_race'
    ]]

    table.to_sql('strategy_shift_features', conn, if_exists='replace', index=False)
    conn.commit()
    conn.close()

    return table


if __name__ == '__main__':
    table = build_strategy_shift_table()
    print(f"strategy_shift_features table built, {len(table)} rows")
    print(f"train, 2018 to 2023: {(table['year'] <= 2023).sum()} rows")
    print(f"test, 2024 to 2025: {(table['year'] >= 2024).sum()} rows")
    print("null counts per column:")
    print(table.isna().sum())
    print(table.describe())
