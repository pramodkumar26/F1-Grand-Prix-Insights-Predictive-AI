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
    # "Lapped" is FastF1's post-2023 spelling of "+1 Lap", a finisher, not a retirement.
    # Omitting it silently dropped 297 legitimate finishers from 2023 onward, most of
    # them in the test years. See the full explanation in build_dnf_rate.py.
    finished = (results['status'].eq('Finished')
                | results['status'].str.startswith('+', na=False)
                | results['status'].eq('Lapped'))
    results = results[finished].copy()
    results = results.dropna(subset=['grid_position', 'finish_position'])
    results['positions_gained'] = results['grid_position'] - results['finish_position']
    return results


def load_primary_degradation(conn):
    # for each driver's race, picks the stint with the most laps as their primary compound trajectory.
    # Reads tyre_degradation_adjusted, not the original tyre_degradation_stints: the raw slope
    # mixes tyre wear with fuel burn and is contaminated by the standing start, badly enough that
    # its median had the wrong SIGN (-0.027 s/lap, i.e. claiming tyres get faster as they age).
    # See feature_engineering/build_fuel_adjusted_degradation.py for both corrections.
    stints = pd.read_sql(
        "SELECT race_id, driver_id, fuel_adjusted_degradation_rate, lap_count FROM tyre_degradation_adjusted",
        conn
    )
    idx = stints.groupby(['race_id', 'driver_id'])['lap_count'].idxmax()
    primary = stints.loc[idx, ['race_id', 'driver_id', 'fuel_adjusted_degradation_rate']]
    return primary.rename(columns={'fuel_adjusted_degradation_rate': 'primary_degradation_rate'})


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
    # built in phase 2 and previously consumed by no model. Erratic pace is a direct
    # signal that a driver hit trouble, which is what the hardest positions_gained
    # values have in common.
    pace = pd.read_sql("SELECT race_id, driver_id, pace_consistency FROM pace_consistency", conn)
    # an extra stop usually means a puncture, damage or a failed strategy, all of which
    # cost track position in a way the existing features can't see
    stops = pd.read_sql("""SELECT race_id, driver_id, MAX(stint_number) AS pit_stop_count
                           FROM fact_laps GROUP BY race_id, driver_id""", conn)

    table = results.merge(aggressiveness, on=['race_id', 'driver_id'], how='inner')
    table = table.merge(difficulty, on='race_id', how='left')
    table = table.merge(pit_delta, on=['race_id', 'driver_id'], how='left')
    table = table.merge(degradation, on=['race_id', 'driver_id'], how='left')
    table = table.merge(strength, on=['race_id', 'team_id'], how='left')
    table = table.merge(form, on=['race_id', 'driver_id'], how='left')
    table = table.merge(quali, on=['race_id', 'driver_id'], how='left')
    table = table.merge(wet, on='race_id', how='left')
    table = table.merge(pace, on=['race_id', 'driver_id'], how='left')
    table = table.merge(stops, on=['race_id', 'driver_id'], how='left')

    # NOT included, deliberately: blue flag count from fact_race_control. It measured as
    # the single largest available gain (R2 0.364 to 0.384, second highest importance),
    # and it is being declined anyway. A blue flag means the driver is being lapped,
    # which is a symptom of the same trouble that costs positions, not a cause of it.
    # It inflates the metric while making the SHAP explanation worse, "you lost
    # positions because you were shown blue flags" is a restatement of the outcome, not
    # an insight. For a project whose whole point is explainability that is a bad trade.
    table = table[[
        'race_id', 'driver_id', 'team_id', 'year',
        'positions_gained', 'strategic_aggressiveness',
        'track_difficulty_index', 'pit_stop_delta',
        'primary_degradation_rate', 'grid_position', 'team_strength', 'driver_form',
        'qualifying_pace_delta', 'is_wet_race', 'pace_consistency', 'pit_stop_count'
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
