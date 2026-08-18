import sqlite3
import pandas as pd

DB_PATH = 'f1.db'
MIN_R_SQUARED = 0.3
MIN_LAP_COUNT = 6


def load_stints(conn):
    query = """
        SELECT s.race_id, s.driver_id, s.stint_number, s.compound_id,
               s.fuel_adjusted_degradation_rate, s.r_squared, s.lap_count, s.stint_start_lap,
               r.year, r.round, r.circuit_id,
               l.team_id
        FROM tyre_degradation_adjusted s
        JOIN dim_race r ON s.race_id = r.race_id
        JOIN (SELECT DISTINCT race_id, driver_id, team_id FROM fact_laps) l
          ON s.race_id = l.race_id AND s.driver_id = l.driver_id
    """
    stints = pd.read_sql(query, conn)
    return stints[(stints.r_squared > MIN_R_SQUARED) & (stints.lap_count >= MIN_LAP_COUNT)].copy()


def add_race_start_weather(stints, conn):
    # conditions at the start of the session only, avoids leaking later-race readings
    weather = pd.read_sql("""
        SELECT race_id, air_temp, track_temp
        FROM fact_weather w
        WHERE w.session_time_seconds <= (
            SELECT MIN(session_time_seconds) + 300 FROM fact_weather w2 WHERE w2.race_id = w.race_id
        )
    """, conn)
    start = weather.groupby('race_id').agg(
        race_start_air_temp=('air_temp', 'mean'),
        race_start_track_temp=('track_temp', 'mean'),
    ).reset_index()
    return stints.merge(start, on='race_id', how='left')


def add_trailing(stints, group_cols, out_col):
    per_race = (stints.groupby(group_cols + ['race_id', 'year', 'round'])
                ['fuel_adjusted_degradation_rate'].mean().reset_index())

    out = []
    for _, group in per_race.groupby(group_cols):
        group = group.sort_values(['year', 'round']).copy()
        group[out_col] = group['fuel_adjusted_degradation_rate'].shift().expanding().mean().values
        out.append(group)

    trailing = pd.concat(out, ignore_index=True)
    return stints.merge(trailing[group_cols + ['race_id', out_col]], on=group_cols + ['race_id'], how='left')


def build_tyre_features_table():
    conn = sqlite3.connect(DB_PATH)

    stints = load_stints(conn)
    stints = add_race_start_weather(stints, conn)
    stints = add_trailing(stints, ['circuit_id'], 'circuit_degradation_trailing')
    stints = add_trailing(stints, ['driver_id'], 'driver_degradation_trailing')

    strength = pd.read_sql("SELECT DISTINCT race_id, team_id, team_strength FROM team_strength", conn)
    stints = stints.merge(strength, on=['race_id', 'team_id'], how='left')

    table = stints[[
        'race_id', 'driver_id', 'team_id', 'stint_number', 'year',
        'fuel_adjusted_degradation_rate',
        'compound_id', 'stint_start_lap',
        'race_start_air_temp', 'race_start_track_temp',
        'circuit_degradation_trailing', 'driver_degradation_trailing', 'team_strength',
    ]]

    table.to_sql('tyre_degradation_features', conn, if_exists='replace', index=False)
    conn.commit()
    conn.close()
    return table


if __name__ == '__main__':
    table = build_tyre_features_table()
    print(f"tyre_degradation_features table built, {len(table)} stints")
    print(f"train 2018-2022: {(table.year <= 2022).sum()}, "
          f"val 2023: {(table.year == 2023).sum()}, "
          f"test 2024+: {(table.year >= 2024).sum()}")
    print("\nnull counts:")
    print(table.isna().sum().to_string())
    print(f"\ntarget (fuel_adjusted_degradation_rate):")
    print(table['fuel_adjusted_degradation_rate'].describe().to_string())
