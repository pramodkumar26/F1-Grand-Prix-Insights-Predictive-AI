import sqlite3
import pandas as pd

DB_PATH = 'f1.db'


def load_results(conn):
    return pd.read_sql("""
        SELECT r.race_id, r.driver_id, r.team_id, r.grid_position, d.year
        FROM fact_race_results r
        JOIN dim_race d ON r.race_id = d.race_id
        WHERE r.grid_position IS NOT NULL
    """, conn)


def build_overtaking_features_table():
    conn = sqlite3.connect(DB_PATH)

    results = load_results(conn)
    overtakes = pd.read_sql("SELECT race_id, driver_id, overtakes_made FROM overtakes_made", conn)
    trailing = pd.read_sql("SELECT race_id, driver_id, driver_overtake_trailing FROM driver_overtake_trailing", conn)
    strength = pd.read_sql("SELECT DISTINCT race_id, team_id, team_strength FROM team_strength", conn)
    quali = pd.read_sql("SELECT race_id, driver_id, qualifying_pace_delta FROM qualifying_pace_delta", conn)
    gap = pd.read_sql("SELECT race_id, driver_id, qualifying_gap_to_pole FROM qualifying_gap_to_pole", conn)
    circuit_trailing = pd.read_sql("SELECT race_id, circuit_overtake_trailing FROM circuit_overtake_trailing", conn)

    table = results.merge(overtakes, on=['race_id', 'driver_id'], how='inner')
    table = table.merge(trailing, on=['race_id', 'driver_id'], how='left')
    table = table.merge(strength, on=['race_id', 'team_id'], how='left')
    table = table.merge(quali, on=['race_id', 'driver_id'], how='left')
    table = table.merge(gap, on=['race_id', 'driver_id'], how='left')
    table = table.merge(circuit_trailing, on='race_id', how='left')

    table = table[[
        'race_id', 'driver_id', 'team_id', 'year',
        'overtakes_made', 'grid_position', 'team_strength', 'driver_overtake_trailing',
        'qualifying_pace_delta', 'qualifying_gap_to_pole', 'circuit_overtake_trailing'
    ]]

    table.to_sql('overtaking_features', conn, if_exists='replace', index=False)
    conn.commit()
    conn.close()
    return table


if __name__ == '__main__':
    table = build_overtaking_features_table()
    print(f"overtaking_features table built, {len(table)} rows")
    print(f"train 2018-2022: {(table.year <= 2022).sum()}, "
          f"val 2023: {(table.year == 2023).sum()}, "
          f"test 2024+: {(table.year >= 2024).sum()}")
    print("\nnull counts:")
    print(table.isna().sum().to_string())
    print(f"\novertakes_made distribution:")
    print(table['overtakes_made'].describe().to_string())
