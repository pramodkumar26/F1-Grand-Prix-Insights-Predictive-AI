import sqlite3
import pandas as pd

DB_PATH = 'f1.db'


def load_results(conn):
    query = """
        SELECT r.race_id, r.driver_id, r.team_id, r.grid_position, r.points, d.year
        FROM fact_race_results r
        JOIN dim_race d ON r.race_id = d.race_id
    """
    results = pd.read_sql(query, conn)
    return results.dropna(subset=['grid_position'])


def build_points_features_table():
    conn = sqlite3.connect(DB_PATH)

    results = load_results(conn)
    strength = pd.read_sql("SELECT DISTINCT race_id, team_id, team_strength FROM team_strength", conn)
    form = pd.read_sql("SELECT race_id, driver_id, driver_form FROM driver_form", conn)
    quali = pd.read_sql("SELECT race_id, driver_id, qualifying_pace_delta FROM qualifying_pace_delta", conn)
    gap_to_pole = pd.read_sql("SELECT race_id, driver_id, qualifying_gap_to_pole FROM qualifying_gap_to_pole", conn)

    table = results.merge(strength, on=['race_id', 'team_id'], how='left')
    table = table.merge(form, on=['race_id', 'driver_id'], how='left')
    table = table.merge(quali, on=['race_id', 'driver_id'], how='left')
    table = table.merge(gap_to_pole, on=['race_id', 'driver_id'], how='left')

    table = table[[
        'race_id', 'driver_id', 'team_id', 'year',
        'points', 'grid_position', 'team_strength', 'driver_form',
        'qualifying_pace_delta', 'qualifying_gap_to_pole'
    ]]

    table.to_sql('points_scored_features', conn, if_exists='replace', index=False)
    conn.commit()
    conn.close()
    return table


if __name__ == '__main__':
    table = build_points_features_table()
    print(f"points_scored_features table built, {len(table)} rows")
    print(f"train 2018-2022: {(table.year <= 2022).sum()}, "
          f"val 2023: {(table.year == 2023).sum()}, "
          f"test 2024+: {(table.year >= 2024).sum()}")
    print("\nnull counts:")
    print(table.isna().sum().to_string())
    print(f"\npoints distribution:")
    print(table['points'].describe().to_string())
    print(f"races with 0 points: {(table.points == 0).mean()*100:.1f}%")
