import sqlite3
import pandas as pd

DB_PATH = 'f1.db'


def load_results(conn):
    # identical to podium/build_features.py's load_results, every driver who actually
    # started (has a grid_position), DNFs included as legitimate is_win=0 rows
    query = """
        SELECT r.race_id, r.driver_id, r.team_id, r.grid_position, r.finish_position, d.year
        FROM fact_race_results r
        JOIN dim_race d ON r.race_id = d.race_id
    """
    results = pd.read_sql(query, conn)
    results = results.dropna(subset=['grid_position'])
    results['is_win'] = (results['finish_position'] == 1).astype(int)
    return results


def build_win_features_table():
    conn = sqlite3.connect(DB_PATH)

    results = load_results(conn)
    # identical pre-race-safe feature set as podium_features.py, reused directly, not
    # rebuilt, same reasoning: only information knowable after qualifying, before the
    # race starts
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
        'is_win', 'grid_position', 'team_strength', 'driver_form',
        'qualifying_pace_delta', 'qualifying_gap_to_pole'
    ]]

    table.to_sql('win_features', conn, if_exists='replace', index=False)
    conn.commit()
    conn.close()
    return table


if __name__ == '__main__':
    table = build_win_features_table()
    print(f"win_features table built, {len(table)} rows")
    print(f"win rate: {table['is_win'].mean():.4f}")
    print(f"train, 2018-2022: {(table.year <= 2022).sum()} rows")
    print(f"val, 2023: {(table.year == 2023).sum()} rows")
    print(f"test, 2024-2025: {(table.year >= 2024).sum()} rows")
    print("null counts:")
    print(table.isna().sum().to_string())
