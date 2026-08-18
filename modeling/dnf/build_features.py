import sqlite3
import pandas as pd

DB_PATH = 'f1.db'


def load_results(conn):
    return pd.read_sql("""
        SELECT r.race_id, r.driver_id, r.team_id, r.grid_position,
               d.year, d.round, d.circuit_id,
               t.dnf, t.team_dnf_rate
        FROM fact_race_results r
        JOIN dim_race d ON r.race_id = d.race_id
        JOIN team_dnf_rate t ON r.race_id = t.race_id AND r.driver_id = t.driver_id
        WHERE r.grid_position IS NOT NULL
    """, conn)


def add_trailing_rate(df, key, out_col):
    per_race = df.groupby([key, 'race_id', 'year', 'round'])['dnf'].mean().reset_index()

    out = []
    for _, group in per_race.groupby(key):
        group = group.sort_values(['year', 'round']).copy()
        group[out_col] = group['dnf'].shift().expanding().mean().values
        out.append(group)

    trailing = pd.concat(out, ignore_index=True)
    return df.merge(trailing[[key, 'race_id', out_col]], on=[key, 'race_id'], how='left')


def build_dnf_features_table():
    conn = sqlite3.connect(DB_PATH)

    df = load_results(conn)
    df = add_trailing_rate(df, 'circuit_id', 'circuit_dnf_trailing')
    df = add_trailing_rate(df, 'driver_id', 'driver_dnf_trailing')

    strength = pd.read_sql("SELECT DISTINCT race_id, team_id, team_strength FROM team_strength", conn)
    quali = pd.read_sql("SELECT race_id, driver_id, qualifying_pace_delta FROM qualifying_pace_delta", conn)
    df = df.merge(strength, on=['race_id', 'team_id'], how='left')
    df = df.merge(quali, on=['race_id', 'driver_id'], how='left')

    table = df[[
        'race_id', 'driver_id', 'team_id', 'year',
        'dnf', 'grid_position', 'team_dnf_rate', 'circuit_dnf_trailing',
        'driver_dnf_trailing', 'team_strength', 'qualifying_pace_delta',
    ]].copy()
    table['dnf'] = table['dnf'].astype(int)

    table.to_sql('dnf_features', conn, if_exists='replace', index=False)
    conn.commit()
    conn.close()
    return table


if __name__ == '__main__':
    table = build_dnf_features_table()
    print(f"dnf_features table built, {len(table)} rows")
    print(f"DNF base rate: {table['dnf'].mean():.3f}")
    print(f"train 2018-2022: {(table.year <= 2022).sum()}, "
          f"val 2023: {(table.year == 2023).sum()}, "
          f"test 2024+: {(table.year >= 2024).sum()}")
    print(f"DNF rate by split: train {table[table.year<=2022].dnf.mean():.3f}, "
          f"val {table[table.year==2023].dnf.mean():.3f}, "
          f"test {table[table.year>=2024].dnf.mean():.3f}")
    print("\nnull counts:")
    print(table.isna().sum().to_string())
