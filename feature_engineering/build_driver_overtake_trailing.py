import sqlite3
import pandas as pd

DB_PATH = 'f1.db'


def load_overtakes(conn):
    return pd.read_sql("""
        SELECT o.race_id, o.driver_id, o.overtakes_made, d.year, d.round
        FROM overtakes_made o
        JOIN dim_race d ON o.race_id = d.race_id
    """, conn)


def compute_trailing(overtakes):
    # strictly prior races only, same no-leakage trailing pattern as build_driver_form.py
    # and build_team_strength.py, this is a driver's own racecraft signal, independent
    # of the car, and it must not see this race's own outcome.
    out = []
    for _, group in overtakes.groupby('driver_id'):
        group = group.sort_values(['year', 'round']).copy()
        group['driver_overtake_trailing'] = group['overtakes_made'].shift().expanding().mean().values
        out.append(group)
    return pd.concat(out, ignore_index=True)


def build_driver_overtake_trailing_table():
    conn = sqlite3.connect(DB_PATH)
    overtakes = load_overtakes(conn)
    trailing = compute_trailing(overtakes)

    output = trailing[['race_id', 'driver_id', 'driver_overtake_trailing']]
    output.to_sql('driver_overtake_trailing', conn, if_exists='replace', index=False)
    conn.commit()
    conn.close()
    return output


if __name__ == '__main__':
    output = build_driver_overtake_trailing_table()
    print(f"driver_overtake_trailing table built, {len(output)} rows")
    print(f"rows with no trailing history yet: {output['driver_overtake_trailing'].isna().sum()}")
    print(output['driver_overtake_trailing'].describe().to_string())
