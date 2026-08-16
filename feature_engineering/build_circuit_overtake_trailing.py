import sqlite3
import pandas as pd

DB_PATH = 'f1.db'


def load_overtakes(conn):
    return pd.read_sql("""
        SELECT o.race_id, o.overtakes_made, r.year, r.round, r.circuit_id
        FROM overtakes_made o
        JOIN dim_race r ON o.race_id = r.race_id
    """, conn)


def compute_trailing(overtakes):
    # strictly prior races at this circuit only, same no-leakage pattern as every other
    # trailing feature in the project. Collapsed to one value per race per circuit first
    # (mean across the field that race), then an all-time expanding mean over prior
    # races. A rolling window was tested (see model_log.md for the full account, an
    # exploratory test script had a bug that made a window look like a real improvement
    # when it wasn't), but with at most 10 races for any circuit across the whole
    # 2018-2025 span, a window of 8+ is functionally almost identical to using all
    # history anyway, most circuits never accumulate enough races for windowing to mean
    # anything different. Kept as expanding, a deliberately equivalent windowed config
    # would be misleading about what the code is actually doing.
    per_race = overtakes.groupby(['circuit_id', 'race_id', 'year', 'round'])['overtakes_made'].mean().reset_index()

    out = []
    for _, group in per_race.groupby('circuit_id'):
        group = group.sort_values(['year', 'round']).copy()
        group['circuit_overtake_trailing'] = group['overtakes_made'].shift().expanding().mean().values
        out.append(group)
    return pd.concat(out, ignore_index=True)


def build_circuit_overtake_trailing_table():
    conn = sqlite3.connect(DB_PATH)
    overtakes = load_overtakes(conn)
    trailing = compute_trailing(overtakes)

    output = trailing[['race_id', 'circuit_overtake_trailing']]
    output.to_sql('circuit_overtake_trailing', conn, if_exists='replace', index=False)
    conn.commit()
    conn.close()
    return output


if __name__ == '__main__':
    output = build_circuit_overtake_trailing_table()
    print(f"circuit_overtake_trailing table built, {len(output)} rows")
    print(f"rows with no trailing history yet (circuit's first race in the dataset): "
          f"{output['circuit_overtake_trailing'].isna().sum()}")
    print(output['circuit_overtake_trailing'].describe().to_string())
