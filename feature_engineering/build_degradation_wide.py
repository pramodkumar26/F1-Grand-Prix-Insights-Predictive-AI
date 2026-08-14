import sqlite3
import pandas as pd

DB_PATH = 'f1.db'


def load_degradation_stints(conn):
    # pulls the granular stint level degradation table along with compound names
    query = """
        SELECT s.race_id, s.driver_id, s.compound_id, s.degradation_rate, s.lap_count, c.compound_name
        FROM tyre_degradation_stints s
        JOIN dim_tyre_compound c ON s.compound_id = c.compound_id
    """
    return pd.read_sql(query, conn)


def weighted_average_by_compound(stints):
    # collapses multiple stints on the same compound in a race into one lap weighted average
    def weighted_mean(group):
        return (group['degradation_rate'] * group['lap_count']).sum() / group['lap_count'].sum()

    grouped = stints.groupby(['race_id', 'driver_id', 'compound_name']).apply(weighted_mean)
    return grouped.reset_index(name='degradation_rate')


def pivot_to_wide(collapsed):
    # one row per race per driver, one column per compound
    wide = collapsed.pivot_table(
        index=['race_id', 'driver_id'],
        columns='compound_name',
        values='degradation_rate'
    ).reset_index()

    wide.columns = [
        f"degradation_rate_{col.lower()}" if col not in ('race_id', 'driver_id') else col
        for col in wide.columns
    ]
    return wide


def build_degradation_wide_table():
    conn = sqlite3.connect(DB_PATH)
    stints = load_degradation_stints(conn)
    collapsed = weighted_average_by_compound(stints)
    wide = pivot_to_wide(collapsed)

    wide.to_sql('degradation_wide', conn, if_exists='replace', index=False)
    conn.commit()
    conn.close()

    return wide


if __name__ == '__main__':
    wide = build_degradation_wide_table()
    print(f"degradation_wide table built, {len(wide)} rows")
    print(wide.columns.tolist())
    print(wide.isna().sum())