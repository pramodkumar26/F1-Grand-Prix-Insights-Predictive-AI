import sqlite3
import pandas as pd

DB_PATH = 'f1.db'


def load_results(conn):
    # pulls driver and team per race, the raw material for figuring out who partnered whom
    return pd.read_sql("SELECT race_id, driver_id, team_id FROM fact_race_results", conn)


def build_pairs(results):
    # for each race and team, pairs up whichever two drivers actually raced together that weekend
    pairs = []
    anomalies = []

    for (race_id, team_id), group in results.groupby(['race_id', 'team_id']):
        drivers = group['driver_id'].tolist()

        if len(drivers) == 2:
            pairs.append({'race_id': race_id, 'team_id': team_id, 'driver_id': drivers[0], 'teammate_driver_id': drivers[1]})
            pairs.append({'race_id': race_id, 'team_id': team_id, 'driver_id': drivers[1], 'teammate_driver_id': drivers[0]})
        elif len(drivers) == 1:
            pairs.append({'race_id': race_id, 'team_id': team_id, 'driver_id': drivers[0], 'teammate_driver_id': None})
        else:
            anomalies.append({'race_id': race_id, 'team_id': team_id, 'driver_count': len(drivers)})

    return pd.DataFrame(pairs), pd.DataFrame(anomalies)


def build_teammate_pairs_table():
    conn = sqlite3.connect(DB_PATH)
    results = load_results(conn)
    pairs, anomalies = build_pairs(results)

    pairs.to_sql('teammate_pairs', conn, if_exists='replace', index=False)
    conn.commit()
    conn.close()

    return pairs, anomalies


if __name__ == '__main__':
    pairs, anomalies = build_teammate_pairs_table()
    print(f"teammate_pairs table built, {len(pairs)} rows")
    print(f"solo entries with no teammate that race: {pairs['teammate_driver_id'].isna().sum()}")
    if len(anomalies):
        print(f"{len(anomalies)} race and team combos had more than two drivers, needs a look")
        print(anomalies)