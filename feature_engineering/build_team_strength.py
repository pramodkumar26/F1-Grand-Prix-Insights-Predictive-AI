import sqlite3
import pandas as pd

DB_PATH = 'f1.db'


def load_results(conn):
    # points per driver per race, plus year/round for chronological ordering
    query = """
        SELECT r.race_id, r.team_id, r.driver_id, r.points, d.year, d.round
        FROM fact_race_results r
        JOIN dim_race d ON r.race_id = d.race_id
    """
    return pd.read_sql(query, conn)


def collapse_to_team_race(results):
    # one row per team per race first, same leakage guard used in build_dnf_rate.py,
    # otherwise two teammates in the same race could leak into each other's history
    team_race = results.groupby(['team_id', 'race_id', 'year', 'round'])['points'].sum().reset_index()
    return team_race.rename(columns={'points': 'team_points'})


def compute_trailing_strength(team_race):
    # trailing average points per race, strictly prior races only, broadcast back to drivers after
    rates = []
    for team_id, group in team_race.groupby('team_id'):
        group = group.sort_values(['year', 'round']).copy()
        group['team_strength'] = group['team_points'].shift().expanding().mean().values
        rates.append(group)

    return pd.concat(rates, ignore_index=True)


def build_team_strength_table():
    conn = sqlite3.connect(DB_PATH)
    results = load_results(conn)

    team_race = collapse_to_team_race(results)
    team_race = compute_trailing_strength(team_race)

    output = results.merge(
        team_race[['team_id', 'race_id', 'team_strength']],
        on=['team_id', 'race_id'],
        how='left'
    )
    output = output[['race_id', 'team_id', 'driver_id', 'team_strength']]

    output.to_sql('team_strength', conn, if_exists='replace', index=False)
    conn.commit()
    conn.close()

    return output


if __name__ == '__main__':
    output = build_team_strength_table()
    print(f"team_strength table built, {len(output)} rows")
    print(f"rows with no trailing history yet: {output['team_strength'].isna().sum()}")
    print(output['team_strength'].describe())
