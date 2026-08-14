import sqlite3
import pandas as pd

DB_PATH = 'f1.db'


def load_results(conn):
    # pulls team, driver, race, and finish status, plus year and round for chronological ordering
    query = """
        SELECT r.race_id, r.team_id, r.driver_id, r.status, d.year, d.round
        FROM fact_race_results r
        JOIN dim_race d ON r.race_id = d.race_id
    """
    return pd.read_sql(query, conn)


def classify_dnf(results):
    # anything other than finishing on the lead lap or a lapped classification counts as a DNF
    finished = results['status'].eq('Finished') | results['status'].str.startswith('+', na=False)
    results['dnf'] = ~finished
    return results


def collapse_to_team_race(results):
    # one row per team per race first, so teammates in the same race can never leak into each other's history
    team_race = results.groupby(['team_id', 'race_id', 'year', 'round'])['dnf'].mean().reset_index()
    return team_race.rename(columns={'dnf': 'team_dnf_fraction'})


def compute_trailing_dnf_rate(team_race):
    # trailing average at the team race level, strictly prior races only, broadcast back to drivers after
    rates = []
    for team_id, group in team_race.groupby('team_id'):
        group = group.sort_values(['year', 'round']).copy()
        group['team_dnf_rate'] = group['team_dnf_fraction'].shift().expanding().mean().values
        rates.append(group)

    return pd.concat(rates, ignore_index=True)


def build_dnf_rate_table():
    conn = sqlite3.connect(DB_PATH)
    results = load_results(conn)
    results = classify_dnf(results)

    team_race = collapse_to_team_race(results)
    team_race = compute_trailing_dnf_rate(team_race)

    output = results.merge(
        team_race[['team_id', 'race_id', 'team_dnf_rate']],
        on=['team_id', 'race_id'],
        how='left'
    )
    output = output[['race_id', 'team_id', 'driver_id', 'dnf', 'team_dnf_rate']]

    output.to_sql('team_dnf_rate', conn, if_exists='replace', index=False)
    conn.commit()
    conn.close()

    return output


if __name__ == '__main__':
    output = build_dnf_rate_table()
    print(f"team_dnf_rate table built, {len(output)} rows")
    print(f"rows with no trailing history yet: {output['team_dnf_rate'].isna().sum()}")
    print(output['team_dnf_rate'].describe())