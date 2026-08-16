import sqlite3
import pandas as pd

DB_PATH = 'f1.db'
MIN_SHARED_RACES = 5


def load_pairs(conn):
    return pd.read_sql("""
        SELECT p.race_id, p.driver_id, p.teammate_driver_id, r.year, t.team_id
        FROM teammate_pairs p
        JOIN dim_race r ON p.race_id = r.race_id
        JOIN fact_race_results t ON p.race_id = t.race_id AND p.driver_id = t.driver_id
        WHERE p.teammate_driver_id IS NOT NULL
    """, conn)


def load_results(conn):
    # "Lapped" is FastF1's post-2023 spelling of "+1 Lap", a finisher. See
    # feature_engineering/build_dnf_rate.py for why omitting it corrupts era comparisons.
    res = pd.read_sql("""
        SELECT race_id, driver_id, grid_position, finish_position, status
        FROM fact_race_results
    """, conn)
    res['finished'] = (res.status.eq('Finished')
                       | res.status.str.startswith('+', na=False)
                       | res.status.eq('Lapped'))
    return res


def head_to_head(pairs, results, col, require_finish, out_prefix):
    # Fairness rule: a head to head only counts when BOTH drivers produced a comparable
    # result. If a teammate retired with a mechanical failure, finishing ahead of them
    # says nothing about relative skill, so that race is excluded rather than scored as
    # a win. Same logic for qualifying, both must have set a valid time.
    own = results[['race_id', 'driver_id', col, 'finished']].rename(
        columns={'driver_id': 'd', col: 'own', 'finished': 'own_fin'})
    mate = results[['race_id', 'driver_id', col, 'finished']].rename(
        columns={'driver_id': 'm', col: 'mate', 'finished': 'mate_fin'})

    df = pairs.merge(own, left_on=['race_id', 'driver_id'], right_on=['race_id', 'd'], how='inner')
    df = df.merge(mate, left_on=['race_id', 'teammate_driver_id'], right_on=['race_id', 'm'], how='inner')
    df = df.dropna(subset=['own', 'mate'])
    if require_finish:
        df = df[df.own_fin & df.mate_fin]

    df[f'{out_prefix}_win'] = (df['own'] < df['mate']).astype(int)
    return df.groupby(['driver_id', 'year']).agg(
        **{f'{out_prefix}_wins': (f'{out_prefix}_win', 'sum'),
           f'{out_prefix}_races': (f'{out_prefix}_win', 'size')}
    ).reset_index()


def build_scorecard():
    conn = sqlite3.connect(DB_PATH)
    pairs = load_pairs(conn)
    results = load_results(conn)

    quali = head_to_head(pairs, results, 'grid_position', False, 'quali')
    race = head_to_head(pairs, results, 'finish_position', True, 'race')

    pace = pd.read_sql("SELECT race_id, driver_id, qualifying_pace_delta FROM qualifying_pace_delta", conn)
    pace = pairs.merge(pace, on=['race_id', 'driver_id'], how='left')
    pace = pace.groupby(['driver_id', 'year'])['qualifying_pace_delta'].median().reset_index()
    pace = pace.rename(columns={'qualifying_pace_delta': 'quali_gap_median'})

    wet = pd.read_sql("SELECT race_id, driver_id, wet_weather_delta FROM wet_weather_delta", conn)
    wet = pairs.merge(wet, on=['race_id', 'driver_id'], how='left')
    wet = wet.groupby(['driver_id', 'year'])['wet_weather_delta'].mean().reset_index()

    tyre = pd.read_sql("SELECT * FROM tyre_wear_delta", conn)
    delta_cols = [c for c in tyre.columns if c.startswith('tyre_wear_delta_')]
    tyre['tyre_wear_advantage'] = -tyre[delta_cols].mean(axis=1, skipna=True)
    tyre = pairs.merge(tyre[['race_id', 'driver_id', 'tyre_wear_advantage']], on=['race_id', 'driver_id'], how='left')
    tyre = tyre.groupby(['driver_id', 'year'])['tyre_wear_advantage'].mean().reset_index()

    card = (quali.merge(race, on=['driver_id', 'year'], how='outer')
                 .merge(pace, on=['driver_id', 'year'], how='left')
                 .merge(wet, on=['driver_id', 'year'], how='left')
                 .merge(tyre, on=['driver_id', 'year'], how='left'))

    card['quali_win_pct'] = card.quali_wins / card.quali_races
    card['race_win_pct'] = card.race_wins / card.race_races
    card = card[card.quali_races >= MIN_SHARED_RACES].copy()

    names = pd.read_sql("SELECT driver_id, driver_code, driver_name FROM dim_driver", conn)
    card = card.merge(names, on='driver_id', how='left')

    card.to_sql('teammate_scorecard', conn, if_exists='replace', index=False)
    conn.commit()
    conn.close()
    return card


if __name__ == '__main__':
    card = build_scorecard()
    print(f"teammate_scorecard built, {len(card)} driver-seasons "
          f"({card.driver_id.nunique()} drivers, {card.year.nunique()} seasons)")
    print(f"\nqualifying H2H is the cleanest signal: same car, same session, minimal luck.")
    print(f"race H2H excludes any race where either driver failed to finish.\n")
    top = card.nlargest(8, 'quali_win_pct')[['driver_code', 'year', 'quali_wins', 'quali_races',
                                             'quali_win_pct', 'race_win_pct', 'quali_gap_median']]
    print("strongest qualifying records vs teammate:")
    print(top.to_string(index=False))
