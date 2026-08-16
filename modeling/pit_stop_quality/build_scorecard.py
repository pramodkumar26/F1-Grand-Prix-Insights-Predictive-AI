"""Team pit stop execution scorecard, descriptive rather than a trained model."""
import sqlite3
import pandas as pd

DB_PATH = 'f1.db'
MIN_STOPS = 10


def load(conn):
    return pd.read_sql("""
        SELECT p.race_id, p.driver_id, p.pit_stop_delta, p.avg_stop_duration, p.stop_count,
               r.year, t.team_id, m.team_name
        FROM pit_stop_delta p
        JOIN dim_race r ON p.race_id = r.race_id
        JOIN (SELECT DISTINCT race_id, driver_id, team_id FROM fact_race_results) t
          ON p.race_id = t.race_id AND p.driver_id = t.driver_id
        JOIN dim_team m ON t.team_id = m.team_id
    """, conn)


def build_scorecard():
    conn = sqlite3.connect(DB_PATH)
    df = load(conn)

    card = df.groupby(['team_id', 'team_name', 'year']).agg(
        races=('race_id', 'nunique'),
        driver_races=('race_id', 'size'),
        total_stops=('stop_count', 'sum'),
        mean_delta=('pit_stop_delta', 'mean'),
        median_delta=('pit_stop_delta', 'median'),
        consistency=('pit_stop_delta', 'std'),
        mean_stop_seconds=('avg_stop_duration', 'mean'),
    ).reset_index()

    card = card[card.total_stops >= MIN_STOPS].copy()
    card['rank_in_season'] = card.groupby('year')['mean_delta'].rank(method='min')
    card.to_sql('pit_stop_scorecard', conn, if_exists='replace', index=False)
    conn.commit()
    conn.close()
    return card


def validate(card):
    c = card.sort_values(['team_id', 'year'])
    c['next_year'] = c.groupby('team_id')['year'].shift(-1)
    c['next_delta'] = c.groupby('team_id')['mean_delta'].shift(-1)
    consec = c[c.next_year == c.year + 1].dropna(subset=['next_delta'])
    return consec.mean_delta.corr(consec.next_delta), len(consec)


if __name__ == '__main__':
    card = build_scorecard()
    print(f"pit_stop_scorecard built, {len(card)} team-seasons "
          f"({card.team_name.nunique()} teams, {card.year.nunique()} seasons)")

    corr, n = validate(card)
    print(f"\nVALIDATION, consecutive-season persistence of team execution: "
          f"corr {corr:+.3f} (n={n})")
    print("  for reference: teammate advantage +0.609, tyre degradation +0.200 (failed)")

    print("\nbest executing team-seasons (mean_delta, negative = faster than field median):")
    print(card.nsmallest(8, 'mean_delta')[
        ['team_name', 'year', 'mean_delta', 'consistency', 'total_stops']
    ].round(3).to_string(index=False))

    print("\n2025 ranking:")
    print(card[card.year == 2025].sort_values('mean_delta')[
        ['team_name', 'mean_delta', 'consistency', 'total_stops']
    ].round(3).to_string(index=False))
