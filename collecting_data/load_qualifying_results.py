import sqlite3
import time
import warnings

import fastf1
import pandas as pd

warnings.filterwarnings('ignore')
fastf1.Cache.enable_cache('cache')

DB_PATH = 'f1.db'
YEARS = [2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025, 2026]


def create_table(conn):
    """Qualifying classification, which exists BEFORE the race is run.

    fact_race_results also carries a grid position, but only after the race has
    happened. This table is what makes a genuine pre-race prediction possible.
    """
    conn.execute("""
        CREATE TABLE IF NOT EXISTS fact_qualifying_results (
            quali_result_id INTEGER PRIMARY KEY AUTOINCREMENT,
            race_id INTEGER,
            driver_id INTEGER,
            team_id INTEGER,
            quali_position INTEGER,
            best_quali_lap REAL,
            UNIQUE (race_id, driver_id),
            FOREIGN KEY (race_id) REFERENCES dim_race(race_id),
            FOREIGN KEY (driver_id) REFERENCES dim_driver(driver_id),
            FOREIGN KEY (team_id) REFERENCES dim_team(team_id)
        )
    """)
    conn.commit()


def best_lap_seconds(row):
    laps = [row.get(seg) for seg in ('Q1', 'Q2', 'Q3')]
    laps = [t.total_seconds() for t in laps if pd.notna(t)]
    return min(laps) if laps else None


def main():
    conn = sqlite3.connect(DB_PATH)
    create_table(conn)

    races = pd.read_sql('SELECT race_id, year, round FROM dim_race', conn)
    race_lookup = {(r.year, r['round']): r.race_id for _, r in races.iterrows()}
    drivers = pd.read_sql('SELECT driver_id, driver_code FROM dim_driver', conn)
    driver_lookup = dict(zip(drivers.driver_code, drivers.driver_id))
    teams = pd.read_sql('SELECT team_id, team_name FROM dim_team', conn)
    team_lookup = dict(zip(teams.team_name, teams.team_id))

    already = set(pd.read_sql('SELECT DISTINCT race_id FROM fact_qualifying_results', conn).race_id)

    def resolve_team_id(team_name, driver_id):
        """FastF1's team naming shifts between versions, so a direct match can miss. Fall
        back to whoever this driver last actually raced for rather than storing a null."""
        team_id = team_lookup.get(team_name)
        if team_id is not None:
            return team_id
        recent = pd.read_sql("""
            SELECT fr.team_id FROM fact_race_results fr
            JOIN dim_race r ON fr.race_id = r.race_id
            WHERE fr.driver_id = ? AND fr.team_id IS NOT NULL
            ORDER BY r.race_date DESC LIMIT 1
        """, conn, params=(driver_id,))
        return int(recent.team_id.iloc[0]) if len(recent) else None

    inserted = 0
    for year in YEARS:
        try:
            schedule = fastf1.get_event_schedule(year)
        except Exception as e:
            print(f'{year}: could not load schedule: {str(e)[:70]}')
            continue

        for round_number in schedule['RoundNumber']:
            if round_number == 0:
                continue
            race_id = race_lookup.get((year, round_number))
            if race_id is None or race_id in already:
                continue

            try:
                session = fastf1.get_session(year, round_number, 'Q')
                # messages=True: without race control messages FastF1 cannot resolve
                # deleted laps and silently returns every position as NaN
                session.load(telemetry=False, weather=False, messages=True)
                results = session.results
            except Exception as e:
                print(f'{year} round {round_number} qualifying failed: {str(e)[:60]}')
                continue

            if results is None or len(results) == 0 or results['Position'].isna().all():
                print(f'{year} round {round_number}: no qualifying classification yet')
                continue

            rows = 0
            for _, r in results.iterrows():
                if pd.isna(r['Position']):
                    continue
                driver_id = driver_lookup.get(r['Abbreviation'])
                if driver_id is None:
                    continue
                conn.execute(
                    'INSERT OR IGNORE INTO fact_qualifying_results '
                    '(race_id, driver_id, team_id, quali_position, best_quali_lap) VALUES (?, ?, ?, ?, ?)',
                    (int(race_id), int(driver_id), resolve_team_id(r['TeamName'], int(driver_id)),
                     int(r['Position']), best_lap_seconds(r))
                )
                rows += 1
            conn.commit()
            inserted += rows
            print(f'{year} round {round_number}: saved {rows} qualifying positions')
            time.sleep(2)

    total = pd.read_sql('SELECT COUNT(*) n FROM fact_qualifying_results', conn).n.iloc[0]
    races_n = pd.read_sql('SELECT COUNT(DISTINCT race_id) n FROM fact_qualifying_results', conn).n.iloc[0]
    print(f'\ninserted this run: {inserted}')
    print(f'fact_qualifying_results: {total} rows across {races_n} races')
    conn.close()


if __name__ == '__main__':
    main()
