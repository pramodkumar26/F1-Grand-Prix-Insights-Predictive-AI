"""Fallback loader for race results, via Jolpica rather than FastF1.

FastF1 reads F1's live timing service, which does not reliably answer requests from
datacenter IPs. Locally it works; from a GitHub Actions runner it returns "Failed to load
timing data" for a race that has genuinely finished, so an automated pull silently never
picks up the result. Jolpica is an ordinary REST API and answers fine from anywhere.

This only fills in results that are missing. Anything FastF1 already loaded is left alone,
since FastF1 is the richer source (laps, telemetry, weather) and stays the primary one.
"""
import json
import sqlite3
import urllib.request

import pandas as pd

DB_PATH = 'f1.db'
CURRENT_SEASON = 2026


def fetch_results(year):
    """Jolpica caps limit at 100, so this has to paginate or it silently truncates."""
    rows = []
    offset, limit = 0, 100
    while True:
        url = f'https://api.jolpi.ca/ergast/f1/{year}/results.json?limit={limit}&offset={offset}'
        with urllib.request.urlopen(url) as resp:
            data = json.load(resp)['MRData']
        for race in data['RaceTable']['Races']:
            for r in race.get('Results', []):
                fastest = r.get('FastestLap', {}).get('Time', {}).get('time')
                rows.append({
                    'round': int(race['round']),
                    'driver_code': r['Driver'].get('code'),
                    'team_name': r['Constructor']['name'],
                    'grid_position': int(r['grid']),
                    'finish_position': int(r['position']),
                    'points': float(r['points']),
                    'status': r['status'],
                    'fastest_lap_time': lap_to_seconds(fastest),
                })
        offset += limit
        if offset >= int(data['total']):
            break
    return rows


def lap_to_seconds(text):
    if not text:
        return None
    try:
        minutes, rest = text.split(':')
        return round(int(minutes) * 60 + float(rest), 3)
    except (ValueError, AttributeError):
        return None


def main():
    conn = sqlite3.connect(DB_PATH)

    races = pd.read_sql('SELECT race_id, round FROM dim_race WHERE year = ?', conn,
                        params=(CURRENT_SEASON,))
    race_lookup = dict(zip(races['round'], races.race_id))
    drivers = pd.read_sql('SELECT driver_id, driver_code FROM dim_driver', conn)
    driver_lookup = dict(zip(drivers.driver_code, drivers.driver_id))
    teams = pd.read_sql('SELECT team_id, team_name FROM dim_team', conn)
    team_lookup = dict(zip(teams.team_name, teams.team_id))

    # only races we have no result for at all, so FastF1's richer data is never overwritten
    have = set(pd.read_sql(
        'SELECT DISTINCT race_id FROM fact_race_results', conn).race_id)

    try:
        rows = fetch_results(CURRENT_SEASON)
    except Exception as e:
        print(f'could not reach Jolpica: {str(e)[:80]}')
        conn.close()
        return

    inserted = skipped_existing = unmatched = 0
    for r in rows:
        race_id = race_lookup.get(r['round'])
        if race_id is None:
            continue
        if race_id in have:
            skipped_existing += 1
            continue
        driver_id = driver_lookup.get(r['driver_code'])
        if driver_id is None:
            unmatched += 1
            continue
        conn.execute("""
            INSERT INTO fact_race_results
            (race_id, driver_id, team_id, grid_position, finish_position, points, status, fastest_lap_time)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (int(race_id), int(driver_id), team_lookup.get(r['team_name']),
              r['grid_position'], r['finish_position'], r['points'],
              r['status'], r['fastest_lap_time']))
        inserted += 1

    conn.commit()
    total = pd.read_sql('SELECT COUNT(*) n FROM fact_race_results', conn).n.iloc[0]
    conn.close()

    print(f'inserted from Jolpica: {inserted}')
    print(f'skipped, already had results: {skipped_existing}')
    if unmatched:
        print(f'driver codes not in dim_driver: {unmatched}')
    print(f'fact_race_results: {total} rows')


if __name__ == '__main__':
    main()
