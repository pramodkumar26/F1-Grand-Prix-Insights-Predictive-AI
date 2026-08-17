import sqlite3
import time
import urllib.request
import json

DB_PATH = 'f1.db'
YEARS = [2021, 2022, 2023, 2024, 2025, 2026]

# Jolpica reports constructors with more verbose or differently-branded names than
# this project's dim_team, and its raw names go back through the same rebrands
# every other loader already consolidates (see team_name_mapping in load_results.py).
# This maps straight from Jolpica's raw name to the final dim_team name in one step.
constructor_name_mapping = {
    'Alpine F1 Team': 'Alpine',
    'Red Bull': 'Red Bull Racing',
    'RB F1 Team': 'RB',
    'Cadillac F1 Team': 'Cadillac',
    'Sauber': 'Kick Sauber',
    'Alfa Romeo': 'Kick Sauber',
    'AlphaTauri': 'RB',
}


def fetch_sprint_races(year):
    races = []
    offset = 0
    limit = 100
    while True:
        url = f'https://api.jolpi.ca/ergast/f1/{year}/sprint.json?limit={limit}&offset={offset}'
        with urllib.request.urlopen(url) as resp:
            data = json.load(resp)
        table = data['MRData']['RaceTable']
        races.extend(table['Races'])
        total = int(data['MRData']['total'])
        offset += limit
        if offset >= total:
            break
        time.sleep(1)
    return races


def main():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute('''
    CREATE TABLE IF NOT EXISTS fact_sprint_results (
        sprint_result_id INTEGER PRIMARY KEY AUTOINCREMENT,
        race_id INTEGER,
        driver_id INTEGER,
        team_id INTEGER,
        sprint_grid_position INTEGER,
        sprint_finish_position INTEGER,
        sprint_points REAL,
        FOREIGN KEY (race_id) REFERENCES dim_race(race_id),
        FOREIGN KEY (driver_id) REFERENCES dim_driver(driver_id),
        FOREIGN KEY (team_id) REFERENCES dim_team(team_id)
    )
    ''')
    conn.commit()

    driver_lookup = cursor.execute('SELECT driver_id, driver_code FROM dim_driver').fetchall()
    driver_map = {code: driver_id for driver_id, code in driver_lookup}

    team_lookup = cursor.execute('SELECT team_id, team_name FROM dim_team').fetchall()
    team_map = {name: team_id for team_id, name in team_lookup}

    race_lookup = cursor.execute('SELECT race_id, year, round FROM dim_race').fetchall()
    race_map = {(year, round_): race_id for race_id, year, round_ in race_lookup}

    loaded_races = cursor.execute('SELECT DISTINCT race_id FROM fact_sprint_results').fetchall()
    loaded_race_ids = {row[0] for row in loaded_races}

    inserted = 0
    skipped = 0
    unmatched_driver = set()
    unmatched_team = set()
    rows_to_insert = []

    for year in YEARS:
        races = fetch_sprint_races(year)
        for race in races:
            season = int(race['season'])
            round_number = int(race['round'])
            race_id = race_map.get((season, round_number))
            if race_id is None:
                skipped += len(race['SprintResults'])
                continue
            if race_id in loaded_race_ids:
                skipped += len(race['SprintResults'])
                continue

            for result in race['SprintResults']:
                code = result['Driver']['code']
                driver_id = driver_map.get(code)
                if driver_id is None:
                    unmatched_driver.add(code)
                    skipped += 1
                    continue

                raw_team_name = result['Constructor']['name']
                team_name = constructor_name_mapping.get(raw_team_name, raw_team_name)
                team_id = team_map.get(team_name)
                if team_id is None:
                    unmatched_team.add(raw_team_name)
                    skipped += 1
                    continue

                rows_to_insert.append((
                    race_id,
                    driver_id,
                    team_id,
                    int(result['grid']),
                    int(result['position']),
                    float(result['points']),
                ))
                inserted += 1

        print(f'{year}: {len(races)} sprint races found')
        time.sleep(1)

    cursor.executemany(
        'INSERT INTO fact_sprint_results (race_id, driver_id, team_id, sprint_grid_position, sprint_finish_position, sprint_points) '
        'VALUES (?, ?, ?, ?, ?, ?)',
        rows_to_insert
    )
    conn.commit()

    print(f'\nsprint results inserted: {inserted}')
    print(f'rows skipped: {skipped}')
    if unmatched_driver:
        print(f'driver codes with no match in dim_driver: {unmatched_driver}')
    if unmatched_team:
        print(f'constructor names with no match in dim_team: {unmatched_team}')

    conn.close()


if __name__ == '__main__':
    main()
