import sqlite3
import pandas as pd

conn = sqlite3.connect('f1.db')
cursor = conn.cursor()

results = pd.read_csv('data/processed/all_results.csv')

team_name_mapping = {
    'Racing Point': 'Aston Martin',
    'Force India': 'Aston Martin',
    'Renault': 'Alpine',
    'Alfa Romeo': 'Kick Sauber',
    'Alfa Romeo Racing': 'Kick Sauber',
    'Toro Rosso': 'RB',
    'AlphaTauri': 'RB'
}

results['TeamName'] = results['TeamName'].replace(team_name_mapping)

driver_lookup = pd.read_sql('SELECT driver_id, driver_code FROM dim_driver', conn)
driver_map = {row['driver_code']: int(row['driver_id']) for _, row in driver_lookup.iterrows()}

team_lookup = pd.read_sql('SELECT team_id, team_name FROM dim_team', conn)
team_map = {row['team_name']: int(row['team_id']) for _, row in team_lookup.iterrows()}

race_lookup = pd.read_sql('SELECT race_id, year, round FROM dim_race', conn)
race_map = {(int(row['year']), int(row['round'])): int(row['race_id']) for _, row in race_lookup.iterrows()}

inserted = 0
skipped = 0

for _, row in results.iterrows():
    race_id = race_map.get((int(row['year']), int(row['round'])))
    driver_id = driver_map.get(row['Abbreviation'])
    team_id = team_map.get(row['TeamName'])

    if race_id is None or driver_id is None or team_id is None:
        skipped += 1
        continue

    cursor.execute(
        'INSERT INTO fact_race_results (race_id, driver_id, team_id, grid_position, finish_position, points, status, fastest_lap_time) VALUES (?, ?, ?, ?, ?, ?, ?, ?)',
        (race_id, driver_id, team_id, row['GridPosition'], row['Position'], row['Points'], row['Status'], row.get('FastestLapTime'))
    )
    inserted += 1

conn.commit()

print('results inserted:', inserted)
print('results skipped:', skipped)

conn.close()