import sqlite3
import pandas as pd

conn = sqlite3.connect('f1.db')
cursor = conn.cursor()

cursor.execute('''
CREATE TABLE IF NOT EXISTS fact_qualifying_laps (
    quali_lap_id INTEGER PRIMARY KEY AUTOINCREMENT,
    race_id INTEGER,
    driver_id INTEGER,
    team_id INTEGER,
    lap_number INTEGER,
    lap_time REAL,
    is_personal_best INTEGER,
    deleted INTEGER,
    is_accurate INTEGER,
    FOREIGN KEY (race_id) REFERENCES dim_race(race_id),
    FOREIGN KEY (driver_id) REFERENCES dim_driver(driver_id),
    FOREIGN KEY (team_id) REFERENCES dim_team(team_id)
)
''')
conn.commit()

quali = pd.read_csv('data/processed/all_qualifying.csv', low_memory=False)
quali['LapTime'] = pd.to_timedelta(quali['LapTime'], errors='coerce').dt.total_seconds()

team_name_mapping = {
    'Racing Point': 'Aston Martin',
    'Force India': 'Aston Martin',
    'Renault': 'Alpine',
    'Alfa Romeo': 'Kick Sauber',
    'Alfa Romeo Racing': 'Kick Sauber',
    'Toro Rosso': 'RB',
    'AlphaTauri': 'RB'
}
quali['Team'] = quali['Team'].replace(team_name_mapping)

driver_lookup = pd.read_sql('SELECT driver_id, driver_code FROM dim_driver', conn)
driver_map = {row['driver_code']: int(row['driver_id']) for _, row in driver_lookup.iterrows()}

team_lookup = pd.read_sql('SELECT team_id, team_name FROM dim_team', conn)
team_map = {row['team_name']: int(row['team_id']) for _, row in team_lookup.iterrows()}

race_lookup = pd.read_sql('SELECT race_id, year, round FROM dim_race', conn)
race_map = {(int(row['year']), int(row['round'])): int(row['race_id']) for _, row in race_lookup.iterrows()}

loaded_races = pd.read_sql('SELECT DISTINCT race_id FROM fact_qualifying_laps', conn)
loaded_race_ids = set(loaded_races['race_id'])

inserted = 0
skipped = 0
rows_to_insert = []

for _, row in quali.iterrows():
    race_id = race_map.get((int(row['year']), int(row['round'])))
    driver_id = driver_map.get(row['Driver'])
    team_id = team_map.get(row['Team'])

    if race_id is None or driver_id is None or team_id is None or race_id in loaded_race_ids:
        skipped += 1
        continue

    rows_to_insert.append((
        race_id,
        driver_id,
        team_id,
        row.get('LapNumber'),
        row.get('LapTime'),
        int(bool(row.get('IsPersonalBest'))),
        int(bool(row.get('Deleted'))),
        int(bool(row.get('IsAccurate')))
    ))
    inserted += 1

cursor.executemany(
    'INSERT INTO fact_qualifying_laps (race_id, driver_id, team_id, lap_number, lap_time, is_personal_best, deleted, is_accurate) VALUES (?, ?, ?, ?, ?, ?, ?, ?)',
    rows_to_insert
)

conn.commit()

print('qualifying laps inserted:', inserted)
print('qualifying laps skipped:', skipped)

conn.close()
