import sqlite3
import pandas as pd

conn = sqlite3.connect('f1.db')
cursor = conn.cursor()

laps = pd.read_csv('data/processed/all_laps.csv')

team_name_mapping = {
    'Racing Point': 'Aston Martin',
    'Force India': 'Aston Martin',
    'Renault': 'Alpine',
    'Alfa Romeo': 'Kick Sauber',
    'Alfa Romeo Racing': 'Kick Sauber',
    'Toro Rosso': 'RB',
    'AlphaTauri': 'RB'
}

laps['Team'] = laps['Team'].replace(team_name_mapping)

driver_lookup = pd.read_sql('SELECT driver_id, driver_code FROM dim_driver', conn)
driver_map = {row['driver_code']: int(row['driver_id']) for _, row in driver_lookup.iterrows()}

team_lookup = pd.read_sql('SELECT team_id, team_name FROM dim_team', conn)
team_map = {row['team_name']: int(row['team_id']) for _, row in team_lookup.iterrows()}

race_lookup = pd.read_sql('SELECT race_id, year, round FROM dim_race', conn)
race_map = {(int(row['year']), int(row['round'])): int(row['race_id']) for _, row in race_lookup.iterrows()}

compound_lookup = pd.read_sql('SELECT compound_id, compound_name FROM dim_tyre_compound', conn)
compound_map = {row['compound_name']: int(row['compound_id']) for _, row in compound_lookup.iterrows()}

inserted = 0
skipped = 0

rows_to_insert = []

for _, row in laps.iterrows():
    race_id = race_map.get((int(row['year']), int(row['round'])))
    driver_id = driver_map.get(row['Driver'])
    team_id = team_map.get(row['Team'])
    compound_id = compound_map.get(row['Compound'])

    if race_id is None or driver_id is None or team_id is None:
        skipped += 1
        continue

    rows_to_insert.append((
        race_id,
        driver_id,
        team_id,
        compound_id,
        row.get('LapNumber'),
        row.get('LapTime'),
        row.get('TyreLife'),
        row.get('Stint'),
        row.get('Position'),
        row.get('PitInTime'),
        row.get('PitOutTime')
    ))
    inserted += 1

cursor.executemany(
    'INSERT INTO fact_laps (race_id, driver_id, team_id, compound_id, lap_number, lap_time, tyre_age, stint_number, track_position, pit_in_time, pit_out_time) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)',
    rows_to_insert
)

conn.commit()

print('laps inserted:', inserted)
print('laps skipped:', skipped)

conn.close()