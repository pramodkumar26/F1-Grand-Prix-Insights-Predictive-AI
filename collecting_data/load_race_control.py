import sqlite3
import pandas as pd

conn = sqlite3.connect('f1.db')
cursor = conn.cursor()

cursor.execute('''
CREATE TABLE IF NOT EXISTS fact_race_control (
    race_control_id INTEGER PRIMARY KEY AUTOINCREMENT,
    race_id INTEGER,
    driver_id INTEGER,
    lap_number INTEGER,
    message_time TEXT,
    category TEXT,
    message TEXT,
    status TEXT,
    flag TEXT,
    scope TEXT,
    FOREIGN KEY (race_id) REFERENCES dim_race(race_id),
    FOREIGN KEY (driver_id) REFERENCES dim_driver(driver_id)
)
''')
conn.commit()

race_control = pd.read_csv('data/processed/all_race_control.csv')

# race control only has RacingNumber (car number), not the 3 letter driver code
# every other loader keys off. dim_driver has no car number column, and a car
# number isn't a stable global key anyway, a reserve driver can take over a
# regular's number mid weekend. Build the number -> code mapping per year/round
# instead, off all_laps.csv, which carries both for the same sessions.
laps = pd.read_csv('data/processed/all_laps.csv', usecols=['year', 'round', 'DriverNumber', 'Driver'])
laps = laps.dropna(subset=['DriverNumber']).drop_duplicates(subset=['year', 'round', 'DriverNumber'])
number_to_code = {
    (int(row.year), int(row.round), int(row.DriverNumber)): row.Driver
    for row in laps.itertuples()
}

driver_lookup = pd.read_sql('SELECT driver_id, driver_code FROM dim_driver', conn)
driver_map = {row['driver_code']: int(row['driver_id']) for _, row in driver_lookup.iterrows()}

race_lookup = pd.read_sql('SELECT race_id, year, round FROM dim_race', conn)
race_map = {(int(row['year']), int(row['round'])): int(row['race_id']) for _, row in race_lookup.iterrows()}

loaded_races = pd.read_sql('SELECT DISTINCT race_id FROM fact_race_control', conn)
loaded_race_ids = set(loaded_races['race_id'])

inserted = 0
skipped = 0
unmatched_driver = 0
rows_to_insert = []

for row in race_control.itertuples():
    race_id = race_map.get((int(row.year), int(row.round)))
    if race_id is None or race_id in loaded_race_ids:
        skipped += 1
        continue

    driver_id = None
    if pd.notna(row.RacingNumber):
        code = number_to_code.get((int(row.year), int(row.round), int(row.RacingNumber)))
        driver_id = driver_map.get(code) if code else None
        if driver_id is None:
            unmatched_driver += 1

    rows_to_insert.append((
        race_id,
        driver_id,
        row.Lap if pd.notna(row.Lap) else None,
        row.Time,
        row.Category,
        row.Message,
        row.Status if pd.notna(row.Status) else None,
        row.Flag if pd.notna(row.Flag) else None,
        row.Scope if pd.notna(row.Scope) else None,
    ))
    inserted += 1

cursor.executemany(
    'INSERT INTO fact_race_control (race_id, driver_id, lap_number, message_time, category, message, status, flag, scope) '
    'VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)',
    rows_to_insert
)

conn.commit()

print('race control messages inserted:', inserted)
print('rows skipped (no matching race):', skipped)
print('messages with a RacingNumber that could not be matched to a driver_id:', unmatched_driver)

conn.close()
