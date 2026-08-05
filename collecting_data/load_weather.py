import sqlite3
import pandas as pd

conn = sqlite3.connect('f1.db')
cursor = conn.cursor()

weather = pd.read_csv('data/processed/all_weather.csv')

race_lookup = pd.read_sql('SELECT race_id, year, round FROM dim_race', conn)
race_map = {(row['year'], row['round']): row['race_id'] for _, row in race_lookup.iterrows()}

inserted = 0
skipped = 0

for _, row in weather.iterrows():
    race_id = race_map.get((row['year'], row['round']))

    if race_id is None:
        skipped += 1
        continue

    cursor.execute(
        'INSERT INTO fact_weather (race_id, air_temp, track_temp, humidity, rainfall, wind_speed) VALUES (?, ?, ?, ?, ?, ?)',
        (race_id, row.get('AirTemp'), row.get('TrackTemp'), row.get('Humidity'), row.get('Rainfall'), row.get('WindSpeed'))
    )
    inserted += 1

conn.commit()

print('inserted:', inserted)
print('skipped:', skipped)

conn.close()