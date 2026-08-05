import sqlite3
import pandas as pd

conn = sqlite3.connect('f1.db')
cursor = conn.cursor()

schedule = pd.read_csv('data/processed/all_schedule.csv')

circuits = schedule[['Location', 'Country']].drop_duplicates(subset='Location')

for _, row in circuits.iterrows():
    cursor.execute(
        'INSERT OR IGNORE INTO dim_circuit (circuit_name, country) VALUES (?, ?)',
        (row['Location'], row['Country'])
    )

conn.commit()

circuit_lookup = pd.read_sql('SELECT circuit_id, circuit_name FROM dim_circuit', conn)
circuit_map = dict(zip(circuit_lookup['circuit_name'], circuit_lookup['circuit_id']))

def get_season_era(year):
    if year < 2020:
        return 'pre_covid'
    elif year == 2020:
        return 'covid'
    else:
        return 'post_covid'

for _, row in schedule.iterrows():
    circuit_id = circuit_map.get(row['Location'])
    season_era = get_season_era(row['year'])

    cursor.execute(
        'INSERT INTO dim_race (year, round, race_name, circuit_id, race_date, season_era) VALUES (?, ?, ?, ?, ?, ?)',
        (row['year'], row['RoundNumber'], row['EventName'], circuit_id, row['EventDate'], season_era)
    )

conn.commit()

cursor.execute('SELECT COUNT(*) FROM dim_circuit')
print('circuits:', cursor.fetchone()[0])

cursor.execute('SELECT COUNT(*) FROM dim_race')
print('races:', cursor.fetchone()[0])

cursor.execute('SELECT year, round, race_name, season_era FROM dim_race LIMIT 5')
print(cursor.fetchall())

conn.close()