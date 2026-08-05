import sqlite3
import pandas as pd

conn = sqlite3.connect('f1.db')
cursor = conn.cursor()

laps = pd.read_csv('data/processed/all_laps.csv')
results = pd.read_csv('data/processed/all_results.csv')
weather = pd.read_csv('data/processed/all_weather.csv')

drivers = results[['Abbreviation', 'FullName']].drop_duplicates(subset='Abbreviation')

for _, row in drivers.iterrows():
    cursor.execute(
        'INSERT OR IGNORE INTO dim_driver (driver_code, driver_name) VALUES (?, ?)',
        (row['Abbreviation'], row['FullName'])
    )

teams = results[['TeamName']].drop_duplicates()

for _, row in teams.iterrows():
    cursor.execute(
        'INSERT OR IGNORE INTO dim_team (team_name) VALUES (?)',
        (row['TeamName'],)
    )

compounds = laps[['Compound']].dropna().drop_duplicates()

for _, row in compounds.iterrows():
    cursor.execute(
        'INSERT OR IGNORE INTO dim_tyre_compound (compound_name) VALUES (?)',
        (row['Compound'],)
    )

conn.commit()

print('drivers, teams, and compounds loaded')