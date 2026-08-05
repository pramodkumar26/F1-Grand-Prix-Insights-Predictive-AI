import sqlite3
import pandas as pd

conn = sqlite3.connect('f1.db')
cursor = conn.cursor()

cursor.execute('DELETE FROM dim_driver')
cursor.execute('DELETE FROM dim_team')
conn.commit()

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

drivers = results[['Abbreviation', 'FullName', 'CountryCode']].drop_duplicates(subset='Abbreviation')

for _, row in drivers.iterrows():
    cursor.execute(
        'INSERT OR IGNORE INTO dim_driver (driver_code, driver_name, nationality) VALUES (?, ?, ?)',
        (row['Abbreviation'], row['FullName'], row['CountryCode'])
    )

teams = results[['TeamName']].drop_duplicates()

for _, row in teams.iterrows():
    cursor.execute(
        'INSERT OR IGNORE INTO dim_team (team_name) VALUES (?)',
        (row['TeamName'],)
    )

conn.commit()

cursor.execute('SELECT COUNT(*) FROM dim_driver')
print('drivers:', cursor.fetchone()[0])

cursor.execute('SELECT COUNT(*) FROM dim_team')
print('teams:', cursor.fetchone()[0])

cursor.execute('SELECT * FROM dim_driver LIMIT 5')
print(cursor.fetchall())

conn.close()