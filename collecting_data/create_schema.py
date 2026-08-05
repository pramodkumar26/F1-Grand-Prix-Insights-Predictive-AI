import sqlite3

conn = sqlite3.connect('f1.db')
cursor = conn.cursor()

cursor.execute('''
CREATE TABLE dim_driver (
    driver_id INTEGER PRIMARY KEY AUTOINCREMENT,
    driver_code TEXT UNIQUE,
    driver_name TEXT,
    nationality TEXT
)
''')

cursor.execute('''
CREATE TABLE dim_team (
    team_id INTEGER PRIMARY KEY AUTOINCREMENT,
    team_name TEXT UNIQUE,
    engine_manufacturer TEXT
)
''')

cursor.execute('''
CREATE TABLE dim_circuit (
    circuit_id INTEGER PRIMARY KEY AUTOINCREMENT,
    circuit_name TEXT UNIQUE,
    country TEXT
)
''')

cursor.execute('''
CREATE TABLE dim_tyre_compound (
    compound_id INTEGER PRIMARY KEY AUTOINCREMENT,
    compound_name TEXT UNIQUE
)
''')

cursor.execute('''
CREATE TABLE dim_race (
    race_id INTEGER PRIMARY KEY AUTOINCREMENT,
    year INTEGER,
    round INTEGER,
    race_name TEXT,
    circuit_id INTEGER,
    race_date TEXT,
    season_era TEXT,
    FOREIGN KEY (circuit_id) REFERENCES dim_circuit(circuit_id)
)
''')

cursor.execute('''
CREATE TABLE fact_laps (
    lap_id INTEGER PRIMARY KEY AUTOINCREMENT,
    race_id INTEGER,
    driver_id INTEGER,
    team_id INTEGER,
    compound_id INTEGER,
    lap_number INTEGER,
    lap_time REAL,
    tyre_age INTEGER,
    stint_number INTEGER,
    track_position INTEGER,
    pit_in_time REAL,
    pit_out_time REAL,
    FOREIGN KEY (race_id) REFERENCES dim_race(race_id),
    FOREIGN KEY (driver_id) REFERENCES dim_driver(driver_id),
    FOREIGN KEY (team_id) REFERENCES dim_team(team_id),
    FOREIGN KEY (compound_id) REFERENCES dim_tyre_compound(compound_id)
)
''')

cursor.execute('''
CREATE TABLE fact_race_results (
    result_id INTEGER PRIMARY KEY AUTOINCREMENT,
    race_id INTEGER,
    driver_id INTEGER,
    team_id INTEGER,
    grid_position INTEGER,
    finish_position INTEGER,
    points REAL,
    status TEXT,
    fastest_lap_time REAL,
    FOREIGN KEY (race_id) REFERENCES dim_race(race_id),
    FOREIGN KEY (driver_id) REFERENCES dim_driver(driver_id),
    FOREIGN KEY (team_id) REFERENCES dim_team(team_id)
)
''')

cursor.execute('''
CREATE TABLE fact_weather (
    weather_id INTEGER PRIMARY KEY AUTOINCREMENT,
    race_id INTEGER,
    air_temp REAL,
    track_temp REAL,
    humidity REAL,
    rainfall INTEGER,
    wind_speed REAL,
    FOREIGN KEY (race_id) REFERENCES dim_race(race_id)
)
''')

conn.commit()
conn.close()

print('schema created')