import sqlite3
import pandas as pd

conn = sqlite3.connect('f1.db')

race_check = pd.read_sql(
    "SELECT race_id, year, round, race_name FROM dim_race WHERE year = 2023 AND round = 14",
    conn
)
print('race row:')
print(race_check)

race_id = race_check['race_id'].iloc[0]

results_check = pd.read_sql(
    f"SELECT * FROM fact_race_results WHERE race_id = {race_id}",
    conn
)
print('results for this race_id:')
print(results_check)

conn.close()