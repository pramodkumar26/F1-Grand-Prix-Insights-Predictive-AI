import sqlite3
import pandas as pd

conn = sqlite3.connect('f1.db')

check = pd.read_sql("SELECT year, round, race_name FROM dim_race WHERE year = 2023", conn)

print(check)

conn.close()