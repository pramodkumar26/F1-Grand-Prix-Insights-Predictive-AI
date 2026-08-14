import sqlite3
import pandas as pd

conn = sqlite3.connect('f1.db')
race_ids = (29, 38, 42, 157, 170, 8, 58)
query = f'SELECT * FROM dim_race WHERE race_id IN {race_ids}'
races = pd.read_sql(query, conn)
conn.close()

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
print(races)