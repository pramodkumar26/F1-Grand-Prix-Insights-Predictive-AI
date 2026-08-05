import sqlite3
import pandas as pd

conn = sqlite3.connect('f1.db')

query = '''
SELECT
    dr.year,
    dr.race_name,
    d.driver_name,
    t.team_name,
    r.grid_position,
    r.finish_position,
    r.points
FROM fact_race_results r
JOIN dim_race dr ON r.race_id = dr.race_id
JOIN dim_driver d ON r.driver_id = d.driver_id
JOIN dim_team t ON r.team_id = t.team_id
WHERE dr.year = 2023 AND dr.race_name = 'Italian Grand Prix'
ORDER BY r.finish_position
LIMIT 10
'''

result = pd.read_sql(query, conn)
print(result)

conn.close()