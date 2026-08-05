import sqlite3
import pandas as pd
import os

conn = sqlite3.connect('f1.db')

os.makedirs('data/export', exist_ok=True)

laps_query = '''
SELECT
    dr.year,
    dr.round,
    dr.race_name,
    dr.season_era,
    d.driver_name,
    d.driver_code,
    t.team_name,
    l.lap_number,
    l.lap_time,
    l.tyre_age,
    c.compound_name,
    l.stint_number,
    l.track_position,
    l.pit_in_time,
    l.pit_out_time
FROM fact_laps l
JOIN dim_race dr ON l.race_id = dr.race_id
JOIN dim_driver d ON l.driver_id = d.driver_id
JOIN dim_team t ON l.team_id = t.team_id
LEFT JOIN dim_tyre_compound c ON l.compound_id = c.compound_id
'''

laps = pd.read_sql(laps_query, conn)
laps['lap_time_seconds'] = pd.to_timedelta(laps['lap_time']).dt.total_seconds()
laps.to_csv('data/export/laps_flat.csv', index=False)
print('laps exported:', laps.shape)

results_query = '''
SELECT
    dr.year,
    dr.round,
    dr.race_name,
    dr.season_era,
    d.driver_name,
    t.team_name,
    r.grid_position,
    r.finish_position,
    r.points,
    r.status,
    r.fastest_lap_time
FROM fact_race_results r
JOIN dim_race dr ON r.race_id = dr.race_id
JOIN dim_driver d ON r.driver_id = d.driver_id
JOIN dim_team t ON r.team_id = t.team_id
'''

results = pd.read_sql(results_query, conn)
results.to_csv('data/export/results_flat.csv', index=False)
print('results exported:', results.shape)

weather_query = '''
SELECT
    dr.year,
    dr.round,
    dr.race_name,
    w.air_temp,
    w.track_temp,
    w.humidity,
    w.rainfall,
    w.wind_speed
FROM fact_weather w
JOIN dim_race dr ON w.race_id = dr.race_id
'''

weather = pd.read_sql(weather_query, conn)
weather.to_csv('data/export/weather_flat.csv', index=False)
print('weather exported:', weather.shape)

conn.close()