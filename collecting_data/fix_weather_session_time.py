import sqlite3
import pandas as pd

DB_PATH = 'f1.db'


def load_csv_times():
    # the raw processed CSV has a Time column that load_weather.py read but never
    # inserted, so fact_weather ended up with ~149 unordered samples per race and no
    # way to tell which reading came before or after a given point in the race.
    weather = pd.read_csv('data/processed/all_weather.csv')
    weather['session_time_seconds'] = pd.to_timedelta(weather['Time'], errors='coerce').dt.total_seconds()
    return weather


def verify_alignment(conn, weather):
    # rows were inserted in CSV order per race, so positional alignment is valid ONLY
    # if the per race counts match exactly. Verify before writing anything.
    race_lookup = pd.read_sql('SELECT race_id, year, round FROM dim_race', conn)
    weather = weather.merge(race_lookup, on=['year', 'round'], how='inner')

    csv_counts = weather.groupby('race_id').size()
    db_counts = pd.read_sql(
        'SELECT race_id, COUNT(*) n FROM fact_weather GROUP BY race_id', conn
    ).set_index('race_id')['n']

    mismatched = [
        rid for rid in csv_counts.index
        if rid not in db_counts.index or csv_counts[rid] != db_counts[rid]
    ]
    return weather, mismatched


def apply_fix(conn, weather):
    existing = [row[1] for row in conn.execute("PRAGMA table_info(fact_weather)")]
    if 'session_time_seconds' not in existing:
        conn.execute("ALTER TABLE fact_weather ADD COLUMN session_time_seconds REAL")

    ids = pd.read_sql('SELECT weather_id, race_id FROM fact_weather ORDER BY race_id, weather_id', conn)
    ordered = weather.sort_values(['race_id']).reset_index(drop=True)
    ids = ids.sort_values(['race_id', 'weather_id']).reset_index(drop=True)
    ids['session_time_seconds'] = ordered['session_time_seconds'].values

    ids[['weather_id', 'session_time_seconds']].to_sql('weather_time_staging', conn, if_exists='replace', index=False)
    conn.execute("CREATE INDEX idx_wts_id ON weather_time_staging(weather_id)")
    conn.execute("""
        UPDATE fact_weather
        SET session_time_seconds = (
            SELECT session_time_seconds FROM weather_time_staging
            WHERE weather_time_staging.weather_id = fact_weather.weather_id
        )
    """)
    conn.execute("DROP TABLE weather_time_staging")
    conn.commit()
    return len(ids)


def main():
    conn = sqlite3.connect(DB_PATH)
    weather = load_csv_times()
    weather, mismatched = verify_alignment(conn, weather)

    if mismatched:
        print(f"ABORTED, {len(mismatched)} races have mismatched row counts between CSV and table")
        print(f"  affected race_ids: {mismatched[:10]}")
        conn.close()
        return

    print("per race row counts match between CSV and fact_weather, positional alignment is safe")
    updated = apply_fix(conn, weather)
    print(f"fact_weather.session_time_seconds backfilled, {updated} rows")

    check = pd.read_sql("""
        SELECT COUNT(*) total,
               SUM(session_time_seconds IS NULL) nulls,
               MIN(session_time_seconds) min_t,
               MAX(session_time_seconds) max_t
        FROM fact_weather
    """, conn)
    print(check.to_string(index=False))
    conn.close()


if __name__ == '__main__':
    main()
