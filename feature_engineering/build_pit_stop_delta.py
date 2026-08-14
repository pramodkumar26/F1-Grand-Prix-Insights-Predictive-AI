import sqlite3
import pandas as pd

DB_PATH = 'f1.db'
MAX_STOP_DURATION = 100


def load_pit_laps(conn):
    # pulls every lap that has either a pit entry or a pit exit recorded
    return pd.read_sql("""
        SELECT race_id, driver_id, lap_number, pit_in_time, pit_out_time
        FROM fact_laps
        WHERE pit_in_time IS NOT NULL OR pit_out_time IS NOT NULL
    """, conn)


def compute_stop_durations(pit_laps):
    # pairs a pit in lap with the following lap's pit out time to get the stop length
    pit_laps = pit_laps.sort_values(['race_id', 'driver_id', 'lap_number'])
    stops = []

    for (race_id, driver_id), group in pit_laps.groupby(['race_id', 'driver_id']):
        group = group.reset_index(drop=True)
        for _, row in group.iterrows():
            if pd.notna(row['pit_in_time']):
                next_lap = group[group['lap_number'] == row['lap_number'] + 1]
                if not next_lap.empty and pd.notna(next_lap.iloc[0]['pit_out_time']):
                    duration = next_lap.iloc[0]['pit_out_time'] - row['pit_in_time']
                    if duration > 0:
                        stops.append({
                            'race_id': race_id,
                            'driver_id': driver_id,
                            'lap_number': row['lap_number'],
                            'stop_duration': duration
                        })

    return pd.DataFrame(stops)


def filter_valid_stops(stops):
    # drops stops inflated by red flags or long stoppages, real stops rarely go past a minute or so
    before = len(stops)
    valid = stops[stops['stop_duration'] <= MAX_STOP_DURATION].copy()
    dropped = before - len(valid)
    if dropped:
        print(f"{dropped} stops excluded as unrealistic, likely red flag or session stoppage")
    return valid


def compute_pit_stop_delta(stops):
    # compares each driver's average stop time in a race to that race's median stop time
    race_medians = stops.groupby('race_id')['stop_duration'].median()
    driver_avg = stops.groupby(['race_id', 'driver_id'])['stop_duration'].mean().reset_index()
    driver_avg = driver_avg.rename(columns={'stop_duration': 'avg_stop_duration'})
    driver_avg['race_median_stop'] = driver_avg['race_id'].map(race_medians)
    driver_avg['pit_stop_delta'] = driver_avg['avg_stop_duration'] - driver_avg['race_median_stop']

    stop_counts = stops.groupby(['race_id', 'driver_id']).size().reset_index(name='stop_count')
    driver_avg = driver_avg.merge(stop_counts, on=['race_id', 'driver_id'])

    return driver_avg


def build_pit_stop_delta_table():
    conn = sqlite3.connect(DB_PATH)
    pit_laps = load_pit_laps(conn)
    stops = compute_stop_durations(pit_laps)
    stops = filter_valid_stops(stops)
    delta = compute_pit_stop_delta(stops)

    delta.to_sql('pit_stop_delta', conn, if_exists='replace', index=False)
    conn.commit()
    conn.close()

    return delta, stops


if __name__ == '__main__':
    delta, stops = build_pit_stop_delta_table()
    print(f"valid stops kept: {len(stops)}")
    print(f"pit_stop_delta table built, {len(delta)} rows")
    print(delta['pit_stop_delta'].describe())