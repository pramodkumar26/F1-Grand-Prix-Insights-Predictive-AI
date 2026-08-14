import sqlite3
import pandas as pd

DB_PATH = 'f1.db'


def load_first_pit_laps(conn):
    # pulls the first pit stop lap for every driver in every race
    query = """
        SELECT race_id, driver_id, team_id, MIN(lap_number) AS first_pit_lap
        FROM fact_laps
        WHERE pit_in_time IS NOT NULL
        GROUP BY race_id, driver_id, team_id
    """
    return pd.read_sql(query, conn)


def compute_strategic_aggressiveness(pit_laps):
    # compares each driver's first stop lap against the field's median first stop lap that race
    race_medians = pit_laps.groupby('race_id')['first_pit_lap'].median()
    pit_laps['field_median_first_pit_lap'] = pit_laps['race_id'].map(race_medians)
    pit_laps['strategic_aggressiveness'] = pit_laps['first_pit_lap'] - pit_laps['field_median_first_pit_lap']
    return pit_laps


def build_strategic_aggressiveness_table():
    conn = sqlite3.connect(DB_PATH)
    pit_laps = load_first_pit_laps(conn)
    result = compute_strategic_aggressiveness(pit_laps)

    result.to_sql('strategic_aggressiveness', conn, if_exists='replace', index=False)
    conn.commit()
    conn.close()

    return result


if __name__ == '__main__':
    result = build_strategic_aggressiveness_table()
    print(f"strategic_aggressiveness table built, {len(result)} rows")
    print(result['strategic_aggressiveness'].describe())