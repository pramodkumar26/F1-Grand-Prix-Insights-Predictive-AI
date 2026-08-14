import sqlite3
import fastf1
import pandas as pd

fastf1.Cache.enable_cache('cache')

DB_PATH = 'f1.db'


def get_race_lookup(conn):
    return pd.read_sql("SELECT race_id, year, round FROM dim_race", conn)


def get_driver_lookup(conn):
    return pd.read_sql("SELECT driver_id, driver_code FROM dim_driver", conn)


def pull_track_status_for_race(year, round_number):
    session = fastf1.get_session(year, round_number, 'R')
    session.load(laps=True, telemetry=False, weather=False, messages=False)
    laps = session.laps[['Driver', 'LapNumber', 'TrackStatus']].copy()
    laps = laps.dropna(subset=['LapNumber'])
    laps['LapNumber'] = laps['LapNumber'].astype(int)
    return laps


def build_lap_flags():
    conn = sqlite3.connect(DB_PATH)
    races = get_race_lookup(conn)
    drivers = get_driver_lookup(conn)
    driver_map = dict(zip(drivers['driver_code'], drivers['driver_id']))

    all_rows = []
    failed_races = []
    unmatched_codes = set()

    for _, race in races.iterrows():
        race_id = race['race_id']
        year = race['year']
        round_number = race['round']

        try:
            laps = pull_track_status_for_race(year, round_number)
        except Exception as error:
            print(f"skipped {year} round {round_number}: {error}")
            failed_races.append((year, round_number, str(error)))
            continue

        for _, lap in laps.iterrows():
            driver_id = driver_map.get(lap['Driver'])
            if driver_id is None:
                unmatched_codes.add(lap['Driver'])
                continue
            all_rows.append({
                'race_id': race_id,
                'driver_id': driver_id,
                'lap_number': lap['LapNumber'],
                'track_status': lap['TrackStatus']
            })

        print(f"done {year} round {round_number}, race_id {race_id}")

    lap_flags = pd.DataFrame(all_rows)
    lap_flags.to_sql('lap_flags', conn, if_exists='replace', index=False)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_lap_flags_lookup ON lap_flags(race_id, driver_id, lap_number)")
    conn.commit()
    conn.close()

    if failed_races:
        print(f"{len(failed_races)} races failed to load, see failed_races")
    if unmatched_codes:
        print(f"driver codes with no match in dim_driver: {unmatched_codes}")

    return lap_flags, failed_races, unmatched_codes


if __name__ == '__main__':
    lap_flags, failed_races, unmatched_codes = build_lap_flags()
    print(f"lap_flags table built, {len(lap_flags)} rows")


# What this does: pulls per lap track status, clear, yellow, safety car, red flag,
# from FastF1 for every race in dim_race, and writes it into a new lap_flags table
# in f1.db. This lets us filter out laps affected by flags before computing things
# like tyre degradation rate, so degradation numbers reflect real pace instead of
# laps run behind a safety car or under yellow.
#
# Why this approach: fact_laps has no flag information at all. TrackStatus already
# exists inside FastF1's laps data, per driver, per lap, so this reads directly from
# that instead of manually parsing race control message timestamps.