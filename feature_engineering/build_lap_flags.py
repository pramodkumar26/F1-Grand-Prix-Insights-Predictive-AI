import sqlite3
import fastf1
import pandas as pd

fastf1.Cache.enable_cache('cache')

DB_PATH = 'f1.db'


def get_race_lookup(conn):
    # only races that have actually been run. A future race has nothing to fetch, and
    # asking for one costs a real API call against a 500/hour limit every single run.
    return pd.read_sql("""
        SELECT race_id, year, round FROM dim_race
        WHERE race_date <= date('now')
    """, conn)


def get_driver_lookup(conn):
    return pd.read_sql("SELECT driver_id, driver_code FROM dim_driver", conn)


def pull_track_status_for_race(year, round_number):
    session = fastf1.get_session(year, round_number, 'R')
    session.load(laps=True, telemetry=False, weather=False, messages=False)
    laps = session.laps[['Driver', 'LapNumber', 'TrackStatus']].copy()
    laps = laps.dropna(subset=['LapNumber'])
    laps['LapNumber'] = laps['LapNumber'].astype(int)
    return laps


def already_built(conn):
    """Races whose flags are already stored, so they never need fetching again."""
    try:
        done = pd.read_sql('SELECT DISTINCT race_id FROM lap_flags', conn)
        return set(done['race_id'])
    except Exception:
        return set()


def build_lap_flags():
    conn = sqlite3.connect(DB_PATH)
    races = get_race_lookup(conn)
    drivers = get_driver_lookup(conn)
    driver_map = dict(zip(drivers['driver_code'], drivers['driver_id']))

    # This is the only feature builder that reads from the FastF1 API rather than the
    # database, so it is the only one that can be throttled (500 calls/hour) or run
    # against a cold cache. It used to refetch every race and write with
    # if_exists='replace', which meant a throttled run silently rewrote the whole table
    # from a partial fetch, gutting every feature table built on top of it. Now it only
    # fetches races it does not already have, and appends.
    stored = already_built(conn)

    all_rows = []
    failed_races = []
    unmatched_codes = set()

    for _, race in races.iterrows():
        race_id = race['race_id']
        year = race['year']
        round_number = race['round']

        if race_id in stored:
            continue

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
    if len(lap_flags):
        # append, never replace - a partial fetch must not be able to destroy what is
        # already stored for every other race
        lap_flags.to_sql('lap_flags', conn, if_exists='append', index=False)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_lap_flags_lookup ON lap_flags(race_id, driver_id, lap_number)")
    conn.commit()

    total = pd.read_sql('SELECT COUNT(*) n FROM lap_flags', conn).n.iloc[0]
    conn.close()

    if failed_races:
        print(f"{len(failed_races)} races failed to load, see failed_races")
    if unmatched_codes:
        print(f"driver codes with no match in dim_driver: {unmatched_codes}")

    return lap_flags, failed_races, unmatched_codes, total


if __name__ == '__main__':
    lap_flags, failed_races, unmatched_codes, total = build_lap_flags()
    print(f"lap_flags: {len(lap_flags)} new rows added, {total} rows total")


# What this does: pulls per lap track status, clear, yellow, safety car, red flag,
# from FastF1 for every race in dim_race, and writes it into a new lap_flags table
# in f1.db. This lets us filter out laps affected by flags before computing things
# like tyre degradation rate, so degradation numbers reflect real pace instead of
# laps run behind a safety car or under yellow.
#
# Why this approach: fact_laps has no flag information at all. TrackStatus already
# exists inside FastF1's laps data, per driver, per lap, so this reads directly from
# that instead of manually parsing race control message timestamps.