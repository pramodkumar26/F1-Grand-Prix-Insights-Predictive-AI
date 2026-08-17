import fastf1
import os
import time

fastf1.Cache.enable_cache('cache')

years = [2025, 2026]

for year in years:
    schedule = fastf1.get_event_schedule(year)

    for round_number in schedule['RoundNumber']:
        if round_number == 0:
            continue

        rc_path = f'data/raw/race_control/{year}_{round_number}.csv'

        if os.path.exists(rc_path):
            print(f'{year} round {round_number} race control already saved, skipping')
            continue

        try:
            session = fastf1.get_session(year, round_number, 'R')
            session.load()

            rc_messages = session.race_control_messages
            rc_messages.to_csv(rc_path, index=False)

            print(f'saved {year} round {round_number} race control')
            time.sleep(8)

        except fastf1.exceptions.RateLimitExceededError:
            print('rate limit hit, pausing for an hour')
            time.sleep(3600)

        except Exception as e:
            print(f'{year} round {round_number} race control failed: {e}')