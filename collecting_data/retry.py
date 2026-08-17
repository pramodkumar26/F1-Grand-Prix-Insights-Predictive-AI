import fastf1
import pandas as pd
import os
import time

fastf1.Cache.enable_cache('cache')

year = 2026

schedule = fastf1.get_event_schedule(year)

for round_number in schedule['RoundNumber']:
    if round_number == 0:
        continue

    laps_path = f'data/raw/laps/{year}_{round_number}.csv'

    if os.path.exists(laps_path):
        print(f'{year} round {round_number} already saved, skipping')
        continue

    try:
        session = fastf1.get_session(year, round_number, 'R')
        session.load()

        laps = session.laps
        results = session.results
        weather = session.weather_data

        laps.to_csv(laps_path, index=False)
        results.to_csv(f'data/raw/results/{year}_{round_number}.csv', index=False)
        weather.to_csv(f'data/raw/weather/{year}_{round_number}.csv', index=False)

        print(f'saved {year} round {round_number}')
        time.sleep(8)

    except fastf1.exceptions.RateLimitExceededError:
        print('rate limit hit, pausing for an hour')
        time.sleep(36)

    except Exception as e:
        print(f'{year} round {round_number} failed: {e}')