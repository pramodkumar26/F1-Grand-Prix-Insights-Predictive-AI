import fastf1
import os
import time

fastf1.Cache.enable_cache('cache')

years = [2023, 2024, 2025, 2026]

for year in years:
    schedule = fastf1.get_event_schedule(year)

    for round_number in schedule['RoundNumber']:
        if round_number == 0:
            continue

        quali_path = f'data/raw/qualifying/{year}_{round_number}.csv'

        if os.path.exists(quali_path):
            print(f'{year} round {round_number} qualifying already saved, skipping')
            continue

        try:
            session = fastf1.get_session(year, round_number, 'Q')
            session.load()

            laps = session.laps
            laps.to_csv(quali_path, index=False)

            print(f'saved {year} round {round_number} qualifying')
            time.sleep(8)

        except fastf1.exceptions.RateLimitExceededError:
            print('rate limit hit, pausing for an hour')
            time.sleep(36)

        except Exception as e:
            print(f'{year} round {round_number} qualifying failed: {e}')