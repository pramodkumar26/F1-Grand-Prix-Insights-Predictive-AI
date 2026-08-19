import fastf1
import os
from datetime import datetime

fastf1.Cache.enable_cache('cache')

years = [2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025, 2026]
current_year = datetime.now().year

os.makedirs('data/raw/schedule', exist_ok=True)

for year in years:
    schedule_path = f'data/raw/schedule/{year}.csv'

    # a finished season's calendar never changes, safe to skip. The current season's
    # calendar can still be amended (added/cancelled/rescheduled races), so it's always
    # re-fetched even if already saved - this keeps a daily automated run fast on a
    # quiet day without ever serving a stale current-season schedule.
    if year != current_year and os.path.exists(schedule_path):
        print(f'{year} schedule already saved, skipping')
        continue

    schedule = fastf1.get_event_schedule(year)
    schedule = schedule[schedule['RoundNumber'] != 0]

    columns_needed = ['RoundNumber', 'EventName', 'Location', 'Country', 'EventDate']
    schedule = schedule[columns_needed]
    schedule['year'] = year

    schedule.to_csv(schedule_path, index=False)
    print(f'saved schedule for {year}')