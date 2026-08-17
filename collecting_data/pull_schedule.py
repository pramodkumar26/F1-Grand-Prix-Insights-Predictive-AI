import fastf1
import os

fastf1.Cache.enable_cache('cache')

years = [2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025, 2026]

os.makedirs('data/raw/schedule', exist_ok=True)

for year in years:
    schedule = fastf1.get_event_schedule(year)
    schedule = schedule[schedule['RoundNumber'] != 0]

    columns_needed = ['RoundNumber', 'EventName', 'Location', 'Country', 'EventDate']
    schedule = schedule[columns_needed]
    schedule['year'] = year

    schedule.to_csv(f'data/raw/schedule/{year}.csv', index=False)
    print(f'saved schedule for {year}')