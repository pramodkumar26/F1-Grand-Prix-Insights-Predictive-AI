import pandas as pd
import os

def combine_files(folder):
    all_rows = []

    for filename in os.listdir(folder):
        if not filename.endswith('.csv'):
            continue

        year, round_number = filename.replace('.csv', '').split('_')
        df = pd.read_csv(os.path.join(folder, filename))
        df['year'] = int(year)
        df['round'] = int(round_number)
        all_rows.append(df)

    combined = pd.concat(all_rows, ignore_index=True)
    return combined

laps = combine_files('data/raw/laps')
results = combine_files('data/raw/results')
weather = combine_files('data/raw/weather')
qualifying = combine_files('data/raw/qualifying')
race_control = combine_files('data/raw/race_control')

laps.to_csv('data/processed/all_laps.csv', index=False)
results.to_csv('data/processed/all_results.csv', index=False)
weather.to_csv('data/processed/all_weather.csv', index=False)
qualifying.to_csv('data/processed/all_qualifying.csv', index=False)
race_control.to_csv('data/processed/all_race_control.csv', index=False)

print('laps:', laps.shape)
print('results:', results.shape)
print('weather:', weather.shape)
print('qualifying:', qualifying.shape)
print('race control:', race_control.shape)