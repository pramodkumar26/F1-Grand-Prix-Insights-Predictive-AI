import pandas as pd
import os

def combine_files(folder):
    all_rows = []

    for filename in os.listdir(folder):
        if not filename.endswith('.csv'):
            continue

        df = pd.read_csv(os.path.join(folder, filename))
        all_rows.append(df)

    combined = pd.concat(all_rows, ignore_index=True)
    return combined

schedule = combine_files('data/raw/schedule')
schedule.to_csv('data/processed/all_schedule.csv', index=False)

print('schedule:', schedule.shape)