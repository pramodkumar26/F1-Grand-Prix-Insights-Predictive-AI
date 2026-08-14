import sqlite3
import pandas as pd
from scipy.stats import linregress

DB_PATH = 'f1.db'
MIN_LAPS = 4


def load_clean_laps(conn):
    # pulls every clean lap, coerces lap_time to numeric, drops anything that fails
    laps = pd.read_sql("SELECT * FROM clean_laps", conn)
    laps['lap_time'] = pd.to_numeric(laps['lap_time'], errors='coerce')
    bad_count = laps['lap_time'].isna().sum()
    if bad_count:
        print(f"{bad_count} rows had a non numeric lap_time, dropping them")
    laps = laps.dropna(subset=['lap_time'])
    return laps


def add_normalized_lap_time(laps):
    # subtracts each race's own median lap time so fuel load and circuit differences wash out
    race_medians = laps.groupby('race_id')['lap_time'].median()
    laps['race_median'] = laps['race_id'].map(race_medians)
    laps['normalized_lap_time'] = laps['lap_time'] - laps['race_median']
    return laps


def compute_stint_degradation(laps):
    # fits a line per race, driver, stint, compound and keeps the slope as the degradation rate
    results = []
    group_cols = ['race_id', 'driver_id', 'stint_number', 'compound_id']

    for keys, group in laps.groupby(group_cols):
        if len(group) < MIN_LAPS:
            continue

        fit = linregress(group['tyre_age'], group['normalized_lap_time'])

        results.append({
            'race_id': keys[0],
            'driver_id': keys[1],
            'stint_number': keys[2],
            'compound_id': keys[3],
            'degradation_rate': fit.slope,
            'intercept': fit.intercept,
            'r_squared': fit.rvalue ** 2,
            'lap_count': len(group)
        })

    return pd.DataFrame(results)


def build_tyre_degradation_table():
    conn = sqlite3.connect(DB_PATH)
    laps = load_clean_laps(conn)
    laps = add_normalized_lap_time(laps)
    degradation = compute_stint_degradation(laps)

    degradation.to_sql('tyre_degradation_stints', conn, if_exists='replace', index=False)
    conn.commit()
    conn.close()

    return degradation


if __name__ == '__main__':
    degradation = build_tyre_degradation_table()
    print(f"tyre_degradation_stints table built, {len(degradation)} rows")
    print(degradation['degradation_rate'].describe())