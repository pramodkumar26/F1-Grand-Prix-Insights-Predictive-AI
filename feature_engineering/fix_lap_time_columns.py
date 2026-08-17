import sqlite3
import pandas as pd

DB_PATH = 'f1.db'
TIME_COLUMNS = ['lap_time', 'pit_in_time', 'pit_out_time']


def load_time_columns(conn):
    # pulls lap_id plus the three columns that need converting
    columns = ', '.join(TIME_COLUMNS)
    return pd.read_sql(f"SELECT lap_id, {columns} FROM fact_laps", conn)


def convert_to_seconds(series):
    # a column can mix already-fixed numeric values (from a prior run) with fresh
    # timedelta strings (from newly loaded races) in the same pandas object column,
    # so the numeric/text split has to happen per value, not once for the whole series
    numeric = pd.to_numeric(series, errors='coerce')
    needs_parsing = numeric.isna() & series.notna()
    if needs_parsing.any():
        parsed = pd.to_timedelta(series[needs_parsing], errors='coerce').dt.total_seconds()
        numeric.loc[needs_parsing] = parsed
    return numeric


def apply_fix(conn, laps):
    # stages the converted values, indexes lap_id so the update below isn't a full scan per row
    laps.to_sql('lap_time_fix_staging', conn, if_exists='replace', index=False)
    conn.execute("CREATE INDEX idx_staging_lap_id ON lap_time_fix_staging(lap_id)")

    conn.execute("""
        UPDATE fact_laps
        SET lap_time = (SELECT lap_time FROM lap_time_fix_staging WHERE lap_time_fix_staging.lap_id = fact_laps.lap_id),
            pit_in_time = (SELECT pit_in_time FROM lap_time_fix_staging WHERE lap_time_fix_staging.lap_id = fact_laps.lap_id),
            pit_out_time = (SELECT pit_out_time FROM lap_time_fix_staging WHERE lap_time_fix_staging.lap_id = fact_laps.lap_id)
    """)

    conn.execute("DROP TABLE lap_time_fix_staging")
    conn.commit()


def fix_lap_time_columns():
    conn = sqlite3.connect(DB_PATH)
    laps = load_time_columns(conn)

    for col in TIME_COLUMNS:
        before_dtype = laps[col].dtype
        laps[col] = convert_to_seconds(laps[col])
        print(f"{col}: was {before_dtype}, now converted")

    apply_fix(conn, laps)
    conn.close()
    return laps


if __name__ == '__main__':
    laps = fix_lap_time_columns()
    print(f"fact_laps time columns converted, {len(laps)} rows processed")
    print(laps[TIME_COLUMNS].describe())