import sqlite3
import pandas as pd

DB_PATH = 'f1.db'


def load_results(conn):
    return pd.read_sql("SELECT result_id, race_id, grid_position FROM fact_race_results", conn)


def fix_pit_lane_starts(results):
    # FastF1 encodes a pit lane start as grid_position 0, not an actual grid slot.
    # Replace with that race's true back of grid slot instead, so anything computed
    # off grid_position (positions_gained, trailing driver form, etc) doesn't read a
    # pit lane start as if the driver started from an impossible P0.
    grid_size = results.groupby('race_id')['grid_position'].transform(lambda s: s[s > 0].max())
    pit_lane_start = results['grid_position'] == 0
    results.loc[pit_lane_start, 'grid_position'] = grid_size[pit_lane_start] + 1
    return results, pit_lane_start.sum()


def apply_fix(conn, results):
    results[['result_id', 'grid_position']].to_sql('grid_position_fix_staging', conn, if_exists='replace', index=False)
    conn.execute("CREATE INDEX idx_staging_result_id ON grid_position_fix_staging(result_id)")

    conn.execute("""
        UPDATE fact_race_results
        SET grid_position = (SELECT grid_position FROM grid_position_fix_staging WHERE grid_position_fix_staging.result_id = fact_race_results.result_id)
        WHERE result_id IN (SELECT result_id FROM grid_position_fix_staging)
    """)

    conn.execute("DROP TABLE grid_position_fix_staging")
    conn.commit()


def fix_grid_position():
    conn = sqlite3.connect(DB_PATH)
    results = load_results(conn)
    results, fixed_count = fix_pit_lane_starts(results)
    apply_fix(conn, results)
    conn.close()
    return fixed_count


def load_sprint_results(conn):
    return pd.read_sql(
        "SELECT sprint_result_id AS result_id, race_id, sprint_grid_position AS grid_position FROM fact_sprint_results",
        conn
    )


def apply_sprint_fix(conn, results):
    results[['result_id', 'grid_position']].to_sql('sprint_grid_position_fix_staging', conn, if_exists='replace', index=False)
    conn.execute("CREATE INDEX idx_sprint_staging_result_id ON sprint_grid_position_fix_staging(result_id)")

    conn.execute("""
        UPDATE fact_sprint_results
        SET sprint_grid_position = (SELECT grid_position FROM sprint_grid_position_fix_staging WHERE sprint_grid_position_fix_staging.result_id = fact_sprint_results.sprint_result_id)
        WHERE sprint_result_id IN (SELECT result_id FROM sprint_grid_position_fix_staging)
    """)

    conn.execute("DROP TABLE sprint_grid_position_fix_staging")
    conn.commit()


def fix_sprint_grid_position():
    conn = sqlite3.connect(DB_PATH)
    results = load_sprint_results(conn)
    results, fixed_count = fix_pit_lane_starts(results)
    apply_sprint_fix(conn, results)
    conn.close()
    return fixed_count


if __name__ == '__main__':
    fixed_count = fix_grid_position()
    print(f"fact_race_results grid_position fixed at source, {fixed_count} pit lane starts converted from 0 to true back of grid slot")

    sprint_fixed_count = fix_sprint_grid_position()
    print(f"fact_sprint_results sprint_grid_position fixed at source, {sprint_fixed_count} pit lane starts converted from 0 to true back of grid slot")
