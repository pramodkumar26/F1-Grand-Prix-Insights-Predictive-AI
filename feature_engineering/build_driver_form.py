import sqlite3
import pandas as pd

DB_PATH = 'f1.db'


def load_results(conn):
    # grid and finish position per driver, plus year/round for chronological ordering
    query = """
        SELECT r.race_id, r.driver_id, r.grid_position, r.finish_position, r.status, d.year, d.round
        FROM fact_race_results r
        JOIN dim_race d ON r.race_id = d.race_id
    """
    results = pd.read_sql(query, conn)
    # "Lapped" is FastF1's post-2023 spelling of "+1 Lap", a finisher, not a retirement.
    # Omitting it silently dropped 297 legitimate finishers from 2023 onward. See the
    # full explanation in build_dnf_rate.py.
    finished = (results['status'].eq('Finished')
                | results['status'].str.startswith('+', na=False)
                | results['status'].eq('Lapped'))
    results = results[finished].copy()
    results = results.dropna(subset=['grid_position', 'finish_position'])
    results['positions_gained'] = results['grid_position'] - results['finish_position']
    return results


def compute_trailing_form(results):
    # trailing average of the driver's own positions_gained, strictly prior races only.
    # team_strength already captures car pace, this is meant to capture racecraft on
    # top of that, a driver who consistently outperforms their grid slot across teams
    # and seasons is showing something that isn't just "had a fast car that day".
    rates = []
    for driver_id, group in results.groupby('driver_id'):
        group = group.sort_values(['year', 'round']).copy()
        group['driver_form'] = group['positions_gained'].shift().expanding().mean().values
        rates.append(group)

    return pd.concat(rates, ignore_index=True)


def build_driver_form_table():
    conn = sqlite3.connect(DB_PATH)
    results = load_results(conn)
    results = compute_trailing_form(results)

    output = results[['race_id', 'driver_id', 'driver_form']]
    output.to_sql('driver_form', conn, if_exists='replace', index=False)
    conn.commit()
    conn.close()

    return output


if __name__ == '__main__':
    output = build_driver_form_table()
    print(f"driver_form table built, {len(output)} rows")
    print(f"rows with no trailing history yet: {output['driver_form'].isna().sum()}")
    print(output['driver_form'].describe())
