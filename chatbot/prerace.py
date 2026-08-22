"""Feature assembly for a race that has qualified but not yet run.

The normal feature tables are built FROM fact_race_results, so a race with no result
row gets no features at all. That is fine for training and backtesting, but it is
exactly what blocks predicting an upcoming race. This module recomputes the two
trailing features on demand for such a race, using only prior races, and reads the
qualifying-derived features from the tables that DO cover an unraced weekend.

The trailing formulas here deliberately mirror feature_engineering/build_team_strength.py
and build_driver_form.py. backtest_prerace_features() checks that they still agree.
"""
import sqlite3

import pandas as pd

DB_PATH = 'f1.db'
ROLLING_WINDOW = 15  # matches build_team_strength.py


def team_strength_before(conn, team_id, year, round_number):
    """Rolling mean of the team's points over its last 15 races before this one."""
    points = pd.read_sql("""
        SELECT d.year, d.round, SUM(r.points) AS team_points
        FROM fact_race_results r
        JOIN dim_race d ON r.race_id = d.race_id
        WHERE r.team_id = ? AND (d.year < ? OR (d.year = ? AND d.round < ?))
        GROUP BY r.race_id, d.year, d.round
        ORDER BY d.year, d.round
    """, conn, params=(team_id, year, year, round_number))
    if len(points) == 0:
        return None
    return float(points.team_points.tail(ROLLING_WINDOW).mean())


def driver_form_before(conn, driver_id, year, round_number):
    """Expanding mean of the driver's positions gained across all prior finished races."""
    results = pd.read_sql("""
        SELECT r.grid_position, r.finish_position, r.status
        FROM fact_race_results r
        JOIN dim_race d ON r.race_id = d.race_id
        WHERE r.driver_id = ? AND (d.year < ? OR (d.year = ? AND d.round < ?))
        ORDER BY d.year, d.round
    """, conn, params=(driver_id, year, year, round_number))
    if len(results) == 0:
        return None
    # same finisher definition as build_driver_form.py, including the post-2023 "Lapped"
    finished = (results.status.eq('Finished')
                | results.status.str.startswith('+', na=False)
                | results.status.eq('Lapped'))
    results = results[finished].dropna(subset=['grid_position', 'finish_position'])
    if len(results) == 0:
        return None
    gained = results.grid_position - results.finish_position
    return float(gained.mean())


def has_qualifying(conn, race_id):
    n = pd.read_sql('SELECT COUNT(*) n FROM fact_qualifying_results WHERE race_id = ?',
                    conn, params=(race_id,)).n.iloc[0]
    return int(n) > 0


def has_result(conn, race_id):
    n = pd.read_sql('SELECT COUNT(*) n FROM fact_race_results WHERE race_id = ?',
                    conn, params=(race_id,)).n.iloc[0]
    return int(n) > 0


def qualifying_grid(conn, race_id):
    """The starting order as qualified. Grid penalties are applied after qualifying and
    are not published anywhere we can read before the race, so this is a close proxy
    rather than the certain grid - measured at roughly 0.003 AUC of accuracy.

    team_id can be null when FastF1 reports a team name that does not match dim_team,
    which varies by FastF1 version. team_strength is looked up by team, so a null there
    would break the prediction entirely. Fall back to the team the driver most recently
    actually raced for, which is right in every case except a mid-season switch.
    """
    grid = pd.read_sql("""
        SELECT q.driver_id, q.team_id, q.quali_position, d.driver_name, t.team_name
        FROM fact_qualifying_results q
        JOIN dim_driver d ON q.driver_id = d.driver_id
        LEFT JOIN dim_team t ON q.team_id = t.team_id
        WHERE q.race_id = ?
        ORDER BY q.quali_position
    """, conn, params=(race_id,))

    missing = grid.team_id.isna()
    if missing.any():
        for idx in grid.index[missing]:
            recent = pd.read_sql("""
                SELECT fr.team_id, t.team_name
                FROM fact_race_results fr
                JOIN dim_race r ON fr.race_id = r.race_id
                LEFT JOIN dim_team t ON fr.team_id = t.team_id
                WHERE fr.driver_id = ? AND fr.team_id IS NOT NULL
                ORDER BY r.race_date DESC LIMIT 1
            """, conn, params=(int(grid.at[idx, 'driver_id']),))
            if len(recent):
                grid.at[idx, 'team_id'] = recent.team_id.iloc[0]
                grid.at[idx, 'team_name'] = recent.team_name.iloc[0]

    return grid.dropna(subset=['team_id'])


def build_prerace_row(conn, driver_id, team_id, race_id, year, round_number, quali_position):
    """The same five features the models were trained on, assembled without a result row."""
    def scalar(query, params):
        row = pd.read_sql(query, conn, params=params)
        value = row.iloc[0, 0] if len(row) else None
        return float('nan') if value is None else value

    row = {
        'grid_position': quali_position,
        'team_strength': team_strength_before(conn, team_id, year, round_number),
        'driver_form': driver_form_before(conn, driver_id, year, round_number),
        'qualifying_pace_delta': scalar(
            'SELECT qualifying_pace_delta FROM qualifying_pace_delta WHERE race_id=? AND driver_id=?',
            (race_id, driver_id)),
        'qualifying_gap_to_pole': scalar(
            'SELECT qualifying_gap_to_pole FROM qualifying_gap_to_pole WHERE race_id=? AND driver_id=?',
            (race_id, driver_id)),
    }
    for key in ('team_strength', 'driver_form'):
        if row[key] is None:
            row[key] = float('nan')
    return pd.DataFrame([row])[['grid_position', 'team_strength', 'driver_form',
                                 'qualifying_pace_delta', 'qualifying_gap_to_pole']]


def backtest_prerace_features(limit_races=40):
    """Do the on-demand trailing features match what the normal pipeline stored?

    Runs against past races, where both paths can be compared directly. If these
    disagree, a prediction for an upcoming race is being built on different numbers
    than the model was trained on, which would invalidate the whole thing.
    """
    conn = sqlite3.connect(DB_PATH)
    stored = pd.read_sql("""
        SELECT ts.race_id, ts.driver_id, ts.team_id, ts.team_strength,
               df.driver_form, r.year, r.round
        FROM team_strength ts
        JOIN dim_race r ON ts.race_id = r.race_id
        LEFT JOIN driver_form df ON df.race_id = ts.race_id AND df.driver_id = ts.driver_id
        WHERE r.year >= 2024
        ORDER BY r.year DESC, r.round DESC
    """, conn)
    stored = stored[stored.race_id.isin(stored.race_id.unique()[:limit_races])]

    strength_diffs, form_diffs = [], []
    for _, s in stored.iterrows():
        recomputed_strength = team_strength_before(conn, s.team_id, s.year, s['round'])
        if pd.notna(s.team_strength) and recomputed_strength is not None:
            strength_diffs.append(abs(recomputed_strength - s.team_strength))
        recomputed_form = driver_form_before(conn, s.driver_id, s.year, s['round'])
        if pd.notna(s.driver_form) and recomputed_form is not None:
            form_diffs.append(abs(recomputed_form - s.driver_form))

    conn.close()
    return {
        'rows_checked': len(stored),
        'team_strength_max_diff': max(strength_diffs) if strength_diffs else None,
        'driver_form_max_diff': max(form_diffs) if form_diffs else None,
    }


if __name__ == '__main__':
    print(backtest_prerace_features())
