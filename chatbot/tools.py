import sqlite3
from datetime import date
import pandas as pd

from chatbot.resolve import resolve_driver, resolve_race, resolve_team
from chatbot.registry import load_models, load_explainers
from chatbot import prerace

DB_PATH = 'f1.db'

SHARED_FEATURES = ['grid_position', 'team_strength', 'driver_form', 'qualifying_pace_delta', 'qualifying_gap_to_pole']
STRATEGY_SHIFT_FEATURES = [
    'strategic_aggressiveness', 'track_difficulty_index', 'pit_stop_delta', 'primary_degradation_rate',
    'grid_position', 'team_strength', 'driver_form', 'qualifying_pace_delta',
    'is_wet_race', 'pace_consistency', 'pit_stop_count',
]
OVERTAKING_FEATURES = [
    'grid_position', 'team_strength', 'driver_overtake_trailing',
    'qualifying_pace_delta', 'qualifying_gap_to_pole', 'circuit_overtake_trailing',
]


def _resolve_driver_and_race(driver, race, year):
    d = resolve_driver(driver)
    r = resolve_race(race, year)
    if d is None:
        return None, None, {'found': False, 'reason': f'no driver matching "{driver}" in the dataset'}
    if isinstance(d, dict) and d.get('ambiguous'):
        return None, None, {'found': False, 'ambiguous': True, 'candidates': d['candidates']}
    if r is None:
        return None, None, {'found': False, 'reason': f'no race matching "{race}" ({year}) in the dataset'}
    if isinstance(r, dict) and r.get('ambiguous'):
        return None, None, {'found': False, 'ambiguous': True, 'candidates': r['candidates']}
    return d, r, None


def _scalar(conn, query, params):
    row = pd.read_sql(query, conn, params=params)
    return row.iloc[0, 0] if len(row) else None


def _nullable(conn, query, params):
    """Like _scalar, but a missing value becomes NaN (float) rather than None, so a single-row
    DataFrame column keeps a proper numeric dtype instead of collapsing to dtype=object - which
    both XGBoost and MLflow's pyfunc schema enforcement reject outright."""
    value = _scalar(conn, query, params)
    return float('nan') if value is None else value


def _season_progress(conn, year):
    """A season only has a champion once every scheduled round has been run."""
    completed = _scalar(conn,
        "SELECT COUNT(DISTINCT fr.race_id) FROM fact_race_results fr JOIN dim_race r ON fr.race_id = r.race_id WHERE r.year = ?",
        (int(year),))
    scheduled = _scalar(conn, "SELECT COUNT(*) FROM dim_race WHERE year = ?", (int(year),))
    completed = int(completed or 0)
    scheduled = int(scheduled or 0)
    return completed, scheduled, (scheduled > 0 and completed >= scheduled)


def _clean(value):
    """Tool results get serialised to JSON, which accepts neither NaN nor numpy scalars."""
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value.item() if hasattr(value, 'item') else value


def _records(df):
    return [{k: _clean(v) for k, v in row.items()} for row in df.to_dict('records')]


def _row(series):
    return {k: _clean(v) for k, v in series.items()}


def get_result_row(driver_id, race_id, conn):
    row = pd.read_sql(
        "SELECT * FROM fact_race_results WHERE race_id=? AND driver_id=?",
        conn, params=(race_id, driver_id)
    )
    return row.iloc[0] if len(row) else None


def build_shared_row(driver_id, race_id, team_id, conn):
    row = {}
    row['grid_position'] = _scalar(conn,
        "SELECT grid_position FROM fact_race_results WHERE race_id=? AND driver_id=?", (race_id, driver_id))
    row['team_strength'] = _nullable(conn,
        "SELECT DISTINCT team_strength FROM team_strength WHERE race_id=? AND team_id=?", (race_id, team_id))
    row['driver_form'] = _nullable(conn,
        "SELECT driver_form FROM driver_form WHERE race_id=? AND driver_id=?", (race_id, driver_id))
    row['qualifying_pace_delta'] = _nullable(conn,
        "SELECT qualifying_pace_delta FROM qualifying_pace_delta WHERE race_id=? AND driver_id=?", (race_id, driver_id))
    row['qualifying_gap_to_pole'] = _nullable(conn,
        "SELECT qualifying_gap_to_pole FROM qualifying_gap_to_pole WHERE race_id=? AND driver_id=?", (race_id, driver_id))
    return pd.DataFrame([row])[SHARED_FEATURES]


def build_strategy_shift_row(driver_id, race_id, team_id, conn):
    row = {}
    row['strategic_aggressiveness'] = _nullable(conn,
        "SELECT strategic_aggressiveness FROM strategic_aggressiveness WHERE race_id=? AND driver_id=?", (race_id, driver_id))
    row['track_difficulty_index'] = _nullable(conn,
        "SELECT track_difficulty_index FROM track_difficulty_index WHERE race_id=?", (race_id,))
    row['pit_stop_delta'] = _nullable(conn,
        "SELECT pit_stop_delta FROM pit_stop_delta WHERE race_id=? AND driver_id=?", (race_id, driver_id))
    row['primary_degradation_rate'] = _nullable(conn,
        """SELECT fuel_adjusted_degradation_rate FROM tyre_degradation_adjusted
           WHERE race_id=? AND driver_id=? ORDER BY lap_count DESC LIMIT 1""", (race_id, driver_id))
    row['grid_position'] = _scalar(conn,
        "SELECT grid_position FROM fact_race_results WHERE race_id=? AND driver_id=?", (race_id, driver_id))
    row['team_strength'] = _nullable(conn,
        "SELECT DISTINCT team_strength FROM team_strength WHERE race_id=? AND team_id=?", (race_id, team_id))
    row['driver_form'] = _nullable(conn,
        "SELECT driver_form FROM driver_form WHERE race_id=? AND driver_id=?", (race_id, driver_id))
    row['qualifying_pace_delta'] = _nullable(conn,
        "SELECT qualifying_pace_delta FROM qualifying_pace_delta WHERE race_id=? AND driver_id=?", (race_id, driver_id))
    row['is_wet_race'] = _nullable(conn,
        "SELECT is_wet_race FROM race_wet_flag WHERE race_id=?", (race_id,))
    row['pace_consistency'] = _nullable(conn,
        "SELECT pace_consistency FROM pace_consistency WHERE race_id=? AND driver_id=?", (race_id, driver_id))
    row['pit_stop_count'] = _nullable(conn,
        "SELECT MAX(stint_number) FROM fact_laps WHERE race_id=? AND driver_id=?", (race_id, driver_id))
    return pd.DataFrame([row])[STRATEGY_SHIFT_FEATURES]


def build_overtaking_row(driver_id, race_id, team_id, conn):
    row = {}
    row['grid_position'] = _scalar(conn,
        "SELECT grid_position FROM fact_race_results WHERE race_id=? AND driver_id=?", (race_id, driver_id))
    row['team_strength'] = _nullable(conn,
        "SELECT DISTINCT team_strength FROM team_strength WHERE race_id=? AND team_id=?", (race_id, team_id))
    row['driver_overtake_trailing'] = _nullable(conn,
        "SELECT driver_overtake_trailing FROM driver_overtake_trailing WHERE race_id=? AND driver_id=?", (race_id, driver_id))
    row['qualifying_pace_delta'] = _nullable(conn,
        "SELECT qualifying_pace_delta FROM qualifying_pace_delta WHERE race_id=? AND driver_id=?", (race_id, driver_id))
    row['qualifying_gap_to_pole'] = _nullable(conn,
        "SELECT qualifying_gap_to_pole FROM qualifying_gap_to_pole WHERE race_id=? AND driver_id=?", (race_id, driver_id))
    row['circuit_overtake_trailing'] = _nullable(conn,
        "SELECT circuit_overtake_trailing FROM circuit_overtake_trailing WHERE race_id=?", (race_id,))
    return pd.DataFrame([row])[OVERTAKING_FEATURES]


def get_shap_reasons(explainer, X, top_n=3):
    shap_values = explainer(X)
    values = shap_values.values[0]
    reasons = sorted(zip(X.columns, values), key=lambda pair: abs(pair[1]), reverse=True)[:top_n]
    return [
        {'feature': f, 'shap_contribution': float(v), 'direction': 'increased' if v > 0 else 'decreased'}
        for f, v in reasons
    ]


def _predict_classifier(driver, race, year, model_key):
    conn = sqlite3.connect(DB_PATH)
    try:
        d, r, error = _resolve_driver_and_race(driver, race, year)
        if error:
            return error
        result = get_result_row(d['driver_id'], r['race_id'], conn)
        if result is None or pd.isna(result.grid_position):
            return {'found': False, 'reason': f"{d['driver_name']} did not start {r['race_name']} {r['year']} (no grid position on record, likely a withdrawal)"}
        X = build_shared_row(d['driver_id'], r['race_id'], int(result.team_id), conn)
        models = load_models()
        explainers = load_explainers()
        probability = float(models[model_key].predict_proba(X)[:, 1][0])
        reasons = get_shap_reasons(explainers[model_key], X)
        return {
            'found': True, 'driver': d['driver_name'], 'race': r['race_name'], 'year': r['year'],
            'probability': probability, 'top_reasons': reasons, 'confidence_tier': 'confident_prediction',
        }
    except Exception as e:
        return {'found': False, 'reason': f'lookup failed: {e}'}
    finally:
        conn.close()


def predict_podium(driver: str, race: str, year: int) -> dict:
    """Predict the probability a driver finishes on the podium (top 3) in a specific past race already in the dataset (2018-2026). Returns probability plus the top reasons behind it."""
    return _predict_classifier(driver, race, year, 'podium')


def predict_win(driver: str, race: str, year: int) -> dict:
    """Predict the probability a driver wins a specific past race already in the dataset (2018-2026). Returns probability plus the top reasons behind it."""
    return _predict_classifier(driver, race, year, 'win')


def predict_points(driver: str, race: str, year: int) -> dict:
    """Predict how many championship points a driver scores in a specific past race already in the dataset (2018-2026). Returns a point estimate plus the top reasons behind it."""
    conn = sqlite3.connect(DB_PATH)
    try:
        d, r, error = _resolve_driver_and_race(driver, race, year)
        if error:
            return error
        result = get_result_row(d['driver_id'], r['race_id'], conn)
        if result is None or pd.isna(result.grid_position):
            return {'found': False, 'reason': f"{d['driver_name']} did not start {r['race_name']} {r['year']} (no grid position on record, likely a withdrawal)"}
        X = build_shared_row(d['driver_id'], r['race_id'], int(result.team_id), conn)
        models = load_models()
        explainers = load_explainers()
        predicted_points = float(models['points_scored'].predict(X)[0])
        reasons = get_shap_reasons(explainers['points_scored'], X)
        return {
            'found': True, 'driver': d['driver_name'], 'race': r['race_name'], 'year': r['year'],
            'predicted_points': predicted_points, 'top_reasons': reasons, 'confidence_tier': 'confident_prediction',
        }
    except Exception as e:
        return {'found': False, 'reason': f'lookup failed: {e}'}
    finally:
        conn.close()


def explain_strategy_shift(driver: str, race: str, year: int) -> dict:
    """Explain why a driver gained or lost positions relative to their grid slot in a specific past race that already happened. This is an explanation of a completed race, never a forecast."""
    conn = sqlite3.connect(DB_PATH)
    try:
        d, r, error = _resolve_driver_and_race(driver, race, year)
        if error:
            return error
        result = get_result_row(d['driver_id'], r['race_id'], conn)
        if result is None or pd.isna(result.grid_position):
            return {'found': False, 'reason': f"{d['driver_name']} did not start {r['race_name']} {r['year']} (no grid position on record, likely a withdrawal)"}
        X = build_strategy_shift_row(d['driver_id'], r['race_id'], int(result.team_id), conn)
        models = load_models()
        explainers = load_explainers()
        predicted_shift = float(models['strategy_shift'].predict(X)[0])
        actual_shift = int(result.grid_position) - int(result.finish_position) if pd.notna(result.finish_position) else None
        reasons = get_shap_reasons(explainers['strategy_shift'], X)
        return {
            'found': True, 'driver': d['driver_name'], 'race': r['race_name'], 'year': r['year'],
            'actual_positions_gained': actual_shift, 'model_estimate': predicted_shift,
            'top_reasons': reasons, 'framing': 'explanatory',
        }
    except Exception as e:
        return {'found': False, 'reason': f'lookup failed: {e}'}
    finally:
        conn.close()


def estimate_overtakes(driver: str, race: str, year: int) -> dict:
    """Estimate how many overtakes a driver makes in a specific past race already in the dataset. This model only explains a modest share of the variation, so the result must be treated as a rough, relative estimate, not a precise number."""
    conn = sqlite3.connect(DB_PATH)
    try:
        d, r, error = _resolve_driver_and_race(driver, race, year)
        if error:
            return error
        result = get_result_row(d['driver_id'], r['race_id'], conn)
        if result is None or pd.isna(result.grid_position):
            return {'found': False, 'reason': f"{d['driver_name']} did not start {r['race_name']} {r['year']} (no grid position on record, likely a withdrawal)"}
        X = build_overtaking_row(d['driver_id'], r['race_id'], int(result.team_id), conn)
        models = load_models()
        point_estimate = float(models['overtaking'].predict(X)[0])

        grid_position = X['grid_position'].iloc[0]
        typical = _scalar(conn, """
            SELECT AVG(o.overtakes_made) FROM overtaking_features o
            WHERE o.grid_position = ?
        """, (int(grid_position),)) if pd.notna(grid_position) else None

        if typical is not None:
            if point_estimate > typical * 1.15:
                relative = 'more than typical for this grid slot'
            elif point_estimate < typical * 0.85:
                relative = 'fewer than typical for this grid slot'
            else:
                relative = 'about typical for this grid slot'
        else:
            relative = 'typical range unknown for this grid slot'

        return {
            'found': True, 'driver': d['driver_name'], 'race': r['race_name'], 'year': r['year'],
            'point_estimate': point_estimate, 'relative_to_typical': relative,
            'confidence_tier': 'hedged_estimate',
        }
    except Exception as e:
        return {'found': False, 'reason': f'lookup failed: {e}'}
    finally:
        conn.close()


def get_teammate_scorecard(driver: str, year: int) -> dict:
    """Look up how a driver compared to their own teammate in a given season (qualifying and race head-to-head). This is a measured fact from historical results, not a model prediction."""
    conn = sqlite3.connect(DB_PATH)
    try:
        d = resolve_driver(driver)
        if d is None:
            return {'found': False, 'reason': f'no driver matching "{driver}" in the dataset'}
        if isinstance(d, dict) and d.get('ambiguous'):
            return {'found': False, 'ambiguous': True, 'candidates': d['candidates']}
        row = pd.read_sql(
            "SELECT * FROM teammate_scorecard WHERE driver_id=? AND year=?",
            conn, params=(d['driver_id'], int(year))
        )
        if len(row) == 0:
            return {'found': False, 'reason': f"no teammate scorecard for {d['driver_name']} in {year} (not enough shared races with a teammate that season)"}
        return {'found': True, 'confidence_tier': 'measured_fact', **_row(row.iloc[0])}
    except Exception as e:
        return {'found': False, 'reason': f'lookup failed: {e}'}
    finally:
        conn.close()


def get_pit_stop_scorecard(year: int, team: str = None) -> dict:
    """Look up a team's pit stop performance in a season, or the best-ranked team that season if no team is given. This is a measured fact from historical results, not a model prediction."""
    conn = sqlite3.connect(DB_PATH)
    try:
        if team:
            t = resolve_team(team)
            if t is None:
                return {'found': False, 'reason': f'no team matching "{team}" in the dataset'}
            if isinstance(t, dict) and t.get('ambiguous'):
                return {'found': False, 'ambiguous': True, 'candidates': t['candidates']}
            row = pd.read_sql(
                "SELECT * FROM pit_stop_scorecard WHERE team_id=? AND year=?",
                conn, params=(t['team_id'], int(year))
            )
            if len(row) == 0:
                return {'found': False, 'reason': f"no pit stop scorecard for {t['team_name']} in {year}"}
            return {'found': True, 'confidence_tier': 'measured_fact', **_row(row.iloc[0])}
        else:
            row = pd.read_sql(
                "SELECT * FROM pit_stop_scorecard WHERE year=? AND rank_in_season=1",
                conn, params=(int(year),)
            )
            if len(row) == 0:
                return {'found': False, 'reason': f"no pit stop ranking available for {year}"}
            return {'found': True, 'confidence_tier': 'measured_fact', **_row(row.iloc[0])}
    except Exception as e:
        return {'found': False, 'reason': f'lookup failed: {e}'}
    finally:
        conn.close()


def get_race_summary(race: str = None, year: int = None) -> dict:
    """Look up the recorded facts about a race: who won, who started on pole, how many drivers finished or retired, how many laps were run and whether it was wet. Give a race and year, or give nothing at all to get the most recent race that has actually been run, which is what to use for questions about the last race. This is a measured fact from recorded data, not a prediction."""
    conn = sqlite3.connect(DB_PATH)
    try:
        if race is None:
            latest = pd.read_sql("""
                SELECT r.race_id, r.race_name, r.year, r.round FROM dim_race r
                WHERE EXISTS (SELECT 1 FROM fact_race_results fr WHERE fr.race_id = r.race_id)
                ORDER BY r.race_date DESC LIMIT 1
            """, conn)
            if len(latest) == 0:
                return {'found': False, 'reason': 'no completed races on record'}
            race_id = int(latest.race_id.iloc[0])
            r = {'race_id': race_id, 'race_name': latest.race_name.iloc[0],
                 'year': int(latest.year.iloc[0]), 'round': int(latest['round'].iloc[0])}
        else:
            r = resolve_race(race, year)
            if r is None:
                return {'found': False, 'reason': f'no race matching "{race}" ({year}) in the dataset'}
            if isinstance(r, dict) and r.get('ambiguous'):
                return {'found': False, 'ambiguous': True, 'candidates': r['candidates']}
            race_id = r['race_id']

        # Results and laps come from different sources and one can arrive without the
        # other, so a missing lap count must not hide a race we do have the result for.
        laps_completed = _scalar(conn, "SELECT MAX(lap_number) FROM fact_laps WHERE race_id=?", (race_id,))
        has_result = _scalar(conn, "SELECT COUNT(*) FROM fact_race_results WHERE race_id=?", (race_id,))
        if not has_result and laps_completed is None:
            return {'found': False, 'reason': f"nothing recorded yet for {r['race_name']} {r['year']}, it may not have been run"}

        winner = pd.read_sql("""
            SELECT d.driver_name FROM fact_race_results fr
            JOIN dim_driver d ON fr.driver_id = d.driver_id
            WHERE fr.race_id=? AND fr.finish_position=1
        """, conn, params=(race_id,))
        pole = pd.read_sql("""
            SELECT d.driver_name FROM fact_race_results fr
            JOIN dim_driver d ON fr.driver_id = d.driver_id
            WHERE fr.race_id=? AND fr.grid_position=1
        """, conn, params=(race_id,))
        finishers = _scalar(conn,
            "SELECT COUNT(*) FROM fact_race_results WHERE race_id=? AND (status='Finished' OR status LIKE '+%' OR status='Lapped')",
            (race_id,))
        entrants = _scalar(conn, "SELECT COUNT(*) FROM fact_race_results WHERE race_id=?", (race_id,))
        is_wet = _scalar(conn, "SELECT is_wet_race FROM race_wet_flag WHERE race_id=?", (race_id,))
        fastest = pd.read_sql("""
            SELECT d.driver_name, fr.fastest_lap_time FROM fact_race_results fr
            JOIN dim_driver d ON fr.driver_id = d.driver_id
            WHERE fr.race_id=? AND fr.fastest_lap_time IS NOT NULL
            ORDER BY fr.fastest_lap_time LIMIT 1
        """, conn, params=(race_id,))

        return {
            'found': True, 'confidence_tier': 'measured_fact',
            'race': r['race_name'], 'year': r['year'], 'round': r['round'],
            'laps_completed': int(laps_completed) if laps_completed is not None else None,
            'winner': winner.driver_name.iloc[0] if len(winner) else None,
            'pole_sitter': pole.driver_name.iloc[0] if len(pole) else None,
            'entrants': int(entrants), 'classified_finishers': int(finishers),
            'retirements': int(entrants) - int(finishers),
            'wet_race': bool(is_wet) if is_wet is not None else None,
            'fastest_lap_driver': fastest.driver_name.iloc[0] if len(fastest) else None,
            'fastest_lap_seconds': round(float(fastest.fastest_lap_time.iloc[0]), 3) if len(fastest) else None,
        }
    except Exception as e:
        return {'found': False, 'reason': f'lookup failed: {e}'}
    finally:
        conn.close()


def get_race_control_messages(race: str, year: int) -> dict:
    """Look up the official race control messages recorded during a race - red flags, safety cars, delayed starts, track conditions and similar official notices. Use this to explain what actually happened during a race, for example why it was stopped, delayed or shortened. These are recorded messages, not interpretation."""
    conn = sqlite3.connect(DB_PATH)
    try:
        r = resolve_race(race, year)
        if r is None:
            return {'found': False, 'reason': f'no race matching "{race}" ({year}) in the dataset'}
        if isinstance(r, dict) and r.get('ambiguous'):
            return {'found': False, 'ambiguous': True, 'candidates': r['candidates']}

        messages = pd.read_sql("""
            SELECT lap_number, category, flag, message
            FROM fact_race_control
            WHERE race_id = ?
              AND (category IN ('SafetyCar', 'CarEvent', 'Other') OR flag IN ('RED', 'CHEQUERED'))
            ORDER BY race_control_id
            LIMIT 60
        """, conn, params=(r['race_id'],))
        if len(messages) == 0:
            return {'found': False, 'reason': f"no race control messages recorded for {r['race_name']} {r['year']}"}

        return {
            'found': True, 'confidence_tier': 'measured_fact',
            'race': r['race_name'], 'year': r['year'],
            'note': 'routine sector yellow and green flags are excluded, these are the notable events only',
            'messages': _records(messages),
        }
    except Exception as e:
        return {'found': False, 'reason': f'lookup failed: {e}'}
    finally:
        conn.close()


def get_driver_season_results(driver: str, year: int = None) -> dict:
    """Look up a driver's season: which team or teams they drove for, and their race-by-race results with grid position, finishing position, points and status. Use this to answer which team a driver races for. If no year is given it uses the most recent season with results. This is a measured fact from historical results, not a prediction."""
    conn = sqlite3.connect(DB_PATH)
    try:
        d = resolve_driver(driver)
        if d is None:
            return {'found': False, 'reason': f'no driver matching "{driver}" in the dataset'}
        if isinstance(d, dict) and d.get('ambiguous'):
            return {'found': False, 'ambiguous': True, 'candidates': d['candidates']}
        if year is None:
            year = _scalar(conn, """
                SELECT MAX(r.year) FROM fact_race_results fr
                JOIN dim_race r ON fr.race_id = r.race_id WHERE fr.driver_id = ?
            """, (d['driver_id'],))
            if year is None:
                return {'found': False, 'reason': f"no race results on record for {d['driver_name']}"}
        results = pd.read_sql("""
            SELECT r.round, r.race_name, t.team_name, fr.grid_position,
                   fr.finish_position, fr.points, fr.status
            FROM fact_race_results fr
            JOIN dim_race r ON fr.race_id = r.race_id
            LEFT JOIN dim_team t ON fr.team_id = t.team_id
            WHERE r.year = ? AND fr.driver_id = ?
            ORDER BY r.round
        """, conn, params=(int(year), d['driver_id']))
        if len(results) == 0:
            return {'found': False, 'reason': f"no race results for {d['driver_name']} in {year}"}
        teams = [t for t in results['team_name'].dropna().unique().tolist()]
        return {
            'found': True, 'driver': d['driver_name'], 'year': int(year),
            'races_entered': len(results), 'total_points': float(results['points'].sum()),
            'teams_driven_for': teams,
            'confidence_tier': 'measured_fact', 'results': _records(results),
        }
    except Exception as e:
        return {'found': False, 'reason': f'lookup failed: {e}'}
    finally:
        conn.close()


def get_championship_standings(year: int = None) -> dict:
    """Look up the driver championship standings (total points, race + sprint combined) for a season, ranked highest first. If no year is given, uses the most recent season that has race results. This is a measured fact from historical results, not a prediction, and only reflects races that have already happened - it is not a final end-of-season result unless the season is over."""
    conn = sqlite3.connect(DB_PATH)
    try:
        if year is None:
            year = _scalar(conn, "SELECT MAX(r.year) FROM fact_race_results fr JOIN dim_race r ON fr.race_id = r.race_id", ())
        standings = pd.read_sql("""
            SELECT dr.driver_name, dr.driver_code,
                   COALESCE(rp.race_points, 0) + COALESCE(sp.sprint_points, 0) AS total_points
            FROM dim_driver dr
            JOIN (
                SELECT fr.driver_id, SUM(fr.points) AS race_points
                FROM fact_race_results fr JOIN dim_race r ON fr.race_id = r.race_id
                WHERE r.year = ? GROUP BY fr.driver_id
            ) rp ON dr.driver_id = rp.driver_id
            LEFT JOIN (
                SELECT fsr.driver_id, SUM(fsr.sprint_points) AS sprint_points
                FROM fact_sprint_results fsr JOIN dim_race r ON fsr.race_id = r.race_id
                WHERE r.year = ? GROUP BY fsr.driver_id
            ) sp ON dr.driver_id = sp.driver_id
            ORDER BY total_points DESC
        """, conn, params=(int(year), int(year)))
        completed, scheduled, finished = _season_progress(conn, year)
        if len(standings) == 0:
            return {'found': False, 'reason': f'no race results found for {year}'}
        return {
            'found': True, 'year': int(year), 'races_completed': completed,
            'races_scheduled': scheduled, 'season_complete': finished,
            'confidence_tier': 'measured_fact', 'standings': _records(standings),
        }
    except Exception as e:
        return {'found': False, 'reason': f'lookup failed: {e}'}
    finally:
        conn.close()


def get_constructor_standings(year: int = None) -> dict:
    """Look up the constructor (team) championship standings (total points, race + sprint combined) for a season, ranked highest first. If no year is given, uses the most recent season that has race results. This is a measured fact from historical results, not a prediction, and only reflects races that have already happened - it is not a final end-of-season result unless the season is over."""
    conn = sqlite3.connect(DB_PATH)
    try:
        if year is None:
            year = _scalar(conn, "SELECT MAX(r.year) FROM fact_race_results fr JOIN dim_race r ON fr.race_id = r.race_id", ())
        standings = pd.read_sql("""
            SELECT dt.team_name,
                   COALESCE(rp.race_points, 0) + COALESCE(sp.sprint_points, 0) AS total_points
            FROM dim_team dt
            JOIN (
                SELECT fr.team_id, SUM(fr.points) AS race_points
                FROM fact_race_results fr JOIN dim_race r ON fr.race_id = r.race_id
                WHERE r.year = ? GROUP BY fr.team_id
            ) rp ON dt.team_id = rp.team_id
            LEFT JOIN (
                SELECT fsr.team_id, SUM(fsr.sprint_points) AS sprint_points
                FROM fact_sprint_results fsr JOIN dim_race r ON fsr.race_id = r.race_id
                WHERE r.year = ? GROUP BY fsr.team_id
            ) sp ON dt.team_id = sp.team_id
            ORDER BY total_points DESC
        """, conn, params=(int(year), int(year)))
        completed, scheduled, finished = _season_progress(conn, year)
        if len(standings) == 0:
            return {'found': False, 'reason': f'no race results found for {year}'}
        return {
            'found': True, 'year': int(year), 'races_completed': completed,
            'races_scheduled': scheduled, 'season_complete': finished,
            'confidence_tier': 'measured_fact', 'standings': _records(standings),
        }
    except Exception as e:
        return {'found': False, 'reason': f'lookup failed: {e}'}
    finally:
        conn.close()


def get_season_calendar(year: int) -> dict:
    """Look up the full race calendar for a season: every round, race name, circuit, country and date, whether or not that race has happened yet. Use this for questions about a season's schedule, not just the single next race. This is measured/scheduled fact, not a prediction."""
    conn = sqlite3.connect(DB_PATH)
    try:
        races = pd.read_sql("""
            SELECT dr.round, dr.race_name, dc.circuit_name, dc.country, dr.race_date,
                   CASE WHEN dr.race_date < date('now') THEN 1 ELSE 0 END AS already_happened
            FROM dim_race dr JOIN dim_circuit dc ON dr.circuit_id = dc.circuit_id
            WHERE dr.year = ?
            ORDER BY dr.round
        """, conn, params=(int(year),))
        if len(races) == 0:
            return {'found': False, 'reason': f'no calendar on record for {year}'}
        return {
            'found': True, 'confidence_tier': 'measured_fact',
            'year': int(year), 'total_races': len(races), 'races': _records(races),
        }
    except Exception as e:
        return {'found': False, 'reason': f'lookup failed: {e}'}
    finally:
        conn.close()


def get_driver_career_stats(driver: str) -> dict:
    """Look up a driver's career totals across every season in the dataset (2018-2026): total races entered, wins, podiums, points, and the first and last season they appear in. Use this for career-spanning questions like total wins, not a single season. This is a measured fact from historical results, not a prediction."""
    conn = sqlite3.connect(DB_PATH)
    try:
        d = resolve_driver(driver)
        if d is None:
            return {'found': False, 'reason': f'no driver matching "{driver}" in the dataset'}
        if isinstance(d, dict) and d.get('ambiguous'):
            return {'found': False, 'ambiguous': True, 'candidates': d['candidates']}
        stats = pd.read_sql("""
            SELECT COUNT(*) AS races_entered,
                   SUM(CASE WHEN fr.finish_position = 1 THEN 1 ELSE 0 END) AS wins,
                   SUM(CASE WHEN fr.finish_position <= 3 THEN 1 ELSE 0 END) AS podiums,
                   SUM(fr.points) AS total_points,
                   MIN(r.year) AS first_season, MAX(r.year) AS last_season
            FROM fact_race_results fr JOIN dim_race r ON fr.race_id = r.race_id
            WHERE fr.driver_id = ?
        """, conn, params=(d['driver_id'],))
        row = stats.iloc[0]
        if len(stats) == 0 or row.races_entered == 0:
            return {'found': False, 'reason': f"no race results on record for {d['driver_name']}"}
        return {
            'found': True, 'driver': d['driver_name'], 'confidence_tier': 'measured_fact',
            'races_entered': int(row.races_entered), 'wins': int(row.wins), 'podiums': int(row.podiums),
            'total_points': float(row.total_points),
            'first_season': int(row.first_season), 'last_season': int(row.last_season),
        }
    except Exception as e:
        return {'found': False, 'reason': f'lookup failed: {e}'}
    finally:
        conn.close()


def get_next_race() -> dict:
    """Look up the date, location, and name of the next race on the calendar after today. This is schedule information only, never a prediction of who will win - the full season calendar is known in advance, but results are not."""
    conn = sqlite3.connect(DB_PATH)
    try:
        today = date.today().isoformat()
        # "next" means the earliest race that has not been run, not simply the earliest
        # date still ahead. A race_date >= today check keeps showing today's race as
        # upcoming for the rest of its own race day, hours after it has finished.
        # Having results is the real signal that a race is done.
        row = pd.read_sql("""
            SELECT dr.year, dr.round, dr.race_name, dr.race_date, dc.circuit_name, dc.country
            FROM dim_race dr JOIN dim_circuit dc ON dr.circuit_id = dc.circuit_id
            WHERE dr.race_date >= ?
              AND NOT EXISTS (SELECT 1 FROM fact_race_results fr WHERE fr.race_id = dr.race_id)
            ORDER BY dr.race_date ASC LIMIT 1
        """, conn, params=(today,))
        if len(row) == 0:
            return {'found': False, 'reason': 'no upcoming race found in the schedule data'}
        return {'found': True, 'confidence_tier': 'measured_fact', **_row(row.iloc[0])}
    except Exception as e:
        return {'found': False, 'reason': f'lookup failed: {e}'}
    finally:
        conn.close()


def predict_from_hypothetical_grid(race: str = None, year: int = None, use_sprint_grid: bool = True) -> dict:
    """Answer a what-if: given a hypothetical starting grid, what would the podium and win chances be? By default it uses the sprint qualifying order for the next unraced sprint weekend, which answers questions like "what if the sprint grid carried over to the race". This is explicitly a hypothetical, not a prediction of the actual race, because the real grid is set by the Grand Prix's own qualifying session."""
    conn = sqlite3.connect(DB_PATH)
    try:
        if race is None:
            target = pd.read_sql("""
                SELECT r.race_id, r.race_name, r.year, r.round FROM dim_race r
                WHERE EXISTS (SELECT 1 FROM fact_sprint_qualifying q WHERE q.race_id = r.race_id)
                  AND NOT EXISTS (SELECT 1 FROM fact_race_results fr WHERE fr.race_id = r.race_id)
                ORDER BY r.race_date LIMIT 1
            """, conn)
            if len(target) == 0:
                return {'found': False, 'reason': 'no upcoming race has a sprint qualifying order to borrow from'}
            t = target.iloc[0]
            race_id, race_name, race_year, rnd = int(t.race_id), t.race_name, int(t.year), int(t['round'])
        else:
            r = resolve_race(race, year)
            if r is None:
                return {'found': False, 'reason': f'no race matching "{race}" ({year}) in the dataset'}
            if isinstance(r, dict) and r.get('ambiguous'):
                return {'found': False, 'ambiguous': True, 'candidates': r['candidates']}
            race_id, race_name, race_year, rnd = r['race_id'], r['race_name'], r['year'], r['round']

        grid = pd.read_sql("""
            SELECT sq.driver_id, sq.team_id, sq.sprint_quali_position AS assumed_position,
                   d.driver_name, t.team_name
            FROM fact_sprint_qualifying sq
            JOIN dim_driver d ON sq.driver_id = d.driver_id
            LEFT JOIN dim_team t ON sq.team_id = t.team_id
            WHERE sq.race_id = ?
            ORDER BY sq.sprint_quali_position
        """, conn, params=(race_id,))
        if len(grid) == 0:
            return {'found': False, 'reason': f'no sprint qualifying order on record for {race_name} {race_year}'}

        models = load_models()
        out = []
        for _, g in grid.iterrows():
            X = prerace.build_prerace_row(conn, int(g.driver_id), int(g.team_id), race_id,
                                           race_year, rnd, int(g.assumed_position))
            # the Grand Prix qualifying session has not run, so its two derived features
            # genuinely do not exist yet. Measured cost of leaving them out: about 0.007 AUC.
            X['qualifying_pace_delta'] = float('nan')
            X['qualifying_gap_to_pole'] = float('nan')
            out.append({
                'driver': g.driver_name, 'team': g.team_name,
                'assumed_start': int(g.assumed_position),
                'podium_chance': round(float(models['podium'].predict_proba(X)[:, 1][0]), 3),
                'win_chance': round(float(models['win'].predict_proba(X)[:, 1][0]), 3),
            })
        out.sort(key=lambda e: e['podium_chance'], reverse=True)

        return {
            'found': True, 'race': race_name, 'year': race_year,
            'confidence_tier': 'hypothetical',
            'basis': 'sprint qualifying order, assumed to carry over to the Grand Prix',
            'caveat': 'this is a what-if, not a prediction of the actual race. The real grid comes from the Grand Prix qualifying session, which has not run. Those qualifying features are therefore missing here, costing roughly 0.007 AUC against a full prediction.',
            'hypothetical_predictions': out,
        }
    except Exception as e:
        return {'found': False, 'reason': f'lookup failed: {e}'}
    finally:
        conn.close()


def predict_upcoming_race(driver: str = None) -> dict:
    """Predict the podium and win chances for the next race that has already qualified but has not been run yet. This is a genuine forward-looking prediction, using the same trained models and the qualifying result that sets the grid. Pass a driver for just that driver, or leave it out for the full field ordered by podium chance. Only works once qualifying for that race has happened."""
    conn = sqlite3.connect(DB_PATH)
    try:
        upcoming = pd.read_sql("""
            SELECT r.race_id, r.race_name, r.year, r.round, r.race_date
            FROM dim_race r
            WHERE EXISTS (SELECT 1 FROM fact_qualifying_results q WHERE q.race_id = r.race_id)
              AND NOT EXISTS (SELECT 1 FROM fact_race_results fr WHERE fr.race_id = r.race_id)
            ORDER BY r.race_date
            LIMIT 1
        """, conn)
        if len(upcoming) == 0:
            return {'found': False, 'reason': 'no upcoming race has qualified yet, so there is nothing to predict from. Qualifying has to happen first.'}

        race = upcoming.iloc[0]
        grid = prerace.qualifying_grid(conn, int(race.race_id))
        if len(grid) == 0:
            return {'found': False, 'reason': f'no qualifying classification stored for {race.race_name} {race.year}'}

        if driver:
            d = resolve_driver(driver)
            if d is None:
                return {'found': False, 'reason': f'no driver matching "{driver}" in the dataset'}
            if isinstance(d, dict) and d.get('ambiguous'):
                return {'found': False, 'ambiguous': True, 'candidates': d['candidates']}
            grid = grid[grid.driver_id == d['driver_id']]
            if len(grid) == 0:
                return {'found': False, 'reason': f"{d['driver_name']} did not qualify for {race.race_name} {race.year}"}

        models = load_models()
        explainers = load_explainers()
        out = []
        for _, g in grid.iterrows():
            X = prerace.build_prerace_row(conn, int(g.driver_id), int(g.team_id),
                                           int(race.race_id), int(race.year), int(race['round']),
                                           int(g.quali_position))
            entry = {
                'driver': g.driver_name, 'team': g.team_name,
                'starts_from': int(g.quali_position),
                'podium_chance': round(float(models['podium'].predict_proba(X)[:, 1][0]), 3),
                'win_chance': round(float(models['win'].predict_proba(X)[:, 1][0]), 3),
                'predicted_points': round(float(models['points_scored'].predict(X)[0]), 1),
            }
            if driver:
                entry['top_reasons'] = get_shap_reasons(explainers['podium'], X)
            out.append(entry)

        out.sort(key=lambda e: e['podium_chance'], reverse=True)
        return {
            'found': True, 'race': race.race_name, 'year': int(race.year),
            'race_date': race.race_date, 'confidence_tier': 'confident_prediction',
            'caveat': 'grid is taken from qualifying. Grid penalties are applied after qualifying and are not published anywhere readable before the race, so a penalised driver may start further back than shown. Measured cost of this is about 0.003 AUC.',
            'predictions': out,
        }
    except Exception as e:
        return {'found': False, 'reason': f'lookup failed: {e}'}
    finally:
        conn.close()


def get_qualifying_results(race: str = None, year: int = None, driver: str = None) -> dict:
    """Look up Grand Prix qualifying results: the session that sets the starting grid for the race, with each driver's position, best lap and gap to pole. Give a race and year, or give nothing to get the most recent qualifying session on record, which is what to use for questions about today's or the latest qualifying. Add a driver to narrow it to one. This is the main qualifying session, which is a different session from sprint qualifying. This is a measured fact, not a prediction."""
    conn = sqlite3.connect(DB_PATH)
    try:
        if race is None:
            latest = pd.read_sql("""
                SELECT q.race_id, r.race_name, r.year FROM fact_qualifying_results q
                JOIN dim_race r ON q.race_id = r.race_id
                ORDER BY r.race_date DESC LIMIT 1
            """, conn)
            if len(latest) == 0:
                return {'found': False, 'reason': 'no qualifying results on record'}
            race_id = int(latest.race_id.iloc[0])
            race_name, race_year = latest.race_name.iloc[0], int(latest.year.iloc[0])
        else:
            r = resolve_race(race, year)
            if r is None:
                return {'found': False, 'reason': f'no race matching "{race}" ({year}) in the dataset'}
            if isinstance(r, dict) and r.get('ambiguous'):
                return {'found': False, 'ambiguous': True, 'candidates': r['candidates']}
            race_id, race_name, race_year = r['race_id'], r['race_name'], r['year']

        rows = pd.read_sql("""
            SELECT q.quali_position, d.driver_name, t.team_name, q.best_quali_lap
            FROM fact_qualifying_results q
            JOIN dim_driver d ON q.driver_id = d.driver_id
            LEFT JOIN dim_team t ON q.team_id = t.team_id
            WHERE q.race_id = ?
            ORDER BY q.quali_position
        """, conn, params=(race_id,))
        if len(rows) == 0:
            return {'found': False, 'reason': f"no qualifying results on record for {race_name} {race_year}, that session may not have run yet"}

        pole = rows.best_quali_lap.min()
        rows['gap_to_pole'] = (rows.best_quali_lap - pole).round(3)

        if driver:
            d = resolve_driver(driver)
            if d is None:
                return {'found': False, 'reason': f'no driver matching "{driver}" in the dataset'}
            if isinstance(d, dict) and d.get('ambiguous'):
                return {'found': False, 'ambiguous': True, 'candidates': d['candidates']}
            rows = rows[rows.driver_name == d['driver_name']]
            if len(rows) == 0:
                return {'found': False, 'reason': f"{d['driver_name']} has no qualifying record for {race_name} {race_year}"}

        return {
            'found': True, 'race': race_name, 'year': race_year,
            'session': 'Grand Prix qualifying, which sets the race starting grid',
            'confidence_tier': 'measured_fact',
            'qualifying_results': _records(rows),
        }
    except Exception as e:
        return {'found': False, 'reason': f'lookup failed: {e}'}
    finally:
        conn.close()


def get_sprint_qualifying(race: str = None, year: int = None, driver: str = None) -> dict:
    """Look up sprint qualifying results: the starting order for a sprint race, with each driver's best sprint qualifying lap and gap to sprint pole. Give a race and year, or give nothing at all to get the most recent sprint qualifying session on record, which is what to use for questions about today's or the upcoming sprint. Add a driver to narrow it to one. Sprint qualifying runs before the sprint, so this is available even when the sprint itself has not been run yet. This is a measured fact, not a prediction of the sprint outcome."""
    conn = sqlite3.connect(DB_PATH)
    try:
        if race is None:
            latest = pd.read_sql("""
                SELECT sq.race_id, r.race_name, r.year FROM fact_sprint_qualifying sq
                JOIN dim_race r ON sq.race_id = r.race_id
                ORDER BY r.race_date DESC LIMIT 1
            """, conn)
            if len(latest) == 0:
                return {'found': False, 'reason': 'no sprint qualifying on record'}
            race_id = int(latest.race_id.iloc[0])
            race_name, race_year = latest.race_name.iloc[0], int(latest.year.iloc[0])
        else:
            r = resolve_race(race, year)
            if r is None:
                return {'found': False, 'reason': f'no race matching "{race}" ({year}) in the dataset'}
            if isinstance(r, dict) and r.get('ambiguous'):
                return {'found': False, 'ambiguous': True, 'candidates': r['candidates']}
            race_id, race_name, race_year = r['race_id'], r['race_name'], r['year']

        rows = pd.read_sql("""
            SELECT sq.sprint_quali_position, d.driver_name, t.team_name, sq.best_sq_lap
            FROM fact_sprint_qualifying sq
            JOIN dim_driver d ON sq.driver_id = d.driver_id
            LEFT JOIN dim_team t ON sq.team_id = t.team_id
            WHERE sq.race_id = ?
            ORDER BY sq.sprint_quali_position
        """, conn, params=(race_id,))
        if len(rows) == 0:
            return {'found': False, 'reason': f"no sprint qualifying on record for {race_name} {race_year}, it may not be a sprint weekend"}

        pole_lap = rows.best_sq_lap.min()
        rows['gap_to_sprint_pole'] = (rows.best_sq_lap - pole_lap).round(3)

        if driver:
            d = resolve_driver(driver)
            if d is None:
                return {'found': False, 'reason': f'no driver matching "{driver}" in the dataset'}
            if isinstance(d, dict) and d.get('ambiguous'):
                return {'found': False, 'ambiguous': True, 'candidates': d['candidates']}
            rows = rows[rows.driver_name == d['driver_name']]
            if len(rows) == 0:
                return {'found': False, 'reason': f"{d['driver_name']} has no sprint qualifying record for {race_name} {race_year}"}

        return {
            'found': True, 'race': race_name, 'year': race_year,
            'confidence_tier': 'measured_fact',
            'note': 'this is the starting order for the sprint, not a prediction of how the sprint will finish',
            'sprint_qualifying': _records(rows),
        }
    except Exception as e:
        return {'found': False, 'reason': f'lookup failed: {e}'}
    finally:
        conn.close()


def get_sprint_result(race: str = None, year: int = None, driver: str = None) -> dict:
    """Look up sprint race results: the full finishing order for a sprint, or one driver's sprint result. Give a race and year for that weekend's sprint, add a driver to narrow it to one, or give nothing at all to get the most recent sprint that has been run. Use this to answer who won a sprint. This is a measured fact from historical results, not a model prediction."""
    conn = sqlite3.connect(DB_PATH)
    try:
        if race is None:
            latest = pd.read_sql("""
                SELECT sr.race_id, r.race_name, r.year FROM fact_sprint_results sr
                JOIN dim_race r ON sr.race_id = r.race_id
                ORDER BY r.race_date DESC LIMIT 1
            """, conn)
            if len(latest) == 0:
                return {'found': False, 'reason': 'no sprint results on record'}
            race_id = int(latest.race_id.iloc[0])
            race_name, race_year = latest.race_name.iloc[0], int(latest.year.iloc[0])
        else:
            r = resolve_race(race, year)
            if r is None:
                return {'found': False, 'reason': f'no race matching "{race}" ({year}) in the dataset'}
            if isinstance(r, dict) and r.get('ambiguous'):
                return {'found': False, 'ambiguous': True, 'candidates': r['candidates']}
            race_id, race_name, race_year = r['race_id'], r['race_name'], r['year']

        rows = pd.read_sql("""
            SELECT sr.sprint_finish_position, d.driver_name, t.team_name,
                   sr.sprint_grid_position, sr.sprint_points
            FROM fact_sprint_results sr
            JOIN dim_driver d ON sr.driver_id = d.driver_id
            LEFT JOIN dim_team t ON sr.team_id = t.team_id
            WHERE sr.race_id = ?
            ORDER BY sr.sprint_finish_position
        """, conn, params=(race_id,))
        if len(rows) == 0:
            return {'found': False, 'reason': f"{race_name} {race_year} was not a sprint weekend, or its sprint has not been run yet"}

        if driver:
            d = resolve_driver(driver)
            if d is None:
                return {'found': False, 'reason': f'no driver matching "{driver}" in the dataset'}
            if isinstance(d, dict) and d.get('ambiguous'):
                return {'found': False, 'ambiguous': True, 'candidates': d['candidates']}
            rows = rows[rows.driver_name == d['driver_name']]
            if len(rows) == 0:
                return {'found': False, 'reason': f"{d['driver_name']} has no sprint record for {race_name} {race_year}"}

        return {
            'found': True, 'race': race_name, 'year': race_year,
            'confidence_tier': 'measured_fact', 'sprint_results': _records(rows),
        }
    except Exception as e:
        return {'found': False, 'reason': f'lookup failed: {e}'}
    finally:
        conn.close()
