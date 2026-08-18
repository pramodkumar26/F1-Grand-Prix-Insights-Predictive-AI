import sqlite3
from datetime import date
import pandas as pd

from chatbot.resolve import resolve_driver, resolve_race, resolve_team
from chatbot.registry import load_models, load_explainers

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
    row['team_strength'] = _scalar(conn,
        "SELECT DISTINCT team_strength FROM team_strength WHERE race_id=? AND team_id=?", (race_id, team_id))
    row['driver_form'] = _scalar(conn,
        "SELECT driver_form FROM driver_form WHERE race_id=? AND driver_id=?", (race_id, driver_id))
    row['qualifying_pace_delta'] = _scalar(conn,
        "SELECT qualifying_pace_delta FROM qualifying_pace_delta WHERE race_id=? AND driver_id=?", (race_id, driver_id))
    row['qualifying_gap_to_pole'] = _scalar(conn,
        "SELECT qualifying_gap_to_pole FROM qualifying_gap_to_pole WHERE race_id=? AND driver_id=?", (race_id, driver_id))
    return pd.DataFrame([row])[SHARED_FEATURES]


def build_strategy_shift_row(driver_id, race_id, team_id, conn):
    row = {}
    row['strategic_aggressiveness'] = _scalar(conn,
        "SELECT strategic_aggressiveness FROM strategic_aggressiveness WHERE race_id=? AND driver_id=?", (race_id, driver_id))
    row['track_difficulty_index'] = _scalar(conn,
        "SELECT track_difficulty_index FROM track_difficulty_index WHERE race_id=?", (race_id,))
    row['pit_stop_delta'] = _scalar(conn,
        "SELECT pit_stop_delta FROM pit_stop_delta WHERE race_id=? AND driver_id=?", (race_id, driver_id))
    row['primary_degradation_rate'] = _scalar(conn,
        """SELECT fuel_adjusted_degradation_rate FROM tyre_degradation_adjusted
           WHERE race_id=? AND driver_id=? ORDER BY lap_count DESC LIMIT 1""", (race_id, driver_id))
    row['grid_position'] = _scalar(conn,
        "SELECT grid_position FROM fact_race_results WHERE race_id=? AND driver_id=?", (race_id, driver_id))
    row['team_strength'] = _scalar(conn,
        "SELECT DISTINCT team_strength FROM team_strength WHERE race_id=? AND team_id=?", (race_id, team_id))
    row['driver_form'] = _scalar(conn,
        "SELECT driver_form FROM driver_form WHERE race_id=? AND driver_id=?", (race_id, driver_id))
    row['qualifying_pace_delta'] = _scalar(conn,
        "SELECT qualifying_pace_delta FROM qualifying_pace_delta WHERE race_id=? AND driver_id=?", (race_id, driver_id))
    row['is_wet_race'] = _scalar(conn,
        "SELECT is_wet_race FROM race_wet_flag WHERE race_id=?", (race_id,))
    row['pace_consistency'] = _scalar(conn,
        "SELECT pace_consistency FROM pace_consistency WHERE race_id=? AND driver_id=?", (race_id, driver_id))
    row['pit_stop_count'] = _scalar(conn,
        "SELECT MAX(stint_number) FROM fact_laps WHERE race_id=? AND driver_id=?", (race_id, driver_id))
    return pd.DataFrame([row])[STRATEGY_SHIFT_FEATURES]


def build_overtaking_row(driver_id, race_id, team_id, conn):
    row = {}
    row['grid_position'] = _scalar(conn,
        "SELECT grid_position FROM fact_race_results WHERE race_id=? AND driver_id=?", (race_id, driver_id))
    row['team_strength'] = _scalar(conn,
        "SELECT DISTINCT team_strength FROM team_strength WHERE race_id=? AND team_id=?", (race_id, team_id))
    row['driver_overtake_trailing'] = _scalar(conn,
        "SELECT driver_overtake_trailing FROM driver_overtake_trailing WHERE race_id=? AND driver_id=?", (race_id, driver_id))
    row['qualifying_pace_delta'] = _scalar(conn,
        "SELECT qualifying_pace_delta FROM qualifying_pace_delta WHERE race_id=? AND driver_id=?", (race_id, driver_id))
    row['qualifying_gap_to_pole'] = _scalar(conn,
        "SELECT qualifying_gap_to_pole FROM qualifying_gap_to_pole WHERE race_id=? AND driver_id=?", (race_id, driver_id))
    row['circuit_overtake_trailing'] = _scalar(conn,
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
        if result is None:
            return {'found': False, 'reason': f"{d['driver_name']} has no result on record for {r['race_name']} {r['year']}"}
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
        if result is None:
            return {'found': False, 'reason': f"{d['driver_name']} has no result on record for {r['race_name']} {r['year']}"}
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
        if result is None:
            return {'found': False, 'reason': f"{d['driver_name']} has no result on record for {r['race_name']} {r['year']}"}
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
        if result is None:
            return {'found': False, 'reason': f"{d['driver_name']} has no result on record for {r['race_name']} {r['year']}"}
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


def get_race_summary(race: str, year: int) -> dict:
    """Look up the recorded facts about a single race: how many laps were completed, who won, who started on pole, how many drivers finished or retired, and whether it was a wet race. This is a measured fact from recorded data, not a prediction."""
    conn = sqlite3.connect(DB_PATH)
    try:
        r = resolve_race(race, year)
        if r is None:
            return {'found': False, 'reason': f'no race matching "{race}" ({year}) in the dataset'}
        if isinstance(r, dict) and r.get('ambiguous'):
            return {'found': False, 'ambiguous': True, 'candidates': r['candidates']}
        race_id = r['race_id']

        laps_completed = _scalar(conn, "SELECT MAX(lap_number) FROM fact_laps WHERE race_id=?", (race_id,))
        if laps_completed is None:
            return {'found': False, 'reason': f"no lap data recorded for {r['race_name']} {r['year']}"}

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
            'laps_completed': int(laps_completed),
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


def get_driver_season_results(driver: str, year: int) -> dict:
    """Look up a driver's race-by-race results for a whole season - every race they entered, with grid position, finishing position, points scored and status (finished, retired, disqualified). This is a measured fact from historical results, not a prediction."""
    conn = sqlite3.connect(DB_PATH)
    try:
        d = resolve_driver(driver)
        if d is None:
            return {'found': False, 'reason': f'no driver matching "{driver}" in the dataset'}
        if isinstance(d, dict) and d.get('ambiguous'):
            return {'found': False, 'ambiguous': True, 'candidates': d['candidates']}
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
        races_so_far = _scalar(conn,
            "SELECT COUNT(DISTINCT fr.race_id) FROM fact_race_results fr JOIN dim_race r ON fr.race_id = r.race_id WHERE r.year=?",
            (int(year),))
        if len(standings) == 0:
            return {'found': False, 'reason': f'no race results found for {year}'}
        return {
            'found': True, 'year': int(year), 'races_completed': int(races_so_far),
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
        races_so_far = _scalar(conn,
            "SELECT COUNT(DISTINCT fr.race_id) FROM fact_race_results fr JOIN dim_race r ON fr.race_id = r.race_id WHERE r.year=?",
            (int(year),))
        if len(standings) == 0:
            return {'found': False, 'reason': f'no race results found for {year}'}
        return {
            'found': True, 'year': int(year), 'races_completed': int(races_so_far),
            'confidence_tier': 'measured_fact', 'standings': _records(standings),
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
        row = pd.read_sql("""
            SELECT dr.year, dr.round, dr.race_name, dr.race_date, dc.circuit_name, dc.country
            FROM dim_race dr JOIN dim_circuit dc ON dr.circuit_id = dc.circuit_id
            WHERE dr.race_date >= ?
            ORDER BY dr.race_date ASC LIMIT 1
        """, conn, params=(today,))
        if len(row) == 0:
            return {'found': False, 'reason': 'no upcoming race found in the schedule data'}
        return {'found': True, 'confidence_tier': 'measured_fact', **_row(row.iloc[0])}
    except Exception as e:
        return {'found': False, 'reason': f'lookup failed: {e}'}
    finally:
        conn.close()


def get_sprint_result(driver: str, race: str, year: int) -> dict:
    """Look up a driver's sprint race result for a given race weekend, if that weekend had a sprint. This is a measured fact from historical results, not a model prediction."""
    conn = sqlite3.connect(DB_PATH)
    try:
        d, r, error = _resolve_driver_and_race(driver, race, year)
        if error:
            return error
        row = pd.read_sql(
            "SELECT sprint_grid_position, sprint_finish_position, sprint_points FROM fact_sprint_results WHERE race_id=? AND driver_id=?",
            conn, params=(r['race_id'], d['driver_id'])
        )
        if len(row) == 0:
            return {'found': False, 'reason': f"{r['race_name']} {r['year']} was not a sprint weekend, or {d['driver_name']} has no sprint record for it"}
        return {
            'found': True, 'driver': d['driver_name'], 'race': r['race_name'], 'year': r['year'],
            'confidence_tier': 'measured_fact', **_row(row.iloc[0]),
        }
    except Exception as e:
        return {'found': False, 'reason': f'lookup failed: {e}'}
    finally:
        conn.close()
