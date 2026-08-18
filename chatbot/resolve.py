import re
import sqlite3
import difflib
import pandas as pd

DB_PATH = 'f1.db'


def _word_match(series, query):
    pattern = r'\b' + re.escape(query) + r'\b'
    return series.str.lower().str.contains(pattern, na=False, regex=True)


def resolve_driver(name_or_code, conn=None):
    close = conn is None
    conn = conn or sqlite3.connect(DB_PATH)
    drivers = pd.read_sql("SELECT driver_id, driver_code, driver_name FROM dim_driver", conn)
    if close:
        conn.close()

    query = name_or_code.strip().lower()

    code_match = drivers[drivers.driver_code.str.lower() == query]
    if len(code_match) == 1:
        return code_match.iloc[0].to_dict()

    exact_name = drivers[drivers.driver_name.str.lower() == query]
    if len(exact_name) == 1:
        return exact_name.iloc[0].to_dict()

    substr_match = drivers[_word_match(drivers.driver_name, query)]
    if len(substr_match) == 1:
        return substr_match.iloc[0].to_dict()
    if len(substr_match) > 1:
        return {'ambiguous': True, 'candidates': substr_match.driver_name.tolist()}

    close_names = difflib.get_close_matches(query, drivers.driver_name.str.lower(), n=1, cutoff=0.6)
    if close_names:
        row = drivers[drivers.driver_name.str.lower() == close_names[0]].iloc[0]
        return row.to_dict()

    return None


def resolve_race(race_text, year=None, conn=None):
    close = conn is None
    conn = conn or sqlite3.connect(DB_PATH)
    races = pd.read_sql("""
        SELECT dr.race_id, dr.year, dr.round, dr.race_name, dc.circuit_name, dc.country
        FROM dim_race dr JOIN dim_circuit dc ON dr.circuit_id = dc.circuit_id
    """, conn)
    if close:
        conn.close()

    if year is not None:
        races = races[races.year == int(year)]

    query = race_text.strip().lower()
    match = races[
        _word_match(races.race_name, query)
        | _word_match(races.circuit_name, query)
        | _word_match(races.country, query)
    ]
    if len(match) == 1:
        row = match.iloc[0].to_dict()
        row.pop('circuit_name', None)
        row.pop('country', None)
        return row
    if len(match) > 1:
        return {'ambiguous': True, 'candidates': match[['year', 'race_name']].to_dict('records')}

    return None


def resolve_team(name, conn=None):
    close = conn is None
    conn = conn or sqlite3.connect(DB_PATH)
    teams = pd.read_sql("SELECT team_id, team_name FROM dim_team", conn)
    if close:
        conn.close()

    query = name.strip().lower()

    exact = teams[teams.team_name.str.lower() == query]
    if len(exact) == 1:
        return exact.iloc[0].to_dict()

    substr_match = teams[_word_match(teams.team_name, query)]
    if len(substr_match) == 1:
        return substr_match.iloc[0].to_dict()
    if len(substr_match) > 1:
        return {'ambiguous': True, 'candidates': substr_match.team_name.tolist()}

    close_names = difflib.get_close_matches(query, teams.team_name.str.lower(), n=1, cutoff=0.6)
    if close_names:
        row = teams[teams.team_name.str.lower() == close_names[0]].iloc[0]
        return row.to_dict()

    return None
