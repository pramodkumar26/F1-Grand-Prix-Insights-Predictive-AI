import sqlite3
import warnings

import fastf1
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')
fastf1.Cache.enable_cache('cache')

DB_PATH = 'f1.db'
TARGET_POINTS = 140
VIEW = 100


def outline_to_path(x, y):
    """Normalise raw telemetry coordinates into an SVG path that fits a square viewBox."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    step = max(1, len(x) // TARGET_POINTS)
    x, y = x[::step], y[::step]

    width = x.max() - x.min()
    height = y.max() - y.min()
    scale = (VIEW * 0.88) / max(width, height)

    px = (x - x.min()) * scale
    py = (y - y.min()) * scale
    # SVG's y axis points down, telemetry's points up
    py = py.max() - py

    px += (VIEW - px.max()) / 2
    py += (VIEW - py.max()) / 2

    points = [f'{a:.1f},{b:.1f}' for a, b in zip(px, py)]
    return 'M' + ' L'.join(points) + ' Z'


def pick_session(conn, circuit_id):
    """Most recent completed race at this circuit, most likely to have cached telemetry."""
    races = pd.read_sql("""
        SELECT r.year, r.race_name FROM dim_race r
        WHERE r.circuit_id = ? AND r.race_date <= date('now')
        ORDER BY r.year DESC
    """, conn, params=(circuit_id,))
    return races


def main():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS circuit_outline (
            circuit_id INTEGER PRIMARY KEY,
            circuit_name TEXT,
            svg_path TEXT,
            source_year INTEGER,
            FOREIGN KEY (circuit_id) REFERENCES dim_circuit(circuit_id)
        )
    """)
    conn.commit()

    circuits = pd.read_sql('SELECT circuit_id, circuit_name FROM dim_circuit ORDER BY circuit_name', conn)
    done = set(pd.read_sql('SELECT circuit_id FROM circuit_outline', conn).circuit_id)

    for _, c in circuits.iterrows():
        if c.circuit_id in done:
            print(f'{c.circuit_name}: already stored, skipping')
            continue

        saved = False
        for _, race in pick_session(conn, c.circuit_id).iterrows():
            try:
                session = fastf1.get_session(int(race.year), race.race_name, 'R')
                session.load(telemetry=True, weather=False, messages=False)
                tel = session.laps.pick_fastest().get_telemetry()
                if len(tel) < 50:
                    continue
                path = outline_to_path(tel.X.values, tel.Y.values)
                cursor.execute(
                    'INSERT OR REPLACE INTO circuit_outline (circuit_id, circuit_name, svg_path, source_year) VALUES (?, ?, ?, ?)',
                    (int(c.circuit_id), c.circuit_name, path, int(race.year))
                )
                conn.commit()
                print(f'{c.circuit_name}: saved from {race.year}, {len(tel)} points')
                saved = True
                break
            except Exception as e:
                print(f'{c.circuit_name} {race.year} failed: {str(e)[:70]}')
                continue

        if not saved:
            print(f'{c.circuit_name}: NO OUTLINE AVAILABLE')

    total = pd.read_sql('SELECT COUNT(*) n FROM circuit_outline', conn).n.iloc[0]
    print(f'\ncircuit_outline table: {total} of {len(circuits)} circuits')
    conn.close()


if __name__ == '__main__':
    main()
