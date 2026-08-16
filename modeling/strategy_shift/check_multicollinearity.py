import sqlite3
import pandas as pd
from sklearn.linear_model import LinearRegression

DB_PATH = 'f1.db'
FEATURES = ['strategic_aggressiveness', 'track_difficulty_index', 'pit_stop_delta', 'primary_degradation_rate', 'grid_position', 'team_strength', 'driver_form', 'qualifying_pace_delta', 'is_wet_race']


def compute_vif(X):
    # VIF > 5 flags concerning collinearity, > 10 is severe
    vifs = {}
    for col in X.columns:
        y = X[col]
        others = X.drop(columns=[col])
        model = LinearRegression().fit(others, y)
        r2 = model.score(others, y)
        vifs[col] = float('inf') if r2 >= 1.0 else 1 / (1 - r2)
    return vifs


def main():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql('SELECT * FROM strategy_shift_features', conn)
    conn.close()

    X = df[FEATURES]

    corr = X.corr()
    print("correlation matrix:")
    print(corr.round(3))

    print("\nstrongest pairwise correlations (|r| >= 0.3), excluding self-pairs:")
    seen = set()
    rows = []
    for a in corr.index:
        for b in corr.columns:
            if a == b or (b, a) in seen:
                continue
            seen.add((a, b))
            rows.append((a, b, corr.loc[a, b]))
    rows.sort(key=lambda r: -abs(r[2]))
    for a, b, r in rows:
        if abs(r) >= 0.3:
            print(f"  {a} <-> {b}: r = {r:.3f}")

    X_filled = X.fillna(X.median())
    vifs = compute_vif(X_filled)
    print("\nVIF per feature (median filled, >5 flagged, >10 severe):")
    for name, vif in sorted(vifs.items(), key=lambda kv: -kv[1]):
        flag = "  <-- flagged" if vif > 5 else ""
        print(f"  {name}: {vif:.2f}{flag}")


if __name__ == '__main__':
    main()
