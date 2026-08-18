import sqlite3
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor

DB_PATH = 'f1.db'
CATEGORICALS = ['circuit_id', 'compound_id', 'team_id']
TARGET = 'fuel_adjusted_degradation_rate'


def load_stints(conn):
    return pd.read_sql("""
        SELECT s.race_id, s.driver_id, s.compound_id,
               s.fuel_adjusted_degradation_rate, r.year, r.circuit_id, l.team_id
        FROM tyre_degradation_adjusted s
        JOIN dim_race r ON s.race_id = r.race_id
        JOIN (SELECT DISTINCT race_id, driver_id, team_id FROM fact_laps) l
          ON s.race_id = l.race_id AND s.driver_id = l.driver_id
        WHERE s.r_squared > 0.3 AND s.lap_count >= 6
    """, conn)


def split(df):
    return df[df.year <= 2022].copy(), df[df.year >= 2024].copy()


def evaluate(name, y_true, y_pred, baseline_pred):
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    base_mae = mean_absolute_error(y_true, baseline_pred)
    lift = 100 * (1 - mae / base_mae)
    print(f"  {name:34s} MAE {mae:.4f}  R2 {r2:+.3f}  vs trivial {lift:+.1f}%")
    return r2


def aggregate_eval(test, y_pred, level, train_mean):
    t = test.copy()
    t['pred'] = y_pred
    g = t.groupby(level).agg(actual=(TARGET, 'mean'), pred=('pred', 'mean'), n=(TARGET, 'size'))
    g = g[g.n >= 5]
    base = np.full(len(g), train_mean)
    print(f"\n  aggregated to {' x '.join(level)}, cells with >=5 test stints: {len(g)}")
    evaluate('additive (aggregated)', g.actual, g.pred, base)
    return g


def main():
    conn = sqlite3.connect(DB_PATH)
    df = load_stints(conn)
    conn.close()

    train, test = split(df)
    print(f"train (2018-2022): {len(train)} stints, test (2024-2025): {len(test)} stints\n")

    enc = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    X_train = enc.fit_transform(train[CATEGORICALS])
    X_test = enc.transform(test[CATEGORICALS])
    y_train, y_test = train[TARGET], test[TARGET]

    train_mean = y_train.mean()
    trivial = np.full(len(y_test), train_mean)

    print("PER-STINT evaluation (can we predict one individual stint?)")
    evaluate('trivial mean predictor', y_test, trivial, trivial)

    additive = Ridge(alpha=1.0).fit(X_train, y_train)
    add_pred = additive.predict(X_test)
    evaluate('additive fixed effects (Ridge)', y_test, add_pred, trivial)

    xgb = XGBRegressor(random_state=42, max_depth=3, n_estimators=300,
                       learning_rate=0.03, subsample=0.8, reg_lambda=1.0)
    xgb.fit(train[CATEGORICALS], y_train)
    evaluate('xgboost (interactions)', y_test, xgb.predict(test[CATEGORICALS]), trivial)

    print("\nAGGREGATE evaluation (can we predict TYPICAL degradation in a context?)")
    aggregate_eval(test, add_pred, ['circuit_id', 'compound_id'], train_mean)

    coefs = pd.Series(additive.coef_, index=enc.get_feature_names_out(CATEGORICALS))
    print("\n  strongest circuit effects (s/lap relative to average):")
    circ = coefs[coefs.index.str.startswith('circuit_id')].sort_values()
    print(f"    lowest:  {circ.head(3).round(4).to_dict()}")
    print(f"    highest: {circ.tail(3).round(4).to_dict()}")
    team = coefs[coefs.index.str.startswith('team_id')].sort_values()
    print(f"  team effect spread: {team.min():+.4f} to {team.max():+.4f} "
          f"(range {team.max()-team.min():.4f})")
    comp = coefs[coefs.index.str.startswith('compound_id')]
    print(f"  compound effect spread: {comp.min():+.4f} to {comp.max():+.4f} "
          f"(range {comp.max()-comp.min():.4f})")


if __name__ == '__main__':
    main()
