import sqlite3
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

DB_PATH = 'f1.db'
FEATURES = ['grid_position', 'team_strength', 'driver_overtake_trailing',
            'qualifying_pace_delta', 'qualifying_gap_to_pole', 'circuit_overtake_trailing']
TARGET = 'overtakes_made'


def load_data(conn):
    return pd.read_sql('SELECT * FROM overtaking_features', conn)


def chronological_split(df):
    train = df[df.year <= 2022].copy()
    val = df[df.year == 2023].copy()
    test = df[df.year >= 2024].copy()
    return train, val, test


def train_baseline(X_train, y_train, X_test):
    medians = X_train.median()
    model = LinearRegression()
    model.fit(X_train.fillna(medians), y_train)
    return model, model.predict(X_test.fillna(medians))


def train_primary(X_train, y_train, X_val, y_val, X_test):
    model = XGBRegressor(
        random_state=42, max_depth=3, n_estimators=500, learning_rate=0.03,
        subsample=0.8, colsample_bytree=1.0, reg_lambda=1.0,
        early_stopping_rounds=30, eval_metric='mae',
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    return model, model.predict(X_test)


BLEND_WEIGHT_LINEAR = 0.6  # selected on the 2023 validation slice, see model_log.md


def report(name, y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    r2 = r2_score(y_true, y_pred)
    print(f'{name}: MAE {mae:.4f}, RMSE {rmse:.4f}, R2 {r2:.4f}')


def main():
    conn = sqlite3.connect(DB_PATH)
    df = load_data(conn)
    conn.close()

    train, val, test = chronological_split(df)
    print(f'train rows: {len(train)}, val rows: {len(val)}, test rows: {len(test)}')

    triv = [train[TARGET].mean()] * len(test)
    report('trivial mean predictor', test[TARGET], triv)

    core = ['grid_position', 'team_strength', 'driver_overtake_trailing']
    m, p = train_baseline(train[core], train[TARGET], test[core])
    report('linear, core 3 features', test[TARGET], p)
    m, p = train_primary(train[core], train[TARGET], val[core], val[TARGET], test[core])
    report('xgboost, core 3 features', test[TARGET], p)

    X_train, y_train = train[FEATURES], train[TARGET]
    X_val, y_val = val[FEATURES], val[TARGET]
    X_test, y_test = test[FEATURES], test[TARGET]

    baseline_model, baseline_pred = train_baseline(X_train, y_train, X_test)
    primary_model, primary_pred = train_primary(X_train, y_train, X_val, y_val, X_test)

    # blend weight was selected on the validation set in a separate run, not here, to
    # avoid ever touching the test set during model selection. Fixed at 0.6 (linear) /
    # 0.4 (xgboost) once chosen. See model_log.md for the selection run.
    blend_pred = BLEND_WEIGHT_LINEAR * baseline_pred + (1 - BLEND_WEIGHT_LINEAR) * primary_pred

    print()
    report('baseline linear regression, all 6 features', y_test, baseline_pred)
    report('xgboost, all 6 features', y_test, primary_pred)
    report(f'blend ({BLEND_WEIGHT_LINEAR:.1f} linear / {1-BLEND_WEIGHT_LINEAR:.1f} xgboost), adopted', y_test, blend_pred)

    print('\nlinear regression coefficients:')
    for name, coef in sorted(zip(FEATURES, baseline_model.coef_), key=lambda kv: -abs(kv[1])):
        print(f'  {name}: {coef:+.4f}')
    print('\nxgboost feature importance:')
    for name, imp in sorted(zip(FEATURES, primary_model.feature_importances_), key=lambda kv: -kv[1]):
        print(f'  {name}: {imp:.4f}')


if __name__ == '__main__':
    main()
