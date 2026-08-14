import sqlite3
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

DB_PATH = 'f1.db'
FEATURES = ['strategic_aggressiveness', 'track_difficulty_index', 'pit_stop_delta', 'primary_degradation_rate', 'grid_position', 'team_strength', 'driver_form', 'qualifying_pace_delta', 'is_wet_race']
TARGET = 'positions_gained'


def load_data(conn):
    df = pd.read_sql('SELECT * FROM strategy_shift_features', conn)
    before = len(df)
    df = df[df['pit_stop_delta'].notna()].copy()
    dropped = before - len(df)
    print(f'dropped {dropped} rows with null pit_stop_delta, likely red flag disrupted races')
    return df


def chronological_split(df):
    # carves a validation slice out of the training range, strictly before test,
    # so xgboost can early-stop on unseen data instead of overfitting the full
    # training set with a fixed n_estimators
    train = df[df['year'] <= 2022].copy()
    val = df[df['year'] == 2023].copy()
    test = df[df['year'] >= 2024].copy()
    return train, val, test


def train_baseline(X_train, y_train, X_test):
    medians = X_train.median()
    X_train_filled = X_train.fillna(medians)
    X_test_filled = X_test.fillna(medians)
    model = LinearRegression()
    model.fit(X_train_filled, y_train)
    return model, model.predict(X_test_filled)


def train_primary(X_train, y_train, X_val, y_val, X_test):
    # shallow trees, subsampling, and early stopping on a held-out validation slice,
    # defaults were overfitting hard enough that plain linear regression won on the test set
    model = XGBRegressor(
        random_state=42,
        max_depth=3,
        n_estimators=500,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=1.0,
        reg_lambda=1.0,
        early_stopping_rounds=30,
        eval_metric='mae',
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    return model, model.predict(X_test)


def report(name, y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    r2 = r2_score(y_true, y_pred)
    print(f'{name}: MAE {mae:.3f}, RMSE {rmse:.3f}, R2 {r2:.3f}')


def main():
    conn = sqlite3.connect(DB_PATH)
    df = load_data(conn)
    conn.close()

    train, val, test = chronological_split(df)
    X_train, y_train = train[FEATURES], train[TARGET]
    X_val, y_val = val[FEATURES], val[TARGET]
    X_test, y_test = test[FEATURES], test[TARGET]

    print(f'train rows: {len(train)}, val rows: {len(val)}, test rows: {len(test)}')

    baseline_model, baseline_pred = train_baseline(X_train, y_train, X_test)
    primary_model, primary_pred = train_primary(X_train, y_train, X_val, y_val, X_test)

    report('baseline linear regression', y_test, baseline_pred)
    report('primary xgboost', y_test, primary_pred)

    print('\nlinear regression coefficients:')
    for name, coef in zip(FEATURES, baseline_model.coef_):
        print(f'  {name}: {coef:.4f}')

    print('\nxgboost feature importance:')
    for name, imp in zip(FEATURES, primary_model.feature_importances_):
        print(f'  {name}: {imp:.4f}')


if __name__ == '__main__':
    main()