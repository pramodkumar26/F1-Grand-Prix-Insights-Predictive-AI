import sqlite3
import mlflow
import mlflow.xgboost
import mlflow.sklearn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from train_model import DB_PATH, FEATURES, TARGET, load_data, chronological_split, train_baseline, train_primary

MLFLOW_TRACKING_URI = 'sqlite:///mlflow.db'
EXPERIMENT_NAME = 'f1-grand-prix-insights'
REGISTERED_MODEL_NAME = 'f1-points-scored'


def regression_metrics(y_true, pred):
    return {
        'mae': mean_absolute_error(y_true, pred),
        'rmse': mean_squared_error(y_true, pred) ** 0.5,
        'r2': r2_score(y_true, pred),
    }


def main():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    conn = sqlite3.connect(DB_PATH)
    df = load_data(conn)
    conn.close()

    train, val, test = chronological_split(df)
    X_train, y_train = train[FEATURES], train[TARGET]
    X_val, y_val = val[FEATURES], val[TARGET]
    X_test, y_test = test[FEATURES], test[TARGET]

    common_params = {
        'features': ','.join(FEATURES), 'target': TARGET,
        'train_rows': len(train), 'val_rows': len(val), 'test_rows': len(test),
        'train_years': '2018-2022', 'val_year': '2023', 'test_years': '2024-2025',
    }

    baseline_model, baseline_pred = train_baseline(X_train, y_train, X_test)
    baseline_metrics = regression_metrics(y_test, baseline_pred)
    with mlflow.start_run(run_name='points-scored-baseline-linear'):
        mlflow.log_params({**common_params, 'model_type': 'LinearRegression'})
        mlflow.log_metrics(baseline_metrics)
        mlflow.sklearn.log_model(baseline_model, name='model')
    print('baseline logged (not registered):')
    for k, v in baseline_metrics.items():
        print(f'  {k}: {v:.4f}')

    model, pred = train_primary(X_train, y_train, X_val, y_val, X_test)
    metrics = regression_metrics(y_test, pred)
    with mlflow.start_run(run_name='points-scored-primary-xgboost'):
        mlflow.log_params({
            **common_params, 'model_type': 'XGBRegressor',
            'max_depth': 3, 'learning_rate': 0.03, 'n_estimators': 500, 'subsample': 0.8,
        })
        mlflow.log_metrics(metrics)
        mlflow.xgboost.log_model(model, name='model', registered_model_name=REGISTERED_MODEL_NAME)

    print(f'\nprimary logged and registered as "{REGISTERED_MODEL_NAME}":')
    for k, v in metrics.items():
        print(f'  {k}: {v:.4f}')
    print(f'\nimprovement over baseline (r2): {metrics["r2"] - baseline_metrics["r2"]:+.4f}')


if __name__ == '__main__':
    main()
