import sqlite3
import mlflow
import mlflow.pyfunc
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from train_model import (DB_PATH, FEATURES, TARGET, BLEND_WEIGHT_LINEAR,
                          load_data, chronological_split, train_baseline, train_primary)

MLFLOW_TRACKING_URI = 'sqlite:///mlflow.db'
EXPERIMENT_NAME = 'f1-grand-prix-insights'
REGISTERED_MODEL_NAME = 'f1-overtaking'


def regression_metrics(y_true, pred):
    return {
        'mae': mean_absolute_error(y_true, pred),
        'rmse': mean_squared_error(y_true, pred) ** 0.5,
        'r2': r2_score(y_true, pred),
    }


class BlendModel(mlflow.pyfunc.PythonModel):
    # wraps the linear+xgboost blend as a single loggable, reloadable model
    def __init__(self, linear_model, xgb_model, fill_medians, weight_linear):
        self.linear_model = linear_model
        self.xgb_model = xgb_model
        self.fill_medians = fill_medians
        self.weight_linear = weight_linear

    def predict(self, context, model_input):
        filled = model_input.fillna(self.fill_medians)
        lin_pred = self.linear_model.predict(filled)
        xgb_pred = self.xgb_model.predict(model_input)
        return self.weight_linear * lin_pred + (1 - self.weight_linear) * xgb_pred


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

    linear_model, linear_pred = train_baseline(X_train, y_train, X_test)
    linear_metrics = regression_metrics(y_test, linear_pred)
    with mlflow.start_run(run_name='overtaking-component-linear'):
        mlflow.log_params({**common_params, 'model_type': 'LinearRegression'})
        mlflow.log_metrics(linear_metrics)
    print('linear component logged (not registered):')
    for k, v in linear_metrics.items():
        print(f'  {k}: {v:.4f}')

    xgb_model, xgb_pred = train_primary(X_train, y_train, X_val, y_val, X_test)
    xgb_metrics = regression_metrics(y_test, xgb_pred)
    with mlflow.start_run(run_name='overtaking-component-xgboost'):
        mlflow.log_params({
            **common_params, 'model_type': 'XGBRegressor',
            'max_depth': 3, 'learning_rate': 0.03, 'n_estimators': 500, 'subsample': 0.8,
        })
        mlflow.log_metrics(xgb_metrics)
    print('\nxgboost component logged (not registered):')
    for k, v in xgb_metrics.items():
        print(f'  {k}: {v:.4f}')

    blend_pred = BLEND_WEIGHT_LINEAR * linear_pred + (1 - BLEND_WEIGHT_LINEAR) * xgb_pred
    blend_metrics = regression_metrics(y_test, blend_pred)
    blend_model = BlendModel(linear_model, xgb_model, X_train.median(), BLEND_WEIGHT_LINEAR)
    with mlflow.start_run(run_name='overtaking-adopted-blend'):
        mlflow.log_params({
            **common_params, 'model_type': 'LinearRegression + XGBRegressor blend',
            'blend_weight_linear': BLEND_WEIGHT_LINEAR,
        })
        mlflow.log_metrics(blend_metrics)
        mlflow.pyfunc.log_model(
            name='model', python_model=blend_model,
            registered_model_name=REGISTERED_MODEL_NAME,
            input_example=X_train.head(3),
        )

    print(f'\nblend logged and registered as "{REGISTERED_MODEL_NAME}" (the adopted model):')
    for k, v in blend_metrics.items():
        print(f'  {k}: {v:.4f}')
    print(f'\nimprovement over the better individual component (r2): '
          f'{blend_metrics["r2"] - max(linear_metrics["r2"], xgb_metrics["r2"]):+.4f}')


if __name__ == '__main__':
    main()
