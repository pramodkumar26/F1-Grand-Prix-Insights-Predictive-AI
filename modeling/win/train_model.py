import sqlite3
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from xgboost import XGBClassifier

DB_PATH = 'f1.db'
FEATURES = ['grid_position', 'team_strength', 'driver_form', 'qualifying_pace_delta', 'qualifying_gap_to_pole']
TARGET = 'is_win'


def load_data(conn):
    return pd.read_sql('SELECT * FROM win_features', conn)


def chronological_split(df):
    train = df[df.year <= 2022].copy()
    val = df[df.year == 2023].copy()
    test = df[df.year >= 2024].copy()
    return train, val, test


def train_baseline(X_train, y_train, X_test):
    medians = X_train.median()
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train.fillna(medians), y_train)
    Xt = X_test.fillna(medians)
    return model, model.predict(Xt), model.predict_proba(Xt)[:, 1]


def train_primary(X_train, y_train, X_val, y_val, X_test):
    model = XGBClassifier(
        random_state=42, max_depth=3, n_estimators=500, learning_rate=0.03,
        subsample=0.8, colsample_bytree=1.0, reg_lambda=1.0,
        early_stopping_rounds=30, eval_metric='logloss',
        scale_pos_weight=(1 - y_train.mean()) / y_train.mean(),
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    return model, model.predict(X_test), model.predict_proba(X_test)[:, 1]


def report(name, y_true, y_pred, y_proba):
    print(f'  {name:32s} acc {accuracy_score(y_true, y_pred):.3f}  '
          f'prec {precision_score(y_true, y_pred, zero_division=0):.3f}  '
          f'rec {recall_score(y_true, y_pred, zero_division=0):.3f}  '
          f'F1 {f1_score(y_true, y_pred, zero_division=0):.3f}  '
          f'AUC {roc_auc_score(y_true, y_proba):.3f}')


def main():
    conn = sqlite3.connect(DB_PATH)
    df = load_data(conn)
    conn.close()

    train, val, test = chronological_split(df)
    X_train, y_train = train[FEATURES], train[TARGET]
    X_val, y_val = val[FEATURES], val[TARGET]
    X_test, y_test = test[FEATURES], test[TARGET]

    print(f'train rows: {len(train)}, val rows: {len(val)}, test rows: {len(test)}')
    print(f'win rate: train {y_train.mean():.4f}, val {y_val.mean():.4f}, test {y_test.mean():.4f}')

    baseline_model, baseline_pred, baseline_proba = train_baseline(X_train, y_train, X_test)
    primary_model, primary_pred, primary_proba = train_primary(X_train, y_train, X_val, y_val, X_test)

    report('baseline logistic regression', y_test, baseline_pred, baseline_proba)
    report('primary xgboost', y_test, primary_pred, primary_proba)

    print('\nxgboost feature importance:')
    for name, imp in sorted(zip(FEATURES, primary_model.feature_importances_), key=lambda kv: -kv[1]):
        print(f'  {name}: {imp:.4f}')


if __name__ == '__main__':
    main()
