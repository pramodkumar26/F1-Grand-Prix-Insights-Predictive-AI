import sqlite3
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, brier_score_loss)
from xgboost import XGBClassifier

DB_PATH = 'f1.db'
FEATURES = ['grid_position', 'team_dnf_rate', 'circuit_dnf_trailing',
            'driver_dnf_trailing', 'team_strength', 'qualifying_pace_delta']
TARGET = 'dnf'


def load_data(conn):
    return pd.read_sql('SELECT * FROM dnf_features', conn)


def chronological_split(df):
    return (df[df.year <= 2022].copy(), df[df.year == 2023].copy(), df[df.year >= 2024].copy())


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
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    return model, model.predict(X_test), model.predict_proba(X_test)[:, 1]


def report(name, y_true, y_pred, y_proba):
    print(f'  {name:32s} acc {accuracy_score(y_true, y_pred):.3f}  '
          f'prec {precision_score(y_true, y_pred, zero_division=0):.3f}  '
          f'rec {recall_score(y_true, y_pred, zero_division=0):.3f}  '
          f'F1 {f1_score(y_true, y_pred, zero_division=0):.3f}  '
          f'AUC {roc_auc_score(y_true, y_proba):.3f}  '
          f'Brier {brier_score_loss(y_true, y_proba):.4f}')


def main():
    conn = sqlite3.connect(DB_PATH)
    df = load_data(conn)
    conn.close()

    train, val, test = chronological_split(df)
    X_train, y_train = train[FEATURES], train[TARGET]
    X_val, y_val = val[FEATURES], val[TARGET]
    X_test, y_test = test[FEATURES], test[TARGET]

    print(f'train {len(train)}, val {len(val)}, test {len(test)}')
    print(f'DNF rate: train {y_train.mean():.3f}, test {y_test.mean():.3f}\n')

    base_proba = np.full(len(y_test), y_train.mean())
    print(f'  {"always predict base rate":32s} '
          f'acc {accuracy_score(y_test, np.zeros(len(y_test))):.3f}  '
          f'AUC 0.500  Brier {brier_score_loss(y_test, base_proba):.4f}')

    bm, bp, bpr = train_baseline(X_train, y_train, X_test)
    pm, pp, ppr = train_primary(X_train, y_train, X_val, y_val, X_test)
    report('baseline logistic regression', y_test, bp, bpr)
    report('primary xgboost', y_test, pp, ppr)

    print('\nlogistic regression coefficients:')
    for n, c in sorted(zip(FEATURES, bm.coef_[0]), key=lambda kv: -abs(kv[1])):
        print(f'  {n}: {c:+.4f}')
    print('\nxgboost feature importance:')
    for n, i in sorted(zip(FEATURES, pm.feature_importances_), key=lambda kv: -kv[1]):
        print(f'  {n}: {i:.4f}')


if __name__ == '__main__':
    main()
