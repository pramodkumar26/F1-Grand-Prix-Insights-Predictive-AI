import sqlite3
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from xgboost import XGBClassifier

DB_PATH = 'f1.db'
FEATURES = ['grid_position', 'team_strength', 'driver_form', 'qualifying_pace_delta', 'qualifying_gap_to_pole']
TARGET = 'is_podium'


def load_data(conn):
    return pd.read_sql('SELECT * FROM podium_features', conn)


def chronological_split(df):
    train = df[df['year'] <= 2022].copy()
    val = df[df['year'] == 2023].copy()
    test = df[df['year'] >= 2024].copy()
    return train, val, test


def train_baseline(X_train, y_train, X_test):
    medians = X_train.median()
    X_train_filled = X_train.fillna(medians)
    X_test_filled = X_test.fillna(medians)
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_filled, y_train)
    return model, model.predict(X_test_filled), model.predict_proba(X_test_filled)[:, 1]


def train_primary(X_train, y_train, X_val, y_val, X_test):
    # same regularization approach as strategy_shift's xgboost, shallow trees plus early
    # stopping on a held out validation slice instead of a fixed tree count
    model = XGBClassifier(
        random_state=42,
        max_depth=3,
        n_estimators=500,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=1.0,
        reg_lambda=1.0,
        early_stopping_rounds=30,
        eval_metric='logloss',
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    return model, model.predict(X_test), model.predict_proba(X_test)[:, 1]


def report(name, y_true, y_pred, y_proba):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    auc = roc_auc_score(y_true, y_proba)
    print(f'{name}: accuracy {acc:.3f}, precision {prec:.3f}, recall {rec:.3f}, F1 {f1:.3f}, ROC-AUC {auc:.3f}')


def main():
    conn = sqlite3.connect(DB_PATH)
    df = load_data(conn)
    conn.close()

    train, val, test = chronological_split(df)
    X_train, y_train = train[FEATURES], train[TARGET]
    X_val, y_val = val[FEATURES], val[TARGET]
    X_test, y_test = test[FEATURES], test[TARGET]

    print(f'train rows: {len(train)}, val rows: {len(val)}, test rows: {len(test)}')
    print(f'podium rate: train {y_train.mean():.3f}, val {y_val.mean():.3f}, test {y_test.mean():.3f}')

    baseline_model, baseline_pred, baseline_proba = train_baseline(X_train, y_train, X_test)
    primary_model, primary_pred, primary_proba = train_primary(X_train, y_train, X_val, y_val, X_test)

    report('baseline logistic regression', y_test, baseline_pred, baseline_proba)
    report('primary xgboost', y_test, primary_pred, primary_proba)

    print('\nlogistic regression coefficients:')
    for name, coef in zip(FEATURES, baseline_model.coef_[0]):
        print(f'  {name}: {coef:.4f}')

    print('\nxgboost feature importance:')
    for name, imp in zip(FEATURES, primary_model.feature_importances_):
        print(f'  {name}: {imp:.4f}')


if __name__ == '__main__':
    main()
