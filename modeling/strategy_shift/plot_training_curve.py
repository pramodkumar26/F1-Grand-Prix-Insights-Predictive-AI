import sqlite3
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from xgboost import XGBRegressor

from train_model import DB_PATH, FEATURES, TARGET, load_data, chronological_split

N_ESTIMATORS = 500
EARLY_STOPPING_ROUNDS = 30

COLOR_TRAIN = '#2a78d6'   # categorical slot 1, blue
COLOR_TEST = '#eb6834'    # categorical slot 2, orange
COLOR_VAL = '#1baf7a'     # categorical slot 3, aqua
INK_PRIMARY = '#0b0b0b'
INK_SECONDARY = '#52514e'
INK_MUTED = '#898781'
GRIDLINE = '#e1e0d9'


def fit_with_full_history(X_train, y_train, X_val, y_val, X_test, y_test):
    # validation set must stay last in eval_set for early stopping to use it
    model = XGBRegressor(
        random_state=42,
        max_depth=3,
        n_estimators=N_ESTIMATORS,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=1.0,
        reg_lambda=1.0,
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        eval_metric='mae',
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_test, y_test), (X_val, y_val)],
        verbose=False,
    )
    return model


def mae_curves(model):
    results = model.evals_result()
    return results['validation_0']['mae'], results['validation_1']['mae'], results['validation_2']['mae']


def r2_curves(model, X_train, y_train, X_val, y_val, X_test, y_test):
    n_rounds = len(model.evals_result()['validation_0']['mae'])
    train_r2, val_r2, test_r2 = [], [], []
    for i in range(1, n_rounds + 1):
        train_r2.append(r2_score(y_train, model.predict(X_train, iteration_range=(0, i))))
        val_r2.append(r2_score(y_val, model.predict(X_val, iteration_range=(0, i))))
        test_r2.append(r2_score(y_test, model.predict(X_test, iteration_range=(0, i))))
    return train_r2, val_r2, test_r2


def plot(train_mae, test_mae, val_mae, train_r2, val_r2, test_r2, best_iteration, out_path):
    rounds = list(range(1, len(train_mae) + 1))

    fig, (ax_mae, ax_r2) = plt.subplots(2, 1, figsize=(9, 8), sharex=True, facecolor='#fcfcfb')

    for ax in (ax_mae, ax_r2):
        ax.set_facecolor('#fcfcfb')
        ax.grid(True, color=GRIDLINE, linewidth=0.8)
        ax.spines[['top', 'right']].set_visible(False)
        ax.spines[['left', 'bottom']].set_color(INK_MUTED)
        ax.tick_params(colors=INK_SECONDARY, labelsize=9)
        ax.axvline(best_iteration, color=INK_MUTED, linestyle='--', linewidth=1)

    ax_mae.plot(rounds, train_mae, color=COLOR_TRAIN, linewidth=2, label='train')
    ax_mae.plot(rounds, test_mae, color=COLOR_TEST, linewidth=2, label='test')
    ax_mae.plot(rounds, val_mae, color=COLOR_VAL, linewidth=2, label='validation')
    ax_mae.set_ylabel('MAE (positions)', color=INK_SECONDARY, fontsize=10)
    ax_mae.set_title(
        'strategy_shift xgboost: error by boosting round\n'
        'xgboost trains in boosting rounds, not epochs, dashed line is the early-stop round used by the real model',
        color=INK_PRIMARY, fontsize=11, loc='left'
    )
    ax_mae.text(best_iteration, ax_mae.get_ylim()[1], f'  round {best_iteration}, early-stop pick',
                color=INK_MUTED, fontsize=8, va='top')
    ax_mae.legend(frameon=False, labelcolor=INK_SECONDARY, fontsize=9)

    ax_r2.plot(rounds, train_r2, color=COLOR_TRAIN, linewidth=2, label='train')
    ax_r2.plot(rounds, test_r2, color=COLOR_TEST, linewidth=2, label='test')
    ax_r2.plot(rounds, val_r2, color=COLOR_VAL, linewidth=2, label='validation')
    ax_r2.set_ylabel('R2', color=INK_SECONDARY, fontsize=10)
    ax_r2.set_xlabel('boosting round', color=INK_SECONDARY, fontsize=10)
    ax_r2.legend(frameon=False, labelcolor=INK_SECONDARY, fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=180, facecolor=fig.get_facecolor())
    plt.close(fig)


def main():
    conn = sqlite3.connect(DB_PATH)
    df = load_data(conn)
    conn.close()

    train, val, test = chronological_split(df)
    X_train, y_train = train[FEATURES], train[TARGET]
    X_val, y_val = val[FEATURES], val[TARGET]
    X_test, y_test = test[FEATURES], test[TARGET]

    model = fit_with_full_history(X_train, y_train, X_val, y_val, X_test, y_test)
    train_mae, test_mae, val_mae = mae_curves(model)
    train_r2, val_r2, test_r2 = r2_curves(model, X_train, y_train, X_val, y_val, X_test, y_test)

    best_iteration = model.best_iteration
    print(f'n_estimators ceiling: {N_ESTIMATORS}')
    print(f'early_stopping_rounds: {EARLY_STOPPING_ROUNDS}')
    print(f'boosting rounds actually trained: {len(train_mae)}')
    print(f'best iteration (used by the real model): {best_iteration}')
    print(f'MAE at best iteration: train {train_mae[best_iteration]:.3f}, val {val_mae[best_iteration]:.3f}, test {test_mae[best_iteration]:.3f}')
    print(f'R2 at best iteration: train {train_r2[best_iteration]:.3f}, val {val_r2[best_iteration]:.3f}, test {test_r2[best_iteration]:.3f}')

    out_path = 'modeling/strategy_shift/plots/training_curve.png'
    plot(train_mae, test_mae, val_mae, train_r2, val_r2, test_r2, best_iteration, out_path)
    print(f'saved to {out_path}')


if __name__ == '__main__':
    main()
