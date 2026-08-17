import sqlite3
import numpy as np
import matplotlib.pyplot as plt
import shap

from train_model import DB_PATH, FEATURES, TARGET, load_data, chronological_split, train_primary

PLOTS_DIR = 'modeling/points_scored/plots'


def main():
    conn = sqlite3.connect(DB_PATH)
    df = load_data(conn)
    conn.close()

    train, val, test = chronological_split(df)
    X_train, y_train = train[FEATURES], train[TARGET]
    X_val, y_val = val[FEATURES], val[TARGET]
    X_test, y_test = test[FEATURES], test[TARGET]

    model, proba = train_primary(X_train, y_train, X_val, y_val, X_test)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_test)

    mean_abs = shap_values.abs.mean(0).values
    print("mean |SHAP value| per feature (points), test set (2024+):")
    for name, value in sorted(zip(FEATURES, mean_abs), key=lambda kv: -kv[1]):
        print(f"  {name}: {value:.4f}")

    plt.figure()
    shap.plots.beeswarm(shap_values, show=False)
    plt.tight_layout()
    plt.savefig(f'{PLOTS_DIR}/shap_beeswarm.png', dpi=180, bbox_inches='tight')
    plt.close()

    plt.figure()
    shap.plots.bar(shap_values, show=False)
    plt.tight_layout()
    plt.savefig(f'{PLOTS_DIR}/shap_bar.png', dpi=180, bbox_inches='tight')
    plt.close()

    best_idx = int(np.argmax(proba))
    error = (proba - y_test.values)
    worst_idx = int(np.argmax(np.abs(error)))

    for idx, tag in [(best_idx, 'best'), (worst_idx, 'worst_miss')]:
        plt.figure()
        shap.plots.waterfall(shap_values[idx], show=False)
        plt.tight_layout()
        plt.savefig(f'{PLOTS_DIR}/shap_waterfall_{tag}.png', dpi=180, bbox_inches='tight')
        plt.close()
        row = test.iloc[idx]
        print(f"\n{tag} example: race_id={int(row.race_id)}, driver_id={int(row.driver_id)}, "
              f"actual points={row[TARGET]}, predicted points={proba[idx]:.2f}, "
              f"error={proba[idx]-row[TARGET]:+.2f}")

    print(f"\nsaved plots to {PLOTS_DIR}/")


if __name__ == '__main__':
    main()
