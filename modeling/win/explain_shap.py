import sqlite3
import numpy as np
import matplotlib.pyplot as plt
import shap

from train_model import DB_PATH, FEATURES, TARGET, load_data, chronological_split, train_primary

PLOTS_DIR = 'modeling/win/plots'


def main():
    conn = sqlite3.connect(DB_PATH)
    df = load_data(conn)
    conn.close()

    train, val, test = chronological_split(df)
    X_train, y_train = train[FEATURES], train[TARGET]
    X_val, y_val = val[FEATURES], val[TARGET]
    X_test, y_test = test[FEATURES], test[TARGET]

    model, _, proba = train_primary(X_train, y_train, X_val, y_val, X_test)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_test)

    mean_abs = shap_values.abs.mean(0).values
    print("mean |SHAP value| per feature (log-odds units), test set (2024+):")
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

    example_idx = int(np.argmax(proba))
    plt.figure()
    shap.plots.waterfall(shap_values[example_idx], show=False)
    plt.tight_layout()
    plt.savefig(f'{PLOTS_DIR}/shap_waterfall_example.png', dpi=180, bbox_inches='tight')
    plt.close()

    row = test.iloc[example_idx]
    print(f"\nwaterfall example: race_id={int(row.race_id)}, driver_id={int(row.driver_id)}, "
          f"actual is_win={int(row.is_win)}, predicted probability={proba[example_idx]:.3f}")
    print(f"saved plots to {PLOTS_DIR}/")


if __name__ == '__main__':
    main()
