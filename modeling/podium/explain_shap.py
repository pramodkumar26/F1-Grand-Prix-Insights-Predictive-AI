import sqlite3
import numpy as np
import matplotlib.pyplot as plt
import shap

from train_model import DB_PATH, FEATURES, TARGET, load_data, chronological_split, train_primary

PLOTS_DIR = 'modeling/podium/plots'


def main():
    conn = sqlite3.connect(DB_PATH)
    df = load_data(conn)
    conn.close()

    train, val, test = chronological_split(df)
    X_train, y_train = train[FEATURES], train[TARGET]
    X_val, y_val = val[FEATURES], val[TARGET]
    X_test, y_test = test[FEATURES], test[TARGET]

    # same fitted model train_model.py reports accuracy/precision/recall/F1/ROC-AUC for
    model, _, proba = train_primary(X_train, y_train, X_val, y_val, X_test)

    # tree_path_dependent explains in log-odds (margin) space, the exact additive
    # decomposition xgboost's trees support natively. Converting to probability isn't
    # additive, so these SHAP values are log-odds contributions, not probability points
    # directly. predict_proba (sigmoid of the margin) is used separately below for the
    # worked example, so the printed number is the one that's actually interpretable.
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_test)

    mean_abs = shap_values.abs.mean(0).values
    ranked = sorted(zip(FEATURES, mean_abs), key=lambda pair: -pair[1])
    print("mean |SHAP value| per feature (log-odds units), test set (2024-2025):")
    for name, value in ranked:
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

    # worked example: the test row the model is most confident is a podium finish,
    # so the explanation walks through a case with a real, readable story
    example_idx = int(np.argmax(proba))
    plt.figure()
    shap.plots.waterfall(shap_values[example_idx], show=False)
    plt.tight_layout()
    plt.savefig(f'{PLOTS_DIR}/shap_waterfall_example.png', dpi=180, bbox_inches='tight')
    plt.close()

    example_row = test.iloc[example_idx]
    actual = int(y_test.iloc[example_idx])
    print(f"\nwaterfall example: race_id={int(example_row.race_id)}, driver_id={int(example_row.driver_id)}, "
          f"actual is_podium={actual}, predicted probability={proba[example_idx]:.3f}")
    print(f"saved plots to {PLOTS_DIR}/shap_beeswarm.png, shap_bar.png, shap_waterfall_example.png")


if __name__ == '__main__':
    main()
