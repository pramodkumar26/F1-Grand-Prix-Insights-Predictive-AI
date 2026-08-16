import sqlite3
import matplotlib.pyplot as plt
import shap

from train_model import DB_PATH, FEATURES, TARGET, load_data, chronological_split, train_primary

PLOTS_DIR = 'modeling/strategy_shift/plots'


def main():
    conn = sqlite3.connect(DB_PATH)
    df = load_data(conn)
    conn.close()

    train, val, test = chronological_split(df)
    X_train, y_train = train[FEATURES], train[TARGET]
    X_val, y_val = val[FEATURES], val[TARGET]
    X_test, y_test = test[FEATURES], test[TARGET]

    model, _ = train_primary(X_train, y_train, X_val, y_val, X_test)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_test)

    mean_abs = shap_values.abs.mean(0).values
    ranked = sorted(zip(FEATURES, mean_abs), key=lambda pair: -pair[1])
    print("mean |SHAP value| per feature, test set (2024-2025):")
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

    example_idx = y_test.reset_index(drop=True).abs().idxmax()
    plt.figure()
    shap.plots.waterfall(shap_values[example_idx], show=False)
    plt.tight_layout()
    plt.savefig(f'{PLOTS_DIR}/shap_waterfall_example.png', dpi=180, bbox_inches='tight')
    plt.close()

    example_row = test.iloc[example_idx]
    predicted = model.predict(X_test.iloc[[example_idx]])[0]
    print(f"\nwaterfall example: race_id={int(example_row.race_id)}, driver_id={int(example_row.driver_id)}, "
          f"actual positions_gained={y_test.iloc[example_idx]:.0f}, predicted={predicted:.2f}")
    print(f"saved plots to {PLOTS_DIR}/shap_beeswarm.png, shap_bar.png, shap_waterfall_example.png")


if __name__ == '__main__':
    main()
