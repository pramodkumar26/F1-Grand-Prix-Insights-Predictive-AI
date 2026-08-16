"""Hybrid predictor for a driver's teammate qualifying record in a season.

Two branches, because one method does not fit both cases:

  has a prior season  -> carry that record forward. Screened in screen_predictive.py:
                         R2 +0.458, and no model beat it by a distinguishable margin.
  no prior season     -> carry-forward is UNDEFINED. 46 of 165 driver-seasons (28%) are
                         rookies, returnees or mid-career gaps. This is where a model is
                         the only option, so this is what the model is built for.

The rookie branch leans on one idea: with no record of your own, the strongest available
signal is how good the teammate you are being measured against is.
"""
import sqlite3
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error

DB_PATH = 'f1.db'
ROOKIE_FEATURES = ['teammate_prior_quali_pct', 'teammate_prior_gap', 'team_strength']


def build_panel(conn):
    card = pd.read_sql("SELECT * FROM teammate_scorecard", conn)
    pairs = pd.read_sql("""
        SELECT DISTINCT p.driver_id, p.teammate_driver_id, r.year
        FROM teammate_pairs p JOIN dim_race r ON p.race_id = r.race_id
        WHERE p.teammate_driver_id IS NOT NULL
    """, conn)
    main = (pairs.groupby(['driver_id', 'year'])['teammate_driver_id']
                 .agg(lambda s: s.value_counts().index[0]).reset_index())
    card = card.merge(main, on=['driver_id', 'year'], how='left')

    card = card.sort_values(['driver_id', 'year'])
    g = card.groupby('driver_id')
    card['prior_quali_pct'] = g['quali_win_pct'].shift()
    card['prior_gap'] = g['quali_gap_median'].shift()
    card['prior_year'] = g['year'].shift()
    card['has_prior'] = (card.prior_year == card.year - 1)

    # the teammate's own record the season before, the key signal for a rookie
    opp = card[['driver_id', 'year', 'prior_quali_pct', 'prior_gap']].rename(
        columns={'driver_id': 'teammate_driver_id',
                 'prior_quali_pct': 'teammate_prior_quali_pct',
                 'prior_gap': 'teammate_prior_gap'})
    card = card.merge(opp, on=['teammate_driver_id', 'year'], how='left')

    # car quality, averaged over that driver's races in the season
    ts = pd.read_sql("""
        SELECT r.year, t.team_id, AVG(t.team_strength) team_strength
        FROM team_strength t JOIN dim_race r ON t.race_id = r.race_id
        GROUP BY r.year, t.team_id
    """, conn)
    res = pd.read_sql("""
        SELECT DISTINCT rr.driver_id, d.year, rr.team_id
        FROM fact_race_results rr JOIN dim_race d ON rr.race_id = d.race_id
    """, conn).drop_duplicates(['driver_id', 'year'])
    card = card.merge(res, on=['driver_id', 'year'], how='left').merge(ts, on=['year', 'team_id'], how='left')
    return card


def fit_rookie_model(train):
    # trained on every season where the features exist, not only rookie seasons, so the
    # relationship between teammate strength and record is learned from all 8 seasons
    # rather than from the handful of rookie rows alone
    t = train.dropna(subset=['teammate_prior_quali_pct'])
    med = t[ROOKIE_FEATURES].median()
    model = Ridge(alpha=1.0).fit(t[ROOKIE_FEATURES].fillna(med), t.quali_win_pct)
    return model, med


def predict(card, model, med, fallback):
    out = np.where(
        card.has_prior,
        card.prior_quali_pct,
        model.predict(card[ROOKIE_FEATURES].fillna(med)),
    )
    return pd.Series(out, index=card.index).fillna(fallback).clip(0, 1)


def bootstrap_gap(y, a, b, n=5000, seed=42):
    rng = np.random.default_rng(seed)
    d = [mean_absolute_error(y[i], a[i]) - mean_absolute_error(y[i], b[i])
         for i in (rng.integers(0, len(y), len(y)) for _ in range(n))]
    return np.mean(d), np.percentile(d, 2.5), np.percentile(d, 97.5)


def main():
    conn = sqlite3.connect(DB_PATH)
    card = build_panel(conn)
    conn.close()

    train, test = card[card.year <= 2022], card[card.year >= 2023]
    model, med = fit_rookie_model(train)
    fallback = train.quali_win_pct.mean()

    print(f"train {len(train)} driver-seasons, test {len(test)}")
    print("\nrookie-branch coefficients (higher teammate strength -> lower own record):")
    for n_, c in zip(ROOKIE_FEATURES, model.coef_):
        print(f"  {n_:28s} {c:+.4f}")

    rook = test[~test.has_prior].copy()
    y = rook.quali_win_pct.values
    pred = model.predict(rook[ROOKIE_FEATURES].fillna(med)).clip(0, 1)
    naive = np.full(len(rook), fallback)

    print(f"\nROOKIE BRANCH, the case carry-forward cannot serve (n={len(rook)})")
    print(f"  assume average driver   MAE {mean_absolute_error(y, naive):.4f}")
    print(f"  model                   MAE {mean_absolute_error(y, pred):.4f}")
    m_, lo, hi = bootstrap_gap(y, naive, pred)
    print(f"  bootstrap improvement   {m_:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]")
    print(f"  -> {'DISTINGUISHABLE from noise' if lo > 0 else 'NOT distinguishable from noise'}")

    full = predict(test, model, med, fallback)
    print(f"\nFULL HYBRID, all {len(test)} test driver-seasons")
    print(f"  assume average driver   MAE {mean_absolute_error(test.quali_win_pct, np.full(len(test), fallback)):.4f}")
    print(f"  hybrid                  MAE {mean_absolute_error(test.quali_win_pct, full):.4f}")

    card['predicted_quali_win_pct'] = predict(card, model, med, fallback)
    conn = sqlite3.connect(DB_PATH)
    card[['driver_id', 'driver_code', 'year', 'quali_win_pct',
          'predicted_quali_win_pct', 'has_prior']].to_sql(
        'teammate_prediction', conn, if_exists='replace', index=False)
    conn.commit()
    conn.close()
    print("\nwrote teammate_prediction table")


if __name__ == '__main__':
    main()
