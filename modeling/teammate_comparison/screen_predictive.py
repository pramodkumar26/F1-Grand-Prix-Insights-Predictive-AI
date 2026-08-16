"""Screens whether next-season teammate advantage is predictable, before building anything."""
import sqlite3
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score

DB_PATH = 'f1.db'


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
    card['prior_teammate'] = g['teammate_driver_id'].shift()
    card['career_races'] = g['quali_races'].cumsum() - card['quali_races']

    card = card[card.prior_year == card.year - 1].copy()
    card['teammate_changed'] = (card.teammate_driver_id != card.prior_teammate).astype(int)

    opp = card[['driver_id', 'year', 'prior_quali_pct', 'prior_gap']].rename(
        columns={'driver_id': 'teammate_driver_id',
                 'prior_quali_pct': 'teammate_prior_quali_pct',
                 'prior_gap': 'teammate_prior_gap'})
    card = card.merge(opp, on=['teammate_driver_id', 'year'], how='left')
    return card


def main():
    conn = sqlite3.connect(DB_PATH)
    card = build_panel(conn)
    conn.close()

    print(f"usable driver-seasons (consecutive, with prior): {len(card)}")
    print(f"  of which the teammate CHANGED: {card.teammate_changed.sum()} "
          f"({100*card.teammate_changed.mean():.0f}%)")
    print(f"  new teammate's own prior record known: {card.teammate_prior_quali_pct.notna().sum()}")

    train, test = card[card.year <= 2022], card[card.year >= 2024]
    print(f"\nchronological split: train {len(train)}, test {len(test)}")
    if len(test) < 20:
        print("  WARNING: test set is very small, treat all numbers below as indicative only")

    y_test = test.quali_win_pct

    print("\nBASELINES (what any model must beat)")
    flat = np.full(len(test), train.quali_win_pct.mean())
    print(f"  predict overall mean       MAE {mean_absolute_error(y_test, flat):.4f}  "
          f"R2 {r2_score(y_test, flat):+.3f}")
    carry = test.prior_quali_pct.fillna(train.quali_win_pct.mean())
    print(f"  carry last season forward  MAE {mean_absolute_error(y_test, carry):.4f}  "
          f"R2 {r2_score(y_test, carry):+.3f}   <-- the real bar")

    FEATS = ['prior_quali_pct', 'prior_gap', 'career_races',
             'teammate_changed', 'teammate_prior_quali_pct']
    med = train[FEATS].median()
    m = Ridge(alpha=1.0).fit(train[FEATS].fillna(med), train.quali_win_pct)
    pred = m.predict(test[FEATS].fillna(med))
    print("\nMODEL")
    print(f"  ridge on {len(FEATS)} features     MAE {mean_absolute_error(y_test, pred):.4f}  "
          f"R2 {r2_score(y_test, pred):+.3f}")
    print("\n  coefficients:")
    for n, c in sorted(zip(FEATS, m.coef_), key=lambda kv: -abs(kv[1])):
        print(f"    {n:28s} {c:+.4f}")

    sub = test[test.teammate_changed == 1]
    if len(sub) >= 5:
        idx = test.teammate_changed == 1
        print(f"\n  restricted to CHANGED-teammate rows only (n={len(sub)}):")
        print(f"    carry forward  MAE {mean_absolute_error(sub.quali_win_pct, carry[idx]):.4f}")
        print(f"    model          MAE {mean_absolute_error(sub.quali_win_pct, pred[idx.values]):.4f}")


if __name__ == '__main__':
    main()
