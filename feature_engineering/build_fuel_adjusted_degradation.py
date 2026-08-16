import sqlite3
import pandas as pd
from sklearn.linear_model import LinearRegression
from scipy.stats import linregress

DB_PATH = 'f1.db'
MIN_LAPS_PER_STINT = 4
MIN_LAPS_PER_RACE = 100
MIN_STINTS_PER_RACE = 20

# physically plausible band for fuel effect, seconds per lap, always negative since
# the car burns weight off and gets faster. Cars carry roughly 100kg at the start and
# burn 1.5-2kg per lap at roughly 0.035s per kg, so around -0.06s/lap is expected, and
# the measured dry race median lands at -0.062, which is the independent check that
# this method is measuring the real thing. Band chosen to keep 96.6% of dry races.
PLAUSIBLE_FUEL_EFFECT = (-0.20, -0.01)

# laps dropped from the front of an opening stint, to remove the standing start
# artifact. See the comment in compute_raw_stint_slopes for the measured justification.
STANDING_START_LAPS_DROPPED = 2


def load_clean_laps(conn):
    laps = pd.read_sql("SELECT * FROM clean_laps", conn)
    laps['lap_time'] = pd.to_numeric(laps['lap_time'], errors='coerce')
    return laps.dropna(subset=['lap_time', 'tyre_age', 'lap_number'])


def add_driver_centered_lap_time(laps):
    # centers each driver's laps on their own median for that race, so the pooled fit
    # below isn't picking up "Verstappen is faster than a Williams" as if it were a
    # fuel or tyre effect. Only within-driver variation is left.
    driver_median = laps.groupby(['race_id', 'driver_id'])['lap_time'].transform('median')
    laps['driver_centered_lap_time'] = laps['lap_time'] - driver_median
    return laps


def estimate_fuel_effect_per_race(laps):
    # The confound this whole script exists to fix: WITHIN a stint, tyre_age and
    # lap_number move in perfect lockstep (measured corr = 1.0000), so a per stint
    # slope is tyre wear plus fuel burn and the two can't be told apart.
    #
    # ACROSS stints they decouple (measured corr median 0.537), because a stint
    # starting on lap 30 sits at tyre_age 1 while lap_number is already 30. So the
    # fuel coefficient IS identifiable once laps are pooled across every stint and
    # driver in a race. Estimated per race, not globally, since fuel burn per lap
    # depends on lap length and circuit, Monaco and Monza are not comparable.
    rows = []
    for race_id, group in laps.groupby('race_id'):
        n_stints = group.groupby(['driver_id', 'stint_number']).ngroups
        if len(group) < MIN_LAPS_PER_RACE or n_stints < MIN_STINTS_PER_RACE:
            continue

        X = group[['tyre_age', 'lap_number']].to_numpy()
        y = group['driver_centered_lap_time'].to_numpy()
        fit = LinearRegression().fit(X, y)

        rows.append({
            'race_id': race_id,
            'tyre_effect_pooled': fit.coef_[0],
            'fuel_effect_per_lap': fit.coef_[1],
            'pooled_r_squared': fit.score(X, y),
            'pooled_lap_count': len(group),
            'pooled_stint_count': n_stints,
        })

    return pd.DataFrame(rows)


def resolve_unreliable_fuel_effects(fuel, conn):
    # Wet races can't give a trustworthy fuel estimate. Track evolution, a wet line
    # drying out or rain arriving mid race, is ALSO a function of lap_number, and it
    # swamps the fuel signal, exactly the same class of confound this script exists to
    # remove. Measured: wet races have 6.7x the spread of dry ones, and 8 of the 12
    # physically implausible estimates are wet against a 12.9% base rate.
    #
    # Substituting a circuit's typical dry estimate is physically justified, not just a
    # patch: fuel burn per lap is a property of the car and circuit, not the weather. A
    # wet race still burns fuel normally. What changes is track evolution, which this
    # term should never have been absorbing in the first place.
    wet = pd.read_sql("SELECT race_id, is_wet_race FROM race_wet_flag", conn)
    circuits = pd.read_sql("SELECT race_id, circuit_id FROM dim_race", conn)
    fuel = fuel.merge(wet, on='race_id', how='left').merge(circuits, on='race_id', how='left')

    low, high = PLAUSIBLE_FUEL_EFFECT
    in_band = fuel['fuel_effect_per_lap'].between(low, high)
    is_dry = fuel['is_wet_race'].fillna(1) == 0
    fuel['own_estimate_reliable'] = in_band & is_dry

    reliable = fuel[fuel['own_estimate_reliable']]
    circuit_median = reliable.groupby('circuit_id')['fuel_effect_per_lap'].median()
    global_median = reliable['fuel_effect_per_lap'].median()

    fallback = fuel['circuit_id'].map(circuit_median).fillna(global_median)
    fuel['fuel_effect_used'] = fuel['fuel_effect_per_lap'].where(fuel['own_estimate_reliable'], fallback)
    fuel['fuel_effect_source'] = 'own'
    substituted = ~fuel['own_estimate_reliable']
    has_circuit = fuel['circuit_id'].map(circuit_median).notna()
    fuel.loc[substituted & has_circuit, 'fuel_effect_source'] = 'circuit_median'
    fuel.loc[substituted & ~has_circuit, 'fuel_effect_source'] = 'global_median'

    return fuel


def compute_raw_stint_slopes(laps):
    # same per stint fit the original build_tyre_degradation.py does, kept identical
    # so the adjustment below is the only thing that changes
    race_medians = laps.groupby('race_id')['lap_time'].transform('median')
    laps['normalized_lap_time'] = laps['lap_time'] - race_medians

    rows = []
    for keys, group in laps.groupby(['race_id', 'driver_id', 'stint_number', 'compound_id']):
        # Opening stints are contaminated by the standing start. Lap 1 includes the
        # launch and first-lap traffic and is abnormally slow, so the field's natural
        # speed-up over the next laps reads as strongly NEGATIVE degradation, tyres
        # apparently getting faster as they age, which is physically backwards.
        # Measured: opening stints came out at a median of -0.324 s/lap against +0.014
        # for later stints. Dropping the first two laps of an opening stint closes that
        # gap to 0.027 and flips the sign to +0.041, in line with later stints. It also
        # INCREASES the number of stints passing the r_squared filter (423 to 639),
        # since the contaminating laps were degrading the fit quality too.
        if group['lap_number'].min() == 1:
            group = group[group['lap_number'] > STANDING_START_LAPS_DROPPED]

        if len(group) < MIN_LAPS_PER_STINT:
            continue
        fit = linregress(group['tyre_age'], group['normalized_lap_time'])
        rows.append({
            'race_id': keys[0],
            'driver_id': keys[1],
            'stint_number': keys[2],
            'compound_id': keys[3],
            'raw_degradation_rate': fit.slope,
            'r_squared': fit.rvalue ** 2,
            'lap_count': len(group),
            'stint_start_lap': int(group['lap_number'].min()),
        })

    return pd.DataFrame(rows)


def apply_fuel_adjustment(stints, fuel):
    # within a stint tyre_age advances exactly 1 per lap_number, so the raw slope is
    # (tyre effect + fuel effect) added together. Subtracting the race's fuel
    # coefficient leaves the tyre-only rate. Fuel effect is negative (car burns off
    # weight and gets faster), so the adjusted rate should sit ABOVE the raw one.
    merged = stints.merge(fuel[['race_id', 'fuel_effect_used', 'fuel_effect_source']], on='race_id', how='left')
    merged['fuel_adjusted_degradation_rate'] = merged['raw_degradation_rate'] - merged['fuel_effect_used']
    return merged


def build_fuel_adjusted_degradation_table():
    conn = sqlite3.connect(DB_PATH)
    laps = load_clean_laps(conn)
    laps = add_driver_centered_lap_time(laps)

    fuel = estimate_fuel_effect_per_race(laps)
    fuel = resolve_unreliable_fuel_effects(fuel, conn)
    stints = compute_raw_stint_slopes(laps)
    adjusted = apply_fuel_adjustment(stints, fuel)

    fuel.to_sql('race_fuel_effect', conn, if_exists='replace', index=False)
    adjusted.to_sql('tyre_degradation_adjusted', conn, if_exists='replace', index=False)
    conn.commit()
    conn.close()

    return adjusted, fuel


if __name__ == '__main__':
    adjusted, fuel = build_fuel_adjusted_degradation_table()

    print(f"race_fuel_effect table built, {len(fuel)} races")
    print("\nfuel effect source (own estimate vs substituted):")
    print(fuel['fuel_effect_source'].value_counts().to_string())
    print("\nfuel effect actually used, per lap (expected NEGATIVE, car gets lighter and faster):")
    print(fuel['fuel_effect_used'].describe())
    print(f"races with a negative (physically sensible) fuel effect used: "
          f"{(fuel['fuel_effect_used'] < 0).sum()} / {len(fuel)}")

    print(f"\ntyre_degradation_adjusted table built, {len(adjusted)} stints")
    print("\nraw vs fuel adjusted degradation rate (s/lap):")
    print(adjusted[['raw_degradation_rate', 'fuel_adjusted_degradation_rate']].describe())
    print(f"\nmean raw:      {adjusted['raw_degradation_rate'].mean():.4f}")
    print(f"mean adjusted: {adjusted['fuel_adjusted_degradation_rate'].mean():.4f}")
    print(f"stints where adjusted > raw (expected, since fuel effect is negative): "
          f"{(adjusted['fuel_adjusted_degradation_rate'] > adjusted['raw_degradation_rate']).sum()} / {len(adjusted)}")
