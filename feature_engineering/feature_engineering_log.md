## Race level features

### build_lap_flags.py

**Why:** fact_laps had no information about race conditions at all. No way to tell if a lap was run under safety car, red flag, or yellow flag, and any pace based calculation, degradation rate especially, would be corrupted by including those laps.

**What it does:** pulls FastF1's per lap TrackStatus field for every race in dim_race and writes it into a new table, lap_flags, keyed by race, driver, and lap number. TrackStatus is FastF1's own computed field, so this reads it directly instead of manually parsing race control message timestamps.

**Why it matters:** this is the foundation everything else in phase 2 sits on. Without it, degradation rate, pace consistency, and track difficulty index would all be computed off contaminated laps.

### build_clean_laps.py

**Why:** needed one trusted definition of "a real racing lap," rather than repeating pit lap and flag filtering logic inside every single feature script.

**What it does:** a SQL view joining fact_laps and lap_flags, dropping pit in laps, pit out laps, flagged laps, and null lap times.

**Why it matters:** every race level feature reads from this view instead of fact_laps directly, so the filtering logic lives in exactly one place and stays consistent across the whole project.

### fix_lap_time_columns.py

**Why:** while building degradation rate, lap_time, pit_in_time, and pit_out_time turned out to be stored as raw text, timedelta strings like "0 days 00:01:19.106000", instead of actual numbers, despite the columns being declared REAL. This crashed any calculation that touched them.

**What it does:** converts all three columns to real numeric seconds directly inside fact_laps, in place, through a staged UPDATE that leaves the table's schema, primary key, and foreign keys untouched.

**Why it matters:** fixes the bug once at the source instead of patching around it in five different feature scripts. This is a genuine, permanent database fix, same category as the four bugs already caught during EDA, byte blob race_id, empty fastest_lap_time, missing nationalities, split team ids.

### build_tyre_degradation.py

**Why:** the actual reason lap_flags and clean_laps exist. Needed a number that captures how fast a driver's pace fell off as their tyres aged.

**What it does:** for every race, driver, stint, and compound, fits a line of normalized lap time (lap time minus that race's median) against tyre age. The slope is the degradation rate, in seconds per lap. Also stores r_squared and lap count so downstream work can judge how reliable each slope is. Minimum four laps required per stint, shorter stints are skipped rather than recorded as noise.

**Why it matters:** core signal for the standalone tyre degradation model, and a feature input for win and podium prediction.

**Known limitation, logged as a follow up task:** this currently mixes tyre wear with fuel burn effect, since within a single stint tyre age and lap number move in lockstep and can't be separated without fuel or telemetry data. Still useful as a combined pace trend signal for ML input today, but needs a fuel adjusted correction before making any standalone explainability claim like "hard tyres degrade at X seconds per lap."

### build_pace_consistency.py

**Why:** wanted a race level number for how steady or erratic a driver's pace was, separate from raw pace itself.

**What it does:** standard deviation of clean lap times per driver per race.

**Why it matters:** feeds driver performance and strategy quality analysis, a driver who's fast but wildly inconsistent is telling you something different from one who's fast and steady.

### build_pit_stop_delta.py

**Why:** wanted a race level number for pit crew execution, whether a driver's stops cost more or less time than the field that day.

**What it does:** pairs each pit in lap with the following lap's pit out time to compute real stop duration, then compares a driver's average stop time against that race's median.

**Bug caught and fixed here too:** a handful of stops were inflated by red flag stoppages, the session clock kept running while cars sat in the garage, making some "stops" look like they took over an hour. Filtered anything over 100 seconds before computing medians, since real stops rarely exceed a minute even with penalties or double stacking.

**Why it matters:** usable directly as a team execution feature, and it's an input to the strategic aggressiveness feature built later at the team level.

### build_track_difficulty_index.py

**Why:** wanted a race level number for how hard or chaotic a given race was, independent of any one driver's performance.

**What it does:** combines how much of the race ran under a non green flag with how much the whole field's pace bounced around that day, standardizes both, and averages them into one index.

**Why it matters:** gives models context on race difficulty as its own factor, and helps interpret other features correctly, a mediocre degradation number means something different at an easy circuit than at a chaotic one.

### build_degradation_wide.py

**Why:** the final model ready feature table needs one row per driver per race, but tyre_degradation_stints is granular, multiple rows per driver per race, one per stint. Needed a usable, model ready version.

**What it does:** collapses multiple stints on the same compound within a race into one lap weighted average, then pivots into one row per driver per race with a separate column per compound, degradation_rate_soft, degradation_rate_hard, and so on.

**Why it matters:** this is what actually joins into the final assembled feature table for win, podium, and other downstream models, keeping compound specific signal intact instead of blending it away into one number.

**Known limitation, logged as a follow up task:** 2018 used absolute compound names, hypersoft, supersoft, ultrasoft, that aren't directly comparable to the relative soft, medium, hard labels used from 2019 onward. Not blocking, 2018 is a small share of total races, revisit before modeling if cross year compound comparisons matter.

## Teammate pairing

### build_teammate_pairs.py

**Why:** the two teammate relative driver level features need to know who was actually racing alongside whom, and that isn't fixed for a whole season since drivers get swapped mid year.

**What it does:** groups fact_race_results by race and team, pairs up whichever two drivers actually raced together that specific weekend, in both directions. Solo entries with no teammate that race are kept with a null teammate rather than dropped.

**Why it matters:** derives teammates per race instead of per season, so mid season swaps are handled automatically with no extra logic. Unblocks both driver level features below, and is the join key every teammate relative feature uses. 3,438 rows, only 2 solo entries, no anomalies.

## Driver level features

### build_tyre_wear_delta.py

**Why:** needed to turn the raw degradation numbers into something closer to actual driver skill, not just car and conditions.

**What it does:** joins each driver's degradation rate per compound against their teammate's for the same race, takes the difference.

**Why it matters:** since teammates share the same car and conditions that race, this delta cancels out most of the fuel burn and track effect confound flagged in the degradation rate work, a meaningfully cleaner signal than the raw number alone. Note: the mean and percentiles in this table are always symmetric around zero by construction, since both directions of each pair are stored as separate rows, that symmetry is expected, not a finding.

### build_wet_weather_delta.py

**Why:** last driver level feature, isolating performance specifically in wet conditions relative to a teammate.

**What it does:** classifies a race as wet if more than 5 percent of that race's weather samples showed rainfall, then compares each driver's positions gained, grid position minus finish position, against their teammate's, restricted to wet races only. Drivers with a missing grid or finish position, mostly early retirements, are excluded.

**Why it matters:** same teammate relative logic as tyre wear delta, cancels out car and strategy effects, isolating something closer to real wet weather skill. 23 wet races identified out of 172 total, swings of up to 25 positions either direction, which tracks with how much wet races typically shuffle the field.

## Team level features

### build_dnf_rate.py

**Why:** first team level feature, a team's mechanical reliability record.

**What it does:** classifies each result as finished or DNF based on the status text, Finished or a plus lap count counts as finished, anything else, engine, collision, disqualified, illness, counts as a DNF. Computes a trailing average per team, using only that team's races strictly before the current one.

**Bug caught and fixed here too:** the first version computed the trailing average at the individual row level, where two teammates in the same race share an identical year and round. That let one driver's row occasionally pick up their teammate's result from the same race as if it were prior history, a small leakage bug. Fixed by collapsing to one row per team per race before computing the trailing average, then broadcasting the corrected value back to both drivers.

**Why it matters:** usable directly as a predictive feature without leaking future information into past predictions, which a flat season long average would have done.

### build_strategic_aggressiveness.py

**Why:** second team level feature, how early or late a team tends to make their first pit stop relative to the rest of the field.

**What it does:** finds each driver's first pit stop lap, compares it against that race's median first stop lap across the whole field. Negative means they pitted earlier than most, an undercut attempt, positive means they stayed out longer, holding track position.

**Why it matters:** captures strategic tendency as its own signal, separate from execution quality, feeds into the strategy related models planned for phase 3.

## Open follow up items

1. Fuel adjusted degradation rate, separate tyre wear from fuel burn effect before making standalone explainability claims.
2. Reconcile 2018 compound naming against the post 2019 relative soft, medium, hard system.
3. lap_time null count came back at 3,156 after the timedelta fix, higher than the roughly 1,700 previously documented during EDA. Worth a quick look to see if a few hundred rows had a genuinely malformed value.

## Phase 2  complete



## Next phase

Phase 3, model training, SHAP, and MLflow. 