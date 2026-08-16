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

## Fuel adjusted degradation, open item 1 below, now closed

### build_fuel_adjusted_degradation.py

**Why:** `build_tyre_degradation.py` fits a slope per stint, and that slope is tyre wear and fuel burn added together with no way to tell them apart. Measured the confound directly to confirm it: WITHIN a stint, correlation between tyre_age and lap_number is exactly 1.0000, perfect lockstep, so no per stint fit can ever separate them. This was logged as the first open follow up item and blocked any standalone claim like "hard tyres degrade at X seconds per lap."

**The insight that unblocks it:** the confound is total within a stint but not across stints. Pooled across every stint in a race, corr(tyre_age, lap_number) drops to a median of 0.537, because a stint starting on lap 30 sits at tyre_age 1 while lap_number is already 30. So the fuel coefficient IS identifiable once laps are pooled, even though it never is within a single stint. 94.9% of race+driver units have 2 or more stints, so this covers nearly the whole dataset.

**What it does:** estimates the fuel coefficient per race (not globally, fuel burn per lap depends on lap length, Monaco and Monza aren't comparable) by pooling all clean laps in that race and regressing driver centered lap time on tyre_age and lap_number together. Lap times are centered on each driver's own race median first so the fit doesn't read "Red Bull is faster than Williams" as a fuel effect. Then, since within a stint tyre_age advances exactly 1 per lap, the raw stint slope is (tyre effect + fuel effect), so subtracting the race's fuel coefficient leaves the tyre-only rate. Writes `race_fuel_effect` and `tyre_degradation_adjusted`, keeping the raw rate alongside the adjusted one so the two are always comparable.

**Independent validation, the part that matters most:** the measured dry race fuel effect is a median of **-0.062 s/lap**. Nothing in the pipeline was tuned toward that number, it fell out of the regression. Real F1 cars carry roughly 100kg of fuel at the start and burn 1.5-2kg per lap at roughly 0.035s per kg, which predicts about -0.06 s/lap. Matching a known engineering value that was never fed into the calculation is real evidence the method is measuring the actual physical thing rather than fitting noise.

**The headline finding:** raw degradation had a median of **-0.027 s/lap**, negative, meaning the original feature claimed tyres get FASTER as they age, which is physically backwards. Fuel adjusted, the median is **+0.031 s/lap**, positive, tyres get slower as they wear, which is correct. The fuel burn effect was large enough to flip the sign of the entire feature. Every one of the 7,782 stints moved in the expected direction.

**Bug caught and fixed here too:** first version produced 6 races with a positive fuel effect (physically impossible, cars don't get slower as they lighten) and one at -1.03 s/lap (absurdly large). Checked them directly: 8 of the 12 implausible races were wet, against a 12.9% base rate of wet races, roughly 5x enrichment, and wet races showed 6.7x the spread of dry ones. The mechanism is the same class of confound this script exists to fix, in a wet race the track drying out or rain arriving is ALSO a function of lap_number, and it swamps the fuel signal. 2025 Belgian (-1.03) was a wet track drying; 2023 Monaco (+0.25) was rain arriving mid race. Fixed by treating wet races and out of band estimates as unreliable and substituting that circuit's median dry race estimate, falling back to the global dry median for the 2 races with no circuit reference. This is physically justified rather than a patch: fuel burn per lap is a property of the car and circuit, not the weather, a wet race still burns fuel normally, what changes is track evolution, which this term should never have been absorbing. After the fix all 171 races have a sensible negative fuel effect and the spread tightened from std 0.101 to 0.021. A `fuel_effect_source` column records own / circuit_median / global_median per race so every value stays auditable, 144 own, 25 circuit, 2 global.

**Known limitation, and it is a significant one, logged as new open item 4:** fuel adjustment does NOT make cross compound claims safe, which was the original motivation for this work. Tested it directly. After adjustment, median degradation by compound comes out SOFT -0.001, MEDIUM -0.010, HARD +0.027, which is backwards from the naive expectation that softer rubber wears faster. Controlling for stint length (11-20 lap stints only) doesn't recover the expected ordering either. The reason: compound labels are relative, not absolute. Pirelli brings 3 of 5 compounds (C1 to C5) to each race, so "HARD" at Monaco is physically different rubber than "HARD" at Silverstone, and compound choice is endogenous, teams pick harder compounds precisely at abrasive circuits. Measured the size of this: for the SAME "HARD" label, degradation ranges from -0.026 s/lap at Sochi to +0.173 at Spa, a spread of 0.199, while the entire spread ACROSS compound labels is only 0.029. **The circuit effect is 7x larger than the compound label effect.** So a claim like "hard tyres degrade at X s/lap" is still not defensible, not because of fuel any more, but because the compound label barely means anything without the circuit and the absolute C1-C5 mapping, which this dataset doesn't carry.

## Open follow up items

1. ~~Fuel adjusted degradation rate, separate tyre wear from fuel burn effect before making standalone explainability claims.~~ **Done**, see above. Note the follow up in item 4, fuel was not the only confound.
2. Reconcile 2018 compound naming against the post 2019 relative soft, medium, hard system.
3. lap_time null count came back at 3,156 after the timedelta fix, higher than the roughly 1,700 previously documented during EDA. Worth a quick look to see if a few hundred rows had a genuinely malformed value.
4. **Cross compound degradation claims are not identifiable, deliberately deferred.**

   **What's blocked:** any statement of the form "compound A degrades faster than compound B," across circuits. Single circuit statements are fine, and so is compound as one input among several in a model.

   **Why, with numbers.** Compound labels in this dataset are relative, not absolute. Pirelli selects 3 of its 5 slick compounds (C1 hardest to C5 softest) for each race weekend and relabels them locally as HARD / MEDIUM / SOFT, so the "HARD" at one race is frequently different rubber from the "HARD" at another. Compound choice is also endogenous, teams pick harder rubber precisely at abrasive circuits, so the label correlates with the very thing it would need to be independent of. Measured after fuel adjustment, on quality filtered stints (r_squared > 0.3, 8+ laps):
   - median degradation by label: SOFT -0.001, MEDIUM -0.010, HARD +0.027 s/lap, which is backwards from the naive expectation that softer rubber wears faster
   - controlling for stint length (11-20 lap stints only) does not recover the expected ordering either
   - for the SAME "HARD" label, median degradation ranges from -0.026 s/lap at Sochi to +0.173 at Spa, a spread of 0.199
   - the entire spread ACROSS the three compound labels is 0.029
   - **the circuit effect is 7.0x larger than the compound label effect**

   **What it would take to close.** A mapping of (race, local label) to absolute Pirelli compound C1 to C5. That is roughly 173 races x 3 labels, about 520 rows. It is published by Pirelli per event and is not derivable from anything currently in `f1.db`, FastF1's lap data carries only the local label. Sourcing options, cheapest first: Pirelli's per event press releases or preview PDFs, the FIA event notes, or a maintained community dataset. Once loaded as a `dim_compound_absolute` style table joined on race_id + compound_id, the analysis above can be rerun grouped on the absolute compound, and cross compound claims become testable rather than assumed.

   **What it does NOT block.** Degradation modelling scoped to circuit and stint context, which is what the tyre model is being built as. Compound stays in as one modest input. The only thing withheld is the cross compound comparative claim.

   **Revisit trigger:** if a use case appears that specifically needs compound to compound comparison, for example a chatbot answer like "should I have used the medium instead," come back to this item first. The evidence above is the reason it was deferred, not an argument that it is unimportant.

5. **`fact_weather` had no time dimension until this was fixed.** `load_weather.py` read the `Time` column from `data/processed/all_weather.csv` and never inserted it, leaving roughly 149 unordered samples per race with no way to tell which reading preceded a given point in the race. Any temperature aggregate built from that table would have silently mixed in readings from later in the race than the moment being predicted from, a leakage vector for any forecasting model. Fixed at the source by `collecting_data/fix_weather_session_time.py`, which adds `session_time_seconds` and backfills all 25,680 rows, verifying per race row counts match between CSV and table before relying on positional alignment. Race start conditions are now unambiguously identifiable. Full time aligned weather, matching a weather sample to a specific lap, is still not possible, since `fact_laps` carries no timestamp either, that would need a similar fix on the lap loader and is not currently blocking anything.

## Phase 2  complete



## Next phase

Phase 3, model training, SHAP, and MLflow. 