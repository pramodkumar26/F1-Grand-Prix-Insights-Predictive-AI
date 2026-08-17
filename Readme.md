# Grand Prix Insights

This is a Formula 1 data project where I pull real race data from 2018
through the current 2026 season and use it to build machine learning
models. The original goal was
five models: podium, tyre wear, teammate comparison, pit stop decision
quality, and how much a driver's race outcome shifted from what their
grid position alone would predict. That goal shifted a bit along the way,
some of those turned out to be better answered with a counted statistic
than a trained model, and two more models got added once the data showed
they were worth building. On top of everything, there's a chatbot planned
that answers questions using the model outputs, not just general
knowledge, that part hasn't been built yet.

## What the data looks like

All the race data comes from FastF1 and is stored in a SQLite database.
It covers lap times, race results, weather, qualifying, and race control
messages (things like flags and safety cars) across 9 seasons, with 2026
still in progress and refreshed as new races happen.

## What I found while exploring the data

Before building anything, I spent time going through the data carefully to
make sure it could actually be trusted. A few things stood out.

Some of the data had real bugs hiding in it. One table had race IDs stored
in a broken format that silently made every join fail. Another important
column was completely empty and had to be rebuilt from other data that was
already there. And two teams had gotten split into duplicate entries
because of a rebrand, which would have quietly messed up any stats for
that team.

Some things that looked wrong at first turned out to be correct once I
understood the reason behind them. For example, a few laps had extremely
long lap times, and it turned out that was because of an actual red flag
and race restart, not bad data.

Once the data was cleaned up, a few real patterns showed up. Tyres do wear
down over a race, but you only see it clearly once you account for the car
getting lighter as fuel burns off. Starting position is a decent predictor
of where a driver finishes, but it's a much stronger predictor near the
front of the grid than in the middle or back of the field.

Overall, this exploration step mattered a lot. It caught real problems
before they could quietly break any model, and it confirmed that the
patterns I'm hoping to model are actually present in the data, not just
assumptions.

## Turning the data into features

Before any model, there was a separate phase just for building features
out of the raw data, things like how strong a team's car has been
recently, how a driver's pace compares to their own teammate that
weekend, how much a driver's tyres actually wore down once fuel burning
off is accounted for, and how early or late a team tends to pit relative
to the rest of the field. Every one of these is built so it only ever
looks at information that would genuinely have been available at the
time, not something borrowed from later in the same race, unless a model
is deliberately meant to explain a race that already happened rather than
predict one that hasn't happened yet.

## The models that worked

Five things ended up predictable enough to actually build.

| model | what it predicts | result |
|---|---|---|
| podium | whether a driver finishes in the top 3 | ranks podium finishers above non-finishers correctly about 94% of the time, the cleanest result of the project, mostly because grid position and car strength are genuinely very predictive of a podium in real F1 |
| win | whether a driver finishes 1st | about 95%, cheap to build once podium already existed, and it showed something genuinely different, a competitive car and a clean race is enough to reach the podium, but winning outright takes real car dominance |
| points scored | championship points earned in a race | explains around 72% of the variation, the strongest predictive result of everything I built, and it fills a real gap podium leaves, most of the grid never actually podiums |
| strategy shift | positions gained or lost versus grid position, and why | explains a bit under half, allowed to use information from during the race itself since its job is explaining a race that already happened, not forecasting one, a real chunk of what's left is other drivers crashing or a safety car landing at the right or wrong moment, which nothing can predict |
| overtaking | track position gained over a race | the weakest of the five at around 16%, and it measures net position gained, not pure passing, some of the signal is just benefiting from a rival's pit stop timing, the two aren't cleanly separable with the data I have. that number moved down from 20% once the 2026 season was folded in, but the actual prediction error barely changed, it's because 2026's races so far have had less overtaking spread to explain, not a weaker model, checked directly rather than just taking the drop at face value |

## Two things that turned out to be counted, not modeled

Teammate comparison and pit stop execution both turned out to be
questions with a directly countable answer. Building a model to estimate
something that can just be measured would only add error, not remove it,
so both of these became scorecards instead, a head to head record between
teammates, and a team level pit stop execution ranking.

## Two things I tried and closed

Tyre degradation and DNF prediction both got a real, thorough attempt and
both got closed, not because the modeling was weak but because the
pattern genuinely isn't predictable in this data. Tyre wear at a given
circuit barely repeats from one season to the next. Team reliability is
stable year over year, but which specific driver retires from which
specific race is dominated by things like crashes, which are close to
random. Both are documented in full rather than abandoned quietly, the
reasoning behind why something doesn't work is still a real result.

## Explaining the predictions, not just stating them

A number on its own isn't that useful. I added SHAP on top of the models
complex enough to actually need it, so a prediction comes with a
breakdown of which factors pushed it up and which pushed it down, not
just the final number on its own. The simplest of the five models
explains itself well enough through its own coefficients that SHAP
wouldn't have added anything.

## Tracking the models properly

I set up MLflow locally to track every model's training runs, the simple
baseline it was compared against and the model that actually got adopted,
not just the final number. The adopted models are registered there by
name, so anything downstream, eventually the chatbot, can ask for the
current version of a model instead of depending on raw files.

## What's next

The models, the scorecards, and the tracking above are the inputs. The
chatbot, the part that actually answers questions using them, hasn't been
built yet.
