# Grand Prix Insights

This is a Formula 1 data project where I pull real race data from 2018 to
2025 and use it to build machine learning models. The end goal is to
predict things like who finishes on the podium, how tyres wear down over a
race, how one teammate compares to another, and whether a pit stop
decision was actually a good call. On top of the models, there's a chatbot
that answers questions using the model outputs, not just general
knowledge.

## What the data looks like

All the race data comes from FastF1 and is stored in a SQLite database.
It covers lap times, race results, weather, qualifying, and race control
messages (things like flags and safety cars) across 8 seasons.

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