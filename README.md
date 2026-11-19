# gta6-trophy-rarity-forecast-01

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/t65244973-cell/gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/t65244973-cell/gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Forecast every Grand Theft Auto VI PlayStation 5 trophy's earn rate on Exophase one week after launch. A per-trophy hierarchical drift model in logit space: it reads the case's Exophase read history, estimates each trophy's recent drift with shrinkage toward its tier mean, projects it to the case target, and sets each quantile's spread from a rolling-origin backtest on the case's own history, so the forecast stays honest when the page goes stale and Exophase refreshes in jumps.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

`main.py` (Python 3.13, standard library only) reads every `<id>.in` case file in the input directory and writes `<id>.out` — five quantiles (5%, 25%, 50%, 75%, 95%) of each of the 55 trophies' earn rate at the case's target — to the output directory. Two runs on the same input write identical bytes.

## How it forecasts

- **Stale reads are events, not noise.** Exophase refreshes its page on its own schedule, so consecutive identical reads are collapsed to refresh events and only the events feed the fit.
- **Drift, shrunk.** Each trophy's logit earn rate is regressed on time with recency weights; the slope is shrunk toward its tier's (platinum / gold / silver / bronze) mean with an empirical-Bayes weight, and the total projected move is softly capped so a 4.5-day horizon cannot run away.
- **Dilution-aware on launch night.** When the history is flat (no refresh yet), the median is shifted down by a launch-night dilution prior: new players dilute every rate, and the shift is larger for rare, late trophies than for early ones everyone earns.
- **Spread from backtesting.** A rolling-origin backtest inside the case's own history picks the spread multiplier that minimises pinball loss; when the history cannot support a backtest, a horizon-growing prior spread is used instead.

## Validation

The model was validated on a simulator of the Exophase process (player arrivals, per-trophy earn curves, Exophase refresh staleness, house reads), calibrated to the observed launch night: 638 players tracked with rates near 1% at the first read, 3,868 at the 04:30 refresh with every rate lower. Mean pinball loss on the logit scale over 8 simulated worlds, two regimes (launch-night dilution strong / mild):

| case | this model | persistence baseline | recent-curve baseline |
| --- | --- | --- | --- |
| practice p00/p01, strong dilution | 0.24 | 0.74 | 0.74 |
| final f00, strong dilution | 0.21 | 0.70 | 0.81 |
| final f00, mild dilution | 0.11 | 0.42 | 0.79 |

The drift window (last ~18 hours) and the dilution prior were the two largest wins: unshrunk curve extrapolation runs away over a 4.5-day horizon, and persistence misses the launch-night refresh drop entirely.

---

<sub>Entered on [make.fun](https://make.fun/?ref=s6yd0164&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
