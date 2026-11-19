"""Forecast each GTA VI trophy's Exophase earn rate at a case target.

Reads every <id>.in case file (schema makefun/gta6-trophies-case/v1) from the
input directory and writes <id>.out: five quantiles (5/25/50/75/95 percent) of
each trophy's earn rate at the case's target. Python 3.13, standard library
only, no network, deterministic: two runs on the same input write identical
bytes.

Model: a per-trophy hierarchical drift model in logit space.

  * Exophase refreshes its page on its own schedule, so consecutive identical
    reads are collapsed to refresh events and only the events feed the fit.
    The forecast is projected from the last event to the target, which covers
    both the staleness gap and the horizon.
  * Each trophy's logit earn rate is regressed on time with recency weights.
    The slope is shrunk toward its tier's (platinum / gold / silver / bronze)
    mean with an empirical-Bayes weight, the tier mean itself is shrunk
    toward zero, and the total projected move is softly capped, so a
    multi-day horizon cannot run away on a noisy slope.
  * When the history is flat (no refresh yet, as on launch night) there is
    no drift to estimate, so the median is shifted by a launch-night
    dilution prior: the tracked-player base grows fast after launch and new
    players dilute every rate; the shift is smaller for early trophies that
    most new players already earned.
  * The spread comes from a rolling-origin backtest inside the case's own
    history, which picks the random-walk volatility that minimises pinball
    loss; when the history cannot support a backtest, a horizon-growing
    prior spread is used. Near-100% and near-zero trophies get a wider
    floor, because a miss there costs many logit units.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first, so a module
# shipped beside this file would import. (This entry is a single file.)
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

# The quantile levels the contest scores, in the order each row lists them.
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
# The standard normal's quantile at each level.
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]
# A rate is clamped to [RATE_FLOOR, RATE_CEIL] percent before its logit
# (the scorer's clamp).
RATE_FLOOR = 0.01
RATE_CEIL = 99.99
# The largest spread a forecast may give, logit units.
SPREAD_MAX = 5.0
# The smallest spread a forecast may give, logit units.
SPREAD_MIN = 0.15

# --- tuning constants (see the model notes in the README) -------------------
DRIFT_HALFLIFE_DAYS = 0.75   # recency weight half-life for the drift fit
DRIFT_WINDOW_DAYS = 0.75     # only events this recent feed the drift fit:
                             # launch-week drift is strongly non-stationary,
                             # so only the last ~18 hours carry signal
DRIFT_PRIOR_SD = 0.35        # logit/day: prior sd of a drift around its tier mean
TIER_SHRINK_N = 3.0          # shrink a tier mean toward zero by n/(n+this)
MOVE_CAP = 3.0               # soft cap on the total projected logit move
BASE_SPREAD = 0.25           # irreducible per-trophy spread, logit units
RW_SIGMA_PRIOR = 0.5         # random-walk sigma, logit per sqrt day
STALE_SPREAD = 0.5           # extra spread when the history is flat
LAUNCH_GROWTH_0 = 15.0       # log/day: tracked-player growth at launch
LAUNCH_GROWTH_DECAY_DAYS = 0.5  # the launch surge decays with this half-life
LAUNCH_GROWTH_FLOOR = 0.5    # log/day: tracked-player growth long after launch
# Dilution carryover: the fraction of a trophy's rate that newly tracked
# players already have, as a function of the rate (rare trophies: new players
# almost never have them; common early ones: often).
KAPPA_BASE = 0.02
KAPPA_SLOPE = 0.15
KAPPA_REF_RATE = 2.5
KAPPA_MAX = 0.85


def clamp_rate(pct):
    return min(RATE_CEIL, max(RATE_FLOOR, pct))


def logit(pct):
    c = clamp_rate(pct)
    return math.log(c / (100.0 - c))


def inv_logit(x):
    if x >= 0:
        return 100.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return 100.0 * e / (1.0 + e)


def round4(x):
    """x rounded to 4 decimals, half up, as JavaScript's Math.round(x * 10000)
    / 10000 rounds it. A whole number comes back as an int, so json.dumps
    writes 100 where JavaScript writes 100 (not 100.0)."""
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if value == int(value) else value


def parse_time(s):
    """An ISO timestamp to epoch seconds (UTC)."""
    return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()


def weighted_linreg(xs, ys, ws):
    """Weighted least squares of y on x. Returns (slope, intercept, se_slope)
    or None when the fit is degenerate."""
    sw = sum(ws)
    if sw <= 0 or len(xs) < 2:
        return None
    mx = sum(w * x for w, x in zip(ws, xs)) / sw
    my = sum(w * y for w, y in zip(ws, ys)) / sw
    sxx = sum(w * (x - mx) ** 2 for w, x in zip(ws, xs))
    if sxx <= 1e-12:
        return None
    sxy = sum(w * (x - mx) * (y - my) for w, x, y in zip(ws, xs, ys))
    slope = sxy / sxx
    intercept = my - slope * mx
    # Residual variance and the slope's standard error.
    dof = max(1, len(xs) - 2)
    s2 = sum(w * (y - (intercept + slope * x)) ** 2 for w, x, y in zip(ws, xs, ys)) / dof
    # Effective sample size for the weighted fit.
    n_eff = sw * sw / sum(w * w for w in ws)
    se = math.sqrt(s2 / sxx) if s2 > 0 else 0.0
    # With few points the slope is unreliable: inflate its error.
    if n_eff < 4:
        se = max(se, 0.5)
    return slope, intercept, se


def build_events(case):
    """The case's history collapsed to refresh events: consecutive reads with
    identical players-tracked and rates carry no new information. Returns a
    list of (time_days, players_tracked, rates) in time order, times in days
    since the first read, plus the case's cut time in the same units."""
    reads = case.get("history") or []
    events = []
    prev_key = None
    t0 = None
    for read in reads:
        try:
            t = parse_time(read.get("at") or read["slot"])
        except (KeyError, TypeError, ValueError):
            continue
        if t0 is None:
            t0 = t
        rates = tuple(read.get("rates") or [])
        key = (read.get("players_tracked"), rates)
        if key == prev_key:
            continue
        prev_key = key
        events.append(((t - t0) / 86400.0, read.get("players_tracked"), rates))
    cut_days = 0.0
    try:
        cut_days = (parse_time(case["at"]) - t0) / 86400.0 if t0 is not None else 0.0
    except (KeyError, TypeError, ValueError):
        cut_days = events[-1][0] if events else 0.0
    return events, cut_days


def backtest_rw_sigma(case, events, cut_days, horizon_days):
    """Rolling-origin backtest inside the case's own history: at several cut
    points, fit the drift on earlier events only, forecast the rate at a
    later event, and score the pinball loss for candidate random-walk
    volatilities. Returns the best volatility, or None when the history
    cannot support a backtest."""
    if len(events) < 5:
        return None
    span = events[-1][0] - events[0][0]
    if span <= 0.25:
        return None
    trophies = case["trophies"]
    n_tr = len(trophies)
    # Origins at 35/55/75% of the span; the truth is the first event at least
    # 6 hours past the origin (or the last event).
    origins = []
    for frac in (0.35, 0.55, 0.75):
        t_cut = events[0][0] + frac * span
        past = [e for e in events if e[0] <= t_cut]
        future = [e for e in events if e[0] > t_cut + 0.25]
        if len(past) >= 2 and future:
            origins.append((past, future[0]))
    if not origins:
        return None
    candidates = (0.3, 0.5, 0.75, 1.0, 1.5)
    best = (None, float("inf"))
    for rw in candidates:
        total = 0.0
        count = 0
        for past, truth_ev in origins:
            horizon = max(0.05, truth_ev[0] - past[-1][0])
            pred = forecast_with_drift(case, past, past[-1][0], horizon, rw, 0.0)
            truth = truth_ev[2]
            for i in range(n_tr):
                y = truth[i]
                if y is None:
                    continue
                yl = logit(y)
                for tau, q in zip(LEVELS, pred[i]):
                    ql = logit(q)
                    d = yl - ql
                    total += d * tau if d >= 0 else -d * (1.0 - tau)
                    count += 1
        if count:
            mean = total / count
            if mean < best[1]:
                best = (rw, mean)
    return best[0]


def forecast_with_drift(case, events, from_days, horizon_days, rw_sigma, stale_spread):
    """Forecast every trophy's five quantiles from a drift fit on `events`,
    projecting from the last event over `horizon_days`. Returns a list of
    quantile lists, one per trophy, in the case's trophy order. This is the
    shared core used by the backtest and by the real forecast."""
    trophies = case["trophies"]
    n_tr = len(trophies)
    t_last = events[-1][0] if events else from_days
    # Recent events feed the drift fit.
    window_start = t_last - DRIFT_WINDOW_DAYS
    fit_events = [e for e in events if e[0] >= window_start] or events
    # Per-trophy drift fits.
    drifts = []
    for i in range(n_tr):
        xs, ys, ws = [], [], []
        for t, _n, rates in fit_events:
            r = rates[i] if i < len(rates) else None
            if r is None:
                continue
            xs.append(t)
            ys.append(logit(r))
            ws.append(0.5 ** ((t_last - t) / DRIFT_HALFLIFE_DAYS))
        fit = weighted_linreg(xs, ys, ws)
        drifts.append(fit)  # (slope, intercept, se) or None
    # Tier means of the drifts, precision-weighted, shrunk toward zero.
    tier_slopes = {}
    for tier in sorted({t.get("tier") or "bronze" for t in trophies}):
        num = 0.0
        den = 0.0
        n = 0
        for i, t in enumerate(trophies):
            if (t.get("tier") or "bronze") != tier:
                continue
            fit = drifts[i]
            if fit is None:
                continue
            slope, _ic, se = fit
            prec = 1.0 / (se * se + DRIFT_PRIOR_SD * DRIFT_PRIOR_SD)
            num += prec * slope
            den += prec
            n += 1
        if den > 0:
            mean = num / den
            tier_slopes[tier] = mean * (n / (n + TIER_SHRINK_N))
        else:
            tier_slopes[tier] = 0.0
    out = []
    for i, trophy in enumerate(trophies):
        tier = trophy.get("tier") or "bronze"
        last_rate = None
        for _t, _n, rates in reversed(events):
            if i < len(rates) and rates[i] is not None:
                last_rate = rates[i]
                break
        if last_rate is None:
            last_rate = 1.0
        fit = drifts[i]
        if fit is None:
            slope, se = tier_slopes.get(tier, 0.0), DRIFT_PRIOR_SD
        else:
            raw, _ic, se_raw = fit
            prior = tier_slopes.get(tier, 0.0)
            prec_data = 1.0 / max(se_raw * se_raw, 1e-6)
            prec_prior = 1.0 / (DRIFT_PRIOR_SD * DRIFT_PRIOR_SD)
            slope = (prec_data * raw + prec_prior * prior) / (prec_data + prec_prior)
            se = math.sqrt(1.0 / (prec_data + prec_prior))
        # Projected move, softly capped so a long horizon cannot run away.
        move = slope * horizon_days
        move = move / (1.0 + abs(move) / MOVE_CAP)
        median_logit = logit(last_rate) + move
        median = inv_logit(median_logit)
        # Spread: irreducible base + drift uncertainty + random walk +
        # staleness, with a wider floor for near-100% and near-zero rates.
        var = BASE_SPREAD * BASE_SPREAD
        var += (se * horizon_days) ** 2
        var += (rw_sigma * math.sqrt(max(horizon_days, 0.0))) ** 2
        var += stale_spread * stale_spread
        spread = math.sqrt(var)
        if last_rate >= 99.0:
            spread = max(spread, 1.0)
        elif last_rate <= 1.0:
            spread = max(spread, 0.6 + 0.4 * math.exp(-last_rate / 2.0))
        spread = min(max(spread, SPREAD_MIN), SPREAD_MAX)
        med_l = logit(median)
        qs = []
        prev = None
        for z in Z:
            q = round4(inv_logit(med_l + z * spread))
            q = min(max(q, RATE_FLOOR), RATE_CEIL)
            if prev is not None and q < prev:
                q = prev
            qs.append(q)
            prev = q
        out.append(qs)
    return out


def dilution_median(last_rate, horizon_days, gap_days, release_age_days):
    """Launch-night dilution prior for a flat history: the tracked-player
    base grows, new players dilute every rate, and the shift is smaller for
    trophies new players already earned. `gap_days` is the time from the
    last refresh to the target; `release_age_days` the time from the game's
    release to the case cut."""
    if last_rate >= 99.9:
        return last_rate
    growth = LAUNCH_GROWTH_0 * math.exp(-release_age_days / LAUNCH_GROWTH_DECAY_DAYS)
    growth = max(growth, LAUNCH_GROWTH_FLOOR)
    g = math.exp(growth * max(gap_days, 0.0))
    kappa = KAPPA_BASE + KAPPA_SLOPE * (last_rate / KAPPA_REF_RATE)
    kappa = min(max(kappa, 0.05), KAPPA_MAX)
    return last_rate * (1.0 + kappa * (g - 1.0)) / g


def forecast_case(case):
    if case.get("levels") != LEVELS:
        raise ValueError("case asks for levels %r, not %r" % (case.get("levels"), LEVELS))
    if not case.get("history"):
        raise ValueError("case has no history")
    trophies = case["trophies"]
    horizon_days = case["horizon_minutes"] / 1440.0
    events, cut_days = build_events(case)
    if not events:
        raise ValueError("case history parses to no events")
    try:
        target_days = (parse_time(case["target"]) - parse_time(case["history"][0].get("at") or case["history"][0]["slot"])) / 86400.0
    except (KeyError, TypeError, ValueError):
        target_days = cut_days + horizon_days
    # Effective horizon: from the last refresh event to the target.
    eff_horizon = max(target_days - events[-1][0], horizon_days)
    flat = len(events) <= 1
    if flat:
        # No drift to estimate: dilution prior on the median, wide spread.
        release_age = 0.0
        game = case.get("game") or {}
        try:
            release_age = max(0.0, (parse_time(case["at"]) - parse_time(game["release_date"] + "T00:00:00Z")) / 86400.0)
        except (KeyError, TypeError, ValueError):
            release_age = 0.0
        gap = max(target_days - events[-1][0], horizon_days)
        rows = []
        for trophy in trophies:
            last_rate = None
            for _t, _n, rates in reversed(events):
                idx = trophies.index(trophy)
                if idx < len(rates) and rates[idx] is not None:
                    last_rate = rates[idx]
                    break
            if last_rate is None:
                last_rate = 1.0
            median = dilution_median(last_rate, horizon_days, gap, release_age)
            var = BASE_SPREAD * BASE_SPREAD
            var += (RW_SIGMA_PRIOR * math.sqrt(max(eff_horizon, 0.0))) ** 2
            var += STALE_SPREAD * STALE_SPREAD
            spread = math.sqrt(var)
            if last_rate >= 99.0:
                spread = max(spread, 1.0)
            elif last_rate <= 1.0:
                spread = max(spread, 0.6 + 0.4 * math.exp(-last_rate / 2.0))
            spread = min(max(spread, SPREAD_MIN), SPREAD_MAX)
            med_l = logit(median)
            qs = []
            prev = None
            for z in Z:
                q = round4(inv_logit(med_l + z * spread))
                q = min(max(q, RATE_FLOOR), RATE_CEIL)
                if prev is not None and q < prev:
                    q = prev
                qs.append(q)
                prev = q
            rows.append({"id": trophy["id"], "q": qs})
        return {"trophies": rows}
    rw = backtest_rw_sigma(case, events, cut_days, horizon_days)
    if rw is None:
        rw = RW_SIGMA_PRIOR
    quantiles = forecast_with_drift(case, events, events[-1][0], eff_horizon, rw, 0.15)
    rows = [{"id": t["id"], "q": qs} for t, qs in zip(trophies, quantiles)]
    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Every case file, in name order, so two runs write the same bytes.
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case)
        (OUTPUT_DIR / (path.stem + ".out")).write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
