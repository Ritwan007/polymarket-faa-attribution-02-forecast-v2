#!/usr/bin/env python3
"""Forecast Polymarket FAA-outage attribution Yes prices, minute by minute.

Entry for make.fun competition polymarket-faa-attribution-02 (program/v2).

Method: per-market persistence median, adjusted by a conditional
sum-reversion shift (see below), with a spread estimated from the market's
own recent history.  For each anchor horizon h in (1, 5, 15, 60, 240)
minutes we take the robust scale (1.4826 * median absolute deviation, a
jump-robust estimator) of the h-minute price changes over the last 720
minutes, weighting each change by recency (half-life 240 minutes: recent
moves matter more in a jumpy regime), convert it to a per-square-root-minute
sigma, and interpolate sigma(h) in log-log space between the anchors (flat
outside them).  The five quantiles are then x + z_tau * sigma(h) * sqrt(h),
clamped to [0, 1] and rounded to 6 decimals, where x is the price at the
checkpoint; the 95% quantile is widened by a factor 1.3 (the upper tail is
where jump risk lives: on the practice the 95% quantile covered only
89.8% of prices, against a nominal 95%).

Sum reversion: the family's markets are near-mutually-exclusive outcomes,
so the sum S of their checkpoint prices should sit near 1.  When S > 1 the
field is overpriced and each market's median is pulled toward its
sum-normalised share x_i / S, with strength 1 - exp(-h/60) over the
horizon; the shift is clipped to +-0.03 (it never binds on the practice,
where the largest shift is 0.026, and it bounds the one dangerous case
on the final, where Iran is a subset of "a foreign actor" so the two
could legitimately sum near 2).  When S <= 1 there is no shift: the
residual "none of these" probability is real, and upward reversion was
measured to hurt.

A 0.85 calibration multiplier on sigma and the other constants were chosen
by walk-forward measurement on the six practice cases (leave-one-out: the
chosen form beat the v1 entry on five of six held-out cases, losing only
p05 by 0.004).

Measured on the practice (truth rebuilt from the next case's overlap for
p00-p04 and from Polymarket's public price history for p05; the baseline
reproduces the RULES figure 0.363589 exactly on this truth):
  house baseline  0.363589
  v1 entry        0.280044
  this entry      0.266431  (per case: p00 0.4117, p01 0.3052, p02 0.3515,
                             p03 0.2666, p04 0.1956, p05 0.1240)

Standard library only, no network, deterministic: two runs are byte-identical.
Reads $MAKEFUN_INPUT_DIR (else /input), writes $MAKEFUN_OUTPUT_DIR (else
/output), one <id>.out per <id>.in found.
"""

import json
import math
import os
from pathlib import Path

QUANTILES = [0.05, 0.25, 0.5, 0.75, 0.95]
# Standard normal quantiles at the five levels, to 6 decimals.
Z = [-1.644854, -0.67449, 0.0, 0.67449, 1.644854]
# Extra widening of the 95% quantile (upper-tail jump risk).
UP95_MULT = 1.3
# How far back the spread estimation looks (minutes).
WINDOW = 720
# Recency half-life (minutes) weighting the price changes.
HALF_LIFE = 240.0
# Anchor horizons (minutes) at which the h-minute change scale is measured.
ANCHORS = (1, 5, 15, 60, 240)
# Minimum per-sqrt-minute sigma, so a market that has never moved still gets
# a non-degenerate forecast.
SIGMA_FLOOR = 0.0002
# Calibration multiplier on the estimated sigma (walk-forward measured).
SIGMA_MULT = 0.85
# Fallback sigma when a market has too little history for any anchor.
FALLBACK_SIGMA = 0.0025
# An anchor needs at least this many h-minute changes to be trusted.
MIN_WINDOWS = 30
# Sum-reversion strength and time constant (minutes); the shift is clipped
# to +-SUMREV_CAP so no median is ever dragged far from its checkpoint.
SUMREV_LAM = 1.0
SUMREV_TAU = 60.0
SUMREV_CAP = 0.03


def median(values):
    """Median of a non-empty list (average of the two middles when even)."""
    ordered = sorted(values)
    n = len(ordered)
    mid = n // 2
    if n % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def weighted_median(pairs):
    """Median of values with weights, linearly interpolated at cum. 0.5.

    pairs is a list of (value, weight) with positive weights.  Matches the
    numpy reference (np.interp(0.5, cum_weights, sorted_values)): the first
    value when the first weight already carries half the total, otherwise
    a linear interpolation between the two bracketing values.
    """
    ordered = sorted(pairs)
    total = sum(w for _, w in ordered)
    cum = 0.0
    prev_value = ordered[0][0]
    prev_cum = 0.0
    for value, weight in ordered:
        cum += weight
        share = cum / total
        if 0.5 <= share:
            if prev_cum == 0.0:
                return value
            if share == prev_cum:
                return value
            t = (0.5 - prev_cum) / (share - prev_cum)
            return prev_value + t * (value - prev_value)
        prev_value = value
        prev_cum = share
    return ordered[-1][0]


def clamp6(x):
    """x clamped to [0, 1] and rounded to 6 decimals (0, never -0)."""
    return round(min(1.0, max(0.0, x)), 6) + 0.0


def anchor_sigmas(prices):
    """Robust per-sqrt-minute sigma at each anchor horizon.

    Returns (horizons, sigmas) with at least 30 sampled changes per anchor,
    or ([], []) when the market is too young for every anchor.  Each change
    is weighted by recency (half-life HALF_LIFE minutes) so the estimate
    follows the current volatility regime.
    """
    n = len(prices)
    lo = max(0, n - WINDOW)
    horizons = []
    sigmas = []
    for h in ANCHORS:
        end = n - h
        if end - lo < MIN_WINDOWS:
            continue
        diffs = []
        for i in range(lo, end):
            age = (n - 1) - (i + h)
            weight = 0.5 ** (age / HALF_LIFE)
            diffs.append((prices[i + h] - prices[i], weight))
        centre = weighted_median(diffs)
        devs = [(abs(d - centre), w) for d, w in diffs]
        mad = weighted_median(devs)
        sigma = 1.4826 * mad / math.sqrt(h)
        horizons.append(h)
        sigmas.append(max(sigma, SIGMA_FLOOR))
    return horizons, sigmas


def sigma_at(h, horizons, sigmas):
    """sigma(h) by log-log interpolation between anchors, flat outside."""
    if not horizons:
        return FALLBACK_SIGMA
    if h <= horizons[0]:
        return sigmas[0]
    if h >= horizons[-1]:
        return sigmas[-1]
    for j in range(len(horizons) - 1):
        h0, h1 = horizons[j], horizons[j + 1]
        if h0 <= h <= h1:
            w = (math.log(h) - math.log(h0)) / (math.log(h1) - math.log(h0))
            return math.exp(math.log(sigmas[j]) * (1.0 - w)
                            + math.log(sigmas[j + 1]) * w)
    return sigmas[-1]


def forecast_case(case):
    """240 rows of five quantiles per market of one case."""
    horizon = case["horizon_minutes"]
    markets = case["markets"]
    total = sum(m["prices"][-1] for m in markets)
    forecasts = []
    for m in markets:
        prices = m["prices"]
        x = prices[-1] if prices else 0.5
        horizons, sigmas = anchor_sigmas(prices) if prices else ([], [])
        rows = []
        for h in range(1, horizon + 1):
            # Sum-reversion shift: only when the field is overpriced
            # (sum > 1), pulling the median toward its normalised share,
            # clipped so it can never drag a market far from its price.
            shift = 0.0
            if total > 1.0:
                target = x / total
                shift = SUMREV_LAM * (target - x) * (
                    1.0 - math.exp(-h / SUMREV_TAU))
                shift = max(-SUMREV_CAP, min(SUMREV_CAP, shift))
            if horizons:
                sigma = sigma_at(h, horizons, sigmas) * SIGMA_MULT
                spread = sigma * math.sqrt(h)
                row = [clamp6(x + shift + z * spread
                              * (UP95_MULT if j == 4 else 1.0))
                       for j, z in enumerate(Z)]
            else:
                # Too young to estimate any anchor: the house baseline's
                # own sigma, unscaled -- a market we know nothing about
                # gets the prior spread, not the calibrated one.
                spread = FALLBACK_SIGMA * math.sqrt(h)
                row = [clamp6(x + shift + z * spread) for z in Z]
            rows.append(row)
        forecasts.append({"id": m["id"], "q": rows})
    return forecasts


def main():
    input_dir = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
    output_dir = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))
    output_dir.mkdir(parents=True, exist_ok=True)
    inputs = sorted(p for p in input_dir.iterdir()
                    if p.is_file() and p.name.endswith(".in"))
    for path in inputs:
        case = json.loads(path.read_text(encoding="utf-8"))
        if list(case["quantiles"]) != QUANTILES:
            raise ValueError(
                f"{path.name} asks for quantiles {case['quantiles']}, "
                f"not {QUANTILES}")
        out = json.dumps({"forecasts": forecast_case(case)},
                         separators=(",", ":")) + "\n"
        (output_dir / (path.stem + ".out")).write_text(out, encoding="utf-8")


if __name__ == "__main__":
    main()
