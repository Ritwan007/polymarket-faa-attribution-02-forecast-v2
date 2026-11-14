# polymarket-faa-attribution-02 — forecast v2

Entry for the make.fun competition `polymarket-faa-attribution-02`:
forecast Polymarket's FAA outage attribution Yes prices, 4 markets,
240 minutes ahead, 5 quantiles, pinball loss.

## Model

- **Median**: last price (persistence) + a conditional sum-reversion shift.
  When the sum of the case's markets' checkpoint prices exceeds 1 (the
  field is overpriced — mutually-exclusive outcomes should sum near 1),
  each median is pulled toward its sum-normalised share `x_i / S` with
  strength `1 − exp(−h/60)`, clipped to ±0.03. No shift when the sum is
  ≤ 1 (upward reversion measured worse).
- **Spread**: per-market, jump-robust MAD of h-minute price changes at
  anchors (1, 5, 15, 60, 240) over the last 720 minutes, with EWMA
  recency weights (half-life 240 min), log-log interpolated, floor 0.0002,
  ×0.85 calibration.
- **Quantiles**: normal z-scores; the 95% quantile widened ×1.3 (upper-tail
  jump risk). Clamped [0, 1], rounded to 6 decimals.
- Too-young markets fall back to the house sigma 0.0025 unscaled.

Pure Python standard library, no network, deterministic.

## Measured practice loss

| entry | loss |
|---|---|
| house baseline | 0.363589 |
| v1 (submitted) | 0.280044 |
| **this entry (v2)** | **0.266431** |

Per case: p00 0.4117, p01 0.3052, p02 0.3515, p03 0.2666, p04 0.1956,
p05 0.1240. Beats v1 on 5 of 6 held-out cases in leave-one-out (only p05
worse, +0.004). Full detail:
`~/workspace/makefun/faa-attribution-02-v2-findings.md`.

## Run locally

```sh
mkdir -p out
MAKEFUN_INPUT_DIR=<kit>/input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

## License

MIT — see `LICENSE`.
