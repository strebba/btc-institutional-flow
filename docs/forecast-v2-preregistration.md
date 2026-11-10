# dealer_flow v2 — pre-registration

Date: **2026-11-10**.

This document is the rule sheet for the shadow forecast. It was written before
v2 is scored on any cohort it produces. The decision below uses only cohorts
created on or after this date. Anything computed on earlier cohorts is
in-sample and is not evidence.

v1 (`source=dealer_flow`) stays the live model: CAUTION still means flat inside
a ±1% band, max-pain targets are unchanged, the probability is still the linear
map of the score. v2 is written beside it as `source=dealer_flow_v2`, same
slot, same inputs, same active weights. `idx_pred_day_unique` is per source,
so the two cohorts do not collide and the v1 catch-up is unchanged: if v1
already exists it is not regenerated. If v2 is missing, it is built from the
stored v1 rows (no second market fetch).

## Rules (fixed)

Weights are frozen (`governance.freeze_weights: true`). The weekly job still
computes metrics. It does not insert a proposed weight version, and
`POST /api/weights/{id}/activate` returns 409. Unfreezing is a separate
decision and is not part of this test.

**(a) Direction.** The call is the sign of `(score − 50)`: above 50 → up,
below 50 → down. If confidence `|score−50|/50` is below **0.10** (score in
the open interval (45, 55)) there is **no direction row**. That is an
abstention.

Why not "map all of CAUTION to a direction", and why not abstain on the whole
CAUTION band (40–64)? CAUTION was ~115 of 127 calls and was defined as flat,
so the direction head matched always-flat (26.8% vs 26.0%) and never said up.
The score already has a sign; using it is the information the model computed.
Abstaining on all of 40–64 would leave only the LONG / RISK_OFF tails and a
coverage too thin to evaluate. The 0.10 cut is the existing confidence
definition: inside five points of 50 the sign is not a call.

The flat band used when the outcome is scored is **not** ±1%. It is
`0.5 × σ_horizon`, where `σ_horizon` is the sample standard deviation of the
last 20 complete daily log-returns strictly before the prediction date,
times `sqrt(horizon)`. Under a normal, ±0.5σ holds about 38% of outcomes, so
"flat" tracks recent volatility instead of a fixed 1% band that contained
only ~26% of 5-day returns. With fewer than 10 returns the band is the prior
**4%**. A floor of 0.1% avoids a zero band. The number is stored on the
prediction (`flat_band_pct`) and is not refit after the fact.

Abstentions are not in the hit-rate denominator. Coverage is
`cohorts with a direction row / cohorts`, reported next to the hit rate.

**(b) Level.** A max-pain target is emitted only when it is at most **4%**
from the reference price. Farther than that, there is no level prediction.
The negative-gamma gamma-flip target is unchanged. The scorecard's 0/51 on
targets more than 4% away is why the cutoff is 4% and not a fitted distance.

**(c) Probability.** Raw `p` is still `clip(0.5 + (score−50)/100, 0.05, 0.95)`.
It is then recalibrated with data available before the prediction only:

- Training rows: matured `dealer_flow` **prob** outcomes whose horizon end is
  strictly before the prediction timestamp. `y = 1` if realized return > 0.
  v2's own outcomes are not in the fit, so the procedure does not chase itself.
- If `n ≥ 30`: Platt scaling, `P(y=1) = sigmoid(a + b·(score−50))`, fit by
  minimising negative log-likelihood plus `(λ/2)(a²+b²)` with `λ = 1`,
  `a ∈ [−5, 5]`, `b ∈ [−2, 2]`. A negative slope is allowed: the recorded
  failure was an inversion (when p>0.5 the up-rate was ~26%), and shrinking
  toward 0.5 cannot flip a ranking.
- If `n < 30`: `p = 0.5·p_raw + 0.5·π`, where `π` is the expanding base rate
  of those same rows, or 0.5 if there are none.
- Clip to [0.05, 0.95]. The method and `n` are stored on the row.

## Metrics

Scored only on matured cohorts with `date ≥ 2026-11-10` where v2 has an
outcome. v1 is reported on the same dates.

| Head | Model | Baselines |
| --- | --- | --- |
| Direction | Hit rate on non-abstaining calls, 95% Wilson interval. Coverage reported separately. | always-flat and persistence (previous cohort's realized class), same stored flat band. Best = the higher of the two. |
| Probability | Mean Brier, 95% CI upper = mean + 1.96·se. | Constant 0.5 (Brier 0.25) and the expanding base rate (each row scored against the mean of earlier `y` in the sample only). Best = the lower Brier. |
| Level | Hit rate, 95% Wilson interval. | Historical touch rate at the same distance and side, estimated only on daily bars **before** 2026-11-10: fraction of past 5-day windows whose high/low range reached a level that far from the window's reference close. |

A head **beats** its baseline only if:

- direction / level: the Wilson lower bound is strictly above the baseline point estimate;
- probability: the Brier CI upper bound is strictly below the better reference Brier.

## Decision

Do not call v2 better until all of the following hold:

1. At least **40** matured v2 cohorts (distinct dates).
2. Direction has at least **40** scored non-abstaining calls, probability at
   least **40** scored rows, level at least **20** scored rows, and each head
   has a baseline (level has no baseline if the price history is missing).
3. Every head beats its best baseline by the rule above.

Otherwise the decision is **not yet**, including when one head is short of
its minimum. If the minima are met and any head fails its baseline, the
decision is **v2 not better**. Beating v1 is not the criterion: v1 did not
beat always-flat, persistence, a constant 0.5, or the historical touch rate.

`scripts/compare_v1_v2.py` prints these numbers. `--replay` appends an
in-sample application of the same rules to v1 cohorts dated before
2026-11-10, using stored realized returns. That block is labelled indicative
and is ignored by the decision. It will tend to look kinder than the forward
sample, because the rules were chosen after seeing that history.
