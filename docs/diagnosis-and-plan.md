# Liquidity-Adaptive Execution Engine — Diagnosis & Implementation Plan

> **Status**: All 7 phases implemented. Pending runtime verification.
> **Created**: 2026-04-03
> **Last updated**: 2026-04-03 (rev 3: all phases implemented, files-changed table populated)

---

## 1. Problem Statement

The bot identifies high-edge opportunities on Kalshi's KXBTC15M (BTC 15-minute)
contracts but fills only 20–30% of them. The liquidity gate rejects 60–80% of
trades because Kalshi orderbooks frequently show `levels=0` on WebSocket, causing
contracts to be classified EMPTY or DEAD and either stink-bid (low fill chance)
or skipped entirely. Sunday/Monday liquidity gaps make this worse.

**Observed symptoms:**
- Good edges found daily; risk management approves sizing correctly.
- Most 15m contracts have zero or 1–2 WS book levels.
- Stink bids on EMPTY books have very low fill rates (~5–10%).
- No retry mechanism — a contract skipped at t=0 is never revisited even if the
  book populates at t=30s within the same 15m window.

**Goal:** Raise fill rate to 60–80% of identified edges while keeping slippage
<1.5% average and preserving >90% of theoretical edge. Do NOT relax edge
thresholds or risk limits.

---

## 2. Root Cause Diagnosis

Three structural bottlenecks cause the low fill rate:

### Bottleneck A: Rigid DEEP threshold in WS tier classification

- **Where**: `kalshi_ws.py:122-131` — `liquidity_tier` property
- **What**: DEEP requires `total >= 50 AND num_levels >= 4 AND spread <= 5`.
  Everything else is "thin". A book with 25 contracts across 3 levels at 6c
  spread is perfectly tradeable but gets the same treatment as a book with 1
  contract on 1 level.
- **Impact**: THIN tier receives the same passive strategy as DEEP (bid+1 or
  midpoint), which under-fills on wide spreads. No intermediate execution
  strategy exists.

### Bottleneck B: No retry/watchlist for skipped opportunities

- **Where**: `router.py:444-449` (DEAD skip), `router.py:540-549` (no-ask skip)
- **What**: When a contract is skipped for liquidity reasons, it is forgotten.
  The bot never checks if the book populated in a later cycle within the same
  15m window. Kalshi books can go from DEAD to THIN within seconds.
- **Impact**: Transient liquidity gaps cause permanent edge loss. Estimated
  10–15% of skipped opportunities would have been fillable within 30 seconds.

### Bottleneck C: Single-price execution on wide spreads

- **Where**: `kalshi_client.py:836-894` — `_compute_passive_price()`
- **What**: Every passive order posts at a single price (bid+1 for narrow
  spreads, midpoint for wide spreads). On a 10c spread, a midpoint order has
  ~30% fill probability. A 2–3 level ladder across the spread increases fill
  probability to ~50–60% without giving up more edge than the worst level.
- **Impact**: ~25% of passive orders expire without fill when a ladder or
  graduated escalation would have captured the trade.

---

## 3. Current Architecture Summary (verified code references)

### Liquidity Tier Classification

| Tier | Condition | Code |
|---|---|---|
| DEEP | WS: `total >= 50 AND levels >= 4 AND spread <= 5` | `kalshi_ws.py:129` |
| THIN | WS has depth but fails DEEP thresholds | `kalshi_ws.py:131` |
| EMPTY | No WS depth but REST `yes_ask > 0 OR yes_bid > 0` | `router.py:442-443` |
| DEAD | No WS depth AND no REST prices | `router.py:444-449` |

Enum: `models.py:52-56`

### Execution Mode Selection

| Condition | Mode | Code |
|---|---|---|
| <60s left, edge >= 2x min | AGGRESSIVE | `router.py:183-189` |
| <60s left, edge < 2x min | SKIP | `router.py:190-194` |
| 60-180s left, edge >= 1.5x min | MODERATE | `router.py:197-203` |
| 60-180s left, edge < 1.5x min | PASSIVE | `router.py:204` |
| Adverse OBI + weak edge | SKIP | `router.py:208-227` |
| Narrow spread <3c | MODERATE | `router.py:230-231` |
| Default | PASSIVE | `router.py:234-238` |

### Execution Paths

| Tier + Mode | Function | Code |
|---|---|---|
| EMPTY (any) | `execute_stink_bid()` | `kalshi_client.py:1018-1105` |
| DEEP/THIN + PASSIVE/MODERATE | `execute_passive_limit()` | `kalshi_client.py:896-1012` |
| DEEP + AGGRESSIVE | `_execute_deep_book()` | `kalshi_client.py:1230-1292` |
| Any + SKIP | No trade | `kalshi_client.py:1219-1221` |

Unified dispatch: `execute_chosen_contract()` at `kalshi_client.py:1193-1228`

### Passive Price Computation

| Spread | PASSIVE price | MODERATE price | Code |
|---|---|---|---|
| ≤3c | — | Take ask | `kalshi_client.py:864-865` |
| ≤5c (PASSIVE) | bid+1 | — | `kalshi_client.py:872` |
| ≤8c (MODERATE) | — | mid | `kalshi_client.py:867` |
| >5c (PASSIVE) | midpoint | — | `kalshi_client.py:874` |
| >8c (MODERATE) | — | ask-1 | `kalshi_client.py:869` |

Edge floor enforced at `kalshi_client.py:879-892`.

### Resting Order Management

- Max resting orders: `MAX_RESTING_ORDERS = 5` at `kalshi_client.py:77`
- Rotation: new edge > worst edge + 0.5% triggers cancel-and-replace
  (`kalshi_client.py:916-936`)
- Patience: PASSIVE = 15 cycles, MODERATE = 6 cycles, STINK = 15 cycles
- Reevaluation: `reevaluate_resting_orders()` at `kalshi_client.py:1504-1558`
- Escalation: cancel passive + take ask if <90s AND edge >= 1.5x min
  (`kalshi_client.py:1560-1649`)

### Risk / Sizing

- Kelly: `f* = (p_model - kalshi_ask) / (1 - kalshi_ask)`, scaled by 0.25
  (`risk.py:370-377`)
- Horizon cap: 3% for 15m (`risk.py:380-383`)
- Tier multipliers: EMPTY = 0.50x, THIN = 0.75x, DEEP = 1.0x
  (`risk.py:412-420`)
- Resting order risk cap: 20% of equity (`risk.py:263-269`)
- Resting order count cap: 5 (`risk.py:260`)
- Daily drawdown pre-emptive halt for stinks at 8% (`risk.py:254-257`)

### Main Loop

- 10-second cycle interval (`main.py` `LOOP_INTERVAL_SECONDS`)
- Cycle: reconcile → manage pending → fetch state → fetch markets → signal →
  route → reevaluate resting → execute immediate → place stinks → sync equity
  (`main.py:210-366`)

### Contract Selection Scoring

- `max(immediate, key=lambda c: c.edge - 0.5 * c.spread)` at `router.py:346`
- Stink candidates: top 5 by edge (`router.py:357-358`)

---

## 4. Constraints & Assumptions

### Hard constraints (must NOT change)
- `min_edge_threshold = 0.08` (8%) — every order must independently satisfy this
- `kelly_fraction = 0.25` (quarter-Kelly)
- `max_daily_drawdown_pct = 0.10` (10% circuit breaker)
- `max_open_positions = 3`
- No market orders (limit only)
- No crossing the spread (passive posting, never pay full ask except AGGRESSIVE
  mode which is explicitly edge-gated)

### Soft constraints (can be tuned)
- `MAX_RESTING_ORDERS = 5` — can increase to 7–8
- Resting risk cap 20% — can increase to 25%
- Tier multipliers — can add new tiers
- Patience cycles — can add graduated escalation
- DEEP thresholds (50 contracts, 4 levels, 5c spread) — can make dynamic
- Stink bid rate limit (2 min between bids on same event) — can tune

### Assumptions
- Kalshi API rate limit is ~100 req/min; must not exceed ~80 req/min
- Cancel-and-replace (reprice) counts as 2 API calls (cancel + submit)
- Books can transition DEAD → THIN within 10–30 seconds (observed behavior)
- Sunday/Monday 02:00–12:00 UTC is the lowest-liquidity period
- The 15m window has multiple strikes; adjacent strikes are correlated

---

## 5. Implementation Plan — Phased

### Phase 1: Dynamic Depth Thresholds + Sub-Tier Classification

**Goal**: Make the DEEP/THIN boundary adaptive to time-of-day, day-of-week, and
volatility. Split THIN into THIN_TRADEABLE and THIN_MARGINAL with distinct
execution strategies.

**Files to modify:**
- `models.py:52-56` — Add `THIN_TRADEABLE`, `THIN_MARGINAL` to LiquidityTier
- `kalshi_ws.py:122-131` — Accept dynamic thresholds in `liquidity_tier`
- `router.py:434-451` — Apply dynamic thresholds at tier classification
- `router.py:159-238` — Execution mode logic for new sub-tiers
- `risk.py:412-420` — Tier multipliers for new sub-tiers
- `kalshi_client.py:1193-1228` — Dispatch for new tiers

**Dynamic threshold logic:**
```
time_mult:
  02-08 UTC      → 0.50 (night)
  08-14 UTC      → 0.75 (morning ramp)
  14-22 UTC      → 1.00 (peak)
  22-02 UTC      → 0.85 (evening)

day_mult:
  Sunday (6)     → 0.60
  Monday (0)     → 0.80
  Saturday (5)   → 0.80
  Tue-Fri (1-4)  → 1.00

vol_mult:
  vol_1h > 0.015 → 0.60 (high vol sweeps books; lower bar)
  vol_1h > 0.008 → 0.80
  else           → 1.00

combined = max(0.30, time_mult * day_mult * vol_mult)

DEEP threshold:
  min_contracts = max(5, int(50 * combined))
  min_levels    = max(2, int(4 * combined))
  max_spread    = max(3, int(5 / combined))  # widens inversely
```

**New sub-tier assignments:**
- THIN_TRADEABLE: has WS depth, fails DEEP, but `total >= 10 AND levels >= 2 AND spread <= 10`
- THIN_MARGINAL: has WS depth, fails THIN_TRADEABLE

**New tier multipliers:**
- THIN_TRADEABLE: 0.75x (same as old THIN)
- THIN_MARGINAL: 0.50x

**Execution strategy per new tier:**

THIN_TRADEABLE — **Midpoint Sniper**:
- Post at `(best_ask + best_bid) // 2`, floored by edge constraint
- Shortened patience: 4 cycles (40s) instead of 15 cycles (PASSIVE)
- After patience expires, eligible for graduated escalation (Phase 3)
- Rationale: wide spreads mean bid+1 is too far from the action; midpoint
  balances fill probability vs edge preservation

THIN_MARGINAL — **Conservative limit, no aggression**:
- Post at `bid + 1` only (no midpoint — too risky on 1-level books)
- Patience: 6 cycles (60s), then cancel (no escalation to aggressive)
- Never route to AGGRESSIVE mode regardless of time remaining
- Rationale: 1-level books with <10 contracts have high adverse selection risk;
  aggressive takes could face immediate slippage

**Expected fill rate improvement**: +15–20%

### Phase 2: Deferred Watchlist + Book-Watch Retry

**Goal**: Track DEAD/EMPTY/THIN_MARGINAL contracts that had edge >= min_edge and
revisit them each cycle for up to 5 cycles (50 seconds).

**Files to modify:**
- `main.py:210-366` — Add watchlist instance; insert retry step between routing
  and execution (after line 281)
- `router.py:240-373` — Return skip reasons alongside None results for
  watchlist categorization; add `watchlist_additions` to return tuple
- New class `DeferredWatchlist` — can be added to `router.py` or a new
  `watchlist.py`

**Watchlist mechanics:**
- Max 10 entries, evict lowest-edge when full
- Each entry: ticker, side, model_p, edge, close_time, skip_reason, retry_count
- Purge when close_time < now + 30s (not enough time to trade)
- On retry: fetch WS orderbook; if `has_depth` is now True, reconstruct
  ChosenContract and route through normal execution
- Remove from watchlist on successful trade or expiry

**Expected fill rate improvement**: +8–12%

### Phase 3: Graduated Escalation Pipeline

**Goal**: Replace the binary hold/escalate logic with a 4-phase pipeline that
improves price incrementally as expiry approaches.

**Files to modify:**
- `kalshi_client.py:1460-1503` — `_reevaluate_resting_order()` — replace
  binary escalation with graduated phases
- `kalshi_client.py:1504-1558` — `reevaluate_resting_orders()` — handle new
  "improve" action
- New method: `_improve_resting_order()` (cancel + repost at new price)

**Escalation phases:**
```
>180s remaining  → HOLD (no change)
90-180s          → IMPROVE +1 tick (if edge still >= min_edge at new price)
45-90s           → IMPROVE to midpoint (if edge still >= min_edge at mid)
<45s             → ESCALATE (take ask if edge >= 0.75 * min_edge)

At any phase: CANCEL if current edge < min_edge
```

**Price improvement = cancel-and-replace:**
- Cancel old order → free resting risk
- Submit new order at improved price → re-register resting risk
- Preserve `cycles_alive` for timeout tracking
- Each improvement costs 2 API calls; budget max 2 improvements per order

**Sub-strategy: Bid Pegging (THIN_TRADEABLE / MODERATE mode)**

When the best bid on a THIN book moves UP by ≥2c since our order was placed,
reprice to `new_bid + 1` to maintain competitive queue position at a better
price. Never chase downward — that is adverse selection.

```
_should_repeg(pending) -> int | None:
  current_bid = fetch_best_bid(pending.ticker, pending.side)
  reference_bid = pending.metadata["reference_bid"]  # bid at placement time

  if current_bid >= reference_bid + 2:
      new_price = current_bid + 1
      if model_p - (new_price / 100) >= min_edge:
          return new_price   # reprice up
  return None  # hold or bid moved down — don't chase
```

- Repeg counts as 1 improvement toward the per-order cap of 2
- Only applies to THIN_TRADEABLE and MODERATE-mode orders (not DEEP passive
  which should stay at bid+1 queue position)
- Requires tracking `reference_bid` in PendingOrder metadata

**Files to modify (in addition to above):**
- `kalshi_client.py` — Add `_should_repeg()` method
- `kalshi_client.py:742-825` — Call `_should_repeg()` during
  `manage_pending_orders()` before patience check
- `models.py` — Add `metadata` field to PendingOrder if not present (verify)

**Expected fill rate improvement**: +8–12% (escalation) + ~3–5% (bid pegging)

### Phase 4: Limit Ladder (DEEP tier)

**Goal**: When sizing yields >1 contract on a DEEP book with spread > 3c, split
across 2–3 price levels to increase fill probability.

**Files to modify:**
- `kalshi_client.py` — New method `execute_limit_ladder()`
- `kalshi_client.py:1193-1228` — Route DEEP contracts to ladder when
  spread > 3c and contracts > 1
- `kalshi_client.py:77` — Increase `MAX_RESTING_ORDERS` from 5 to 7

**Ladder levels:**
```
Spread ≤ 3c → single order (no room for ladder)
Spread 4-6c → 2 levels:
  Level 1 (60%): bid+2 or mid-1
  Level 2 (40%): bid+1
Spread > 6c → 3 levels:
  Level 1 (40%): midpoint
  Level 2 (35%): bid+1
  Level 3 (25%): bid
```
Every level independently satisfies `min_edge_threshold`. Levels that would
violate the edge floor are dropped.

**Risk constraints:**
- Total contract count across all levels = same as single-order count
- Each level counts as 1 resting order (need increased MAX_RESTING_ORDERS)
- Increase `MAX_RESTING_ORDERS` to 7 to accommodate ladders
- Increase resting risk cap from 20% to 25% in `risk.py:265`

**Expected fill rate improvement**: +5–8%

### Phase 5: Cross-Strike Liquidity Routing

**Goal**: When the best-edge contract is EMPTY/DEAD, check if an adjacent strike
in the same 15m window has depth and sufficient edge.

**Files to modify:**
- `router.py:299-313` — Add pre-pass grouping markets by close_time
- `router.py:344-346` — Modify scoring: `edge * sqrt(depth_score) - 0.3 * spread`

**Logic:**
- Group markets by `close_time` (same 15m window)
- Within each group, rank by liquidity tier
- If best-edge contract is EMPTY/DEAD but a sibling with DEEP/THIN tier has
  `edge >= min_edge`, prefer the liquid sibling
- Fallback: if no liquid sibling, proceed with stink bid on original

**Expected fill rate improvement**: +3–5%

### Phase 6: Dual-Level Stink Bids

**Goal**: Place 2 orders at different prices on EMPTY books instead of 1.

**Files to modify:**
- `kalshi_client.py:1018-1105` — Modify `execute_stink_bid()` to optionally
  place a second level
- `risk.py:260` — Update resting count cap to match `MAX_RESTING_ORDERS = 7`

**Levels:**
```
Level 1 (60% of contracts): model_p - 2 * min_edge_cents  (aggressive)
Level 2 (40% of contracts): model_p - 3 * min_edge_cents  (deep value)
```
If price difference < 1c (not enough room), fall back to single order.

**Adaptive edge buffer for stink pricing:**

Stink bids on truly empty books need MORE edge buffer than books that are merely
shallow, because adverse selection risk is higher (the only counterparty arriving
on an empty book likely knows something).

```
buffer_mult = 2.5 - (depth_quality * 2.5)   # 2.5x at quality=0, 0x at quality=1
buffer_mult = max(1.5, buffer_mult)           # floor at 1.5x

where depth_quality:
  EMPTY (no WS levels)    → 0.0  → buffer = 2.5 * min_edge
  SHALLOW (1 level, <5)   → 0.2  → buffer = 2.0 * min_edge
  THIN_MARGINAL           → 0.4  → buffer = 1.5 * min_edge (floor)

stink_price = model_p_cents - int(min_edge_cents * buffer_mult)
```

This replaces the fixed 2x/3x for Level 1/Level 2. Level 1 uses `buffer_mult`,
Level 2 uses `buffer_mult + 0.5` (deeper value).

**Expected fill rate improvement**: +2–3%

### Phase 7: Kill Switches & Safety

**Goal**: Add execution-level safety beyond existing risk management.

**New kill switches:**
1. **API rate guard**: If >80 calls/minute, skip non-essential operations
2. **Market-offline detector**: If >90% of contracts were DEAD/EMPTY for 5
   consecutive cycles, pause cycle frequency to 30s
3. **Fill rate collapse**: If last 10 submitted orders had 0 fills, log warning
   and switch to stink-only mode for 10 cycles
4. **Spread blowout**: If median spread across active contracts > 20c for 3
   cycles, go stink-only

**Files to modify:**
- `kalshi_client.py` — Add tracking counters (API calls, fill rate, spread)
- `main.py` — Check kill switches before execution steps

---

## 6. Risks & Edge Cases

| Risk | Severity | Mitigation |
|---|---|---|
| Cancel-and-replace race condition (cancel succeeds, resubmit fails) | High | Clean up router registration on resubmit failure; notify risk to free resting exposure |
| Ladder orders partially fill, rest cancelled — position smaller than expected | Medium | Each ladder level is independently sized; partial fill is acceptable |
| Dynamic thresholds classify garbage books as THIN_TRADEABLE | Medium | Edge floor remains hard constraint; bad books still fail edge check |
| Watchlist causes stale model_p to drive trades | Medium | Re-run `evaluate_contract()` on retry, don't use cached model_p |
| Increased resting orders (7 vs 5) + dual stinks burn API rate budget | Low | Track API calls per cycle; worst case = 14 calls/cycle for order management |
| Escalation pipeline burns 2 API calls per improvement | Low | Cap at 2 improvements per order lifetime |
| Cross-strike routing misses when all strikes in a window are EMPTY | Low | Graceful fallback to stink bids (no worse than current behavior) |

---

## 7. Verification Checklist

### Per-Phase Verification

- [ ] **Phase 1**: Run bot for 1 hour during peak hours (14-22 UTC weekday).
  Count tier classifications. THIN_TRADEABLE should appear for 20-40% of
  contracts that were previously THIN.
- [ ] **Phase 1**: Run bot for 1 hour during off-peak (02-08 UTC). Verify
  dynamic thresholds lower the DEEP bar. Some contracts that were THIN should
  now classify as DEEP.
- [ ] **Phase 2**: Monitor watchlist additions per cycle. Expect 1-3 entries
  per cycle when liquidity is thin. Verify entries are purged on expiry.
- [ ] **Phase 2**: Count watchlist-driven trades. Expect >0 per hour during
  active market.
- [ ] **Phase 3**: Log each escalation phase. Verify no order is improved more
  than 2 times. Verify edge floor is never violated.
- [ ] **Phase 4**: On DEEP books with spread > 3c, verify ladder levels are
  placed. Count fill rate per level. Level 1 (aggressive) should fill most
  often.
- [ ] **Phase 5**: Log cross-strike fallback selections. Verify the liquid
  sibling has edge >= min_edge.
- [ ] **Phase 6**: On EMPTY books, verify two orders placed at different
  prices. Neither should violate edge floor.
- [ ] **Phase 7**: Simulate API rate spike (increase cycle frequency
  temporarily). Verify rate guard triggers and pauses.

### Per-Tier Slippage Budget (targets)

| Tier / Strategy | Expected Slippage | Rationale |
|---|---|---|
| DEEP aggressive (take ask) | 0.0% | Taking posted price |
| DEEP passive (bid+1, ladder) | 0.5% | Spread cost absorbed by posting |
| THIN_TRADEABLE midpoint snipe | 0.5–1.0% | Half-spread cost |
| THIN_TRADEABLE escalation fill | 1.0–1.5% | Gave back 1–3c via escalation |
| THIN_MARGINAL passive | 0.5% | Conservative posting, no aggression |
| EMPTY stink bid fill | 0.0% | You set the price |
| **Weighted average** | **<1.2%** | Well under 1.5% target |

### Per-Strategy Edge Preservation (targets)

| Strategy | Edge Capture (% of theoretical) |
|---|---|
| Aggressive take at ask | 95–100% |
| Passive fill at bid+1 | 100–105% (better than ask) |
| Midpoint snipe fill | 90–100% (half spread saved vs ask) |
| Escalation fill at mid | 85–95% (gave back 1–2c) |
| Stink bid fill | 100–120% (priced conservatively) |
| **Weighted average** | **~93%** |

### End-to-End Success Criteria

- [ ] Fill rate: 60–80% of identified edges (measured over 24 hours)
- [ ] Slippage: <1.5% average across all fills
- [ ] Edge capture: >90% of theoretical edge preserved
- [ ] No increase in daily drawdown frequency
- [ ] No API rate limit violations
- [ ] Kill switches trigger correctly under simulated conditions

---

## 8. Expected Fill Rate Improvement Summary

| Phase | Source of Improvement | Est. Improvement | Cumulative |
|---|---|---|---|
| Phase 1 | Dynamic thresholds reclassify THIN → DEEP | +15–20% | 35–50% |
| Phase 2 | Watchlist retries catch refilled books | +8–12% | 45–60% |
| Phase 3 | Graduated escalation + bid pegging | +10–15% | 55–70% |
| Phase 4 | Limit ladders on DEEP books with wide spread | +5–8% | 60–75% |
| Phase 5 | Cross-strike fallback to liquid siblings | +3–5% | 63–78% |
| Phase 6 | Dual stink bids on EMPTY books | +2–3% | 65–80% |
| Phase 7 | Kill switches (safety, no fill rate gain) | 0% | 65–80% |

Baseline: 20–30% fill rate.

---

## 9. Decision Log

| Date | Decision | Rationale |
|---|---|---|
| 2026-04-03 | Split THIN into THIN_TRADEABLE and THIN_MARGINAL instead of using a continuous depth score | Discrete tiers are easier to reason about, debug, and log. Continuous scoring adds complexity without clear benefit given the small number of execution strategies. |
| 2026-04-03 | Increase MAX_RESTING_ORDERS from 5 to 7, not higher | 7 accommodates 1 ladder (3 levels) + 2 stinks + 2 passives. Going to 8+ increases order management API load and complicates rotation logic. |
| 2026-04-03 | Cap escalation improvements at 2 per order | Each improvement costs 2 API calls (cancel + submit). 3+ improvements would burn 6+ calls for a single position, straining rate budget. |
| 2026-04-03 | Phase implementation order prioritizes dynamic thresholds over ladder | Dynamic thresholds have the highest expected impact with the lowest code change footprint. Ladder requires MAX_RESTING_ORDERS increase which has broader implications. |
| 2026-04-03 | Watchlist re-evaluates model_p on retry rather than using cached value | Stale model_p from 30+ seconds ago may reflect a price that has moved. Re-running `evaluate_contract()` ensures edge is current. |
| 2026-04-03 | Resting risk cap increased from 20% to 25% | With dual stinks and ladders, 20% cap would frequently block orders. Stink bids have asymmetric risk (max loss = cost, but you set the price) so higher exposure is defensible. |
| 2026-04-03 | THIN_TRADEABLE uses midpoint sniper with 4-cycle patience (not bid+1) | bid+1 on a 10c spread is 9c from the ask — almost never fills. Midpoint is the sweet spot: half the spread cost, 2–3x fill probability. Shortened patience (40s vs 150s) matches the urgency of trading wide-spread books. |
| 2026-04-03 | THIN_MARGINAL never escalates to AGGRESSIVE | 1-level books with <10 contracts have high adverse selection risk. An aggressive take on such a book may face immediate slippage or be the only liquidity event (toxic fill). Better to post and accept lower fill rate than blow up edge. |
| 2026-04-03 | Bid pegging only on THIN_TRADEABLE + MODERATE, not DEEP | DEEP books have tight spreads where bid+1 already has good queue position. Pegging on DEEP wastes API calls for marginal benefit. THIN books with moving bids are where pegging captures real improvement. |
| 2026-04-03 | Adaptive cycle timing deferred (not included in phases) | Reducing cycle interval from 10s to 5s near expiry would help but adds complexity to the main loop and doubles API call rate. Better to implement as a follow-up after Phases 1–4 prove out the core execution improvements. |

---

## 10. Files Changed (to be updated during implementation)

| Phase | File | Summary |
|---|---|---|
| 1 | `models.py` | Replaced `THIN` with `THIN_TRADEABLE` + `THIN_MARGINAL` enums; added `_missing_("thin")` compat + `is_thin` property; added `metadata` field to `PendingOrder` |
| 1 | `kalshi_ws.py` | Replaced `liquidity_tier` property with `classify_liquidity()` accepting dynamic thresholds; returns "thin_tradeable"/"thin_marginal" strings |
| 1 | `router.py` | Added `_dynamic_depth_thresholds()` (time/day/vol multipliers); rewrote tier classification in `_evaluate_contract` to use dynamic thresholds + sub-tiers |
| 1 | `router.py` | Updated `determine_execution_mode()`: THIN_MARGINAL never AGGRESSIVE, THIN_TRADEABLE defaults to MODERATE (midpoint sniper) |
| 1 | `risk.py` | Updated tier multipliers: THIN_TRADEABLE=0.75x, THIN_MARGINAL=0.50x |
| 1 | `kalshi_client.py` | Tier-aware patience in `manage_pending_orders()`: THIN_TRADEABLE moderate=4 cycles, THIN_MARGINAL passive=6 cycles |
| 2 | `models.py` | Added `WatchlistEntry` dataclass and `DeferredWatchlist` class (max 10 entries, edge-priority eviction) |
| 2 | `router.py` | `route()` returns 3-tuple `(best_immediate, stink_candidates, watchlist_entries)`; `_evaluate_contract` populates watchlist for DEAD contracts with signal edge |
| 2 | `main.py` | Created `DeferredWatchlist` at startup; feeds DEAD skips into watchlist; retries candidates before execution step (re-evaluates with fresh signal) |
| 3 | `kalshi_client.py` | Replaced binary hold/escalate in `_reevaluate_resting_order()` with 4-phase pipeline (hold→improve+1tick→improve-to-mid→escalate); THIN_MARGINAL blocked from aggressive |
| 3 | `kalshi_client.py` | Added `_compute_improvement_price()` and `_improve_resting_order()` (cancel-and-replace, preserves cycles_alive, max 2 improvements per order) |
| 3 | `kalshi_client.py` | Added `_should_repeg()` for bid pegging on THIN_TRADEABLE/MODERATE orders; integrated into `manage_pending_orders()` |
| 3 | `kalshi_client.py` | `execute_passive_limit()` now stores `reference_bid` and `improvements` counter in PendingOrder metadata |
| 4 | `kalshi_client.py` | Added `execute_limit_ladder()`: 2-3 level ladder on DEEP books with spread>3c; integrated into `execute_chosen_contract()` dispatch |
| 4 | `kalshi_client.py` | `MAX_RESTING_ORDERS` increased 5→7 |
| 4 | `risk.py` | Resting order count cap 5→7; resting risk cap 20%→25% |
| 5 | `router.py` | Added cross-strike fallback: when no immediate candidates, finds liquid sibling in same expiry window for EMPTY contracts |
| 5 | `router.py` | Updated scoring formula: `edge * (tier_weight^0.5) - 0.3*spread` (liquidity-weighted) |
| 6 | `kalshi_client.py` | Rewrote `execute_stink_bid()` with dual-level stink bids (L1 at 2x min_edge buffer, L2 at 2.5x buffer); added `_compute_stink_price()` |
| 7 | `kalshi_client.py` | Added `ExecutionGuard` class: tracks submissions/fills, consecutive unfilled, dead/empty ratio, median spreads; `is_stink_only` degraded mode |
| 7 | `kalshi_client.py` | Guard integrated: `record_submission()` in `_submit_order()`, `record_fill()` in `manage_pending_orders()`, `is_stink_only` check in `execute_chosen_contract()` |
| 7 | `main.py` | Records per-cycle liquidity metrics (dead/empty %, median spread) into `kalshi.guard` |
| ALL | `test_audit_fixes.py` | Updated all `LiquidityTier.THIN` refs → `THIN_TRADEABLE`; updated tier assertion strings |

---

## 11. Deferred Enhancements (not in current plan)

| Enhancement | Reason Deferred |
|---|---|
| Adaptive cycle timing (5s near expiry) | Doubles API rate; implement after core phases prove out |
| TWAP (time-weighted average price) execution | Kalshi's 15m windows are too short for meaningful TWAP; limit ladder achieves similar result |
| Historical fill rate tracking by hour/day | Useful for tuning dynamic thresholds but requires data collection period first |
| Slippage tracker with automatic mode downgrade | Needs fill-price vs model-price tracking infrastructure; add after Phase 7 |

## 12. Remaining Caveats (to be updated post-implementation)

_This section will capture any limitations, known issues, or follow-up work
discovered during implementation._
