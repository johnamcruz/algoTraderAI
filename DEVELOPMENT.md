# Development Workflow

This project follows a **test-driven + data-driven** discipline. Every behavioural change goes through two gates before it ships: unit tests that prove the code does what it claims, and backtest comparison runs that prove it actually improves (or at minimum doesn't hurt) real trading outcomes.

---

## The Two-Gate Rule

```
Idea → Unit Tests → Backtest Comparison → Decision → Commit
```

Neither gate alone is sufficient:

- **Unit tests without data** — code is correct but the feature may hurt live performance (e.g. breakeven-on-1R passed all tests but dropped win rate 33–47% across regimes).
- **Backtests without unit tests** — results look right but edge-case behaviour is untested and the pre-commit hook won't catch regressions.

Both gates must pass before a change goes to `main`.

---

## Unit Tests

Run the full suite before every commit:

```bash
python -m pytest tests/
```

The pre-commit hook enforces this automatically — a failing test aborts the commit.

### Test files

| File | What it covers |
|---|---|
| `tests/test_trading_bot_base.py` | Breakeven logic, position state, stop checks |
| `tests/test_realtime_bot.py` | Live order placement, `_modify_order`, bracket logic |
| `tests/test_backtest.py` | `backtest.py` command builder, scenario coverage |
| `tests/test_smoke.py` | CISD OTE strategy warmup and feature computation |

### Guidelines

- Mock at the boundary (`requests.post`, broker API calls). Never mock internal logic.
- One assertion per test. Test name should read as a sentence describing the expected behaviour.
- When a live API behaviour is confirmed (e.g. `Order/modify` preserves OCO linkage), add a unit test that documents the contract even if the API itself isn't called in the test.

---

## Backtest Comparison Runs

Use `backtest.py` to run predefined market-regime scenarios:

```bash
# Run all fast scenarios (sequential)
python backtest.py

# Run specific scenarios in parallel
python backtest.py --scenario banking_2023 --parallel
python backtest.py --scenario oos_2021 --min_vty_regime 0.90 --parallel

# Side-by-side comparison: vary one parameter, hold everything else constant
python backtest.py --scenario banking_2023 --entry_conf 0.70 --parallel &
python backtest.py --scenario banking_2023 --entry_conf 0.75 --parallel &
wait
```

### Scenario suite

| Key | Period | Purpose |
|---|---|---|
| `banking_2023` | Mar–May 2023 | High-vol chop; SVB collapse |
| `selloff_2024` | Jul–Sep 2024 | Sharp selloff + recovery |
| `recent_120d` | Jun–Oct 2025 | Most recent 120-day window |
| `recovery_2023` | Full year 2023 | Post-rate-hike uptrend (slow) |
| `oos_2021` | Full year 2021 | Out-of-sample low-vol uptrend (slow) |
| `bear_2022` | Jan–Oct 2022 | Persistent bear market (slowest) |

Run the three fast scenarios for quick iteration. Add `oos_2021` when the change specifically targets low-vol regime filtering. Run the full suite before changing a default.

### Decision criteria

- A parameter change is **accepted** if it improves win rate without meaningfully reducing trade count or net P&L across the majority of scenarios.
- A parameter change is **rejected** if it filters winners alongside losers, or if the win rate gain doesn't offset the opportunity cost.
- A feature is **disabled by default** if comparison data shows it hurts the majority of scenarios — but the flag is kept for future experimentation.

---

## What We've Learned (Running Log)

### Breakeven-on-1R (April 2026)

Implemented `_modify_order()` to atomically move the bracket stop to entry once 1R profit is reached. API confirmed working on OCO brackets.

**Comparison result:** Disabled by default after data showed 33–47% win rate drop across all tested regimes. Root cause: price regularly touches 1R then retraces to entry before continuing to target — the stop converts winners into scratch trades.

**Decision:** Keep the `--breakeven` flag for future tuning (candidate thresholds: 1.5R, 2R). Do not enable by default until a regime-specific improvement is demonstrated.

### Entry confidence 0.70 → 0.75 (April 2026)

Motivated by feature analysis showing winners average prob=0.87 vs losers 0.75 — right at the threshold.

**Comparison result:** Rejected. Raised win rate to 100% in Banking Crisis but cut 9 trades including 7 winners, reducing net P&L by ~$2k. Signal volume is already limited — filtering winners is too costly.

**Decision:** Stay at 0.70.

### Volatility regime gate 0.75 → 0.90 (April 2026)

Motivated by 2021 OOS disaster (26.7% win rate, MLL hit). Winners average vty_regime=1.54 vs losers 0.80.

**Comparison result:** Rejected. Barely improved 2021 (still MLL hit at 28%) while cutting P&L in every profitable scenario. The 2021 problem is a fundamental model-fit issue in sustained low-vol uptrends, not a vol filter problem.

**Decision:** Stay at 0.75. The 2021 regime requires a different approach (e.g. htf trend gate, model retrain on that regime).

---

## Parameter Reference (Current Defaults)

| Parameter | Default | Flag |
|---|---|---|
| `entry_conf` | 0.70 | `--entry_conf` |
| `min_vty_regime` | 0.75 | `--min_vty_regime` |
| `min_entry_distance` | 3.0 | `--min_entry_distance` |
| `min_stop_atr` | 0.5 | `--min_stop_atr` |
| `min_stop_pts` | 1.0 | `--min_stop_pts` |
| `breakeven_on_1r` | False | `--breakeven` |
| `high_conf_multiplier` | 2.0 | `--high_conf_multiplier` |
| `max_contracts` | 5 | `--max_contracts` |

---

## Commit Rules

- All tests must pass before committing.
- Never commit API keys or credentials (e.g. `test_breakeven.py`).
- Parameter default changes require comparison backtest data in the commit message.
- Commit as the user — no co-author credits.
