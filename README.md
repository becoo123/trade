# Hot Money Trading Models

Hot Money Trading Models is a Python research repository for studying short-term A-share market behavior, especially sector rotation, limit-up effects, strong-stock pullbacks, first-board continuation, and speculative leader identification. The project is built around a practical question: when liquidity, theme narratives, and daily limit-up behavior concentrate in a small group of stocks, can we describe that concentration with repeatable data features and test whether those features lead to tradable next-day returns?

This repository is not a black-box prediction engine and it is not a live trading system. It is a research toolkit for converting local market data into structured factors, running event-driven backtests, comparing different short-term patterns, and exporting results as CSV tables and performance charts. Most scripts are intentionally direct and readable so that individual assumptions can be inspected, changed, and retested.

## Core Idea

The project focuses on the "hot money" style often seen in China's A-share market: capital rotates quickly across themes, industries, and high-attention stocks. Instead of only looking at index-level returns, the code measures whether money-making effects are broad, persistent, and supported by actual participation.

Several principles appear throughout the repository:

- Sector strength should not be measured only by price movement. Breadth, limit-up density, turnover, and continuation matter.
- Limit-up events are useful sentiment signals, but they must be separated into different patterns such as sealed boards, broken boards, first boards, repeated boards, and low-participation one-price limit-up stocks.
- Short-term strategies should avoid future leakage. Signal features should use information available at the signal date, while forward-looking return fields should be used only for evaluation.
- A good backtest should report not only average return, but also win rate, drawdown, trade count, daily equity curve, and sensitivity to filters.

## Data Layer

The data interface is centered on `StockDataManager` in `data_core.py`. It loads local daily and minute-level data from Parquet files and provides a unified API for stock-level and sector-level queries.

Expected local data files are described in detail in `DATA_MANUAL.md`:

- `data/daydata.parquet`: daily bars, prices, amount, volume, limit-up flag, limit price, and return fields.
- `data/mindata.parquet`: minute bars for intraday analysis.
- `data/stocks.parquet`: stock universe metadata.
- `data/codes.csv`: code list.
- `data/blocks`: sector, industry, concept, and theme membership data.

The default raw-data source is Tongdaxin, controlled by the `TDX_PATH` environment variable. The preprocessing code supports Shanghai and Shenzhen daily/minute files and writes normalized Parquet outputs for fast research queries. The configured stock universe currently covers codes starting with `0`, `3`, and `6`.

The project supports several block classification systems through `BlockManager` in `BK.py`, including:

- Shenwan level-1 and level-2 industries.
- CSRC level-1 and level-2 industries.
- Concept blocks.
- Theme concept blocks.

This makes it possible to compare which classification better captures real capital flow for a given strategy. For example, a short-term momentum strategy may work better on concept blocks, while broader trend analysis may be more stable on industry blocks.

## Strategy Modules

The repository contains multiple research scripts. They can be used independently or as building blocks for new experiments.

`block.py` implements a sector money-making effect analyzer. It calculates limit-up count, touched-limit count, broken-board rate, next-day performance of limit-up stocks, sector turnover, turnover concentration, relative amount strength, and top contributing stocks. This is useful for answering whether a sector is truly active or only lifted by one or two leaders.

`BlockEffect.py` and `BlockEffectSim.py` explore a broader sector-effect scoring framework. The underlying idea is to combine price strength, participation breadth, limit-up density, capital activity, and continuation quality into a ranking system. The generated outputs such as `block_strategy_summary.csv`, `block_strategy_signals.csv`, and `block_strategy_returns.png` are examples of this workflow.

`hot_leader.py` implements a hot-sector leader strategy. It first ranks sectors by abnormal turnover and new-high breadth, then ranks stocks inside the strongest sectors by participation, price position, and freshness of capital activity. The basic backtest timing is T-day close selection, T+1 open entry, and T+2 open exit.

`dip_buy_backtest.py` studies a strong-stock low-volume pullback setup. It looks for stocks with an existing uptrend, recent limit-up history, controlled recent gains, and a moderate pullback on shrinking volume. The goal is to test whether a pullback after confirmed strength offers better next-day open-to-open returns than chasing strength directly.

`gem_pullback_backtest.py` tests a first-board pullback/reversal structure. It identifies recent first-board stocks, waits for several days of low-volume adjustment, then looks for a reversal signal before simulating a next-open entry and following-open exit. Despite the filename, the current version is written as an all-A-share first-board pullback backtest with configurable sector filters.

`DragonStock.py` ranks potential market leaders over a selected period. It emphasizes tradable leadership instead of simply counting limit-up days. The score includes limit-up frequency, next-day return behavior, consecutive-board strength, average amount on limit-up days, and next-day amplitude. This helps reduce the ranking of low-participation one-price limit-up stocks that are difficult to trade.

`stock_to_block.py` and `stock_to_block_backtest.py` analyze the relationship between individual stock signals and their sector background. They are useful when testing whether a stock-level pattern improves after adding hot-sector filters.

`dip_buy_select.py` is a selection-oriented script for applying the low-pullback logic to recent data and producing candidate targets.

## Outputs

The repository includes several generated result files, including backtest charts and trade logs:

- `*_backtest.png`: equity curves or strategy visualizations.
- `*_trades.csv`: individual trade records.
- `block_strategy_*.csv`: sector strategy summaries and signals.
- `today_targets.csv`: example daily target output.
- `param_sweep_results.csv`: parameter sweep results.

These files are useful examples for understanding the intended workflow. When publishing or extending the project, consider whether large generated CSV files should remain in the repository or be moved to releases, artifacts, or ignored by Git.

## Quick Start

Install the main Python dependencies:

```bash
pip install pandas numpy pyarrow matplotlib scipy tqdm numba
```

Prepare local data according to `DATA_MANUAL.md`. At minimum, the daily Parquet file is required for most strategies:

```text
data/daydata.parquet
data/stocks.parquet or data/codes.csv
data/blocks/
```

Optionally set the Tongdaxin source path before preprocessing:

```powershell
$env:TDX_PATH = "C:\new_tdx\vipdoc"
```

Check whether the processed data files can be found:

```bash
python data_core.py
```

Run a strategy script directly, or import the relevant class in a notebook or another Python file:

```python
from hot_leader import HotLeaderStrategy

strategy = HotLeaderStrategy()
signals = strategy.scan("2025-01-10")
print(signals)
```

Some scripts expose backtest classes rather than command-line interfaces. In that case, instantiate the class and call `run_backtest()` with your own date range and parameters.

## Methodology

Most backtests follow a conservative event timeline:

1. Calculate signals using information available on day T.
2. Enter at the next trading day's open, T+1.
3. Exit at T+2 open, or use the script-specific exit rule.
4. Deduct commissions where the module supports it.
5. Filter abnormal price jumps that are likely caused by corporate actions or bad adjustment data.

The daily data contains forward return fields for convenience. Fields with a leading underscore, such as `_O2O` and `_O2C`, should be treated as evaluation-only fields because they contain future information. They must not be used as signal conditions.

## Known Data Notes

Please read `DATA_MANUAL.md` before relying on any result. One important warning is that historical preprocessing logic may misclassify ChiNext limit-up behavior if growth-board stocks are treated as 10 percent limit-up stocks instead of 20 percent limit-up stocks. Any strategy that depends on `flag == 1` should verify the preprocessing version and regenerate the Parquet files if necessary.

The project also does not include a complete production data pipeline for corporate actions, survivorship-bias control, transaction constraints, slippage modeling, or real-time execution. Those issues should be handled before using any research conclusion outside an offline study.

## Project Status

This is an experimental quant research workspace. The code favors fast iteration and transparent assumptions over framework abstraction. It is suitable for:

- Testing A-share short-term strategy ideas.
- Comparing sector classification systems.
- Studying limit-up and broken-board effects.
- Building daily candidate lists.
- Producing charts and CSV logs for manual review.

It is not suitable as-is for automated trading, portfolio execution, or investment advice. All results depend heavily on local data quality, preprocessing choices, commission assumptions, market regime, and whether signals can actually be executed at the assumed prices.

## Disclaimer

This repository is for research and educational purposes only. It does not provide financial advice, investment recommendations, or guaranteed trading signals. Markets are risky, backtests can be misleading, and short-term A-share strategies are especially sensitive to liquidity, trading restrictions, slippage, and crowding. Use the code to inspect hypotheses, not to replace independent judgment.
