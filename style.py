import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

from data_core import StockDataManager

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False


class StyleBacktester:
    def __init__(
        self,
        lookback_days: int = 20,
        stock_top_n: int = 40,
        block_type: str = "sw2_industry",
        block_top_n: int = 10,
        block_strength: str = "strong",
        signal_mode: str = "OPEN",
    ):
        self.dm = StockDataManager()
        self.lookback_days = lookback_days
        self.stock_top_n = stock_top_n
        self.block_type = block_type
        self.block_top_n = block_top_n
        self.block_strength = block_strength
        self.signal_mode = signal_mode.upper() if signal_mode else "OPEN"
        self.exclude_prefixes: list[str] | None = None
        self.open_gap_strategy: str = "none"

    def _offset_date(self, date_str: str, offset_days: int) -> str:
        d = datetime.strptime(date_str, "%Y-%m-%d")
        d2 = d + timedelta(days=offset_days)
        return d2.strftime("%Y-%m-%d")

    def _prepare_signals(self, df: pd.DataFrame, lookback_days: int) -> pd.DataFrame:
        df = df.sort_values(["code", "day"]).copy()

        def per_code(g: pd.DataFrame) -> pd.DataFrame:
            g = g.sort_values("day").copy()
            g["high_N"] = (
                g["high"].shift(1).rolling(window=lookback_days, min_periods=lookback_days).max()
            )
            g["low_N"] = (
                g["low"].shift(1).rolling(window=lookback_days, min_periods=lookback_days).min()
            )
            g["amount_lag"] = g["amount"].shift(1)
            mode = self.signal_mode
            if mode == "EOD":
                ref = g["preclose"]
            else:
                ref = g["open"]
            g["up_strength"] = ref / g["high_N"] - 1.0
            g["down_strength"] = ref / g["low_N"] - 1.0
            return g

        return df.groupby("code", group_keys=False).apply(per_code)

    def _assign_size_group(self, group: pd.DataFrame) -> pd.DataFrame:
        n = len(group)
        if n == 0:
            group["size_group"] = np.nan
            return group
        # 使用前一天的成交额作为市值代理，避免未来函数
        if "amount_lag" not in group.columns:
            # 如果amount_lag不存在，不应回退到amount（那是未来数据），而是返回NaN或跳过
            # 但由于_prepare_signals已计算amount_lag，理论上不应为空（除非是上市首日等，但会被filter掉）
            return group
        
        group = group.dropna(subset=["amount_lag"]).copy()
        n = len(group)
        if n == 0:
            group["size_group"] = np.nan
            return group
        group = group.sort_values("amount_lag").copy()
        k = max(int(n * 0.3), 1)
        labels = np.full(n, np.nan, dtype=object)
        labels[:k] = "small"
        labels[-k:] = "big"
        group["size_group"] = labels
        return group

    def _build_code_blocks(self, codes: pd.Series, block_type: str) -> dict:
        mapping: dict[str, list[str]] = {}
        unique_codes = pd.Series(codes).dropna().unique()
        for code in unique_codes:
            blocks = self.dm.get_blocks_by_stock(code)
            if not blocks:
                mapping[str(code)] = []
                continue
            names = blocks.get(block_type, [])
            if isinstance(names, str):
                names = [names]
            names = [n for n in names if n]
            mapping[str(code)] = names
        return mapping

    def _select_blocks_by_breakout(
        self,
        df: pd.DataFrame,
        code_blocks: dict,
        top_blocks: int,
    ) -> dict:
        result: dict[str, set[str]] = {}
        if top_blocks <= 0:
            return result

        # 根据用户要求，只保留强势板块，移除weak和both选项
        mode = "strong"

        for day, g in df.groupby("day"):
            block_total: dict[str, int] = {}
            block_breakout: dict[str, int] = {}
            for _, row in g[["code", "up_strength"]].iterrows():
                code = str(row["code"])
                blocks = code_blocks.get(code, [])
                if not blocks:
                    continue
                is_up = row["up_strength"] > 0
                for b in blocks:
                    block_total[b] = block_total.get(b, 0) + 1
                    if is_up:
                        block_breakout[b] = block_breakout.get(b, 0) + 1
            if not block_total:
                result[str(day)] = set()
                continue
            rates = {}
            for b, tot in block_total.items():
                if tot > 0:
                    rates[b] = block_breakout.get(b, 0) / tot
            if not rates:
                result[str(day)] = set()
                continue
            sorted_desc = sorted(rates.items(), key=lambda x: x[1], reverse=True)
            sorted_asc = list(reversed(sorted_desc))
            if mode == "strong":
                chosen = [b for b, _ in sorted_desc[:top_blocks]]
            elif mode == "weak":
                chosen = [b for b, _ in sorted_asc[:top_blocks]]
            else:
                top_list = [b for b, _ in sorted_desc[:top_blocks]]
                bottom_list = [b for b, _ in sorted_asc[:top_blocks]]
                chosen = top_list + bottom_list
            result[str(day)] = set(chosen)
        return result

    def _filter_by_blocks(
        self,
        df: pd.DataFrame,
        block_groups_by_day: dict,
        code_blocks: dict,
    ) -> pd.DataFrame:
        if df.empty or not block_groups_by_day or not code_blocks:
            return df

        def keep(row) -> bool:
            day_key = str(row["day"])
            day_blocks = block_groups_by_day.get(day_key)
            if not day_blocks:
                return True
            blocks = code_blocks.get(str(row["code"]), [])
            if not blocks:
                return False
            for b in blocks:
                if b in day_blocks:
                    return True
            return False

        mask = df.apply(keep, axis=1)
        return df[mask].copy()

    def _calc_daily_returns(self, df_style: pd.DataFrame) -> pd.Series:
        if df_style.empty:
            return pd.Series(dtype=float)
        df_style = df_style.copy()
        # 修复：使用open_to_open_return_1d（基于开盘价买入，次日开盘卖出），避免使用open_return导致的未来函数（吃掉跳空缺口）
        df_style["ret"] = df_style["open_to_open_return_1d"] / 100.0
        df_style = df_style[np.isfinite(df_style["ret"])]
        if df_style.empty:
            return pd.Series(dtype=float)
        return df_style.groupby("day")["ret"].mean().sort_index()

    def _apply_exclusions(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        result = df.copy()
        if self.exclude_prefixes:
            prefixes = tuple(self.exclude_prefixes)
            result = result[~result["code"].astype(str).str.startswith(prefixes)]
        if self.open_gap_strategy and self.open_gap_strategy != "none":
            if "gap_open" in result.columns:
                if self.open_gap_strategy == "gap_up_1":
                    result = result[result["gap_open"] >= 0.01]
                elif self.open_gap_strategy == "gap_down_1":
                    result = result[result["gap_open"] <= -0.01]
        return result

    def _build_trades(
        self,
        up_candidates: pd.DataFrame,
        down_candidates: pd.DataFrame,
        code_blocks: dict,
    ) -> pd.DataFrame:
        frames = []
        if up_candidates is not None and not up_candidates.empty:
            for size in ["big", "small"]:
                df_part = up_candidates[up_candidates["size_group"] == size].copy()
                if df_part.empty:
                    continue
                df_part["style"] = f"breakout_{size}"
                frames.append(df_part)
        if down_candidates is not None and not down_candidates.empty:
            for size in ["big", "small"]:
                df_part = down_candidates[down_candidates["size_group"] == size].copy()
                if df_part.empty:
                    continue
                df_part["style"] = f"breakdown_{size}"
                frames.append(df_part)
        if not frames:
            return pd.DataFrame()
        trades = pd.concat(frames, ignore_index=True)
        if not code_blocks:
            all_codes = trades["code"].unique()
            code_blocks = self._build_code_blocks(all_codes, self.block_type)
        block_names = []
        for code in trades["code"]:
            names = code_blocks.get(str(code), [])
            if isinstance(names, str):
                names = [names]
            names = [n for n in names if n]
            block_names.append(";".join(names))
        trades["blocks"] = block_names
        cols = [
            "day",
            "style",
            "code",
            "blocks",
            "size_group",
            "amount",
            "open_to_open_return_1d",
            "up_strength",
            "down_strength",
        ]
        cols = [c for c in cols if c in trades.columns]
        trades = trades[cols].sort_values(["day", "style", "code"]).reset_index(drop=True)
        return trades

    def _plot_curves(
        self,
        curves: dict,
        start_date: str,
        end_date: str,
        save_path: str | None = None,
        show: bool = True,
    ) -> None:
        if not curves:
            return

        labels = {
            "breakout_big": "突破-大盘股",
            "breakout_small": "突破-小盘股",
            "breakdown_big": "跌破-大盘股",
            "breakdown_small": "跌破-小盘股",
        }

        plt.figure(figsize=(14, 8))
        for key, series in curves.items():
            if series.empty:
                continue
            x = pd.to_datetime(series.index)
            plt.plot(x, series.values, label=labels.get(key, key))

        plt.title(f"投资风格模拟净值曲线\n{start_date} 到 {end_date}")
        plt.xlabel("交易日期")
        plt.ylabel("累计收益倍数")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150)
        if show:
            plt.show()
        plt.close()

    def run(
        self,
        start_date: str,
        end_date: str,
        lookback_days: int | None = None,
        stock_top_n: int | None = None,
        save_path: str | None = None,
        show: bool = True,
    ) -> dict:
        lookback = lookback_days or self.lookback_days
        stock_top_n_val = stock_top_n or self.stock_top_n

        start_ext = self._offset_date(start_date, -lookback * 2)

        df = self.dm.load_day_data(
            start_date=start_ext,
            end_date=end_date,
            columns=[
                "code",
                "day",
                "open",
                "high",
                "low",
                "close",
                "limit_price",
                "amount",
                "preclose",
                "open_return",
                "open_to_open_return_1d",
            ],
        )
        if df.empty:
            print("无日线数据，检查日期区间或数据文件")
            return {}

        df = self._prepare_signals(df, lookback)

        df = df[
            (df["day"] >= start_date)
            & (df["day"] <= end_date)
        ].copy()

        df = df.dropna(subset=["high_N", "low_N", "open_return"])
        if df.empty:
            print("无有效数据：high_N/low_N/open_return 为空")
            return {}

        df["limit_open"] = (df["limit_price"] > 0) & (df["open"] >= df["limit_price"])
        df = df[~df["limit_open"]].copy()
        if df.empty:
            print("全部为开盘即涨停，无可交易标的")
            return {}

        df["gap_open"] = np.where(
            df["preclose"] > 0,
            df["open"] / df["preclose"] - 1.0,
            np.nan,
        )

        code_blocks = {}
        block_groups_by_day = {}
        if self.block_top_n and self.block_type:
            code_blocks = self._build_code_blocks(df["code"], self.block_type)
            block_groups_by_day = self._select_blocks_by_breakout(
                df,
                code_blocks,
                self.block_top_n,
            )

        up_candidates = df.copy()
        down_candidates = df.copy()

        if code_blocks and block_groups_by_day:
            up_candidates = self._filter_by_blocks(up_candidates, block_groups_by_day, code_blocks)
            down_candidates = self._filter_by_blocks(down_candidates, block_groups_by_day, code_blocks)

        up_candidates = self._apply_exclusions(up_candidates)
        down_candidates = self._apply_exclusions(down_candidates)

        if up_candidates.empty and down_candidates.empty:
            print("在给定条件下无突破或跌破候选股票")
            return {}

        if not up_candidates.empty:
            up_candidates = up_candidates.sort_values(["day", "up_strength"], ascending=[True, False])
            up_candidates = up_candidates.groupby("day", group_keys=False).head(stock_top_n_val)
            up_candidates = up_candidates.groupby("day", group_keys=False).apply(self._assign_size_group)
        else:
            up_candidates["size_group"] = np.nan

        if not down_candidates.empty:
            down_candidates = down_candidates.sort_values(
                ["day", "down_strength"], ascending=[True, True]
            )
            down_candidates = down_candidates.groupby("day", group_keys=False).head(stock_top_n_val)
            down_candidates = down_candidates.groupby("day", group_keys=False).apply(self._assign_size_group)
        else:
            down_candidates["size_group"] = np.nan

        trades = self._build_trades(up_candidates, down_candidates, code_blocks)

        up_big = self._calc_daily_returns(
            up_candidates[up_candidates.get("size_group") == "big"]
            if not up_candidates.empty
            else up_candidates
        )
        up_small = self._calc_daily_returns(
            up_candidates[up_candidates.get("size_group") == "small"]
            if not up_candidates.empty
            else up_candidates
        )
        down_big = self._calc_daily_returns(
            down_candidates[down_candidates.get("size_group") == "big"]
            if not down_candidates.empty
            else down_candidates
        )
        down_small = self._calc_daily_returns(
            down_candidates[down_candidates.get("size_group") == "small"]
            if not down_candidates.empty
            else down_candidates
        )

        all_index = (
            set(up_big.index)
            | set(up_small.index)
            | set(down_big.index)
            | set(down_small.index)
        )
        if not all_index:
            print("无任何风格的日收益数据")
            return {}

        all_days = sorted(all_index)
        idx = pd.Index(all_days, name="day")

        curves = {}
        for key, series in [
            ("breakout_big", up_big),
            ("breakout_small", up_small),
            ("breakdown_big", down_big),
            ("breakdown_small", down_small),
        ]:
            if series.empty:
                curves[key] = pd.Series(index=idx, data=np.ones(len(idx)))
            else:
                s = series.reindex(idx).fillna(0.0)
                cum = (1.0 + s).cumprod()
                curves[key] = cum
                avg_ret = s.mean()
                pos_ratio = (s > 0).mean() if len(s) > 0 else 0.0
                total_ret = cum.iloc[-1] - 1.0 if len(cum) > 0 else 0.0
                print(
                    f"{key}: 日均收益 {avg_ret:.4%}, "
                    f"正收益天数比例 {pos_ratio:.2%}, "
                    f"总收益 {total_ret:.2%}"
                )

        self._plot_curves(curves, start_date, end_date, save_path=save_path, show=show)

        if not trades.empty:
            file_name = f"style_trades_{start_date}_{end_date}.csv"
            trades.to_csv(file_name, index=False, encoding="utf-8-sig")
            print(f"交易明细已保存: {file_name}")

        return curves


def run_style_analysis(
    start_date: str,
    end_date: str,
    lookback_days: int = 20,
    stock_top_n: int = 20,
    block_top_n: int = 10,
    block_strength: str = "strong",
    signal_mode: str = "OPEN",
    exclude_prefixes: list[str] | None = None,
    open_gap_strategy: str = "none",
    save_path: str | None = "style.png",
    show: bool = True,
) -> dict:
    backtester = StyleBacktester(
        lookback_days=lookback_days,
        stock_top_n=stock_top_n,
        block_top_n=block_top_n,
        block_strength=block_strength,
        signal_mode=signal_mode,
    )
    backtester.exclude_prefixes = exclude_prefixes or []
    backtester.open_gap_strategy = open_gap_strategy
    return backtester.run(
        start_date=start_date,
        end_date=end_date,
        lookback_days=lookback_days,
        stock_top_n=stock_top_n,
        save_path=save_path,
        show=show,
    )


def style(
    start_date: str,
    end_date: str,
    lookback_days: int = 20,
    stock_top_n: int = 20,
    block_top_n: int = 10,
    block_strength: str = "strong",
    signal_mode: str = "OPEN",
    exclude_prefixes: list[str] | None = None,
    open_gap_strategy: str = "none",
    save_path: str | None = "style.png",
    show: bool = True,
) -> dict:
    return run_style_analysis(
        start_date=start_date,
        end_date=end_date,
        lookback_days=lookback_days,
        stock_top_n=stock_top_n,
        block_top_n=block_top_n,
        block_strength=block_strength,
        signal_mode=signal_mode,
        exclude_prefixes=exclude_prefixes,
        open_gap_strategy=open_gap_strategy,
        save_path=save_path,
        show=show,
    )


if __name__ == "__main__":
    start_date = "2023-08-02"
    end_date = "2026-01-30"
    lookback_days = 20
    stock_top_n = 20
    block_top_n = 2
    block_strength = "strong"
    signal_mode = "EOD"
    exclude_prefixes = ["688"]
    open_gap_strategy = "gap_down_1"
    
    print("开始运行风格回测")
    curves = style(
        start_date=start_date,
        end_date=end_date,
        lookback_days=lookback_days,
        stock_top_n=stock_top_n,
        block_top_n=block_top_n,
        block_strength=block_strength,
        signal_mode=signal_mode,
        exclude_prefixes=exclude_prefixes,
        open_gap_strategy=open_gap_strategy,
        save_path="style.png",
        show=True,
    )
    print("风格回测结束，是否得到曲线：", bool(curves))
