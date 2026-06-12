# -*- coding: utf-8 -*-
"""
创业板首板缩量调整回测 v3
策略逻辑：
  1. 标的：创业板股票（代码以 '3' 开头）
  2. 活跃股预筛：按历史均量 ma_amount 取 Top active_pct%（换手活跃代理）
  3. 首板识别：flag==1 且过去 first_board_window 天内无涨停（非连板）
  4. 10天内首板 + 缩量调整连续 n_adj 天：
     过去 first_board_window 天内曾有首板，且今天是连续缩量调整（amount<MA、未涨停）
     的第 n_adj 天（streak 恰好 == n_adj）
  5. 涨停排除：若 T+1（入场日）开盘价 >= 涨停价，视为一字板无法买入，跳过
  6. 按 ma_amount 降序取 top_k（换手活跃优先）
  7. 买入：信号日次日开盘（T+1 Open）
  8. 卖出：买入次日开盘（T+2 Open），持仓 1 天

信号时序示意（n_adj=3，首板在10天内任意一天）：
  T-k : 首板（涨停，k ∈ [n_adj, first_board_window]）
  ...  : 中间可有非缩量天
  T-2 : 缩量调整
  T-1 : 缩量调整
  T   : 缩量调整（连续第3天）← 信号日
  T+1 : 买入开盘
  T+2 : 卖出开盘
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from data_core import StockDataManager


class GEMPullbackBacktester:

    def __init__(self):
        self.dm = StockDataManager()

    def run_backtest(
        self,
        start_date: str,
        end_date: str,
        n_adj: int = 3,                  # 缩量调整天数
        first_board_window: int = 10,    # 首板回溯窗口：过去 N 天无涨停才算首板
        vol_ma_window: int = 20,         # 均量计算窗口
        active_pct: float = 1.0,         # 活跃股预筛比例（历史均量 Top N%）
        top_k: int = 5,                  # 每日最多持仓数
        commission: float = 0.0005,      # 单边手续费
        sector_type: str = 'sw1_industry',  # 板块热度使用的分类
        sector_top_n: int = 10,             # 每日取前 N 热门板块（相对强度排序）
        sector_lookback: int = 10,          # 板块热度计算的回溯天数
        max_drop_pct: float = 3.0,          # 缩量期内单日最大跌幅（超过则打断缩量）
    ):
        print(f"\n{'='*60}")
        print(f"全A股首板缩量调整回测 v5: {start_date} ~ {end_date}")
        print(f"参数: 缩量≥{n_adj}天  首板窗口={first_board_window}天  "
              f"量均={vol_ma_window}天  热门板块Top{sector_top_n}({sector_type})  "
              f"持仓≤{top_k}只  手续费={commission*100:.3f}%/边  "
              f"缩量期最大跌幅={max_drop_pct}%")
        print('='*60)

        # 1. 加载数据（buffer 要足够覆盖均线预热 + 首板回溯）
        buffer = max(vol_ma_window, first_board_window, n_adj) + 30
        load_start = self._offset_date(start_date, -buffer)

        print("正在加载数据...")
        cols = ['code', 'day', 'open', 'close', 'amount', 'close_return',
                'flag', 'limit_price']
        df = self.dm.load_day_data(start_date=load_start, end_date=end_date, columns=cols)

        if df.empty:
            print("❌ 未加载到数据")
            return

        # 2. 全A股（0/3/6 开头）
        df = df.sort_values(['code', 'day']).reset_index(drop=True)
        print(f"全A股行数: {len(df):,}  |  股票数: {df['code'].nunique()}")

        # 3. 计算因子
        print("计算因子...")
        g = df.groupby('code', sort=False)

        # 历史均量（不含当日）—— 用于活跃股预筛 + 缩量判断
        df['ma_amount'] = g['amount'].transform(
            lambda x: x.rolling(vol_ma_window, min_periods=vol_ma_window // 2).mean().shift(1)
        )

        # 涨停标记（flag == 1）
        df['is_limit_up'] = (df['flag'] == 1).astype(np.int8)

        # 首板：今天涨停 + 过去 first_board_window 天内无涨停
        df['had_limit_up'] = g['is_limit_up'].transform(
            lambda x: x.shift(1).rolling(first_board_window, min_periods=1).max()
        ).fillna(0)
        df['is_first_board'] = (df['is_limit_up'] == 1) & (df['had_limit_up'] == 0)

        # 缩量调整（含质量过滤）：缩量 + 未涨停 + 非大阴线（跌幅不超过 max_drop_pct%）
        # close_return 单位为百分比（如 -4.60 表示 -4.6%）
        df['shrink_adj'] = (
            (df['amount'] < df['ma_amount']) &
            df['ma_amount'].notna() &
            (df['is_limit_up'] == 0) &
            (df['close_return'] > -max_drop_pct)   # 不允许大阴线
        ).astype(np.int8)

        # 连续缩量天数 streak（当天缩量=streak继续，否则归零）
        def _streak(s):
            s = s.astype(int)
            return s.groupby((s == 0).cumsum()).cumcount() + 1
        df['shrink_streak'] = g['shrink_adj'].transform(_streak) * df['shrink_adj']

        # 过去 first_board_window 天内是否有首板（不含当日）
        df['fb_in_window'] = g['is_first_board'].transform(
            lambda x: x.shift(1).rolling(first_board_window, min_periods=1).max()
        ).fillna(0).astype(bool)

        # 前一日是否已完成 n_adj 天以上的缩量蓄力
        df['prev_had_enough_shrink'] = (
            g['shrink_streak'].transform(lambda x: x.shift(1)).fillna(0) >= n_adj
        )

        # 止跌信号：今日放量阳线（收正 + 成交额 >= 均量 + 非涨停）
        df['reversal_signal'] = (
            (df['close_return'] > 0) &
            (df['amount'] >= df['ma_amount']) &
            df['ma_amount'].notna() &
            (df['is_limit_up'] == 0)
        )

        # 综合信号：10天内有首板 + 完成≥n_adj天缩量 + 今日放量阳线止跌
        df['signal'] = (
            df['fb_in_window'] &
            df['prev_had_enough_shrink'] &
            df['reversal_signal'] &
            (df['amount'] > 0)
        )

        # 交易价格：T+1 开（入场）和 T+2 开（出场）
        df['next_open']  = g['open'].shift(-1)
        df['next2_open'] = g['open'].shift(-2)

        # 涨停排除：T+1 开盘价 >= T+1 涨停价（一字板无法买入）
        df['next_limit_price'] = g['limit_price'].shift(-1)
        df['entry_is_limit'] = (
            df['next_open'].notna() &
            df['next_limit_price'].notna() &
            (df['next_open'] >= df['next_limit_price'] * 0.999)
        )

        # 交易收益（扣双边手续费）
        df['trade_return'] = df['next2_open'] / df['next_open'] - 1 - 2 * commission
        # 过滤价格跳变异常（复权/分拆导致），单日持仓收益不应超过±50%
        df.loc[df['trade_return'].abs() > 0.5, 'trade_return'] = np.nan

        # 4. 预计算热门板块（无未来函数：relative_strength[T] 用 T 日收盘数据，
        #    在 T+1 开盘前已可知，用于过滤 T 日信号）
        hot_map, code_to_block = self._build_hot_sector_filter(
            load_start, end_date, sector_type, sector_top_n, sector_lookback
        )

        # 5. 按日遍历
        trade_dates = sorted(df[df['day'] >= start_date]['day'].unique())
        print(f"回测区间共 {len(trade_dates)} 个交易日\n")

        daily_results = []
        all_trades    = []

        for i, trade_date in enumerate(trade_dates):
            day_df = df[df['day'] == trade_date]

            # 活跃股预筛（历史均量 Top active_pct%）
            valid = day_df[day_df['ma_amount'].notna() & (day_df['amount'] > 0)]
            if valid.empty:
                daily_results.append({'day': trade_date, 'return': 0.0, 'count': 0})
                continue

            top_n     = max(1, int(len(valid) * active_pct))
            active_df = valid.nlargest(top_n, 'ma_amount')

            # 在活跃股中筛信号，排除涨停入场
            sig_df = active_df[
                active_df['signal'] &
                ~active_df['entry_is_limit'] &
                active_df['next_open'].notna() &
                active_df['next2_open'].notna()
            ].copy()

            # 热门板块过滤：只保留属于今日热门板块的信号
            if hot_map:
                hot_today = hot_map.get(trade_date, set())
                sig_df = sig_df[
                    sig_df['code'].map(lambda c: code_to_block.get(c, '') in hot_today)
                ]

            if sig_df.empty:
                daily_results.append({'day': trade_date, 'return': 0.0, 'count': 0})
                continue

            # 按 ma_amount 取 top_k
            top_df    = sig_df.nlargest(top_k, 'ma_amount')
            daily_ret = top_df['trade_return'].mean()
            if pd.isna(daily_ret):
                daily_ret = 0.0

            daily_results.append({'day': trade_date, 'return': daily_ret, 'count': len(top_df)})

            for _, row in top_df.iterrows():
                ret = row['trade_return']
                all_trades.append({
                    'signal_date':  trade_date,
                    'code':         row['code'],
                    'signal_close': row['close'],
                    'signal_cr':    row['close_return'],
                    'ma_amount':    row['ma_amount'],
                    'buy_price':    row['next_open'],
                    'sell_price':   row['next2_open'],
                    'return':       ret if not pd.isna(ret) else 0.0,
                    'return_pct':   (ret * 100) if not pd.isna(ret) else 0.0,
                })

            if (i + 1) % 30 == 0:
                print(f"  {trade_date} | 活跃 {top_n:4d} 只 → 信号 {len(sig_df):3d} 只 → "
                      f"持仓 {len(top_df)} 只 | 收益 {daily_ret*100:+.2f}%")

        # 5. 汇总输出
        res_df    = pd.DataFrame(daily_results)
        trades_df = pd.DataFrame(all_trades)
        res_df['cum_return'] = (1 + res_df['return']).cumprod()

        self._print_summary(res_df, trades_df)
        self._save_outputs(res_df, trades_df)
        self._plot(res_df)

        return res_df, trades_df

    # ------------------------------------------------------------------
    def _print_summary(self, res_df: pd.DataFrame, trades_df: pd.DataFrame):
        print("\n" + "="*60)
        print("回测结果摘要")
        print("="*60)

        active = res_df[res_df['count'] > 0]
        cum    = res_df['cum_return'].iloc[-1]

        print(f"总交易日:      {len(res_df)}")
        print(f"有信号交易日:  {len(active)}  ({len(active)/len(res_df)*100:.1f}%)")
        print(f"日均持仓:      {res_df['count'].mean():.2f} 只")
        print(f"\n累积收益:      {(cum - 1)*100:.2f}%")

        if len(res_df) > 0:
            avg   = res_df['return'].mean()
            std   = res_df['return'].std()
            ann   = (pow(1 + avg, 250) - 1) * 100
            sharp = avg / std * np.sqrt(250) if std > 0 else 0

            cummax = res_df['cum_return'].cummax()
            dd     = (res_df['cum_return'] - cummax) / cummax
            max_dd = dd.min() * 100

            print(f"年化收益:      {ann:.2f}%")
            print(f"夏普比率:      {sharp:.2f}")
            print(f"最大回撤:      {max_dd:.2f}%")

        if not trades_df.empty:
            wr    = (trades_df['return'] > 0).mean() * 100
            wins  = trades_df[trades_df['return'] > 0]['return']
            loss  = trades_df[trades_df['return'] < 0]['return']
            avg_w = wins.mean() * 100 if len(wins) > 0 else 0
            avg_l = loss.mean() * 100 if len(loss) > 0 else 0
            plr   = abs(avg_w / avg_l) if avg_l != 0 else float('inf')

            print(f"\n总交易笔数:    {len(trades_df)}")
            print(f"胜率:          {wr:.2f}%")
            print(f"平均盈利:      {avg_w:.2f}%")
            print(f"平均亏损:      {avg_l:.2f}%")
            print(f"盈亏比:        {plr:.2f}")

            print("\nTop 10 盈利交易:")
            for _, row in trades_df.nlargest(10, 'return').iterrows():
                print(f"  {row['signal_date']} {row['code']}  "
                      f"买={row['buy_price']:.2f}  卖={row['sell_price']:.2f}  "
                      f"收益={row['return_pct']:+.2f}%")

        print("="*60)

    def _save_outputs(self, res_df: pd.DataFrame, trades_df: pd.DataFrame):
        res_df.to_csv('gem_pullback_backtest.csv', index=False, encoding='utf-8-sig')
        print(f"\n日度结果已保存: gem_pullback_backtest.csv")
        if not trades_df.empty:
            trades_df.to_csv('gem_pullback_trades.csv', index=False, encoding='utf-8-sig')
            print(f"交易明细已保存: gem_pullback_trades.csv  ({len(trades_df)} 笔)")

    def _plot(self, res_df: pd.DataFrame):
        try:
            fig, axes = plt.subplots(2, 1, figsize=(14, 9),
                                     gridspec_kw={'height_ratios': [3, 1]})
            dates   = pd.to_datetime(res_df['day'])
            cum_pct = (res_df['cum_return'] - 1) * 100

            ax0 = axes[0]
            ax0.plot(dates, cum_pct, color='steelblue', linewidth=1.5, label='Strategy')
            ax0.axhline(0, color='gray', linestyle='--', linewidth=0.8)
            ax0.fill_between(dates, cum_pct, 0,
                             where=(cum_pct >= 0), alpha=0.15, color='green')
            ax0.fill_between(dates, cum_pct, 0,
                             where=(cum_pct <  0), alpha=0.15, color='red')
            ax0.set_title('GEM First-Board Pullback v3 – Cumulative Return (%)', fontsize=13)
            ax0.set_ylabel('Cumulative Return (%)')
            ax0.grid(True, alpha=0.3)
            ax0.legend()

            ax1 = axes[1]
            ax1.bar(dates, res_df['count'], color='steelblue', alpha=0.6, width=1.5)
            ax1.set_title('Daily Position Count')
            ax1.set_ylabel('Positions')
            ax1.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig('gem_pullback_backtest.png', dpi=150, bbox_inches='tight')
            print("收益曲线已保存: gem_pullback_backtest.png")
            plt.close(fig)
        except Exception as e:
            print(f"绘图失败: {e}")

    def _build_hot_sector_filter(
        self,
        load_start: str,
        end_date: str,
        sector_type: str,
        sector_top_n: int,
        sector_lookback: int,
    ):
        """
        预计算每日热门板块，返回:
          hot_map:      {day_str -> set(hot_block_names)}
          code_to_block: {code_str -> block_name}

        无未来函数：relative_strength[T] 使用 rolling 窗口，包含 T 日成交数据，
        在 T 日收盘后可知，用于过滤 T 日信号（T+1 开盘执行）。
        """
        from block import BlockAnalyzer

        print(f"预计算板块热度 ({sector_type}, 回溯{sector_lookback}天)...")
        self.dm._ensure_block_manager()
        bm = self.dm.block_manager

        if sector_type not in bm.blocks:
            print(f"  ⚠️  未找到 {sector_type}，跳过板块过滤")
            return {}, {}

        block_names = self.dm.get_block_list(sector_type)
        if not block_names:
            return {}, {}

        analyzer = BlockAnalyzer(self.dm)
        hist = analyzer.get_block_history(
            block_names=block_names,
            start_date=load_start,
            end_date=end_date,
            block_type=sector_type,
            lookback_days=sector_lookback,
        )

        if hist.empty:
            print("  ⚠️  板块历史数据为空，跳过板块过滤")
            return {}, {}

        # 每日取 relative_strength 前 sector_top_n 的板块
        hot_map = {}
        for day, day_df in hist.groupby('day'):
            top = day_df.nlargest(sector_top_n, 'relative_strength')
            hot_map[day] = set(top['block_name'].tolist())

        print(f"  板块热度计算完成，覆盖 {len(hot_map)} 个交易日")

        # 构建 股票代码 → 板块名 映射（静态，取第一个板块）
        sw_df = bm.blocks[sector_type][['stock_code', 'block_name']].copy()
        sw_df['code'] = sw_df['stock_code'].str.split('.').str[0].str.zfill(6)
        sw_df = sw_df.drop_duplicates('code')
        code_to_block = sw_df.set_index('code')['block_name'].to_dict()

        return hot_map, code_to_block

    @staticmethod
    def _offset_date(date_str: str, calendar_days: int) -> str:
        dt = datetime.strptime(date_str, '%Y-%m-%d')
        result = dt + timedelta(days=int(calendar_days * 1.5))
        return result.strftime('%Y-%m-%d')


if __name__ == '__main__':
    import sys

    backtester = GEMPullbackBacktester()

    start = '2023-01-01'
    end   = '2026-03-01'

    if len(sys.argv) >= 3:
        start, end = sys.argv[1], sys.argv[2]

    backtester.run_backtest(
        start_date=start,
        end_date=end,
        n_adj=3,
        first_board_window=10,
        vol_ma_window=20,
        active_pct=1.0,
        top_k=5,
        commission=0.0005,
        sector_type='sw1_industry',
        sector_top_n=10,
        sector_lookback=10,
    )
