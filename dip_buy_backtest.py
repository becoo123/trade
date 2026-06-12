# -*- coding: utf-8 -*-
"""
强势股缩量回调策略回测（乔帮主低吸战法量化版）

核心逻辑（基于因子分析验证）：

  背景（强势股筛选）：
    1. close > MA10 > MA20           ← 多头排列，上升趋势
    2. 近20天内有涨停板               ← 强势动量股（游资/机构关注）
    3. 近5日涨幅 ret_5d ∈ [8%, 20%]  ← 甜点区间（>20%反而回调后效果差）

  T 日信号（回调质量过滤）：
    4. close_return ∈ [-5%, -1%]     ← 中等幅度回调（太小=无意义，太大=破位）
    5. r_vol < r_vol_max (0.8)       ← 缩量（量比<1 = 筹码稳，无杀跌盘）
    6. flag == 0                     ← 非涨停/非跌停日
    7. 热门板块过滤（可选）

  执行：T+1 开盘买入，T+2 开盘卖出（Open-to-Open）

信号时序：
  T-k  : 涨停板（近20天内，确立强势股身份）
  T-4~T-1: 上涨积累5日涨幅 8-15%
  T    : 缩量回调 -5%~-1%  ← 信号日（洗盘或短线获利盘离场）
  T+1  : 开盘买入（低吸入场）
  T+2  : 开盘卖出

因子分析结论（全样本 2023-2026，n=17,827笔）：
  条件              均值      胜率    盈亏比
  缩量回调 -1~-0.5%  +0.04%   46.5%   1.16
  缩量回调 -2~-1%    +0.15%   48.5%   1.15
  缩量回调 -3~-2%    +0.29%   50.4%   1.20  ← 甜点
  缩量回调 -5~-3%    +0.33%   51.4%   1.15  ← 甜点
  追涨 close>3%     -0.34%   40.8%   1.21  ← 不可行（T+1开买太贵）
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from data_core import StockDataManager


class StrongPullbackBacktester:

    def __init__(self):
        self.dm = StockDataManager()

    def run_backtest(
        self,
        start_date: str,
        end_date: str,
        # --- 强势股背景条件 ---
        ret_5d_min: float = 0.08,        # 5日涨幅下限（8%），太低=趋势不强
        ret_5d_max: float = 0.20,        # 5日涨幅上限（20%），太高=回调后效果差
        limit_count_min: int = 1,        # 近 recent_limit_days 天内最少涨停次数
        recent_limit_days: int = 20,     # 涨停回溯窗口
        trend_ma_short: int = 10,        # 趋势短期均线
        trend_ma_long: int = 20,         # 趋势长期均线
        # --- 回调信号条件 ---
        pullback_min: float = -5.0,      # 当日涨跌幅下限（%）
        pullback_max: float = -1.0,      # 当日涨跌幅上限（%）
        r_vol_min: float = 0.0,          # 量比下限（追涨放量用），默认不限
        r_vol_max: float = 0.8,          # 量比上限（缩量判断），默认 < 0.8
        r_vol_prev_max: float = None,    # 前 shrink_window-1 日量比上限，None=不检查
        shrink_window: int = 1,          # 连续缩量天数（1=仅当日，2=今+昨，依此类推）
        # --- 组合管理 ---
        vol_ma_window: int = 20,         # 均量计算窗口
        active_pct: float = 1.0,         # 活跃股预筛比例
        top_k: int = 5,                  # 每日最多持仓数
        commission: float = 0.0005,      # 单边手续费
        # --- 板块热度过滤 ---
        sector_type: str = 'sw1_industry',
        sector_top_n: int = 10,
        sector_lookback: int = 10,
    ):
        print(f"\n{'='*65}")
        print(f"强势股缩量回调策略回测: {start_date} ~ {end_date}")
        print(f"强势背景: 5日涨幅[{ret_5d_min*100:.0f}%,{ret_5d_max*100:.0f}%]  "
              f"近{recent_limit_days}天≥{limit_count_min}次涨停  "
              f"趋势均线{trend_ma_short}/{trend_ma_long}")
        print(f"回调信号: 跌幅[{pullback_min}%,{pullback_max}%]  量比<{r_vol_max}  "
              f"热门板块Top{sector_top_n}({sector_type})")
        print(f"执行: T+1开买→T+2开卖  持仓≤{top_k}只  手续费={commission*100:.3f}%/边")
        print('='*65)

        # 1. 加载数据
        buffer = max(vol_ma_window, trend_ma_long, recent_limit_days, 10) + 30
        load_start = self._offset_date(start_date, -buffer)

        print("正在加载数据...")
        cols = ['code', 'day', 'open', 'close', 'amount', 'close_return',
                'flag', 'limit_price']
        df = self.dm.load_day_data(start_date=load_start, end_date=end_date, columns=cols)

        if df.empty:
            print("❌ 未加载到数据")
            return

        df = df.sort_values(['code', 'day']).reset_index(drop=True)
        print(f"全A股行数: {len(df):,}  |  股票数: {df['code'].nunique()}")

        # 2. 计算因子
        print("计算因子...")
        g = df.groupby('code', sort=False)

        # 历史均量（不含当日）
        df['ma_amount'] = g['amount'].transform(
            lambda x: x.rolling(vol_ma_window, min_periods=vol_ma_window//2).mean().shift(1)
        )

        # 趋势均线（不含当日）
        df['ma_short'] = g['close'].transform(
            lambda x: x.rolling(trend_ma_short, min_periods=trend_ma_short//2).mean().shift(1)
        )
        df['ma_long'] = g['close'].transform(
            lambda x: x.rolling(trend_ma_long, min_periods=trend_ma_long//2).mean().shift(1)
        )

        # 多头排列
        df['uptrend'] = (
            (df['close'] > df['ma_short']) &
            (df['ma_short'] > df['ma_long']) &
            df['ma_short'].notna() & df['ma_long'].notna()
        )

        # 近5日涨幅（收盘价，不含当日）—— 强势股核心指标
        df['ret_5d'] = g['close'].transform(
            lambda x: x.pct_change(5).shift(1)
        )

        # 量比（当日成交 / 历史均量）
        df['r_vol'] = df['amount'] / df['ma_amount']

        # 预计算各股连续缩量天数（1-5），供分级统计使用
        r_vol_prev1 = g['r_vol'].transform(lambda x: x.shift(1))
        df['shrink_days'] = 1
        for _w in range(1, 5):
            _peak = g['r_vol'].transform(
                lambda x, w=_w: x.shift(1).rolling(w, min_periods=w).max()
            )
            df.loc[_peak.notna() & (_peak < 1.0), 'shrink_days'] = _w + 1
        # shrink_days=1 表示仅当日缩量（前一日量比 >= 1.0 或无数据）
        df.loc[r_vol_prev1.isna() | (r_vol_prev1 >= 1.0), 'shrink_days'] = 1

        # 前 shrink_window-1 日量比最大值（用于信号过滤）
        if r_vol_prev_max is not None and shrink_window > 1:
            df['r_vol_prev_peak'] = g['r_vol'].transform(
                lambda x: x.shift(1).rolling(shrink_window - 1, min_periods=shrink_window - 1).max()
            )
        else:
            df['r_vol_prev_peak'] = np.nan

        # 近期涨停次数
        df['is_limit_up'] = (df['flag'] == 1).astype(np.int8)
        df['limit_count'] = g['is_limit_up'].transform(
            lambda x: x.shift(1).rolling(recent_limit_days, min_periods=1).sum()
        )

        # 连续缩量条件（可选）：前 shrink_window-1 天的量比峰值 <= r_vol_prev_max
        prev_shrink = (
            (df['r_vol_prev_peak'] <= r_vol_prev_max) & df['r_vol_prev_peak'].notna()
            if (r_vol_prev_max is not None and shrink_window > 1)
            else pd.Series(True, index=df.index)
        )

        # 综合信号
        df['signal'] = (
            df['uptrend'] &
            (df['ret_5d'] >= ret_5d_min) &
            (df['ret_5d'] <= ret_5d_max) &
            (df['limit_count'] >= limit_count_min) &
            (df['close_return'] >= pullback_min) &
            (df['close_return'] <= pullback_max) &
            (df['r_vol'] >= r_vol_min) &
            (df['r_vol'] <= r_vol_max) &
            prev_shrink &
            (df['flag'] == 0) &
            (df['amount'] > 0) &
            df['ret_5d'].notna()
        )

        # 交易价格：T+1 开（入场），T+2 开（出场）
        df['next_open']  = g['open'].shift(-1)
        df['next2_open'] = g['open'].shift(-2)

        # 排除 T+1 一字涨停
        df['next_limit_price'] = g['limit_price'].shift(-1)
        df['entry_is_limit'] = (
            df['next_open'].notna() &
            df['next_limit_price'].notna() &
            (df['next_open'] >= df['next_limit_price'] * 0.999)
        )

        # 交易收益
        df['trade_return'] = df['next2_open'] / df['next_open'] - 1 - 2 * commission
        df.loc[df['trade_return'].abs() > 0.5, 'trade_return'] = np.nan

        # 3. 预计算热门板块
        hot_map, code_to_block = self._build_hot_sector_filter(
            load_start, end_date, sector_type, sector_top_n, sector_lookback
        )

        # 4. 按日遍历
        trade_dates = sorted(df[df['day'] >= start_date]['day'].unique())
        print(f"回测区间共 {len(trade_dates)} 个交易日\n")

        daily_results = []
        all_trades    = []

        for i, trade_date in enumerate(trade_dates):
            day_df = df[df['day'] == trade_date]

            valid = day_df[day_df['ma_amount'].notna() & (day_df['amount'] > 0)]
            if valid.empty:
                daily_results.append({'day': trade_date, 'return': 0.0, 'count': 0})
                continue

            top_n     = max(1, int(len(valid) * active_pct))
            active_df = valid.nlargest(top_n, 'ma_amount')

            sig_df = active_df[
                active_df['signal'] &
                ~active_df['entry_is_limit'] &
                active_df['next_open'].notna() &
                active_df['next2_open'].notna()
            ].copy()

            # 热门板块过滤
            if hot_map:
                hot_today = hot_map.get(trade_date, set())
                sig_df = sig_df[
                    sig_df['code'].map(lambda c: code_to_block.get(c, '') in hot_today)
                ]

            if sig_df.empty:
                daily_results.append({'day': trade_date, 'return': 0.0, 'count': 0})
                continue

            # 按 ret_5d 排序（近期最强的优先），再按 ma_amount 打破平局
            sig_df = sig_df.sort_values(['ret_5d', 'ma_amount'], ascending=[False, False])
            top_df    = sig_df.head(top_k)
            daily_ret = top_df['trade_return'].mean()
            if pd.isna(daily_ret):
                daily_ret = 0.0

            daily_results.append({'day': trade_date, 'return': daily_ret, 'count': len(top_df)})

            for _, row in top_df.iterrows():
                ret = row['trade_return']
                all_trades.append({
                    'signal_date':  trade_date,
                    'code':         row['code'],
                    'ret_5d_pct':   round(row['ret_5d'] * 100, 2),    # 近5日涨幅
                    'pullback_pct': row['close_return'],                # 当日回调幅度
                    'r_vol':        round(row['r_vol'], 2),             # 量比
                    'shrink_days':  int(row.get('shrink_days', 1)),     # 连续缩量天数
                    'limit_count':  row['limit_count'],                 # 近期涨停次数
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
        print("\n" + "="*65)
        print("回测结果摘要")
        print("="*65)

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
            print(f"期望值:        {(wr/100*avg_w + (1-wr/100)*avg_l):.3f}%/笔")

            if 'pullback_pct' in trades_df.columns:
                print(f"\n信号日回调分布 (pullback_pct):")
                print(f"  均值={trades_df['pullback_pct'].mean():.2f}%  "
                      f"中位={trades_df['pullback_pct'].median():.2f}%")
            if 'ret_5d_pct' in trades_df.columns:
                print(f"近5日涨幅分布 (ret_5d_pct):")
                print(f"  均值={trades_df['ret_5d_pct'].mean():.2f}%  "
                      f"中位={trades_df['ret_5d_pct'].median():.2f}%")

            # 按连续缩量天数分级统计
            if 'shrink_days' in trades_df.columns:
                print(f"\n{'─'*55}")
                print(f"{'缩量天数':<6} {'笔数':>6} {'胜率':>7} {'均盈利':>8} {'均亏损':>8} {'期望值':>8}")
                print(f"{'─'*55}")
                for sd in sorted(trades_df['shrink_days'].unique()):
                    sub = trades_df[trades_df['shrink_days'] == sd]
                    sw  = (sub['return'] > 0).mean() * 100
                    sw_ = sub[sub['return'] > 0]['return'].mean() * 100 if (sub['return'] > 0).any() else 0
                    sl_ = sub[sub['return'] < 0]['return'].mean() * 100 if (sub['return'] < 0).any() else 0
                    ev  = sw / 100 * sw_ + (1 - sw / 100) * sl_
                    print(f"  {sd}日{'★'*sd:<5}  {len(sub):>5}  {sw:>6.1f}%  {sw_:>7.2f}%  {sl_:>7.2f}%  {ev:>+7.3f}%")
                print(f"{'─'*55}")

            print("\nTop 10 盈利交易:")
            for _, row in trades_df.nlargest(10, 'return').iterrows():
                print(f"  {row['signal_date']} {row['code']}  "
                      f"缩量{row.get('shrink_days',1)}日  "
                      f"5日={row.get('ret_5d_pct', 0):+.1f}%  "
                      f"回调={row.get('pullback_pct', 0):.1f}%  "
                      f"量比={row.get('r_vol', 0):.2f}  "
                      f"买={row['buy_price']:.2f} 卖={row['sell_price']:.2f}  "
                      f"收益={row['return_pct']:+.2f}%")

        print("="*65)

    def _save_outputs(self, res_df: pd.DataFrame, trades_df: pd.DataFrame):
        res_df.to_csv('dip_buy_backtest.csv', index=False, encoding='utf-8-sig')
        print(f"\n日度结果已保存: dip_buy_backtest.csv")
        if not trades_df.empty:
            trades_df.to_csv('dip_buy_trades.csv', index=False, encoding='utf-8-sig')
            print(f"交易明细已保存: dip_buy_trades.csv  ({len(trades_df)} 笔)")

    def _plot(self, res_df: pd.DataFrame):
        try:
            fig, axes = plt.subplots(2, 1, figsize=(14, 9),
                                     gridspec_kw={'height_ratios': [3, 1]})
            dates   = pd.to_datetime(res_df['day'])
            cum_pct = (res_df['cum_return'] - 1) * 100

            ax0 = axes[0]
            ax0.plot(dates, cum_pct, color='steelblue', linewidth=1.5, label='Strong Pullback Strategy')
            ax0.axhline(0, color='gray', linestyle='--', linewidth=0.8)
            ax0.fill_between(dates, cum_pct, 0, where=(cum_pct >= 0), alpha=0.15, color='green')
            ax0.fill_between(dates, cum_pct, 0, where=(cum_pct <  0), alpha=0.15, color='red')
            ax0.set_title('Strong Stock Pullback Strategy – Cumulative Return (%)', fontsize=13)
            ax0.set_ylabel('Cumulative Return (%)')
            ax0.grid(True, alpha=0.3)
            ax0.legend()

            ax1 = axes[1]
            ax1.bar(dates, res_df['count'], color='steelblue', alpha=0.6, width=1.5)
            ax1.set_title('Daily Position Count')
            ax1.set_ylabel('Positions')
            ax1.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig('dip_buy_backtest.png', dpi=150, bbox_inches='tight')
            print("收益曲线已保存: dip_buy_backtest.png")
            plt.close(fig)
        except Exception as e:
            print(f"绘图失败: {e}")

    def _build_hot_sector_filter(self, load_start, end_date, sector_type, sector_top_n, sector_lookback):
        from block import BlockAnalyzer

        print(f"预计算板块热度 ({sector_type}, 回溯{sector_lookback}天)...")
        self.dm._ensure_block_manager()
        bm = self.dm.block_manager

        if sector_type not in bm.blocks:
            print(f"  未找到 {sector_type}，跳过板块过滤")
            return {}, {}

        block_names = self.dm.get_block_list(sector_type)
        if not block_names:
            return {}, {}

        analyzer = BlockAnalyzer(self.dm)
        hist = analyzer.get_block_history(
            block_names=block_names, start_date=load_start, end_date=end_date,
            block_type=sector_type, lookback_days=sector_lookback,
        )

        if hist.empty:
            return {}, {}

        hot_map = {}
        for day, day_df in hist.groupby('day'):
            top = day_df.nlargest(sector_top_n, 'relative_strength')
            hot_map[day] = set(top['block_name'].tolist())

        print(f"  板块热度计算完成，覆盖 {len(hot_map)} 个交易日")

        sw_df = bm.blocks[sector_type][['stock_code', 'block_name']].copy()
        sw_df['code'] = sw_df['stock_code'].str.split('.').str[0].str.zfill(6)
        sw_df = sw_df.drop_duplicates('code')
        code_to_block = sw_df.set_index('code')['block_name'].to_dict()

        return hot_map, code_to_block

    @staticmethod
    def _offset_date(date_str: str, calendar_days: int) -> str:
        dt = datetime.strptime(date_str, '%Y-%m-%d')
        delta = timedelta(days=int(abs(calendar_days) * 1.5))
        result = dt - delta if calendar_days < 0 else dt + delta
        return result.strftime('%Y-%m-%d')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='强势股缩量回调策略回测')
    parser.add_argument('--start',    default='2023-01-01', help='回测开始日期（默认 2023-01-01）')
    parser.add_argument('--end',      default='2026-03-01', help='回测结束日期（默认 2026-03-01）')
    parser.add_argument('--sector',   default='sw1_industry',
                        help='板块类型（sw1_industry/sw2_industry/concept/…，默认 sw1_industry）')
    parser.add_argument('--top',      type=int,   default=10,     help='热门板块前N名（默认10）')
    parser.add_argument('--lookback', type=int,   default=10,     help='板块热度回溯天数（默认10）')
    parser.add_argument('--topk',     type=int,   default=9999,   help='每日最多持仓数（默认不限）')
    parser.add_argument('--shrink',      type=int,   default=1,    help='最小连续缩量天数（默认1）')
    parser.add_argument('--ret-min',     type=float, default=0.08, help='5日涨幅下限（默认0.08）')
    parser.add_argument('--ret-max',     type=float, default=0.20, help='5日涨幅上限（默认0.20）')
    parser.add_argument('--pct-min',     type=float, default=-5.0, help='当日涨跌幅下限（默认-5.0，追涨时改正值）')
    parser.add_argument('--pct-max',     type=float, default=-1.0, help='当日涨跌幅上限（默认-1.0，追涨时改正值）')
    parser.add_argument('--rvol-min',    type=float, default=0.0,  help='量比下限（默认0，追涨时设>1）')
    parser.add_argument('--rvol-max',    type=float, default=0.8,  help='量比上限（默认0.8，追涨时设99）')
    args = parser.parse_args()

    # shrink>1 时，自动启用前序缩量过滤
    prev_max = 1.0 if args.shrink > 1 else None

    bt = StrongPullbackBacktester()
    bt.run_backtest(
        start_date=args.start,
        end_date=args.end,
        ret_5d_min=args.ret_min,
        ret_5d_max=args.ret_max,
        limit_count_min=0,
        recent_limit_days=20,
        trend_ma_short=10,
        trend_ma_long=20,
        pullback_min=args.pct_min,
        pullback_max=args.pct_max,
        r_vol_min=args.rvol_min,
        r_vol_max=args.rvol_max,
        r_vol_prev_max=prev_max,
        shrink_window=args.shrink,
        vol_ma_window=20,
        active_pct=1.0,
        top_k=args.topk,
        commission=0.0005,
        sector_type=args.sector,
        sector_top_n=args.top,
        sector_lookback=args.lookback,
    )
