# -*- coding: utf-8 -*-
"""
强势股缩量回调选股器

用法：
    python dip_buy_select.py [日期] [--sector 板块类型] [--top N] [--lookback N]

示例：
    python dip_buy_select.py                              # 最新交易日，sw1 Top10
    python dip_buy_select.py 2026-03-03                   # 指定日期，默认参数
    python dip_buy_select.py 2026-03-03 --sector sw2_industry --top 20
    python dip_buy_select.py --sector sw2_industry --top 20 --lookback 5

支持的板块类型（--sector）：
    sw1_industry   申万一级（约30个）   默认
    sw2_industry   申万二级（约100个）
    sw3_industry   申万三级
    csrc1_industry 证监会一级
    csrc2_industry 证监会二级
    concept        概念板块
    theme_concept  主题板块
"""

import sys
import argparse
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from data_core import StockDataManager

# ── 固定策略参数（不通过命令行调整）──────────────────
RET_5D_MIN      = 0.08    # 近5日涨幅下限
RET_5D_MAX      = 0.20    # 近5日涨幅上限
PULLBACK_MIN    = -5.0    # 当日跌幅下限（%）
PULLBACK_MAX    = -1.0    # 当日跌幅上限（%）
R_VOL_MAX       = 0.8     # 当日量比上限
VOL_MA_WINDOW   = 20      # 均量窗口
TREND_SHORT     = 10      # 短期均线
TREND_LONG      = 20      # 长期均线
# ──────────────────────────────────────────────────────


def get_hot_sectors(dm, signal_date: str, sector_type: str, sector_top_n: int, sector_lookback: int) -> set:
    """返回信号日热门板块名称集合（Top N by relative_strength）"""
    from block import BlockAnalyzer
    lookback_start = (datetime.strptime(signal_date, '%Y-%m-%d')
                      - timedelta(days=sector_lookback * 2)).strftime('%Y-%m-%d')
    block_names = dm.get_block_list(sector_type)
    analyzer = BlockAnalyzer(dm)
    hist = analyzer.get_block_history(
        block_names=block_names,
        start_date=lookback_start,
        end_date=signal_date,
        block_type=sector_type,
        lookback_days=sector_lookback,
    )
    if hist.empty:
        return set()
    day_df = hist[hist['day'] == hist['day'].max()]
    top = day_df.nlargest(sector_top_n, 'relative_strength')
    return set(top['block_name'].tolist())


def build_code_to_industry(dm, sector_type: str) -> dict:
    """返回 code -> 行业名 的映射"""
    dm._ensure_block_manager()
    bm = dm.block_manager
    if sector_type not in bm.blocks:
        return {}
    sw_df = bm.blocks[sector_type][['stock_code', 'block_name']].copy()
    sw_df['code'] = sw_df['stock_code'].str.split('.').str[0].str.zfill(6)
    sw_df = sw_df.drop_duplicates('code')
    return sw_df.set_index('code')['block_name'].to_dict()


def select(signal_date: str, sector_type: str = 'sw1_industry',
           sector_top_n: int = 10, sector_lookback: int = 10):
    dm = StockDataManager()

    # 加载足够的历史数据用于计算均线/均量
    buffer_days = max(VOL_MA_WINDOW, TREND_LONG) + 15
    load_start = (datetime.strptime(signal_date, '%Y-%m-%d')
                  - timedelta(days=int(buffer_days * 1.8))).strftime('%Y-%m-%d')

    print(f"\n加载数据（{load_start} ~ {signal_date}）...")
    cols = ['code', 'day', 'open', 'close', 'amount', 'close_return', 'flag']
    df = dm.load_day_data(start_date=load_start, end_date=signal_date, columns=cols)

    if df.empty:
        print("未加载到数据，请检查日期是否为交易日。")
        return

    df = df.sort_values(['code', 'day']).reset_index(drop=True)
    actual_last = df['day'].max()
    if actual_last != signal_date:
        print(f"注意：{signal_date} 无交易数据，使用最近交易日 {actual_last}")
        signal_date = actual_last

    # ── 计算因子 ──────────────────────────────────────
    g = df.groupby('code', sort=False)

    df['ma_amount'] = g['amount'].transform(
        lambda x: x.rolling(VOL_MA_WINDOW, min_periods=VOL_MA_WINDOW // 2).mean().shift(1)
    )
    df['ma_short'] = g['close'].transform(
        lambda x: x.rolling(TREND_SHORT, min_periods=TREND_SHORT // 2).mean().shift(1)
    )
    df['ma_long'] = g['close'].transform(
        lambda x: x.rolling(TREND_LONG, min_periods=TREND_LONG // 2).mean().shift(1)
    )
    df['ret_5d'] = g['close'].transform(lambda x: x.pct_change(5).shift(1))
    df['r_vol']  = df['amount'] / df['ma_amount']

    # 连续缩量天数（1-5）：过去 N-1 天量比峰值
    df['r_vol_prev1'] = g['r_vol'].transform(lambda x: x.shift(1))
    for w in range(2, 5):
        df[f'r_vol_peak_{w}'] = g['r_vol'].transform(
            lambda x, w=w: x.shift(1).rolling(w, min_periods=w).max()
        )

    # ── 取信号日当日数据 ─────────────────────────────
    today = df[df['day'] == signal_date].copy()
    if today.empty:
        print(f"无法获取 {signal_date} 数据。")
        return

    # 基础过滤
    cond = (
        (today['close'] > today['ma_short']) &
        (today['ma_short'] > today['ma_long']) &
        today['ma_short'].notna() & today['ma_long'].notna() &
        (today['ret_5d'] >= RET_5D_MIN) & (today['ret_5d'] <= RET_5D_MAX) &
        (today['close_return'] >= PULLBACK_MIN) &
        (today['close_return'] <= PULLBACK_MAX) &
        (today['r_vol'] <= R_VOL_MAX) &
        (today['flag'] == 0) &
        (today['amount'] > 0) &
        today['ret_5d'].notna() &
        today['ma_amount'].notna()
    )
    candidates = today[cond].copy()

    if candidates.empty:
        print(f"\n{signal_date} 无满足条件的候选股票。")
        return

    # ── 连续缩量天数标注 ─────────────────────────────
    def shrink_days(row):
        if pd.isna(row['r_vol_prev1']) or row['r_vol_prev1'] >= 1.0:
            return 1
        # 前1日也<1.0，继续往前查
        for w in [2, 3, 4]:
            col = f'r_vol_peak_{w}'
            if col not in row or pd.isna(row[col]) or row[col] >= 1.0:
                return w + 1  # 满足前 w 天，共 w+1 天
        return 5  # 前4天峰值都<1.0

    candidates['shrink_days'] = candidates.apply(shrink_days, axis=1)

    # ── 热门板块过滤 ─────────────────────────────────
    print(f"计算热门板块（{sector_type} Top{sector_top_n}，回溯{sector_lookback}天）...")
    hot_sectors = get_hot_sectors(dm, signal_date, sector_type, sector_top_n, sector_lookback)
    code_to_industry = build_code_to_industry(dm, sector_type)

    candidates['industry'] = candidates['code'].map(
        lambda c: code_to_industry.get(c, '未知')
    )
    candidates['in_hot'] = candidates['industry'].isin(hot_sectors)

    hot_df  = candidates[candidates['in_hot']].copy()
    cold_df = candidates[~candidates['in_hot']].copy()

    # ── 输出 ─────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"信号日：{signal_date}  →  T+1 开盘买入，T+2 开盘卖出")
    print(f"板块标准：{sector_type}  Top{sector_top_n}")
    print(f"热门板块：{', '.join(sorted(hot_sectors))}")
    print(f"{'='*65}")

    def fmt_row(row):
        strength_bar = '★' * row['shrink_days']
        return (f"  {row['code']}  "
                f"缩量{row['shrink_days']}日 {strength_bar:<5}  "
                f"行业:{row['industry']:<8}  "
                f"5日涨{row['ret_5d']*100:+.1f}%  "
                f"今日{row['close_return']:+.1f}%  "
                f"量比{row['r_vol']:.2f}")

    hot_df  = hot_df.sort_values(['shrink_days', 'ret_5d'], ascending=[False, False])
    cold_df = cold_df.sort_values(['shrink_days', 'ret_5d'], ascending=[False, False])

    print(f"\n【热门板块内】共 {len(hot_df)} 只  ← 推荐优先")
    if hot_df.empty:
        print("  （无）")
    else:
        for _, row in hot_df.iterrows():
            print(fmt_row(row))

    print(f"\n【非热门板块】共 {len(cold_df)} 只  ← 参考")
    if cold_df.empty:
        print("  （无）")
    else:
        for _, row in cold_df.iterrows():
            print(fmt_row(row))

    total = len(hot_df) + len(cold_df)
    print(f"\n合计：{total} 只候选（热门板块 {len(hot_df)} 只）")
    print(f"{'='*65}\n")

    # 保存CSV
    out = pd.concat([hot_df, cold_df])[[
        'code', 'industry', 'in_hot', 'shrink_days',
        'ret_5d', 'close_return', 'r_vol', 'close', 'amount'
    ]].copy()
    out['ret_5d'] = (out['ret_5d'] * 100).round(2)
    out['amount_亿'] = (out['amount'] / 1e8).round(2)
    out = out.drop(columns='amount')
    fname = f"select_{signal_date}.csv"
    out.to_csv(fname, index=False, encoding='utf-8-sig')
    print(f"结果已保存：{fname}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='强势股缩量回调选股器')
    parser.add_argument('date', nargs='?', default=None,
                        help='信号日期 YYYY-MM-DD（默认：今天）')
    parser.add_argument('--sector', default='sw2_industry',
                        help='板块类型（默认：sw1_industry）')
    parser.add_argument('--top', type=int, default=20,
                        help='热门板块前N名（默认：10）')
    parser.add_argument('--lookback', type=int, default=10,
                        help='板块热度回溯天数（默认：10）')
    args = parser.parse_args()

    date_input = args.date or datetime.today().strftime('%Y-%m-%d')
    if not args.date:
        print(f"未指定日期，使用今天：{date_input}")

    select(date_input,
           sector_type=args.sector,
           sector_top_n=args.top,
           sector_lookback=args.lookback)