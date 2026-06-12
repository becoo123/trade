# -*- coding: utf-8 -*-
"""
按 SW1 行业分类分析创业板缩量调整策略的收益分布
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from data_core import StockDataManager


def run():
    # 1. 加载交易明细
    trades = pd.read_csv('gem_pullback_trades.csv', encoding='utf-8-sig')
    trades['signal_date'] = pd.to_datetime(trades['signal_date'])
    print(f"共 {len(trades)} 笔交易，{trades['code'].nunique()} 只股票")

    # 2. 获取每只股票的 SW1 行业
    dm = StockDataManager()
    dm._ensure_block_manager()
    bm = dm.block_manager

    # 从板块数据中提取 sw1_industry 映射
    if 'sw1_industry' not in bm.blocks:
        print("❌ 未找到 sw1_industry 数据")
        return

    sw1_df = bm.blocks['sw1_industry'][['stock_code', 'block_name']].copy()
    sw1_df.columns = ['code', 'sw1']
    # 去除代码中的交易所后缀
    sw1_df['code'] = sw1_df['code'].str.split('.').str[0]
    # 一只股票可能属于多个板块，取第一个
    sw1_df = sw1_df.drop_duplicates('code')

    # 3. 合并行业信息（统一 code 为 6 位字符串）
    trades['code'] = trades['code'].astype(str).str.zfill(6)
    sw1_df['code'] = sw1_df['code'].astype(str).str.zfill(6)
    trades = trades.merge(sw1_df, on='code', how='left')
    no_industry = trades['sw1'].isna().sum()
    print(f"无法匹配行业的交易: {no_industry} 笔")
    trades['sw1'] = trades['sw1'].fillna('未分类')

    # 4. 按行业统计
    def industry_stats(g):
        n         = len(g)
        win_rate  = (g['return'] > 0).mean() * 100
        avg_ret   = g['return'].mean() * 100
        avg_win   = g[g['return'] > 0]['return'].mean() * 100 if (g['return'] > 0).any() else 0
        avg_loss  = g[g['return'] < 0]['return'].mean() * 100 if (g['return'] < 0).any() else 0
        cum_ret   = (1 + g['return']).prod() - 1   # 等权累积（每笔独立）
        return pd.Series({
            '交易笔数':   n,
            '胜率%':      round(win_rate, 1),
            '平均收益%':  round(avg_ret, 3),
            '平均盈利%':  round(avg_win, 3),
            '平均亏损%':  round(avg_loss, 3),
            '累积收益%':  round(cum_ret * 100, 2),
        })

    stats = trades.groupby('sw1').apply(industry_stats).reset_index()
    stats = stats.sort_values('累积收益%', ascending=False)

    print("\n" + "="*70)
    print("按 SW1 行业分类的策略表现（按累积收益排序）")
    print("="*70)
    print(stats.to_string(index=False))

    # 5. 保存
    stats.to_csv('gem_sector_stats.csv', index=False, encoding='utf-8-sig')
    print(f"\n行业统计已保存: gem_sector_stats.csv")

    # 6. 按行业计算日度累积收益曲线（用于绘图）
    #    思路：把每笔交易按信号日归到对应行业，每个信号日该行业收益 = 当日持仓等权平均
    trades['year_month'] = trades['signal_date'].dt.to_period('M')

    # 取交易笔数 >= 30 的行业画图
    top_industries = stats[stats['交易笔数'] >= 30]['sw1'].tolist()

    _plot_by_industry(trades, top_industries)


def _plot_by_industry(trades: pd.DataFrame, industries: list):
    """按行业绘制每月平均收益热力图 + 累积收益曲线"""
    if not industries:
        print("没有足够数据绘图")
        return

    # --- 图1: 各行业累积收益曲线 ---
    fig, ax = plt.subplots(figsize=(14, 7))

    for ind in industries:
        sub = trades[trades['sw1'] == ind].copy()
        # 按信号日等权汇总 → 日度收益
        daily = sub.groupby('signal_date')['return'].mean().reset_index()
        daily = daily.sort_values('signal_date')
        daily['cum'] = (1 + daily['return']).cumprod() - 1
        ax.plot(daily['signal_date'], daily['cum'] * 100,
                linewidth=1.2, label=ind, alpha=0.8)

    ax.axhline(0, color='black', linestyle='--', linewidth=0.8)
    ax.set_title('Cumulative Return by SW1 Industry (%)', fontsize=13)
    ax.set_ylabel('Cumulative Return (%)')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, ncol=2, loc='lower left')

    plt.tight_layout()
    plt.savefig('gem_sector_cum_return.png', dpi=150, bbox_inches='tight')
    print("行业累积收益曲线已保存: gem_sector_cum_return.png")
    plt.close(fig)

    # --- 图2: 各行业关键指标横向对比（条形图）---
    stats_sub = trades[trades['sw1'].isin(industries)].groupby('sw1').apply(
        lambda g: pd.Series({
            '累积收益%': round(((1 + g['return']).prod() - 1) * 100, 2),
            '胜率%':     round((g['return'] > 0).mean() * 100, 1),
            '笔数':      len(g),
        })
    ).reset_index().sort_values('累积收益%', ascending=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, max(5, len(stats_sub) * 0.4 + 1)))

    colors_cum = ['green' if v >= 0 else 'red' for v in stats_sub['累积收益%']]
    axes[0].barh(stats_sub['sw1'], stats_sub['累积收益%'], color=colors_cum, alpha=0.75)
    axes[0].axvline(0, color='black', linewidth=0.8)
    axes[0].set_title('Cumulative Return % by Industry')
    axes[0].set_xlabel('Cumulative Return (%)')
    axes[0].grid(True, axis='x', alpha=0.3)

    axes[1].barh(stats_sub['sw1'], stats_sub['胜率%'], color='steelblue', alpha=0.75)
    axes[1].axvline(50, color='gray', linestyle='--', linewidth=0.8)
    axes[1].set_title('Win Rate % by Industry')
    axes[1].set_xlabel('Win Rate (%)')
    axes[1].set_xlim(30, 70)
    axes[1].grid(True, axis='x', alpha=0.3)

    plt.tight_layout()
    plt.savefig('gem_sector_comparison.png', dpi=150, bbox_inches='tight')
    print("行业对比图已保存: gem_sector_comparison.png")
    plt.close(fig)


if __name__ == '__main__':
    run()
