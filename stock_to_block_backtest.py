# -*- coding: utf-8 -*-
"""
Stock to Block Strategy Backtest
策略逻辑：
1. 每天计算前一日的相对成交量 (prev_r_vol) 和当日的开盘涨幅 (open_gap)
2. 筛选 prev_r_vol 前 50% 的股票
3. 在其中筛选 open_gap 前 20% 的股票
4. 统计这些股票所属板块，按活跃度（入选数/总数）排序，取 Top 5 板块
5. 买入这 Top 5 板块中入选的股票
6. 交易方式：当日开盘买入，次日开盘卖出 (Open to Open)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict
from datetime import datetime, timedelta
from data_core import StockDataManager
import os

class StockToBlockBacktester:
    def __init__(self):
        self.dm = StockDataManager()
        
    def run_backtest(self, 
                     start_date: str, 
                     end_date: str, 
                     lookback_days: int = 20, 
                     top_vol_pct: float = 0.5, 
                     top_gain_pct: float = 0.2,
                     block_types: List[str] = ['sw1_industry']):
        
        print(f"开始回测: {start_date} 至 {end_date}")
        print(f"策略参数: 回溯{lookback_days}天, 昨量Top{int(top_vol_pct*100)}%, 开盘涨幅Top{int(top_gain_pct*100)}%")
        
        # 1. 加载数据
        # 需要额外加载数据以计算移动平均和前一日数据
        load_start_date = self._get_start_date(start_date, lookback_days + 30)
        
        print("正在加载数据...")
        cols = ['code', 'day', 'open', 'close', 'amount', 'flag']
        df = self.dm.load_day_data(start_date=load_start_date, end_date=end_date, columns=cols)
        
        if df.empty:
            print("❌ 未加载到数据")
            return

        # 2. 预计算指标
        print("计算因子和收益率...")
        df = df.sort_values(['code', 'day'])
        
        # 计算前一日收盘价 (用于计算开盘涨幅)
        df['pre_close'] = df.groupby('code')['close'].shift(1)
        
        # 计算开盘涨幅 (Open Gap)
        df['open_gap'] = df['open'] / df['pre_close'] - 1
        
        # 计算过去N日均量 (不含当日)
        # 原始 ma_vol 是 rolling(N).mean().shift(1) -> 这里的 shift(1) 是相对于 T 而言的 T-1
        # 但我们需要的是 "前一日的相对成交量"，即 T-1 日的 amount / T-1 日的 ma_vol
        # 所以我们先算每日的 r_vol，然后 shift(1)
        
        # 步骤 A: 计算每日均量 (MA of previous N days relative to 'current' row)
        # transform(lambda x: x.rolling(...).mean().shift(1)) 得到的是当日视角的"过去N日均量"
        df['ma_vol'] = df.groupby('code')['amount'].transform(
            lambda x: x.rolling(window=lookback_days, min_periods=max(1, lookback_days//2)).mean().shift(1)
        )
        
        # 步骤 B: 计算每日相对成交量 (Amount / MA_Vol)
        # 这里处理除0
        df['r_vol'] = np.where(df['ma_vol'] > 0, df['amount'] / df['ma_vol'], np.nan)
        
        # 步骤 C: 获取 "前一日" 的相对成交量 (Prev R_Vol)
        df['prev_r_vol'] = df.groupby('code')['r_vol'].shift(1)
        
        # 计算 Open to Open 收益率
        # T日买入(Open_T)，T+1日卖出(Open_T+1)
        # Return = Open_T+1 / Open_T - 1
        df['next_open'] = df.groupby('code')['open'].shift(-1)
        df['open_ret'] = df['next_open'] / df['open'] - 1
        
        # 3. 确保 BlockManager 已加载
        if not hasattr(self.dm, 'block_manager') or self.dm.block_manager is None:
            self.dm._ensure_block_manager()
        bm = self.dm.block_manager
        
        # 预先缓存板块成分股数量，加速计算
        block_total_counts = {}
        for b_type in block_types:
            if b_type in bm.blocks:
                # {block_name: count}
                counts = bm.blocks[b_type]['block_name'].value_counts().to_dict()
                if b_type not in block_total_counts:
                    block_total_counts[b_type] = {}
                block_total_counts[b_type].update(counts)
        
        # 4. 按日回测
        trade_dates = sorted(df[df['day'] >= start_date]['day'].unique())
        daily_returns = []
        
        print(f"共 {len(trade_dates)} 个交易日")
        
        all_trades = []
        
        for trade_date in trade_dates:
            # 筛选当日数据
            day_df = df[df['day'] == trade_date].copy()
            
            # 基础过滤：有 prev_r_vol, open_gap, 且非停牌(amount>0)
            valid_mask = (day_df['prev_r_vol'].notna()) & \
                         (day_df['open_gap'].notna()) & \
                         (day_df['amount'] > 0)
            day_df = day_df[valid_mask]
            
            if day_df.empty:
                daily_returns.append({'day': trade_date, 'return': 0.0, 'count': 0})
                continue
                
            total_stocks = len(day_df)
            
            # --- 筛选逻辑 ---
            
            # 1. 昨量 Top %
            top_vol_n = int(total_stocks * top_vol_pct)
            top_vol_n = max(1, top_vol_n)
            
            vol_df = day_df.sort_values('prev_r_vol', ascending=False).head(top_vol_n)
            
            # 2. 开盘涨幅 Top %
            top_gain_n = int(len(vol_df) * top_gain_pct)
            top_gain_n = max(1, top_gain_n)
            
            final_df = vol_df.sort_values('open_gap', ascending=False).head(top_gain_n)
            
            selected_codes = final_df['code'].tolist()
            
            # --- 板块定位 ---
            # 统计入选股票的板块分布
            block_scores = {} # (block_type, block_name) -> {count, stocks}
            
            for code in selected_codes:
                blocks_map = self.dm.get_blocks_by_stock(code)
                if not blocks_map:
                    continue
                
                for b_type in block_types:
                    type_blocks = blocks_map.get(b_type, [])
                    if isinstance(type_blocks, str):
                        type_blocks = [type_blocks]
                    
                    for block in type_blocks:
                        key = (b_type, block)
                        if key not in block_scores:
                            block_scores[key] = {'count': 0, 'stocks': []}
                        block_scores[key]['count'] += 1
                        block_scores[key]['stocks'].append(code)
            
            # 计算活跃度并排序
            ranked_blocks = []
            for (b_type, block), info in block_scores.items():
                count = info['count']
                
                # 获取总数
                total = block_total_counts.get(b_type, {}).get(block, 0)
                if total == 0:
                    # 尝试实时获取
                    total = len(self.dm.get_stocks_in_block(block, b_type))
                
                if total == 0: continue
                
                # 过滤小样本
                if count < 3: continue
                
                active_ratio = count / total
                ranked_blocks.append({
                    'type': b_type,
                    'block': block,
                    'active_ratio': active_ratio,
                    'stocks': info['stocks']
                })
            
            # 按活跃度排序，取 Top 5
            ranked_blocks.sort(key=lambda x: x['active_ratio'], reverse=True)
            top_blocks = ranked_blocks[:5]
            
            # --- 构建投资组合 ---
            # 取 Top 5 板块中的入选股票
            target_stocks = set()
            stock_source_block = {} # 记录股票来源板块
            
            for b in top_blocks:
                for stock in b['stocks']:
                    target_stocks.add(stock)
                    # 记录主要板块（如果有多个，记录第一个遇到的Top板块）
                    if stock not in stock_source_block:
                        stock_source_block[stock] = b['block']
            
            # 计算收益
            if not target_stocks:
                daily_ret = 0.0
                stock_count = 0
            else:
                # 获取这些股票的 open_ret
                target_df = day_df[day_df['code'].isin(target_stocks)].copy()
                
                # 记录每笔交易
                for _, row in target_df.iterrows():
                    trade_ret = row['open_ret']
                    if pd.isna(trade_ret):
                        trade_ret = 0.0
                        
                    all_trades.append({
                        'code': row['code'],
                        'buy_date': trade_date,
                        'buy_price': row['open'],
                        'sell_price': row['next_open'],
                        'return': trade_ret,
                        'return_pct': trade_ret * 100,
                        'block': stock_source_block.get(row['code'], '')
                    })
                
                # 组合收益（等权）
                daily_ret = target_df['open_ret'].mean()
                stock_count = len(target_df)
                
                # 如果全是 NaN (比如最后一天没有 next_open)，则收益为0
                if pd.isna(daily_ret):
                    daily_ret = 0.0
            
            daily_returns.append({
                'day': trade_date,
                'return': daily_ret,
                'count': stock_count,
                'top_blocks': [b['block'] for b in top_blocks]
            })
            
            if len(daily_returns) % 10 == 0:
                print(f"{trade_date}: 持仓 {stock_count} 只, 收益 {daily_ret*100:.2f}%")
                
        # 5. 结果分析
        # 保存交易明细
        trades_df = pd.DataFrame(all_trades)
        if not trades_df.empty:
            trades_df.to_csv("stock_to_block_trades.csv", index=False, encoding='utf-8-sig')
            print(f"\n交易明细已保存至 stock_to_block_trades.csv (共 {len(trades_df)} 笔交易)")
            
            # 交易统计
            win_rate = (trades_df['return'] > 0).mean() * 100
            avg_profit = trades_df[trades_df['return'] > 0]['return'].mean() * 100
            avg_loss = trades_df[trades_df['return'] <= 0]['return'].mean() * 100
            
            print("\n" + "-"*30)
            print("交易统计")
            print("-"*30)
            print(f"胜率: {win_rate:.2f}%")
            print(f"平均盈利: {avg_profit:.2f}%")
            print(f"平均亏损: {avg_loss:.2f}%")
            print(f"盈亏比: {abs(avg_profit/avg_loss):.2f}" if avg_loss != 0 else "盈亏比: Inf")
            
            # 显示表现最好的前10笔交易
            print("\nTop 10 盈利交易:")
            top_wins = trades_df.sort_values('return', ascending=False).head(10)
            for _, row in top_wins.iterrows():
                print(f"{row['buy_date']} {row['code']} ({row['block']}): {row['return_pct']:.2f}%")
        
        res_df = pd.DataFrame(daily_returns)
        res_df['cum_return'] = (1 + res_df['return']).cumprod()
        
        print("\n" + "="*50)
        print("回测结果摘要")
        print("="*50)
        print(f"总交易日: {len(res_df)}")
        print(f"累积收益: {(res_df['cum_return'].iloc[-1] - 1)*100:.2f}%")
        if len(res_df) > 0:
            avg_daily = res_df['return'].mean()
            sharpe = avg_daily / res_df['return'].std() * np.sqrt(250) if res_df['return'].std() > 0 else 0
            print(f"年化收益: {(pow(1+avg_daily, 250)-1)*100:.2f}%")
            print(f"夏普比率: {sharpe:.2f}")
            print(f"日均持仓: {res_df['count'].mean():.1f} 只")
        
        # 保存结果
        res_df.to_csv("stock_to_block_backtest.csv", index=False, encoding='utf-8-sig')
        print(f"详细结果已保存至 stock_to_block_backtest.csv")
        
        # 简单绘图
        try:
            plt.figure(figsize=(12, 6))
            plt.plot(pd.to_datetime(res_df['day']), res_df['cum_return'], label='Strategy')
            plt.title('Stock to Block Strategy (Open-to-Open)')
            plt.grid(True)
            plt.legend()
            plt.savefig('stock_to_block_backtest.png')
            print("收益曲线已保存至 stock_to_block_backtest.png")
        except Exception as e:
            print(f"绘图失败: {e}")

    def _get_start_date(self, end_date: str, days: int) -> str:
        dt = datetime.strptime(end_date, "%Y-%m-%d")
        start_dt = dt - timedelta(days=days * 1.6 + 20) 
        return start_dt.strftime("%Y-%m-%d")

if __name__ == "__main__":
    backtester = StockToBlockBacktester()
    
    # 设置回测区间
    end_date = "2026-03-02"
    start_date = "2025-10-01" # 回测最近半年
    
    import sys
    if len(sys.argv) > 2:
        start_date = sys.argv[1]
        end_date = sys.argv[2]
        
    backtester.run_backtest(
        start_date=start_date,
        end_date=end_date,
        top_vol_pct=0.5,
        top_gain_pct=0.2,
        block_types=['sw1_industry', 'sw2_industry'] # 混合行业级别
    )
