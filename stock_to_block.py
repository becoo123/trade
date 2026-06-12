# -*- coding: utf-8 -*-
"""
从个股到板块的自下而上分析策略
1. 计算所有股票的相对成交量
2. 取相对成交量前10%的股票
3. 在这些股票中取涨幅前20%的股票
4. 统计这些股票所属的板块，定位最热闹的板块
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
from collections import Counter
from datetime import datetime, timedelta
from data_core import StockDataManager

class StockToBlockAnalyzer:
    def __init__(self):
        self.dm = StockDataManager()
        
    def analyze(self, 
                target_date: str, 
                lookback_days: int = 20, 
                top_vol_pct: float = 0.1, 
                top_gain_pct: float = 0.2,
                block_types: List[str] = ['concept', 'sw2_industry']):
        """
        执行分析
        
        Args:
            target_date: 目标日期
            lookback_days: 回溯天数（用于计算均量）
            top_vol_pct: 相对成交量筛选比例 (0.1 = Top 10%)
            top_gain_pct: 涨幅筛选比例 (0.2 = Top 20%)
            block_types: 需要分析的板块类型列表
        """
        print(f"正在分析 {target_date} 的市场热度...")
        print(f"参数设置: 回溯{lookback_days}天, 相对量Top{int(top_vol_pct*100)}%, 涨幅Top{int(top_gain_pct*100)}%")
        
        # 1. 加载数据
        start_date = self._get_start_date(target_date, lookback_days + 20)
        
        cols = ['code', 'day', 'open', 'close', 'amount', 'flag']
        df = self.dm.load_day_data(start_date=start_date, end_date=target_date, columns=cols)
        
        if df.empty:
            print("❌ 未加载到数据")
            return
            
        print(f"加载数据: {len(df)} 行, {df['code'].nunique()} 只股票")
        
        # 2. 计算指标 (全量计算)
        print("计算技术指标...")
        df = df.sort_values(['code', 'day'])
        
        # 计算前一日收盘价 (用于计算涨幅)
        df['pre_close'] = df.groupby('code')['close'].shift(1)
        df['pct_chg'] = (df['close'] - df['pre_close']) / df['pre_close']
        
        # 计算过去N日均量 (不含当日，shift 1)
        # min_periods 设为 lookback_days 的一半，避免初期数据不足
        df['ma_vol'] = df.groupby('code')['amount'].transform(
            lambda x: x.rolling(window=lookback_days, min_periods=max(1, lookback_days//2)).mean().shift(1)
        )
        
        # 3. 筛选当日数据
        daily_df = df[df['day'] == target_date].copy()
        
        # 过滤无效数据
        # 必须有 ma_vol 且 > 0
        # 必须有 pct_chg (非停牌/首日)
        valid_mask = (daily_df['ma_vol'] > 0) & (daily_df['pct_chg'].notna()) & (daily_df['amount'] > 0)
        daily_df = daily_df[valid_mask].copy()
        
        # 计算相对成交量
        daily_df['r_vol'] = daily_df['amount'] / daily_df['ma_vol']
        
        total_stocks = len(daily_df)
        print(f"当日有效股票数: {total_stocks}")
        
        if total_stocks == 0:
            print("❌ 当日无有效数据")
            return

        # 4. 第一步筛选：相对成交量 Top 10%
        top_vol_n = int(total_stocks * top_vol_pct)
        top_vol_n = max(1, top_vol_n)
        
        vol_df = daily_df.sort_values('r_vol', ascending=False).head(top_vol_n)
        
        print(f"\n【第一步】筛选相对成交量前 {int(top_vol_pct*100)}% ({top_vol_n}只)")
        print(f"  - 阈值: {vol_df['r_vol'].min():.2f}")
        print(f"  - 均值: {vol_df['r_vol'].mean():.2f}")
        
        # 5. 第二步筛选：涨幅 Top 20%
        top_gain_n = int(len(vol_df) * top_gain_pct)
        top_gain_n = max(1, top_gain_n)
        
        final_df = vol_df.sort_values('pct_chg', ascending=False).head(top_gain_n)
        
        print(f"\n【第二步】筛选其中涨幅前 {int(top_gain_pct*100)}% ({top_gain_n}只)")
        print(f"  - 阈值: {final_df['pct_chg'].min()*100:.2f}%")
        print(f"  - 均值: {final_df['pct_chg'].mean()*100:.2f}%")
        
        selected_codes = final_df['code'].tolist()
        
        # 6. 统计板块热度
        print("\n" + "="*50)
        print("🔥 热门板块分析结果")
        print("="*50)
        
        for b_type in block_types:
            block_counts = Counter()
            stock_lists = {} # block -> [stock_code, ...]
            
            # 遍历选中的股票，统计板块
            for code in selected_codes:
                blocks_map = self.dm.get_blocks_by_stock(code)
                if not blocks_map:
                    continue
                    
                # 获取指定类型的板块列表
                type_blocks = blocks_map.get(b_type, [])
                
                # 统一转为列表
                if isinstance(type_blocks, str):
                    type_blocks = [type_blocks]
                elif not isinstance(type_blocks, list):
                    continue
                    
                for block in type_blocks:
                    block_counts[block] += 1
                    if block not in stock_lists:
                        stock_lists[block] = []
                    stock_lists[block].append(code)
            
            if not block_counts:
                print(f"\n【{b_type}】无数据")
                continue
                
            # 统计并计算活跃度
            block_stats = []
            
            # 获取板块总股数以计算活跃度
            # 为避免重复读取文件，简单缓存一下板块信息
            # 注意：这里假设 block_manager 已加载数据
            if not hasattr(self.dm, 'block_manager') or self.dm.block_manager is None:
                self.dm._ensure_block_manager()
            
            bm = self.dm.block_manager
            
            for block, count in block_counts.items():
                # 获取该板块总股票数
                # 优先直接查 DataFrame，如果不行则调用 get_stocks_in_block
                total_count = 0
                if b_type in bm.blocks:
                    df_block = bm.blocks[b_type]
                    # 快速筛选
                    total_count = len(df_block[df_block['block_name'] == block])
                
                if total_count == 0:
                    # 降级方案
                    stocks = self.dm.get_stocks_in_block(block, b_type)
                    total_count = len(stocks)
                
                if total_count == 0:
                    continue
                    
                active_ratio = count / total_count
                
                # 过滤掉入选数量过少的板块（例如少于5只），除非活跃度极高
                # 这里设置一个简单的阈值：至少3只股票入选
                if count < 3:
                    continue
                    
                block_stats.append({
                    'block': block,
                    'count': count,
                    'total': total_count,
                    'active_ratio': active_ratio,
                    'stocks': stock_lists[block]
                })
            
            # 按活跃度排序
            block_stats.sort(key=lambda x: x['active_ratio'], reverse=True)
            
            if not block_stats:
                print(f"\n【{b_type}】无满足条件的数据")
                continue

            # 输出该类型的前10热门板块
            print(f"\n【{b_type}】热门板块 Top 10 (按活跃度排序):")
            print(f"{'板块名称':<15} {'入选/总数':<12} {'活跃度(%)':<10} {'主要贡献股'}")
            print("-" * 70)
            
            for item in block_stats[:10]:
                block = item['block']
                count = item['count']
                total = item['total']
                ratio = item['active_ratio'] * 100
                
                # 获取该板块下的所有选中股票，并按涨幅排序
                stocks_in_block = item['stocks']
                block_stocks_df = final_df[final_df['code'].isin(stocks_in_block)].sort_values('pct_chg', ascending=False)
                
                top_stocks = block_stocks_df['code'].head(3).tolist()
                stocks_show = ",".join(top_stocks)
                if len(stocks_in_block) > 3:
                    stocks_show += f"等{len(stocks_in_block)}只"
                
                print(f"{block:<15} {count}/{total:<11} {ratio:<10.1f} {stocks_show}")

    def _get_start_date(self, end_date: str, days: int) -> str:
        dt = datetime.strptime(end_date, "%Y-%m-%d")
        # 简单推算，多预留一些天数应对假期
        start_dt = dt - timedelta(days=days * 1.6 + 20) 
        return start_dt.strftime("%Y-%m-%d")

if __name__ == "__main__":
    analyzer = StockToBlockAnalyzer()
    # 默认使用最近的一个交易日
    target_date = "2026-03-02" 
    
    # 支持命令行参数或直接修改
    import sys
    if len(sys.argv) > 1:
        target_date = sys.argv[1]
    
    # 可选板块类型: 
    # 'concept' (概念), 'theme_concept' (主题概念)
    # 'sw1_industry' (申万一级), 'sw2_industry' (申万二级), 'sw3_industry' (申万三级)
    # 'csrc1_industry' (证监会一级), 'csrc2_industry' (证监会二级)
    selected_block_types = ['sw1_industry']
    
    analyzer.analyze(
        target_date, 
        lookback_days=20,
        top_vol_pct=0.50,  # 相对成交量前50%
        top_gain_pct=0.2,  # 涨幅前20%
        block_types=selected_block_types
    )
