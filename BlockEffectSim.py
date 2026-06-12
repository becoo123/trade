import pandas as pd
import numpy as np
from typing import List, Optional
from data_core import StockDataManager

class BlockEffectCalculator:
    """板块赚钱效应计算引擎（精简实盘版）"""
    
    # 权重配置（便于调参）
    WEIGHTS = {
        'rs': 0.40,
        'breadth': 0.15,
        'tradable': 0.20,
        'flow': 0.20,
        'edge': 0.05   # 实盘降低权重，防止噪声
    }
    
    def __init__(self, manager: StockDataManager):
        self.manager = manager
    
    def calculate_block_effect(
        self,
        block_name: str,
        block_type: str,
        start_date: str,
        end_date: str,
        benchmark_codes: Optional[List[str]] = None
    ) -> pd.DataFrame:
        # 1. 获取成分股
        block_stocks = self.manager.get_stocks_in_block(block_name, block_type)
        if not block_stocks:
            raise ValueError(f"板块 {block_name} 无成分股")
        
        # 2. 加载板块数据
        df_block = self.manager.load_day_data(
            start_date=start_date, end_date=end_date, codes=block_stocks,
            columns=['code', 'day', 'open', 'high', 'low', 'close', 'amount',
                     'preclose', 'flag', 'open_return']  # open_return 为次日开盘收益（已shift）
        )
        df_block = df_block.sort_values(['code', 'day'])
        
        # 3. 基准收益
        df_bench = self.manager.load_day_data(start_date=start_date, end_date=end_date,
                                              codes=benchmark_codes, columns=['day', 'close_return'])
        bench_ret = df_bench.groupby('day')['close_return'].mean()        
        # 4. 每日指标计算（向量化）
        def _process_day(group: pd.DataFrame, day: str):
            group['ret'] = (group['close'] / group['preclose'] - 1) * 100    
            # 基础指标
            block_ret = group['ret'].mean()
            rs_1d = block_ret - bench_ret.get(day, 0)
            up_ratio = (group['ret'] > 0).mean()
            
            # 可交易子集
            tradable_mask = ((group['open'] != group['limit_price']))
            tradable = group[tradable_mask]
            tradable_ratio = tradable_mask.mean()
            
            close_strength = 0.5
            if len(tradable) > 0:
                hl_range = tradable['high'] - tradable['low']
                close_strength = np.where(hl_range > 0, 
                                         (tradable['close'] - tradable['low']) / hl_range, 0.5).mean()
            
            exp_next_ret = tradable['open_return'].mean() if len(tradable) > 0 else 0
            
            # 资金指标
            up_amount_ratio = group.loc[group['ret'] > 0, 'amount'].sum() / group['amount'].sum()
            up_amount_ratio = 0.5 if pd.isna(up_amount_ratio) else up_amount_ratio
            limit_up_ratio = (group['flag'] == 1).mean()
            total_amount = group['amount'].sum()
            
            return pd.Series({
                'day': day,
                'rs_1d': rs_1d,
                'up_ratio': up_ratio,
                'tradable_ratio': tradable_ratio,
                'close_strength': close_strength,
                'exp_next_ret': exp_next_ret,
                'up_amount_ratio': up_amount_ratio,
                'limit_up_ratio': limit_up_ratio,
                'total_amount': total_amount
            })
        
        daily_metrics = df_block.groupby('day').apply(lambda g: _process_day(g, g.name),include_groups=False)
        daily_metrics = daily_metrics.reset_index(drop=True).sort_values('day')
        
        # 5. 滚动特征
        daily_metrics['rs_5d'] = daily_metrics['rs_1d'].rolling(5, min_periods=3).mean()
        daily_metrics['rs_10d'] = daily_metrics['rs_1d'].rolling(10, min_periods=5).mean()
        daily_metrics['rs_20d'] = daily_metrics['rs_1d'].rolling(20, min_periods=10).mean()
        daily_metrics['rs_score'] = (0.4 * daily_metrics['rs_5d'] + 
                                    0.3 * daily_metrics['rs_10d'] + 
                                    0.3 * daily_metrics['rs_20d']).fillna(0)
        
        daily_metrics['breadth_score'] = daily_metrics['up_ratio'].rolling(5).mean().fillna(0.5)
        
        ma20_amount = daily_metrics['total_amount'].rolling(20).mean()
        daily_metrics['rel_amount'] = daily_metrics['total_amount'] / ma20_amount
        
        daily_metrics['tradable_score'] = (daily_metrics['tradable_ratio'] * 0.5 + 
                                          daily_metrics['close_strength'] * 0.5)
        
        daily_metrics['flow_score'] = (daily_metrics['up_amount_ratio'] * 0.5 +
                                      (daily_metrics['rel_amount'].clip(0.5, 2) - 0.5)/1.5 * 0.3 +
                                      (1 - daily_metrics['limit_up_ratio'].clip(0, 0.4)) * 0.2)
        
        # 6. 实盘edge估计（滚动历史次日收益）
        daily_metrics['edge_score'] = daily_metrics['exp_next_ret'].rolling(30).mean()
        daily_metrics['edge_score'] = (daily_metrics['edge_score'].clip(-3, 6) + 3) / 9  # 归一到0-1
        
        # 7. 最终得分
        def sigmoid(x, center=0, scale=2):
            return 1 / (1 + np.exp(-(x - center) / scale))
        
        daily_metrics['effect_score'] = (
            sigmoid(daily_metrics['rs_score'], center=1, scale=2) * self.WEIGHTS['rs'] +
            daily_metrics['breadth_score'] * self.WEIGHTS['breadth'] +
            daily_metrics['tradable_score'].clip(0, 1) * self.WEIGHTS['tradable'] +
            daily_metrics['flow_score'].clip(0, 1) * self.WEIGHTS['flow'] +
            daily_metrics['edge_score'] * self.WEIGHTS['edge']
        ) * 100
        
        daily_metrics['block_name'] = block_name
        return daily_metrics

class MultiBlockEffectAnalyzer:
    """多板块分析器（批量优化版）"""
    
    def __init__(self, manager: StockDataManager):
        self.manager = manager
        self.calculator = BlockEffectCalculator(manager)
    
    def analyze_all_blocks(
        self,
        block_type: str,
        start_date: str,
        end_date: str,
        top_n: int = 10
    ) -> pd.DataFrame:
        block_list = self.manager.get_block_list(block_type)
        all_results = []
        
        for block_name in block_list:
            try:
                df_effect = self.calculator.calculate_block_effect(
                    block_name=block_name,
                    block_type=block_type,
                    start_date=start_date,
                    end_date=end_date
                )
                all_results.append(df_effect)
            except Exception as e:
                print(f"计算 {block_name} 失败: {e}")
                continue
        
        if not all_results:
            return pd.DataFrame()
        
        df_all = pd.concat(all_results, ignore_index=True)
        df_all['rank'] = df_all.groupby('day')['effect_score'].rank(ascending=False, method='min')
        return df_all
    
    def get_top_blocks_by_date(
        self,
        df_all: pd.DataFrame,
        date: str,
        top_n: int = 5,
        min_score: float = 50
    ) -> List[str]:
        df_day = df_all[(df_all['day'] == date) & (df_all['effect_score'] >= min_score)]
        return df_day.nsmallest(top_n, 'rank')['block_name'].tolist()

if __name__ == "__main__":
    manager = StockDataManager()  # 你的数据管理器
    analyzer = MultiBlockEffectAnalyzer(manager)
    
    df_all = analyzer.analyze_all_blocks(
        block_type='sw2_industry',
        start_date='2025-10-01',
        end_date='2026-02-02'
    )
    
    latest_date = df_all['day'].max()
    top_blocks = analyzer.get_top_blocks_by_date(df_all, latest_date, top_n=5, min_score=50)
    
    print(f"{latest_date} 赚钱效应 Top5 板块：")
    print(top_blocks)