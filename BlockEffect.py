"""
板块赚钱效应量化框架
基于 StockDataManager 实现
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
from data_core import StockDataManager
from scipy import stats

class BlockEffectCalculator:
    """板块赚钱效应计算引擎"""
    
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
        """
        计算单个板块的赚钱效应指标
        
        Parameters:
        -----------
        block_name : 板块名称，如 'SW1家用电器'
        block_type : 板块类型，如 'sw1_industry'
        start_date : 开始日期 'YYYY-MM-DD'
        end_date : 结束日期
        benchmark_codes : 基准指数成分股（用于计算相对强弱），默认用全市场
        
        Returns:
        --------
        DataFrame with columns:
            - day: 交易日
            - rs_score: 相对强弱得分
            - breadth_score: 广度得分
            - tradable_score: 可交易性得分
            - flow_score: 资金流向得分
            - effect_score: 综合赚钱效应得分 (0-100)
        """
        # 1. 获取板块成分股
        block_stocks = self.manager.get_stocks_in_block(block_name, block_type)
        if not block_stocks:
            raise ValueError(f"板块 {block_name} 无成分股")
        
        # 2. 加载板块日线数据（只读必要字段，提升性能）
        df_block = self.manager.load_day_data(
            start_date=start_date,
            end_date=end_date,
            codes=block_stocks,
            columns=['code', 'day', 'open', 'volume', 'preclose', 'flag',
                    'open_return', 'close_return', 'high_return', 'low_return','limit_price']
        )
        
        # 3. 加载基准数据
        if benchmark_codes is None:
            # 使用全市场作为基准（可优化为只读指数）
            df_bench = self.manager.load_day_data(
                start_date=start_date,
                end_date=end_date,
                columns=['day', 'close_return']
            )
            bench_return = df_bench.groupby('day')['close_return'].mean()
        else:
            df_bench = self.manager.load_day_data(
                start_date=start_date,
                end_date=end_date,
                codes=benchmark_codes,
                columns=['day', 'close_return']
            )
            bench_return = df_bench.groupby('day')['close_return'].mean()
        
        # 4. 按日期聚合计算各项指标
        daily_metrics = []
        
        for day, group in df_block.groupby('day'):
            metrics = {
                'day': day,
                'stock_count': len(group)
            }
            
            # === 模块1: 收益强度 ===
            rs_metrics = self._calculate_rs_score(group, bench_return, day)
            metrics.update(rs_metrics)
            
            # === 模块2: 可交易性 ===
            tradable_metrics = self._calculate_tradable_score(group)
            metrics.update(tradable_metrics)
            
            # === 模块3: 资金流向 ===
            flow_metrics = self._calculate_flow_score(group)
            metrics.update(flow_metrics)
            
            daily_metrics.append(metrics)
        
        df_metrics = pd.DataFrame(daily_metrics).sort_values('day')
        
        # 5. 计算滚动窗口指标（需要历史数据）
        df_metrics = self._add_rolling_features(df_metrics)
        
        # 6. 计算溢价质量（需要未来收益，用于回测评估）
        df_metrics = self._calculate_edge_quality(df_block, df_metrics)
        
        # 7. 综合评分
        df_metrics = self._calculate_final_score(df_metrics)
        
        return df_metrics
    
    def _calculate_rs_score(
        self, 
        group: pd.DataFrame, 
        bench_return: pd.Series,
        day: str
    ) -> Dict:
        """计算相对强弱得分"""
        # 板块当日平均收益（用preclose计算当日收益）
        group = group.copy()
        group['daily_return'] = (group['close'] - group['preclose']) / group['preclose'] * 100
        block_return = group['daily_return'].mean()
        
        # 相对强弱
        bench_ret = bench_return.get(day, 0)
        rs_1d = block_return - bench_ret
        
        # 上涨广度
        up_ratio = (group['daily_return'] > 0).sum() / len(group)
        
        return {
            'block_return': block_return,
            'bench_return': bench_ret,
            'rs_1d': rs_1d,
            'up_ratio': up_ratio
        }
    
    def _calculate_tradable_score(self, group: pd.DataFrame) -> Dict:
        """计算可交易性得分"""
        # 剔除一字涨停（开盘=最高=最低=收盘）
        tradable = group[
            ~((group['open'] == group['high']) & 
              (group['high'] == group['low']) & 
              (group['low'] == group['close']) &
              (group['flag'] == 1))
        ].copy()
        
        tradable_ratio = len(tradable) / len(group) if len(group) > 0 else 0
        
        # 收盘强度（收盘价接近最高价的程度）
        if len(tradable) > 0:
            tradable['close_to_high'] = np.where(
                tradable['high'] > tradable['low'],
                (tradable['close'] - tradable['low']) / (tradable['high'] - tradable['low']),
                0.5
            )
            avg_close_strength = tradable['close_to_high'].mean()
            weak_close_ratio = (tradable['close_to_high'] < 0.3).sum() / len(tradable)
        else:
            avg_close_strength = 0
            weak_close_ratio = 1.0
        
        # 可交易个股的次日期望收益（使用已shift的open_return）
        if len(tradable) > 0:
            expected_next_return = tradable['open_return'].mean()  # 次日开盘收益
        else:
            expected_next_return = 0
        
        return {
            'tradable_ratio': tradable_ratio,
            'avg_close_strength': avg_close_strength,
            'weak_close_ratio': weak_close_ratio,
            'expected_next_return': expected_next_return
        }
    
    def _calculate_flow_score(self, group: pd.DataFrame) -> Dict:
        """计算资金流向得分"""
        group = group.copy()
        group['daily_return'] = (group['close'] - group['preclose']) / group['preclose'] * 100
        
        # 上涨成交额占比
        up_stocks = group[group['daily_return'] > 0]
        up_amount_ratio = up_stocks['amount'].sum() / group['amount'].sum() if group['amount'].sum() > 0 else 0.5
        
        # 涨停比例
        limit_up_ratio = (group['flag'] == 1).sum() / len(group)
        
        # 相对成交（需要历史均值，这里先占位）
        avg_amount = group['amount'].mean()
        
        return {
            'up_amount_ratio': up_amount_ratio,
            'limit_up_ratio': limit_up_ratio,
            'avg_amount': avg_amount,
            'total_amount': group['amount'].sum()
        }
    
    def _add_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """添加滚动窗口特征"""
        df = df.sort_values('day').copy()
        
        # RS多周期加权
        df['rs_5d'] = df['rs_1d'].rolling(5, min_periods=1).mean()
        df['rs_10d'] = df['rs_1d'].rolling(10, min_periods=1).mean()
        df['rs_20d'] = df['rs_1d'].rolling(20, min_periods=1).mean()
        df['rs_score'] = 0.4 * df['rs_5d'] + 0.3 * df['rs_10d'] + 0.3 * df['rs_20d']
        
        # 广度趋势
        df['breadth_ma5'] = df['up_ratio'].rolling(5, min_periods=1).mean()
        df['breadth_slope'] = df['up_ratio'].rolling(10, min_periods=2).apply(
            lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) >= 2 else 0,
            raw=True
        )
        df['breadth_score'] = df['breadth_ma5'] * 0.6 + df['breadth_slope'] * 10 * 0.4  # 斜率放大
        
        # 相对成交放大
        df['amount_ma20'] = df['avg_amount'].rolling(20, min_periods=1).mean()
        df['relative_amount'] = df['avg_amount'] / df['amount_ma20']
        
        # 可交易性综合
        df['tradable_score'] = (
            df['tradable_ratio'] * 0.4 +
            df['avg_close_strength'] * 0.3 +
            (1 - df['weak_close_ratio']) * 0.3
        )
        
        # 资金流向综合
        df['flow_score'] = (
            df['up_amount_ratio'] * 0.5 +
            df['relative_amount'].clip(0, 2) / 2 * 0.3 +  # 归一化到0-1
            (1 - df['limit_up_ratio'].clip(0, 0.5) / 0.5) * 0.2  # 过热惩罚
        )
        
        return df
    
    def _calculate_edge_quality(
        self, 
        df_block: pd.DataFrame, 
        df_metrics: pd.DataFrame
    ) -> pd.DataFrame:
        """
        计算溢价质量（Edge IR）
        这是回测评估指标，实盘时用历史滚动估计
        """
        # 按日期聚合可交易股票的次日收益
        daily_edge = []
        
        for day in df_metrics['day'].unique():
            day_stocks = df_block[df_block['day'] == day].copy()
            
            # 剔除一字板
            tradable = day_stocks[
                ~((day_stocks['open'] == day_stocks['high']) & 
                  (day_stocks['high'] == day_stocks['low']) & 
                  (day_stocks['low'] == day_stocks['close']) &
                  (day_stocks['flag'] == 1))
            ]
            
            if len(tradable) > 0:
                # 次日收益（已经shift过）
                edge_1d = tradable['open_return'].mean()
                edge_3d = (tradable['open_return'] + tradable['close_return']).mean()  # 简化
            else:
                edge_1d = 0
                edge_3d = 0
            
            daily_edge.append({
                'day': day,
                'edge_1d': edge_1d,
                'edge_3d': edge_3d
            })
        
        df_edge = pd.DataFrame(daily_edge)
        df_metrics = df_metrics.merge(df_edge, on='day', how='left')
        
        # 滚动IR（信息比率）
        df_metrics['edge_1d_ma20'] = df_metrics['edge_1d'].rolling(20, min_periods=5).mean()
        df_metrics['edge_1d_std20'] = df_metrics['edge_1d'].rolling(20, min_periods=5).std()
        df_metrics['edge_ir'] = df_metrics['edge_1d_ma20'] / (df_metrics['edge_1d_std20'] + 1e-6)
        
        # 归一化到0-1（IR一般在-2到2之间）
        df_metrics['edge_score'] = (df_metrics['edge_ir'].clip(-2, 2) + 2) / 4
        
        return df_metrics
    
    def _calculate_final_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算最终效应得分（0-100）"""
        # 归一化各子得分到0-1
        df['rs_score_norm'] = (df['rs_score'] - df['rs_score'].rolling(60).min()) / \
                               (df['rs_score'].rolling(60).max() - df['rs_score'].rolling(60).min() + 1e-6)
        df['breadth_score_norm'] = df['breadth_score'].clip(0, 1)
        df['tradable_score_norm'] = df['tradable_score'].clip(0, 1)
        df['flow_score_norm'] = df['flow_score'].clip(0, 1)
        df['edge_score_norm'] = df['edge_score'].fillna(0.5).clip(0, 1)
        
        # 加权合成（可调整权重）
        df['effect_score'] = (
            df['rs_score_norm'] * 0.25 +
            df['breadth_score_norm'] * 0.15 +
            df['tradable_score_norm'] * 0.20 +
            df['flow_score_norm'] * 0.20 +
            df['edge_score_norm'] * 0.20
        ) * 100
        
        return df


class MultiBlockEffectAnalyzer:
    """多板块赚钱效应分析器"""
    
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
        """
        分析某类型下所有板块的赚钱效应
        
        Returns:
        --------
        DataFrame: 每个板块每日的效应得分，可用于排序选板块
        """
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
                df_effect['block_name'] = block_name
                all_results.append(df_effect)
            except Exception as e:
                print(f"计算板块 {block_name} 失败: {e}")
                continue
        
        if not all_results:
            return pd.DataFrame()
        
        df_all = pd.concat(all_results, ignore_index=True)
        
        # 每日排名
        df_all['rank'] = df_all.groupby('day')['effect_score'].rank(ascending=False, method='min')
        
        return df_all
    
    def get_top_blocks_by_date(
        self,
        df_all: pd.DataFrame,
        date: str,
        top_n: int = 5,
        min_score: float = 60
    ) -> List[str]:
        """获取某日Top N板块"""
        df_day = df_all[df_all['day'] == date].copy()
        df_day = df_day[df_day['effect_score'] >= min_score]
        df_day = df_day.nsmallest(top_n, 'rank')
        return df_day['block_name'].tolist()
    
if __name__ == "__main__":
    manager = StockDataManager()
    analyzer = MultiBlockEffectAnalyzer(manager)
    # 初始化
    df_all_blocks = analyzer.analyze_all_blocks(
    block_type='sw1_industry',
    start_date='2025-10-01',
    end_date='2026-02-24'
    )
    
    # 查看最近一个交易日的板块排名
    latest_date = df_all_blocks['day'].max()
    top_blocks = analyzer.get_top_blocks_by_date(
        df_all=df_all_blocks,
        date=latest_date,
        top_n=5,
        min_score=35
    )
    
    print(f"\n{latest_date} 赚钱效应Top5板块:")
    print(top_blocks)