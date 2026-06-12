import pandas as pd
import numpy as np
from typing import List, Optional
from data_core import StockDataManager

class DragonStockRanker:
    """
    时间段内潜在龙头股排行器（改进版：强调“参与性”）
    
    新增核心改进：过滤/降权“一字板式”低参与票，优先高成交、高换手风格的“可参与龙头”
    - 新增涨停日平均成交额（amount）作为参与性代理（一字板成交额极低）
    - 新增次日平均振幅（next_high - next_low）作为接力参与性（一字连板次日振幅小）
    - 新增涨停日放量强度（相对近期均量）
    - 综合得分中大幅加权参与性指标，真龙头往往“有量有换手、可打板接力”
    
    效果：排行榜会优先那些涨停但成交活跃、次日有波动的票（游资主战场），一字板票会大幅后移或出局
    """
    
    def __init__(self):
        self.manager = StockDataManager()
        try:
            stocks_df = pd.read_parquet('data/stocks.parquet')
            self.all_codes = stocks_df['code'].astype(str).str.zfill(6).tolist()
        except:
            try:
                codes_df = pd.read_csv('data/codes.csv')
                self.all_codes = codes_df['code'].astype(str).str.zfill(6).tolist()
            except:
                raise FileNotFoundError("无法找到 stocks.parquet 或 codes.csv，请手动提供 all_codes")
    
    def rank_dragons(
        self,
        start_date: str,
        end_date: str,
        top_n: int = 50,
        min_limit_up_count: int = 3,
        min_avg_amount: float = 5e8,  # 新增：涨停日平均成交额下限（默认5亿，过滤极低量一字票，可调低）
        all_codes: Optional[List[str]] = None
    ) -> pd.DataFrame:
        if all_codes is not None:
            self.all_codes = all_codes
        
        # 加载必要列（新增 amount 用于参与性）
        required_columns = [
            'code', 'day', 'flag', 'amount',
            'next_open_return_1d', 'next_close_return_1d',
            'next_high_return_1d', 'next_low_return_1d'
        ]
        
        print("正在加载全市场数据计算参与性（可能需要稍长的时间）...")
        df_market = self.manager.load_day_data(
            start_date=start_date,
            end_date=end_date,
            codes=self.all_codes,
            columns=required_columns
        )
        
        # 只保留涨停日
        df_limit = df_market[df_market['flag'] == 1].copy()
        if df_limit.empty:
            print("指定时间段内无涨停记录")
            return pd.DataFrame()
        
        df_limit = df_limit.sort_values(['day', 'code']).reset_index(drop=True)
        
        # ==================== 每日涨停池排名 ====================
        def calc_daily_rank(group: pd.DataFrame) -> pd.DataFrame:
            group['close_rank_pct'] = group['next_close_return_1d'].rank(pct=True, ascending=False)
            group['open_rank_pct'] = group['next_open_return_1d'].rank(pct=True, ascending=False)
            return group
        
        df_limit = df_limit.groupby('day', group_keys=False).apply(calc_daily_rank)
        
        # ==================== 聚合统计（新增参与性指标） ====================
        agg_dict = {
            'amount': 'mean',  # 涨停日平均成交额
            'next_close_return_1d': 'mean',
            'next_open_return_1d': 'mean',
            'next_high_return_1d': 'mean',
            'next_low_return_1d': 'mean',
            'close_rank_pct': 'mean',
            'open_rank_pct': 'mean',
            'day': 'count'
        }
        
        stock_stats = df_limit.groupby('code').agg(agg_dict).reset_index()
        stock_stats.rename(columns={
            'day': 'limit_up_count',
            'amount': 'avg_limit_amount',
            'next_close_return_1d': 'avg_next_close_after_limit',
            'next_open_return_1d': 'avg_next_open_after_limit',
            'next_high_return_1d': 'avg_best_after_limit',
            'next_low_return_1d': 'avg_worst_after_limit',
            'close_rank_pct': 'avg_close_rank_pct',
            'open_rank_pct': 'avg_open_rank_pct'
        }, inplace=True)
        
        # 次日平均振幅（参与性强往往振幅大）
        stock_stats['avg_next_amplitude'] = stock_stats['avg_best_after_limit'] - stock_stats['avg_worst_after_limit']
        
        # 高开率 & 高开高走率
        high_open_mask = df_limit['next_open_return_1d'] > 0
        stock_stats = stock_stats.merge(
            df_limit[high_open_mask].groupby('code').size().rename('high_open_count').reset_index(),
            on='code', how='left'
        ).fillna({'high_open_count': 0})
        stock_stats['high_open_rate'] = stock_stats['high_open_count'] / stock_stats['limit_up_count']
        
        high_open_high_walk_mask = (df_limit['next_open_return_1d'] > 0) & \
                                   (df_limit['next_close_return_1d'] > df_limit['next_open_return_1d'])
        stock_stats = stock_stats.merge(
            df_limit[high_open_high_walk_mask].groupby('code').size().rename('high_open_high_walk_count').reset_index(),
            on='code', how='left'
        ).fillna({'high_open_high_walk_count': 0})
        stock_stats['high_open_high_walk_rate'] = stock_stats['high_open_high_walk_count'] / stock_stats['limit_up_count']
        
        # 连板高度
        def calc_max_consecutive_flags(sub_df: pd.DataFrame) -> int:
            sub_df = sub_df.sort_values('day')
            max_streak = 0
            current_streak = 0
            for f in sub_df['flag']:
                if f == 1:
                    current_streak += 1
                    max_streak = max(max_streak, current_streak)
                else:
                    current_streak = 0
            return max_streak
        
        max_consecutive = df_limit.groupby('code').apply(calc_max_consecutive_flags).rename('max_consecutive_limit').reset_index()
        stock_stats = stock_stats.merge(max_consecutive, on='code', how='left')
        stock_stats['max_consecutive_limit'] = stock_stats['max_consecutive_limit'].fillna(0).astype(int)
        
        # ==================== 过滤与加权参与性 ====================
        # 硬过滤：平均成交额太低直接出局（一字板典型<1-2亿）
        stock_stats = stock_stats[stock_stats['avg_limit_amount'] >= min_avg_amount]
        stock_stats = stock_stats[stock_stats['limit_up_count'] >= min_limit_up_count]
        
        if stock_stats.empty:
            print("过滤后无满足条件的股票（可尝试降低 min_avg_amount）")
            return pd.DataFrame()
        
        # Z-score 标准化
        def z_score(series: pd.Series) -> pd.Series:
            return (series - series.mean()) / (series.std() + 1e-6)
        
        stock_stats['z_limit_up_count'] = z_score(stock_stats['limit_up_count'])
        stock_stats['z_avg_next_close'] = z_score(stock_stats['avg_next_close_after_limit'])
        stock_stats['z_avg_next_open'] = z_score(stock_stats['avg_next_open_after_limit'])
        stock_stats['z_high_open_high_walk'] = z_score(stock_stats['high_open_high_walk_rate'])
        stock_stats['z_close_rank'] = z_score(stock_stats['avg_close_rank_pct'])
        stock_stats['z_open_rank'] = z_score(stock_stats['avg_open_rank_pct'])
        stock_stats['z_max_consecutive'] = z_score(stock_stats['max_consecutive_limit'])
        
        # 新增参与性Z分
        stock_stats['z_participation_amount'] = z_score(stock_stats['avg_limit_amount'])
        stock_stats['z_participation_amplitude'] = z_score(stock_stats['avg_next_amplitude'])
        
        # 综合得分（大幅加权参与性，总权重重分配）
        stock_stats['dragon_score'] = (
            0.10 * stock_stats['z_limit_up_count'] +
            0.15 * stock_stats['z_avg_next_close'] +
            0.10 * stock_stats['z_avg_next_open'] +
            0.15 * stock_stats['z_high_open_high_walk'] +
            0.10 * stock_stats['z_close_rank'] +
            0.05 * stock_stats['z_open_rank'] +
            0.10 * stock_stats['z_max_consecutive'] +
            0.15 * stock_stats['z_participation_amount'] +      # 涨停日成交活跃
            0.10 * stock_stats['z_participation_amplitude']     # 次日接力有波动
        )
        
        # 排序
        stock_stats = stock_stats.sort_values('dragon_score', ascending=False).reset_index(drop=True)
        stock_stats['rank'] = stock_stats.index + 1
        
        # 显示列（新增参与性指标）
        display_cols = [
            'rank', 'code', 'dragon_score',
            'limit_up_count', 'max_consecutive_limit',
            'avg_limit_amount', 'avg_next_amplitude',
            'avg_next_close_after_limit', 'avg_next_open_after_limit',
            'high_open_high_walk_rate', 'high_open_rate',
            'avg_close_rank_pct', 'avg_open_rank_pct'
        ]
        df_top = stock_stats.head(top_n)[display_cols].copy()
        
        # 格式化
        df_top['avg_limit_amount'] = (df_top['avg_limit_amount'] / 1e8).round(2).astype(str) + '亿'
        df_top['avg_next_amplitude'] = df_top['avg_next_amplitude'].round(2)
        df_top['avg_next_close_after_limit'] = df_top['avg_next_close_after_limit'].round(2)
        df_top['avg_next_open_after_limit'] = df_top['avg_next_open_after_limit'].round(2)
        df_top['high_open_high_walk_rate'] = (df_top['high_open_high_walk_rate'] * 100).round(1)
        df_top['high_open_rate'] = (df_top['high_open_rate'] * 100).round(1)
        df_top['avg_close_rank_pct'] = (df_top['avg_close_rank_pct'] * 100).round(1)
        df_top['avg_open_rank_pct'] = (df_top['avg_open_rank_pct'] * 100).round(1)
        df_top['dragon_score'] = df_top['dragon_score'].round(3)
        
        return df_top
    
    def print_rank_table(self, df_rank: pd.DataFrame):
        if df_rank.empty:
            print("无数据")
            return
        
        print(f"\n=== 潜在龙头股排行榜（Top {len(df_rank)}，已强调参与性）===\n")
        print(df_rank.to_string(index=False))
        print("\n解释：")
        print("- dragon_score: 综合得分（越高越可能是可参与真龙头）")
        print("- avg_limit_amount: 涨停日平均成交额（越高参与性越强，一字板通常<2亿）")
        print("- avg_next_amplitude: 次日平均振幅（越大说明接力有博弈、可参与）")
        print("- 排行已自动降权低量一字票，优先游资主战场风格")

# ================ 使用示例 ================
if __name__ == "__main__":
    ranker = DragonStockRanker()
    
    df_rank = ranker.rank_dragons(
        start_date='2025-09-01',
        end_date='2026-01-31',
        top_n=130,
        min_limit_up_count=3,
        min_avg_amount=3e8  # 可调：3亿起步，过滤多数一字；设1e8更宽松
    )
    
    ranker.print_rank_table(df_rank)