# -*- coding: utf-8 -*-
"""
板块赚钱效应分析（指标版）

核心指标：
1. 涨停维度：涨停数、炸板率、次日收益
2. 资金维度：成交额、占比、相对强度
3. 个股历史：涨停胜率、平均收益
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from data_core import StockDataManager


# ==================== 数据结构 ====================

@dataclass
class BlockMetrics:
    """板块指标"""
    block_name: str
    block_type: str
    trade_date: str
    stock_count: int
    
    # === 涨停维度 ===
    limit_count: int                      # 涨停数
    limit_stocks: List[str] = field(default_factory=list)
    touch_count: int = 0                  # 触板数
    broken_count: int = 0                 # 炸板数
    broken_rate: float = 0.0              # 炸板率 (%)
    
    # === 涨停次日表现 ===
    limit_next_open: float = 0.0          # 次日开盘 (%)
    limit_next_close: float = 0.0         # 次日收盘 (%)
    limit_next_high: float = 0.0          # 次日最高 (%)
    limit_next_low: float = 0.0           # 次日最低 (%)
    limit_win_rate: float = 0.0           # 次日收盘为正比例 (%)
    
    # === 成交量维度 ===
    block_amount: float = 0.0             # 板块成交额（亿）
    limit_amount: float = 0.0             # 涨停股成交额（亿）
    limit_amount_ratio: float = 0.0       # 涨停成交占比 (%)
    
    # === 资金相对强度 ===
    amount_ratio_market: float = 0.0      # 板块成交占全市场比例 (%)
    amount_chg_5d: float = 0.0            # 板块5日成交变化 (%)
    market_chg_5d: float = 0.0            # 市场5日成交变化 (%)
    relative_strength: float = 0.0        # 相对强度 = (板块变化/市场变化-1)×100 (%)
    top5_contribution: float = 0.0        # 成交增加前5股票对change_pct的贡献率 (%)
    top5_stocks: List[str] = field(default_factory=list)  # 成交增加前5股票代码
    rs_top5_stocks: List[str] = field(default_factory=list)  # 相对强度(量比)前5股票代码


@dataclass
class StockMetrics:
    """个股指标"""
    code: str
    block_name: str
    
    # === 当日表现 ===
    amount: float = 0.0                   # 成交额（亿）
    is_limit: bool = False
    
    # === 次日表现 ===
    next_open: float = 0.0
    next_close: float = 0.0
    next_high: float = 0.0
    next_low: float = 0.0
    
    # === 历史涨停统计 ===
    hist_limit_count: int = 0             # 历史涨停次数
    hist_win_rate: float = 0.0            # 历史次日收盘为正比例 (%)
    hist_avg_open: float = 0.0            # 历史次日平均开盘 (%)
    hist_avg_close: float = 0.0           # 历史次日平均收盘 (%)
    hist_avg_high: float = 0.0            # 历史次日平均最高 (%)
    
    # === 板块强度 ===
    block_limit_count: int = 0
    block_relative_strength: float = 0.0


# ==================== 核心分析器 ====================

class BlockAnalyzer:
    """板块分析器"""
    
    COLS = ['code', 'day', 'open', 'high', 'low', 'close', 'limit_price', 
            'flag', 'amount', 'volume',
            'open_return', 'close_return', 'high_return', 'low_return']
    
    def __init__(self, manager: StockDataManager = None):
        self.dm = manager or StockDataManager()
    
    # ==================== 板块分析 ====================
    
    def get_block_metrics(self,
                          trade_date: str,
                          block_type: str = 'sw2_industry',
                          min_limits: int = 2,
                          lookback_days: int = 30) -> List[BlockMetrics]:
        """
        获取板块指标
        
        Args:
            trade_date: 交易日期
            block_type: 板块类型
            min_limits: 最少涨停数
            lookback_days: 成交量回溯天数
        """
        # 1. 加载当日数据
        df = self.dm.load_day_data(
            start_date=trade_date,
            end_date=trade_date,
            columns=self.COLS
        )
        
        if df.empty:
            return []
        
        # 2. 加载历史成交数据（用于计算相对强度）
        start_date = self._offset_date(trade_date, -lookback_days * 2)
        hist_df = self.dm.load_day_data(
            start_date=start_date,
            end_date=trade_date,
            columns=['code', 'day', 'amount']
        )
        
        # 3. 计算市场整体成交变化
        market_amount = self._calc_market_amount(hist_df, trade_date, lookback_days)
        
        # 4. 标记涨停/炸板
        df['touch_limit'] = df['high'] >= df['limit_price']
        df['is_broken'] = df['touch_limit'] & (df['flag'] != 1)
        
        # 5. 按板块统计涨停
        limit_codes = df[df['flag'] == 1]['code'].tolist()
        if not limit_codes:
            return []
        
        block_limits = self._group_by_block(limit_codes, block_type)
        
        # 6. 计算每个板块指标
        results = []
        for block_name, codes in block_limits.items():
            if len(codes) < min_limits:
                continue
            
            metrics = self._calc_block_metrics(
                df, hist_df, block_name, block_type, codes, 
                trade_date, lookback_days, market_amount
            )
            if metrics:
                results.append(metrics)
        
        # 按相对强度排序
        results.sort(key=lambda x: x.relative_strength, reverse=True)
        return results
    
    def _calc_block_metrics(self, df, hist_df, block_name, block_type, 
                            limit_codes, trade_date, lookback_days, market_amount) -> Optional[BlockMetrics]:
        """计算单个板块的指标"""
        
        all_stocks = self.dm.get_stocks_in_block(block_name, block_type)
        if not all_stocks:
            return None
        
        block_df = df[df['code'].isin(all_stocks)]
        if block_df.empty:
            return None
        
        limit_df = block_df[block_df['flag'] == 1]
        
        # === 涨停维度 ===
        touch_count = int(block_df['touch_limit'].sum())
        broken_count = int(block_df['is_broken'].sum())
        broken_rate = broken_count / touch_count if touch_count > 0 else 0
        
        # === 次日表现 ===
        limit_next_open = self._safe_mean(limit_df['open_return'])
        limit_next_close = self._safe_mean(limit_df['close_return'])
        limit_next_high = self._safe_mean(limit_df['high_return'])
        limit_next_low = self._safe_mean(limit_df['low_return']) 
        limit_win_rate = (limit_df['close_return'] > 0).mean() if not limit_df.empty else 0
        
        # === 成交量 ===
        block_amount = block_df['amount'].sum() / 1e8
        limit_amount = limit_df['amount'].sum() / 1e8
        limit_amount_ratio = limit_amount / block_amount if block_amount > 0 else 0
        
        # === 资金相对强度 ===
        amount_ratio_market = block_amount / market_amount['today'] * 100 if market_amount['today'] > 0 else 0
        
        block_hist = hist_df[hist_df['code'].isin(all_stocks)]
        
        # 计算板块基础统计
        block_amount_stats = self._calc_amount_change(block_hist, trade_date, lookback_days)
        amount_chg_5d = block_amount_stats['change_pct']
        market_chg_5d = market_amount['change_pct']
        
        # 计算Top5贡献率
        top5_result = self._calc_top5_contribution(block_hist, trade_date, lookback_days, market_amount)
        relative_strength = top5_result['full_strength']
        top5_contribution = top5_result['top5_contribution']
        top5_stocks = top5_result['top5_stocks']
        rs_top5_stocks = top5_result['rs_top5_stocks']
        
        return BlockMetrics(
            block_name=block_name,
            block_type=block_type,
            trade_date=trade_date,
            stock_count=len(block_df),
            limit_count=len(limit_codes),
            limit_stocks=limit_codes,
            touch_count=touch_count,
            broken_count=broken_count,
            broken_rate=round(broken_rate, 1),
            limit_next_open=round(limit_next_open, 2),
            limit_next_close=round(limit_next_close, 2),
            limit_next_high=round(limit_next_high, 2),
            limit_next_low=round(limit_next_low, 2),
            limit_win_rate=round(limit_win_rate, 1),
            block_amount=round(block_amount, 2),
            limit_amount=round(limit_amount, 2),
            limit_amount_ratio=round(limit_amount_ratio, 1),
            amount_ratio_market=round(amount_ratio_market, 2),
            amount_chg_5d=round(amount_chg_5d, 1),
            market_chg_5d=round(market_chg_5d, 1),
            relative_strength=round(relative_strength, 1),
            top5_contribution=top5_contribution,
            top5_stocks=top5_stocks,
            rs_top5_stocks=rs_top5_stocks
        )
    
    
    def _calc_market_amount(self, hist_df, trade_date, lookback_days) -> Dict:
        """计算市场整体成交量变化"""
        return self._calc_amount_change(hist_df, trade_date, lookback_days)
    
    def _calc_amount_change(self, df, end_date, lookback_days, exclude_codes: List[str] = None) -> Dict:
        """
        计算成交量变化率
        
        Args:
            df: 历史成交数据
            end_date: 截止日期
            lookback_days: 长期窗口天数
            exclude_codes: 要排除的股票代码列表
        """
        # 如果需要排除某些股票
        if exclude_codes:
            df = df[~df['code'].isin(exclude_codes)]
        
        daily = df.groupby('day')['amount'].sum().reset_index()
        daily = daily.sort_values('day')
        if daily.empty:
            return {
                'today': 0.0,
                'past_avg': 0.0,
                'ratio': 0.0,
                'change_pct': 0.0
            }
        daily['amount'] = daily['amount'] / 1e8
        today_amount = daily[daily['day'] == end_date]['amount'].sum()
        total_len = len(daily)
        long_window = min(lookback_days, total_len)
        short_window = min(2, long_window)
        recent = daily.tail(short_window)
        long = daily.tail(long_window)
        short_avg = recent['amount'].mean()
        long_avg = long['amount'].mean() if not long.empty else short_avg
        ratio = short_avg / long_avg if long_avg > 0 else 0.0
        change_pct = (ratio - 1.0) * 100
        
        return {
            'today': today_amount,
            'past_avg': long_avg,
            'ratio': ratio,
            'change_pct': change_pct
        }
    
    def _calc_top5_contribution(self, df, end_date, lookback_days, market_amount) -> Dict:
        """
        计算Top5股票对relative_strength的贡献率      
        Returns:
            {
                'top5_contribution': float,  # 贡献率(%)
                'top5_stocks': List[str],    # Top5股票代码(按成交增量)
                'rs_top5_stocks': List[str], # Top5股票代码(按相对强度/量比)
                'full_strength': float,      # 完整板块相对强度
                'exclude_strength': float    # 剔除Top5后相对强度
            }
        """
        if df.empty:
            return {
                'top5_contribution': 0.0,
                'top5_stocks': [],
                'rs_top5_stocks': [],
                'full_strength': 0.0,
                'exclude_strength': 0.0
            }
        
        # 第一步：按股票计算短期和长期平均成交
        total_len = len(df['day'].unique())
        long_window = min(lookback_days, total_len)
        short_window = min(5, long_window)
        
        # 获取日期范围
        all_dates = sorted(df['day'].unique())
        short_dates = all_dates[-short_window:]
        long_dates = all_dates[-long_window:]
        
        # 计算每只股票的短期和长期平均成交
        stock_amounts = []
        for code in df['code'].unique():
            code_df = df[df['code'] == code]
            
            # 短期平均
            short_data = code_df[code_df['day'].isin(short_dates)]
            short_avg = short_data['amount'].sum() / short_window if not short_data.empty else 0
            
            # 长期平均
            long_data = code_df[code_df['day'].isin(long_dates)]
            long_avg = long_data['amount'].sum() / long_window if not long_data.empty else 0
            
            # 成交增量
            amount_change = short_avg - long_avg
            
            # 量比 (相对强度)
            ratio = short_avg / long_avg if long_avg > 0 else 0.0
            
            stock_amounts.append({
                'code': code,
                'short_avg': short_avg,
                'long_avg': long_avg,
                'amount_change': amount_change,
                'ratio': ratio
            })
        
        # 第二步：找出成交增量最大的前5只
        stock_amounts_df = pd.DataFrame(stock_amounts)
        
        # 按成交增量排序
        stock_amounts_df_amt = stock_amounts_df.sort_values('amount_change', ascending=False)
        top5 = stock_amounts_df_amt.head(5)
        top5_codes = top5['code'].tolist()
        
        # 按量比(相对强度)排序
        stock_amounts_df_ratio = stock_amounts_df.sort_values('ratio', ascending=False)
        rs_top5 = stock_amounts_df_ratio.head(5)
        rs_top5_codes = rs_top5['code'].tolist()
        
        # 第三步：计算完整板块的relative_strength
        block_stats = self._calc_amount_change(df, end_date, lookback_days, exclude_codes=None)
        block_ratio = block_stats['ratio']
        market_ratio = market_amount['ratio']
        
        if market_ratio > 0:
            full_strength = (block_ratio / market_ratio - 1.0) * 100
        else:
            full_strength = 0.0
        
        # 第四步：剔除Top5后重新计算relative_strength
        exclude_stats = self._calc_amount_change(df, end_date, lookback_days, exclude_codes=top5_codes)
        exclude_ratio = exclude_stats['ratio']
        
        if market_ratio > 0:
            exclude_strength = (exclude_ratio / market_ratio - 1.0) * 100
        else:
            exclude_strength = 0.0
        
        # 第五步：计算贡献率
        # 贡献率 = (完整强度 - 剔除Top5强度)  * 100%
        if full_strength != 0:
            contribution = full_strength - exclude_strength
        else:
            contribution = 0.0
        
        return {
            'top5_contribution': round(contribution, 1),
            'top5_stocks': top5_codes,
            'rs_top5_stocks': rs_top5_codes,
            'full_strength': round(full_strength, 1),
            'exclude_strength': round(exclude_strength, 1)
        }
    
    # ==================== 个股分析 ====================
    
    def get_stock_metrics(self,
                          trade_date: str,
                          block_type: str = 'concept',
                          min_limits: int = 2,
                          history_days: int = 60) -> List[StockMetrics]:
        """
        获取个股指标（含历史涨停统计）
        仅包含涨停股
        """
        # 1. 获取板块指标
        block_metrics = self.get_block_metrics(trade_date, block_type, min_limits)
        if not block_metrics:
            return []
        
        # 2. 加载当日数据
        df = self.dm.load_day_data(
            start_date=trade_date,
            end_date=trade_date,
            columns=self.COLS
        )
        
        # 3. 加载历史数据（计算个股涨停历史）
        start_date = self._offset_date(trade_date, -history_days * 2)
        hist_df = self.dm.load_day_data(
            start_date=start_date,
            end_date=trade_date,
            columns=['code', 'day', 'flag', 'open_return', 'close_return', 'high_return']
        )
        
        results = []
        
        for block in block_metrics[:5]:  # 取前5个板块
            # 仅处理涨停股
            for code in block.limit_stocks:
                row = df[df['code'] == code]
                if row.empty:
                    continue
                row = row.iloc[0]
                
                hist = self._calc_stock_history(hist_df, code, trade_date)
                
                results.append(StockMetrics(
                    code=code,
                    block_name=block.block_name,
                    amount=round(row['amount'] / 1e8, 2),
                    is_limit=True,
                    next_open=round(row['open_return'], 2) if pd.notna(row['open_return']) else 0,
                    next_close=round(row['close_return'], 2) if pd.notna(row['close_return']) else 0,
                    next_high=round(row['high_return'], 2) if pd.notna(row['high_return']) else 0,
                    next_low=round(row['low_return'], 2) if pd.notna(row['low_return']) else 0,
                    hist_limit_count=hist['count'],
                    hist_win_rate=hist['win_rate'],
                    hist_avg_open=hist['avg_open'],
                    hist_avg_close=hist['avg_close'],
                    hist_avg_high=hist['avg_high'],
                    block_limit_count=block.limit_count,
                    block_relative_strength=block.relative_strength
                ))
        
        return results
    
    def _calc_stock_history(self, hist_df, code, end_date) -> Dict:
        """计算个股历史涨停统计"""
        stock_df = hist_df[(hist_df['code'] == code) & (hist_df['day'] < end_date)]
        limit_df = stock_df[stock_df['flag'] == 1]
        
        if limit_df.empty:
            return {'count': 0, 'win_rate': 0, 'avg_open': 0, 'avg_close': 0, 'avg_high': 0}
        
        n = len(limit_df)
        valid_close = limit_df['close_return'].dropna()
        valid_open = limit_df['open_return'].dropna()
        valid_high = limit_df['high_return'].dropna()
        
        return {
            'count': n,
            'win_rate': round((valid_close > 0).mean(), 1) if len(valid_close) > 0 else 0,
            'avg_open': round(valid_open.mean(), 2) if len(valid_open) > 0 else 0,
            'avg_close': round(valid_close.mean(), 2) if len(valid_close) > 0 else 0,
            'avg_high': round(valid_high.mean(), 2) if len(valid_high) > 0 else 0
        }
    
    def get_block_history(self, 
                          block_names: List[str], 
                          start_date: str, 
                          end_date: str, 
                          block_type: str = 'concept',
                          lookback_days: int = 30) -> pd.DataFrame:
        """
        获取板块历史热度数据 (优化版：向量化计算)
        
        Args:
            block_names: 板块名称列表
            start_date: 开始日期
            end_date: 结束日期
            block_type: 板块类型
            lookback_days: 相对强度计算的回溯天数
            
        Returns:
            DataFrame columns: [day, block_name, relative_strength, limit_count, block_amount]
        """
        # 1. 获取所有涉及的股票代码
        all_codes = set()
        block_stocks_map = {}
        for name in block_names:
            stocks = self.dm.get_stocks_in_block(name, block_type)
            if stocks:
                all_codes.update(stocks)
                block_stocks_map[name] = stocks
        
        if not all_codes:
            print("❌ 未找到任何板块成分股")
            return pd.DataFrame()
            
        # 2. 加载数据
        # 需要额外加载 lookback_days 的数据用于计算相对强度
        real_start_date = self._offset_date(start_date, -lookback_days * 2)
        
        # 加载成分股数据
        stock_df = self.dm.load_day_data(
            start_date=real_start_date,
            end_date=end_date,
            codes=list(all_codes),
            columns=self.COLS 
        )
        
        if stock_df.empty:
             print("❌ 未加载到任何股票数据")
             return pd.DataFrame()

        # 预计算涨停/炸板标记
        if 'touch_limit' not in stock_df.columns:
             stock_df['touch_limit'] = stock_df['high'] >= stock_df['limit_price']
        if 'is_broken' not in stock_df.columns:
             stock_df['is_broken'] = stock_df['touch_limit'] & (stock_df['flag'] != 1)
             
        # 加载全市场成交额数据（仅需要 day, amount）
        market_df = self.dm.load_day_data(
            start_date=real_start_date,
            end_date=end_date,
            columns=['day', 'amount']
        )
        
        # 计算市场每日总成交及移动平均
        # 注意：这里我们简化了市场成交量的计算，直接用load_day_data返回的数据sum
        # 之前的方法是 _calc_market_amount -> _calc_amount_change，逻辑一致
        daily_market = market_df.groupby('day')['amount'].sum().sort_index()
        daily_market = daily_market / 1e8  # 换算为亿
        
        # 计算市场长短周期均线
        # short_window = 2, long_window = lookback_days
        # 注意：rolling min_periods设为1以确保初期有数据
        market_short_ma = daily_market.rolling(window=2, min_periods=1).mean()
        market_long_ma = daily_market.rolling(window=lookback_days, min_periods=1).mean()
        
        # 避免除以0
        market_long_ma = market_long_ma.replace(0, np.nan)
        market_ratio = market_short_ma / market_long_ma
        
        results = []
        
        print(f"正在计算 {len(block_names)} 个板块数据 (向量化加速)...")
        
        for block_name in block_names:
            codes = block_stocks_map.get(block_name, [])
            if not codes:
                continue
            
            # 筛选该板块的股票
            block_df = stock_df[stock_df['code'].isin(codes)].copy()
            if block_df.empty:
                continue
            
            # 按日期聚合
            # 1. 总成交额 (亿)
            block_daily_amount = block_df.groupby('day')['amount'].sum().sort_index() / 1e8
            
            # 2. 涨停数
            limit_counts = block_df[block_df['flag'] == 1].groupby('day')['code'].count()
            
            # 3. 炸板数 & 触板数
            broken_counts = block_df[block_df['is_broken']].groupby('day')['code'].count()
            touch_counts = block_df[block_df['touch_limit']].groupby('day')['code'].count()
            
            # 4. 涨停股成交额
            limit_amounts = block_df[block_df['flag'] == 1].groupby('day')['amount'].sum() / 1e8
            
            # 对齐索引 (确保所有日期都有值，填充0)
            all_days = block_daily_amount.index
            limit_counts = limit_counts.reindex(all_days, fill_value=0)
            broken_counts = broken_counts.reindex(all_days, fill_value=0)
            touch_counts = touch_counts.reindex(all_days, fill_value=0)
            limit_amounts = limit_amounts.reindex(all_days, fill_value=0)
            
            # 计算板块长短周期均线
            block_short_ma = block_daily_amount.rolling(window=2, min_periods=1).mean()
            block_long_ma = block_daily_amount.rolling(window=lookback_days, min_periods=1).mean()
            
            block_long_ma = block_long_ma.replace(0, np.nan)
            block_ratio = block_short_ma / block_long_ma
            
            # 计算相对强度
            # (block_ratio / market_ratio - 1) * 100
            # 注意索引对齐
            common_idx = block_ratio.index.intersection(market_ratio.index)
            
            # 截取 common_idx 中对应的 series
            b_ratio_aligned = block_ratio[common_idx]
            m_ratio_aligned = market_ratio[common_idx]
            
            rel_strength = (b_ratio_aligned / m_ratio_aligned - 1.0) * 100
            
            # 组装结果
            # 只需要 start_date 到 end_date 之间的数据
            valid_days = [d for d in common_idx if start_date <= d <= end_date]
            
            for d in valid_days:
                t_count = touch_counts.get(d, 0)
                b_count = broken_counts.get(d, 0)
                broken_rate = b_count / t_count if t_count > 0 else 0.0
                
                blk_amt = block_daily_amount.get(d, 0)
                lmt_amt = limit_amounts.get(d, 0)
                lmt_ratio = lmt_amt / blk_amt if blk_amt > 0 else 0.0
                
                results.append({
                    'day': d,
                    'block_name': block_name,
                    'relative_strength': round(rel_strength.get(d, 0), 1),
                    'limit_count': int(limit_counts.get(d, 0)),
                    'broken_rate': round(broken_rate, 1),
                    'block_amount': round(blk_amt, 2),
                    'limit_amount_ratio': round(lmt_ratio, 1)
                })
        
        return pd.DataFrame(results)

    def print_history(self, 
                      block_names: List[str], 
                      start_date: str, 
                      end_date: str, 
                      block_type: str = 'concept'):
        """打印历史热度趋势"""
        df = self.get_block_history(block_names, start_date, end_date, block_type)
        
        if df.empty:
            print("❌ 无数据")
            return
            
        print(f"\n{'='*80}")
        print(f"📈 板块历史热度 ({start_date} ~ {end_date})")
        print(f"{'='*80}")
        
        # 按板块打印
        for name in block_names:
            block_data = df[df['block_name'] == name].sort_values('day')
            if block_data.empty:
                print(f"\n📌 {name}: 无数据")
                continue
                
            print(f"\n📌 {name}")
            print(f"{'日期':<12} {'强度':<8} {'趋势图':<40} {'成交(亿)':<10}")
            print("-" * 80)
            
            for _, row in block_data.iterrows():
                # 强度可视化：每2%一个刻度，最大显示40个刻度(80%)
                bar_len = int(max(0, row['relative_strength']) / 2)
                strength_bar = "█" * bar_len
                # 限制长度
                if len(strength_bar) > 35: strength_bar = strength_bar[:35] + '+'
                
                print(f"{row['day']:<12} "
                      f"{row['relative_strength']:>6.1f} {strength_bar:<40} "
                      f"{row['block_amount']:<10.1f}")

    # ==================== 输出 ====================
    
    def print_blocks(self, trade_date: str, block_type: str = 'concept', top_n: int = 10):
        """打印板块指标"""
        metrics = self.get_block_metrics(trade_date, block_type, min_limits=2)
        
        print(f"\n{'='*120}")
        print(f"📊 {trade_date} 板块赚钱效应 ({block_type})")
        print(f"{'='*120}")
        
        if not metrics:
            print("❌ 无符合条件的板块")
            return
        
        print(f"{'板块':<12} │ {'涨停':>4} {'炸板':>5} │ "
              f"{'次开':>6} {'次收':>6} │ "
              f"{'股票数':>6} │ "
              f"{'强度':>6} {'Top5':>5}")
        print("─" * 120)
        
        for m in metrics[:top_n]:
            strength_flag = "🔥" if m.relative_strength > 20 else ("↑" if m.relative_strength > 0 else "↓")
            top5_flag = "⭐" if m.top5_contribution > 80 else ("✓" if m.top5_contribution > 50 else "")
            
            print(f"{m.block_name:<12} │ "
                  f"{m.limit_count:>4} {m.broken_rate:>4.0f}% │ "
                  f"{m.limit_next_open:>+5.1f}% {m.limit_next_close:>+5.1f}% │ "
                  f"{m.stock_count:>6} │ "
                  f"{m.relative_strength:>+5.0f}%{strength_flag} {m.top5_contribution:>4.0f}%{top5_flag}")
        
        print(f"\n💡 说明:")
        print(f"   - 强度 = (板块5日变化/市场5日变化-1)×100%，🔥>20% 表示资金显著流入")
        print(f"   - Top5 = 剔除成交增加前5股票后，板块强度的下降比例，⭐>80% 表示高度依赖龙头")
        print(f"   - 计算方式：贡献率 = (完整强度 - 剔除Top5强度) / |完整强度| × 100%")
        
        # 显示第一个板块的Top5股票详情
        if metrics:
            for i in range(len(metrics)):
                m = metrics[i]
               # if m.top5_stocks:
                   # print(f"\n📌 {m.block_name} 成交增加Top5股票: {', '.join(m.top5_stocks)}")
                if m.rs_top5_stocks:
                    print(f"\n📌 {m.block_name} 相对成交增加Top5股票: {', '.join(m.rs_top5_stocks)}")
  
    def print_stocks(self, trade_date: str, block_type: str = 'concept', top_n: int = 15):
        """打印个股指标（仅涨停股）"""
        stocks = self.get_stock_metrics(trade_date, block_type)
        
        print(f"\n{'='*110}")
        print(f"📊 {trade_date} 个股指标 ({block_type})")
        print(f"{'='*110}")
        
        if not stocks:
            print("❌ 无符合条件的个股")
            return
        
        print(f"{'代码':<8} {'板块':<12} │ "
              f"{'成交亿':>6} │ "
              f"{'次开':>6} {'次收':>6} {'次高':>6} │ "
              f"{'历史次数':>6} {'胜率':>5} {'均开':>6} {'均收':>6} │ "
              f"{'板块强度':>7}")
        print("─" * 110)
        
        for s in stocks[:top_n]:
            hist_str = f"{s.hist_limit_count}" if s.hist_limit_count > 0 else "-"
            win_str = f"{s.hist_win_rate:.0f}%" if s.hist_limit_count > 0 else "-"
            avg_open_str = f"{s.hist_avg_open:+.1f}%" if s.hist_limit_count > 0 else "-"
            avg_close_str = f"{s.hist_avg_close:+.1f}%" if s.hist_limit_count > 0 else "-"
            
            print(f"{s.code:<8} {s.block_name:<12} │ "
                  f"{s.amount:>6.1f} │ "
                  f"{s.next_open:>+5.1f}% {s.next_close:>+5.1f}% {s.next_high:>+5.1f}% │ "
                  f"{hist_str:>6} {win_str:>5} {avg_open_str:>6} {avg_close_str:>6} │ "
                  f"{s.block_relative_strength:>+6.0f}%")
    
    def to_dataframe(self, trade_date: str, block_type: str = 'concept') -> tuple:
        """导出为DataFrame"""
        blocks = self.get_block_metrics(trade_date, block_type)
        stocks = self.get_stock_metrics(trade_date, block_type)
        
        block_df = pd.DataFrame([vars(b) for b in blocks]) if blocks else pd.DataFrame()
        stock_df = pd.DataFrame([vars(s) for s in stocks]) if stocks else pd.DataFrame()
        
        return block_df, stock_df
    
    # ==================== 工具方法 ====================
    
    def _group_by_block(self, codes: List[str], block_type: str) -> Dict[str, List[str]]:
        """按板块分组"""
        block_limits = {}
        for code in codes:
            blocks = self.dm.get_blocks_by_stock(code)
            if blocks is None:
                continue
            block_names = blocks.get(block_type, [])
            if isinstance(block_names, str):
                block_names = [block_names]
            for name in (block_names or []):
                if name:
                    block_limits.setdefault(name, []).append(code)
        return block_limits
    
    def _offset_date(self, date_str: str, days: int) -> str:
        """日期偏移"""
        dt = pd.to_datetime(date_str) + pd.Timedelta(days=days)
        return dt.strftime('%Y-%m-%d')
    
    @staticmethod
    def _safe_mean(series) -> float:
        """安全计算均值"""
        valid = series.dropna()
        return valid.mean() if len(valid) > 0 else 0


# ==================== 便捷函数 ====================

def blocks(date: str, block_type: str = 'concept', top_n: int = 20):
    """查看板块指标"""
    BlockAnalyzer().print_blocks(date, block_type, top_n)

def stocks(date: str, block_type: str = 'concept', top_n: int = 15):
    """查看个股指标"""
    BlockAnalyzer().print_stocks(date, block_type, top_n)

def export(date: str, block_type: str = 'concept'):
    """导出DataFrame"""
    return BlockAnalyzer().to_dataframe(date, block_type)

def get_blocks(date: str, block_type: str = 'concept') -> List[BlockMetrics]:
    """获取板块指标对象"""
    return BlockAnalyzer().get_block_metrics(date, block_type)

def get_stocks(date: str, block_type: str = 'concept') -> List[StockMetrics]:
    """获取个股指标对象"""
    return BlockAnalyzer().get_stock_metrics(date, block_type)

def history(block_names: List[str], start_date: str, end_date: str, block_type: str = 'concept'):
    """查看板块历史热度"""
    BlockAnalyzer().print_history(block_names, start_date, end_date, block_type)



if __name__ == '__main__':
    date = '2026-03-06'
    # 板块指标
    blocks(date, 'sw1_industry')
    #history(['银行', '基建'], start_date='2025-10-01', end_date='2026-02-24', block_type='sw2_industry')
    # 个股指标
    # stocks(date, 'sw2_industry')