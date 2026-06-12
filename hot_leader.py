# -*- coding: utf-8 -*-
"""
热点板块龙头识别策略（追涨）

核心理念（乔帮主心法量化版）：
  1. 资金决定一切：成交额是上涨的唯一原动力
  2. 无套牢盘：创新高的股票上方没有压力，追涨成功率最高
  3. 全部用截面排名（rank），不用绝对阈值

两层筛选：
  Layer 1 — 板块热度排名（sw2申万二级）
    A. 板块资金异常度 = mean(个股 amount / MA20(amount))
    B. 板块创新高广度 = 板块内创20日新高股票比例
    板块间截面 rank → 等权合成 → 选 top N 热点板块

  Layer 2 — 龙头个股排名（热点板块内）
    f1: 资金异常度 = amount / MA20(amount)
    f2: 创新高程度 = (close - min20) / (max20 - min20)
    f3: 起爆新鲜度 = 1 - 近5日放量天数/5（刚起爆优于持续放量）
    板块内截面 rank → 0.45*r1 + 0.35*r2 + 0.20*r3 → top 5 龙头

回测：T日收盘选股 → T+1开盘买入 → T+2开盘卖出 (open_to_open_return_1d)
"""

import sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import List, Dict, Optional, Tuple
from data_core import StockDataManager


class HotLeaderStrategy:
    """热点板块龙头识别策略"""

    def __init__(self, manager: Optional[StockDataManager] = None):
        self.manager = manager or StockDataManager()
        self._sector_map: Optional[Dict[str, List[str]]] = None  # 板块名 → 代码列表
        self._code_to_sector: Optional[Dict[str, str]] = None     # 代码 → 板块名

    # =========================================================
    # 板块映射（构建一次，反复使用）
    # =========================================================
    def _build_sector_map(self, block_type: str = 'sw2_industry'):
        """构建 code ↔ sector 双向映射"""
        if self._sector_map is not None:
            return
        sectors = self.manager.get_block_list(block_type)
        self._sector_map = {}
        self._code_to_sector = {}
        for name in sectors:
            codes = self.manager.get_stocks_in_block(name, block_type)
            if codes:
                self._sector_map[name] = codes
                for c in codes:
                    self._code_to_sector[c] = name

    # =========================================================
    # 因子计算
    # =========================================================
    def _calc_stock_factors(self, df: pd.DataFrame, target_date: str) -> pd.DataFrame:
        """
        计算个股三因子（仅返回 target_date 当天的截面）

        Parameters:
            df: 含至少40个交易日历史的日线数据，需含 code, day, close, amount, flag, close_return
            target_date: 目标日期 'YYYY-MM-DD'

        Returns:
            DataFrame with columns: code, close, amount, flag, close_return,
                f_excess_amount, f_price_position, f_amount_contin,
                + 回测用前瞻字段（如果存在）
        """
        df = df.sort_values(['code', 'day']).copy()

        # --- 滚动指标（按 code 分组）---
        grp = df.groupby('code', sort=False)

        # MA20(amount)
        df['amount_ma20'] = grp['amount'].transform(
            lambda x: x.rolling(20, min_periods=10).mean()
        )
        # 20日最高/最低 close
        df['close_max20'] = grp['close'].transform(
            lambda x: x.rolling(20, min_periods=10).max()
        )
        df['close_min20'] = grp['close'].transform(
            lambda x: x.rolling(20, min_periods=10).min()
        )
        # 近5日每天是否放量（amount > amount_ma20）
        df['is_above_ma20'] = (df['amount'] > df['amount_ma20']).astype(float)
        df['amount_contin_5d'] = grp['is_above_ma20'].transform(
            lambda x: x.rolling(5, min_periods=1).sum()
        )

        # --- 截取目标日 ---
        today = df[df['day'] == target_date].copy()
        if today.empty:
            return today

        # --- 计算三因子 ---
        # f1: 资金异常度
        today['f_excess_amount'] = today['amount'] / today['amount_ma20'].replace(0, np.nan)

        # f2: 创新高程度 (0~1, 1=20日新高)
        price_range = today['close_max20'] - today['close_min20']
        today['f_price_position'] = np.where(
            price_range > 0,
            (today['close'] - today['close_min20']) / price_range,
            0.5
        )

        # f3: 资金持续性 (0~1)
        today['f_amount_contin'] = today['amount_contin_5d'] / 5.0

        # 清理中间列
        keep_cols = ['code', 'close', 'amount', 'flag', 'close_return',
                     'f_excess_amount', 'f_price_position', 'f_amount_contin']
        # 保留回测前瞻字段（如果存在）
        for col in ['open_to_open_return_1d', 'next_open_return_1d']:
            if col in today.columns:
                keep_cols.append(col)
        return today[keep_cols].reset_index(drop=True)

    # =========================================================
    # 板块热度排名
    # =========================================================
    def rank_sectors(self, stock_factors: pd.DataFrame,
                     block_type: str = 'sw2_industry') -> pd.DataFrame:
        """
        对所有 sw2 板块计算热度得分并排名

        Returns:
            DataFrame: sector, stock_count, avg_excess_amount, new_high_breadth,
                       rank_A, rank_B, composite_rank (排序后)
        """
        self._build_sector_map(block_type)

        # 添加板块标签
        stock_factors = stock_factors.copy()
        stock_factors['sector'] = stock_factors['code'].map(self._code_to_sector)
        stock_factors = stock_factors.dropna(subset=['sector'])

        # 板块聚合
        rows = []
        for sector, grp in stock_factors.groupby('sector'):
            if len(grp) < 3:  # 板块太小，跳过
                continue
            rows.append({
                'sector': sector,
                'stock_count': len(grp),
                'avg_excess_amount': grp['f_excess_amount'].mean(),
                'new_high_breadth': (grp['f_price_position'] >= 0.95).mean(),
            })

        if not rows:
            return pd.DataFrame()

        sector_df = pd.DataFrame(rows)

        # 截面排名
        sector_df['rank_A'] = sector_df['avg_excess_amount'].rank(pct=True)
        sector_df['rank_B'] = sector_df['new_high_breadth'].rank(pct=True)
        sector_df['composite_rank'] = 0.5 * sector_df['rank_A'] + 0.5 * sector_df['rank_B']

        return sector_df.sort_values('composite_rank', ascending=False).reset_index(drop=True)

    # =========================================================
    # 龙头个股排名
    # =========================================================
    def rank_leaders(self, stock_factors: pd.DataFrame,
                     sector_codes: List[str]) -> pd.DataFrame:
        """
        在指定板块成员中做龙头排名

        Returns:
            DataFrame: code, 三因子, r1, r2, r3, composite_rank (排序后)
        """
        sub = stock_factors[stock_factors['code'].isin(sector_codes)].copy()
        if sub.empty:
            return sub

        # 截面排名
        sub['r1'] = sub['f_excess_amount'].rank(pct=True)
        sub['r2'] = sub['f_price_position'].rank(pct=True)
        sub['r3'] = sub['f_amount_contin'].rank(pct=True)
        sub['composite_rank'] = 0.4 * sub['r1'] + 0.4 * sub['r2'] + 0.2 * sub['r3']

        return sub.sort_values('composite_rank', ascending=False).reset_index(drop=True)

    # =========================================================
    # 主扫描流程
    # =========================================================
    def scan(self, target_date: str,
             top_n_sectors: int = 5,
             top_n_stocks: int = 2,
             df_all: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        完整扫描：热点板块 → 龙头个股

        Parameters:
            target_date: 扫描日期
            top_n_sectors: 选取的热点板块数量
            top_n_stocks: 每个板块选取的龙头数量
            df_all: 预加载的全量数据（回测时传入，避免重复IO）

        Returns:
            DataFrame: sector, code, 三因子, composite_rank, 前瞻收益
        """
        self._build_sector_map()

        # 1. 加载数据
        if df_all is None:
            # 单日扫描模式：只加载近60天
            from datetime import datetime, timedelta
            pre_start = (datetime.strptime(target_date, '%Y-%m-%d') - timedelta(days=60)).strftime('%Y-%m-%d')
            df_all = self.manager.load_day_data(
                start_date=pre_start, end_date=target_date,
                columns=['code', 'day', 'close', 'amount', 'flag', 'close_return',
                         'open_to_open_return_1d', 'next_open_return_1d']
            )

        # 2. 计算因子
        stock_factors = self._calc_stock_factors(df_all, target_date)
        if stock_factors.empty:
            return pd.DataFrame()

        # 3. 板块排名
        sector_ranks = self.rank_sectors(stock_factors)
        if sector_ranks.empty:
            return pd.DataFrame()

        top_sectors = sector_ranks.head(top_n_sectors)

        # 4. 板块内龙头排名
        results = []
        for _, row in top_sectors.iterrows():
            sector_name = row['sector']
            sector_codes = self._sector_map.get(sector_name, [])
            leaders = self.rank_leaders(stock_factors, sector_codes)
            if leaders.empty:
                continue

            # 质量过滤：非一字涨停 + 收盘为正
            leaders = leaders[
                (leaders['flag'] != 1) &
                (leaders['close_return'] > 0)
            ]

            top = leaders.head(top_n_stocks).copy()
            top['sector'] = sector_name
            top['sector_rank'] = row['composite_rank']
            results.append(top)

        if not results:
            return pd.DataFrame()

        return pd.concat(results, ignore_index=True)

    # =========================================================
    # 回测引擎
    # =========================================================
    def _calc_all_factors_vectorized(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        向量化计算所有日期的个股因子（一次性完成，避免逐日循环）

        Returns:
            df 增加列: f_excess_amount, f_price_position, f_amount_contin
        """
        df = df.sort_values(['code', 'day']).copy()
        grp = df.groupby('code', sort=False)

        # MA20(amount)
        df['amount_ma20'] = grp['amount'].transform(
            lambda x: x.rolling(20, min_periods=10).mean()
        )
        # 20日最高/最低 close
        df['close_max20'] = grp['close'].transform(
            lambda x: x.rolling(20, min_periods=10).max()
        )
        df['close_min20'] = grp['close'].transform(
            lambda x: x.rolling(20, min_periods=10).min()
        )
        # 近5日放量天数
        df['is_above_ma20'] = (df['amount'] > df['amount_ma20']).astype(float)
        df['amount_contin_5d'] = grp['is_above_ma20'].transform(
            lambda x: x.rolling(5, min_periods=1).sum()
        )

        # 三因子
        df['f_excess_amount'] = df['amount'] / df['amount_ma20'].replace(0, np.nan)
        price_range = df['close_max20'] - df['close_min20']
        df['f_price_position'] = np.where(
            price_range > 0,
            (df['close'] - df['close_min20']) / price_range,
            0.5
        )
        df['f_amount_contin'] = df['amount_contin_5d'] / 5.0

        # --- f4: 股性（120日涨停历史） ---
        df['flag_1'] = (df['flag'] == 1).astype(float)
        df['limit_meat'] = np.where(df['flag'] == 1, grp['close_return'].shift(-1), 0.0)
        df['f_limit_count'] = grp['flag_1'].transform(
            lambda x: x.rolling(120, min_periods=20).sum()
        )
        limit_meat_sum = grp['limit_meat'].transform(
            lambda x: x.rolling(120, min_periods=20).sum()
        )
        df['f_limit_avg_meat'] = np.where(
            df['f_limit_count'] > 0,
            limit_meat_sum / df['f_limit_count'],
            np.nan
        )

        # --- 精细化买入价逻辑 ---
        # 信号日T选股，执行日T+1：
        #   Case A: T+1 open > T close (跳空高开) → 买入价 = T+1 open
        #   Case B: T+1 open <= T close, T+1 high > T close (盘中突破) → 买入价 = T close
        #   Case C: T+1 high <= T close (全天未突破) → 不交易
        # 卖出: T+2 open

        # 在T行上，shift获取T+1的数据
        df['next_open'] = grp['open'].shift(-1)      # T+1 开盘价
        df['next_high'] = grp['high'].shift(-1)       # T+1 最高价
        df['next2_open'] = grp['open'].shift(-2)      # T+2 开盘价

        # Case A: 跳空高开 → 买入T+1 open, 卖出T+2 open
        cond_a = df['next_open'] > df['close']
        # Case B: 非跳空但盘中突破 → 买入T close, 卖出T+2 open
        cond_b = (df['next_open'] <= df['close']) & (df['next_high'] > df['close'])
        # Case C: 不交易
        cond_c = ~cond_a & ~cond_b

        df['buy_price'] = np.where(cond_a, df['next_open'],
                          np.where(cond_b, df['close'], np.nan))
        df['sell_price'] = df['next2_open']

        df['trade_return'] = np.where(
            df['buy_price'].notna() & df['sell_price'].notna(),
            (df['sell_price'] - df['buy_price']) / df['buy_price'] * 100,
            np.nan
        )
        df['entry_type'] = np.where(cond_a, 'gap_up',
                           np.where(cond_b, 'breakout', 'no_trade'))

        # 市场情绪指标移到 backtest() 中计算（因子计算层不做全局聚合）

        # 清理中间列
        drop_cols = ['amount_ma20', 'close_max20', 'close_min20',
                     'is_above_ma20', 'amount_contin_5d',
                     'flag_1', 'limit_meat',
                     'next_open', 'next_high', 'next2_open',
                     'buy_price', 'sell_price']
        df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)
        return df

    def backtest(self, start_date: str, end_date: str,
                 top_n_sectors: int = 5, top_n_stocks: int = 2,
                 verbose: bool = True) -> dict:
        """
        历史回测

        收益定义：T日收盘选股 → T+1开盘买 → T+2开盘卖
        使用字段：open_to_open_return_1d

        Returns:
            dict with keys: trades, stats, equity_curve
        """
        import time
        t0 = time.time()

        if verbose:
            print("加载数据...")

        # 只加载回测区间 + 40天前置窗口（而非全量）
        cols = ['code', 'day', 'open', 'high', 'close', 'amount', 'volume', 'flag',
                'close_return', 'open_return', 'preclose',
                'open_to_open_return_1d', 'next_open_return_1d']

        # 先加载少量数据确定日期，再算前置日期
        from datetime import datetime, timedelta
        pre_start = (datetime.strptime(start_date, '%Y-%m-%d') - timedelta(days=200)).strftime('%Y-%m-%d')
        df_all = self.manager.load_day_data(
            start_date=pre_start, end_date=end_date, columns=cols
        )

        if verbose:
            print(f"  数据加载耗时: {time.time()-t0:.1f}s, shape: {df_all.shape}")

        self._build_sector_map()

        # 向量化计算所有日期的因子（一次性完成）
        if verbose:
            print("计算因子...")
        t1 = time.time()
        df_all = self._calc_all_factors_vectorized(df_all)
        if verbose:
            print(f"  因子计算耗时: {time.time()-t1:.1f}s")

        # 添加板块标签
        df_all['sector'] = df_all['code'].map(self._code_to_sector)

        # 获取回测日期列表
        all_dates = sorted(df_all['day'].unique())
        bt_dates = [d for d in all_dates if start_date <= d <= end_date]

        if verbose:
            print(f"回测区间: {bt_dates[0]} ~ {bt_dates[-1]}, 共 {len(bt_dates)} 个交易日")
            print(f"板块数: {len(self._sector_map)}, 全市场股票数: {df_all['code'].nunique()}")

        # --- 市场情绪过滤（日度） ---
        # 预计算每日情绪指标
        daily_emotion = df_all.groupby('day').agg(
            limit_count=('flag', lambda x: (x == 1).sum()),
            market_mean_return=('close_return', 'mean'),
        ).reset_index()
        # 昨日涨停→今日平均收益（赚钱效应）
        limit_stocks = df_all[df_all['flag'] == 1][['code', 'day']].copy()
        limit_stocks['next_day'] = limit_stocks['day'].map(
            dict(zip(all_dates[:-1], all_dates[1:]))
        )
        # 合并次日收益
        next_day_rets = df_all[['code', 'day', 'close_return']].rename(
            columns={'day': 'next_day', 'close_return': 'limit_next_ret'}
        )
        limit_profits = limit_stocks.merge(next_day_rets, on=['code', 'next_day'], how='inner')
        limit_profit_daily = limit_profits.groupby('next_day')['limit_next_ret'].mean().rename('limit_profit_effect')
        daily_emotion = daily_emotion.join(limit_profit_daily, on='day', how='left')
        daily_emotion['limit_profit_effect'] = daily_emotion['limit_profit_effect'].fillna(0)
        # 3日滚动赚钱效应（更平滑，减少噪音）
        daily_emotion = daily_emotion.sort_values('day')
        daily_emotion['emo_ma3'] = daily_emotion['limit_profit_effect'].rolling(3, min_periods=1).mean()
        # 市场涨跌广度：全市场上涨比例
        market_breadth = df_all.groupby('day')['close_return'].apply(
            lambda x: (x > 0).mean()
        ).rename('up_ratio')
        daily_emotion = daily_emotion.join(market_breadth, on='day', how='left')

        if verbose:
            avg_limit = daily_emotion.loc[daily_emotion['day'].isin(bt_dates), 'limit_count'].mean()
            avg_effect = daily_emotion.loc[daily_emotion['day'].isin(bt_dates), 'limit_profit_effect'].mean()
            print(f"市场情绪: 日均涨停{avg_limit:.0f}家, 昨涨停今均涨{avg_effect:.2f}%")

        emotion_map = daily_emotion.set_index('day')
        skip_count = 0

        all_trades = []

        for i, date in enumerate(bt_dates):
            # === 市场情绪过滤（严格版） ===
            emo = emotion_map.loc[date] if date in emotion_map.index else None
            if emo is not None:
                skip = False
                # 条件1: 3日滚动赚钱效应 <= 0 → 空仓（必须持续为正才交易）
                if emo['emo_ma3'] <= 0:
                    skip = True
                # 条件2: 全市场上涨比例 < 50%（多数股票下跌 = 弱市）
                if emo.get('up_ratio', 0.5) < 0.50:
                    skip = True
                # 条件3: 单日赚钱效应严重恶化 < -2%
                if emo['limit_profit_effect'] < -2.0:
                    skip = True
                if skip:
                    skip_count += 1
                    continue

            # 直接取当日已算好因子的截面
            today = df_all[df_all['day'] == date].copy()
            today = today.dropna(subset=['f_excess_amount', 'f_price_position'])

            if today.empty:
                continue

            # 板块排名（直接用已标记的sector列）
            sector_agg = []
            for sector, grp in today.dropna(subset=['sector']).groupby('sector'):
                if len(grp) < 3:
                    continue
                sector_agg.append({
                    'sector': sector,
                    'avg_excess_amount': grp['f_excess_amount'].mean(),
                    'new_high_breadth': (grp['f_price_position'] >= 0.95).mean(),
                })
            if not sector_agg:
                continue

            sector_df = pd.DataFrame(sector_agg)
            sector_df['rank_A'] = sector_df['avg_excess_amount'].rank(pct=True)
            sector_df['rank_B'] = sector_df['new_high_breadth'].rank(pct=True)
            sector_df['composite_sector'] = 0.5 * sector_df['rank_A'] + 0.5 * sector_df['rank_B']
            top_sectors = sector_df.nlargest(top_n_sectors, 'composite_sector')

            # 龙头排名：板块内 top 2 → 全局 top 5
            candidates = []
            for _, srow in top_sectors.iterrows():
                sname = srow['sector']
                sub = today[today['sector'] == sname].copy()
                if sub.empty:
                    continue

                sub['r1'] = sub['f_excess_amount'].rank(pct=True)
                sub['r2'] = sub['f_price_position'].rank(pct=True)
                sub['r3'] = sub['f_amount_contin'].rank(pct=True)
                sub['composite_rank'] = 0.4 * sub['r1'] + 0.4 * sub['r2'] + 0.2 * sub['r3']

                # 质量过滤：非一字板 + 当日涨幅 1-8%（避免过热吹顶）
                sub = sub[
                    (sub['flag'] != 1) &
                    (sub['close_return'] >= 1.0) &
                    (sub['close_return'] <= 8.0)
                ]
                top_in_sector = sub.nlargest(top_n_stocks, 'composite_rank')
                if not top_in_sector.empty:
                    top_in_sector = top_in_sector.copy()
                    top_in_sector['sector_rank'] = srow['composite_sector']
                    candidates.append(top_in_sector)

            if not candidates:
                continue
            pool = pd.concat(candidates, ignore_index=True)

            # 股性加分 + 全局 top 5
            pool['r4_count'] = pool['f_limit_count'].fillna(0).rank(pct=True)
            pool['r4_meat'] = pool['f_limit_avg_meat'].fillna(0).rank(pct=True)
            pool['r4'] = 0.4 * pool['r4_count'] + 0.6 * pool['r4_meat']
            pool['final_rank'] = 0.60 * pool['composite_rank'] + 0.20 * pool['r4'] + 0.20 * pool['sector_rank']
            top = pool.nlargest(min(5, len(pool)), 'final_rank').copy()
            top['signal_date'] = date
            all_trades.append(top)

            if verbose and (i + 1) % 50 == 0:
                print(f"  进度: {i+1}/{len(bt_dates)}, 已选{sum(len(t) for t in all_trades)}笔, 跳过{skip_count}天")

        if verbose:
            print(f"  情绪过滤跳过: {skip_count}/{len(bt_dates)} 天 ({skip_count/len(bt_dates)*100:.1f}%)")

        if not all_trades:
            print("无交易信号！")
            return {'trades': pd.DataFrame(), 'stats': {}, 'equity_curve': pd.Series()}

        trades = pd.concat(all_trades, ignore_index=True)

        # --- 收益字段 ---
        # trade_return: 精细化买入价（gap_up用T+1 open, breakout用T close）→ T+2 open卖出
        if 'trade_return' in trades.columns:
            trades['return'] = trades['trade_return']
        else:
            trades['return'] = np.nan
        # 过滤 no_trade（T+1全天未突破T close的股票）
        if 'entry_type' in trades.columns:
            before = len(trades)
            trades = trades[trades['entry_type'] != 'no_trade']
            if verbose and len(trades) < before:
                print(f"  过滤未突破: {before - len(trades)}笔")
        trades = trades.dropna(subset=['return'])

        # --- 统计 ---
        stats = self._calc_stats(trades)

        # --- 累计净值 ---
        daily_ret = trades.groupby('signal_date')['return'].mean()
        equity = (1 + daily_ret / 100).cumprod()

        if verbose:
            self._print_stats(stats, equity)

        return {
            'trades': trades,
            'stats': stats,
            'equity_curve': equity
        }

    # =========================================================
    # 统计与输出
    # =========================================================
    @staticmethod
    def _calc_stats(trades: pd.DataFrame) -> dict:
        """计算回测统计指标"""
        rets = trades['return']
        wins = rets[rets > 0]
        losses = rets[rets <= 0]

        return {
            'total_trades': len(rets),
            'win_rate': (rets > 0).mean() * 100,
            'mean_return': rets.mean(),
            'median_return': rets.median(),
            'std_return': rets.std(),
            'avg_win': wins.mean() if len(wins) > 0 else 0,
            'avg_loss': losses.mean() if len(losses) > 0 else 0,
            'profit_loss_ratio': abs(wins.mean() / losses.mean()) if len(losses) > 0 and losses.mean() != 0 else float('inf'),
            'max_single_win': rets.max(),
            'max_single_loss': rets.min(),
            'daily_mean': trades.groupby('signal_date')['return'].mean().mean(),
            'trading_days': trades['signal_date'].nunique(),
        }

    @staticmethod
    def _print_stats(stats: dict, equity: pd.Series):
        """打印回测结果"""
        print("\n" + "=" * 60)
        print("回测结果")
        print("=" * 60)
        print(f"  交易天数:   {stats['trading_days']}")
        print(f"  交易总数:   {stats['total_trades']}")
        print(f"  胜率:       {stats['win_rate']:.1f}%")
        print(f"  日均收益:   {stats['daily_mean']:.3f}%")
        print(f"  单笔均值:   {stats['mean_return']:.3f}%")
        print(f"  单笔中位:   {stats['median_return']:.3f}%")
        print(f"  盈亏比:     {stats['profit_loss_ratio']:.2f}")
        print(f"  最大单笔盈: {stats['max_single_win']:.2f}%")
        print(f"  最大单笔亏: {stats['max_single_loss']:.2f}%")
        if len(equity) > 0:
            total_ret = (equity.iloc[-1] - 1) * 100
            max_dd = ((equity / equity.cummax()) - 1).min() * 100
            print(f"  累计收益:   {total_ret:.1f}%")
            print(f"  最大回撤:   {max_dd:.1f}%")
        print("=" * 60)

    def plot_equity(self, equity: pd.Series, save_path: str = None):
        """绘制净值曲线"""
        fig, ax = plt.subplots(figsize=(14, 6))
        ax.plot(range(len(equity)), equity.values, linewidth=1.5)
        ax.set_title('Hot Leader Strategy - Equity Curve', fontsize=14)
        ax.set_xlabel('Trading Days')
        ax.set_ylabel('Cumulative Return')
        ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
        ax.grid(True, alpha=0.3)

        # x轴标签：每50天标注日期
        tick_idx = list(range(0, len(equity), max(1, len(equity) // 10)))
        ax.set_xticks(tick_idx)
        ax.set_xticklabels([equity.index[i] for i in tick_idx], rotation=45, fontsize=8)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150)
            print(f"净值曲线已保存: {save_path}")
        plt.close()

    # =========================================================
    # 分组检验（验证因子单调性）
    # =========================================================
    @staticmethod
    def group_analysis(trades: pd.DataFrame, n_groups: int = 5):
        """按 composite_rank 分组，验证高rank组收益 > 低rank组"""
        if trades.empty or 'composite_rank' not in trades.columns:
            print("无数据或缺少 composite_rank 列")
            return

        trades = trades.dropna(subset=['return', 'composite_rank']).copy()
        trades['rank_group'] = pd.qcut(trades['composite_rank'], n_groups,
                                        labels=[f'G{i+1}' for i in range(n_groups)],
                                        duplicates='drop')

        print("\n分组检验（按 composite_rank 分组，G1最弱 → G5最强）:")
        print("-" * 55)
        print(f"{'组别':>6} {'样本数':>8} {'均值%':>8} {'胜率%':>8} {'中位%':>8}")
        print("-" * 55)
        for g, grp in trades.groupby('rank_group', observed=True):
            r = grp['return']
            print(f"{g:>6} {len(r):>8} {r.mean():>8.3f} {(r>0).mean()*100:>8.1f} {r.median():>8.3f}")
        print("-" * 55)


# =============================================================
# 命令行入口
# =============================================================
if __name__ == '__main__':
    strategy = HotLeaderStrategy()

    if len(sys.argv) > 1 and sys.argv[1] == 'backtest':
        # 回测模式
        start = sys.argv[2] if len(sys.argv) > 2 else '2024-01-01'
        end = sys.argv[3] if len(sys.argv) > 3 else '2026-03-05'
        result = strategy.backtest(start, end)
        if not result['trades'].empty:
            strategy.group_analysis(result['trades'])
            strategy.plot_equity(result['equity_curve'], 'hot_leader_equity.png')
    else:
        # 单日扫描模式
        date = sys.argv[1] if len(sys.argv) > 1 else '2026-03-05'
        picks = strategy.scan(date)
        if picks.empty:
            print(f"{date}: 无信号")
        else:
            print(f"\n{'='*70}")
            print(f"热点龙头扫描结果 — {date}")
            print(f"{'='*70}")
            display_cols = ['sector', 'code', 'close', 'close_return',
                           'f_excess_amount', 'f_price_position', 'f_amount_contin',
                           'composite_rank']
            display_cols = [c for c in display_cols if c in picks.columns]
            print(picks[display_cols].to_string(index=False, float_format='%.3f'))
