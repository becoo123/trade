# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.dataset as ds
from typing import Dict, List, Set, Tuple
from dataclasses import dataclass
import logging
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from functools import partial
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataset import *
from numba import jit
from ContextInfo import ContextInfo
from block import SectorAnalyzer # Import SectorAnalyzer

# 配置日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

#==================== 策略核心逻辑 ====================#
def init(context: ContextInfo):
    """初始化函数"""
    logging.info("策略初始化开始")
    # 设置股票池
    context.stocklist = context.concept.stocks_df['code'].tolist()
    print('股票池',context.stocklist)
    logging.info(f"初始资金：{context.capital:,}")
    # 初始化板块分析器
    context.sector_analyzer = SectorAnalyzer(context) # Instantiate SectorAnalyzer

def handlebar(context: ContextInfo):
    """逐K线运行的主逻辑"""
    try:
        current_time = pd.to_datetime(context.current_time)
        # 每日9:30获取涨停分析指标
        if current_time.time().strftime('%H:%M') == '09:30':
            # 获取涨停分析指标
            limit_up_indicators = integrate_limit_up_analysis(context)
            # 将指标保存到上下文中，供后续使用
            context.limit_up_indicators = limit_up_indicators
            
            # 记录市场情绪指标
            if limit_up_indicators and 'market_sentiment' in limit_up_indicators:
                logging.info(f"市场情绪指标: {limit_up_indicators['market_sentiment']:.2f}")
                # 根据市场情绪调整仓位控制
                if limit_up_indicators['market_sentiment'] < 0.3:
                    logging.warning("市场情绪低迷，降低仓位")
                    context.position_limit = 0.5  # 降低仓位上限
                elif limit_up_indicators['market_sentiment'] > 0.7:
                    logging.info("市场情绪高涨，提高仓位")
                    context.position_limit = 1.0  # 提高仓位上限
                else:
                    context.position_limit = 0.8  # 默认仓位上限
        
        # 每日9:31执行选股逻辑
        if current_time.time().strftime('%H:%M') == '09:31':
            print('开始选股')
            daily_selection(context)   
            # 根据涨停分析指标调整选股结果
            if hasattr(context, 'limit_up_indicators') and context.limit_up_indicators:
                _adjust_selection_by_limit_up(context)
        # 每日9:32执行买入
        elif current_time.time().strftime('%H:%M') == '09:32':
            _execute_buy(context)            

        # 持仓管理和净值更新（只在收盘时执行）
        if current_time.time().strftime('%H:%M') == '15:00':
            _manage_positions(context)
            _update_netvalue(context)
    except Exception as e:
        logging.error(f"处理K线时发生异常：{e}")

def daily_selection(context: ContextInfo):
    # 获取当前日期
    current_date_str = context.current_time.split(' ')[0] # Get current date string
    # 找到所有小于等于当前日期的日期，并取最后一个的索引
    valid_dates = [d for d in context.day if d <= current_date_str] # Use current_date_str
    current_idx = len(valid_dates) - 1 if valid_dates else 0
    # 计算开始日期
    if current_idx >= 60:
        start_date = context.day[current_idx - 60]
    else:
        # 如果少于60天，则使用最早的日期
        start_date = context.day[0]
    # 获取候选股票列表
    available_codes = [
        code for code in context.stocklist
        if code not in context.MarketPosition
    ]
    if not available_codes:
        # 如果没有可用股票，清空候选列表并返回
        context.candidates = []
        return

    logging.info(f"开始对 {len(available_codes)} 只可用股票进行指标计算和板块分析")

    # 批量计算MACD红柱数量，使用SectorAnalyzer中的方法
    batch_size = 1000  # 可以调整这个数值
    stock_indicator_results = {} # Use a more descriptive name

    for i in range(0, len(available_codes), batch_size):
        batch_codes = available_codes[i:i + batch_size]
        # 调用 calculate_macd_red_counts，传递当前日期
        batch_results = context.sector_analyzer.calculate_macd_red_counts(
            context.data_api, batch_codes, current_date_str # Pass current_date_str
        )
        stock_indicator_results.update(batch_results)
        logging.info(f"已处理 {min(i+batch_size, len(available_codes))}/{len(available_codes)} 只股票的MACD数据")

    # 调用分层板块分析方法
    logging.info("开始进行分层板块分析")
    hierarchical_strong_sectors = context.sector_analyzer.find_strongest_hierarchical_sectors(
        stock_indicator_results=stock_indicator_results,
        indicator_threshold=3,  # MACD红柱阈值
        min_stocks_per_sector=25,
        top_n_primary=5,
        top_n_secondary=3
    )

    # 将最强板块存储到context中
    context.top_sectors = []
    for primary_sector, secondary_sectors in hierarchical_strong_sectors.items():
        context.top_sectors.append(primary_sector) # Add primary sectors
        for secondary_sector, _ in secondary_sectors:
            context.top_sectors.append(secondary_sector) # Add secondary sectors

    logging.info(f"分析完成，找到的顶尖板块: {context.top_sectors}")

    # 根据指标结果筛选股票作为初步候选
    preliminary_candidates = [code for code, result in stock_indicator_results.items() if result >= 3]
    logging.info(f"MACD指标满足条件的初步候选股票数量: {len(preliminary_candidates)}")

    # 最终候选股票：既满足指标条件，又属于顶尖板块
    context.candidates = [code for code in preliminary_candidates if any(s in context.sector_analyzer.get_stock_industry(code) for s in context.top_sectors)]
    logging.info(f"最终候选股票数量 (满足指标且属于顶尖板块): {len(context.candidates)}")

    print(f"candidates: {context.candidates}")
    # daily_selection函数不再返回DataFrame，而是直接更新context.candidates
    # return df # Remove this line
@jit(nopython=True)
def calculate_rank(array, value):
    """使用Numba加速的排名计算（pandas-version需要转为numpy array）"""
    return np.sum(array <= value) / len(array) * 100

def calc_historical_thresholds(hist_data: pd.DataFrame) -> dict:
    """计算每只股票的最后一天价格和成交量的百分位数"""
    percentiles = {}
    # 按股票分组处理
    for code, group in hist_data.groupby('code'):
        if len(group) < 60:  # 数据点太少的股票跳过
            continue            
        try:
            # 获取最后一天的价格和成交量
            pct_changes = (group['close']-group['preclose'])/group['preclose']
            last_close = pct_changes.iloc[-1]
            last_volume = group['volume'].iloc[-1] 
            # 计算收盘价和成交量的历史百分位数
            close_percentile = calculate_rank(pct_changes.values, last_close)
            volume_percentile = calculate_rank(group['volume'].values, last_volume)
            
            # 记录结果
            percentiles[code] = {
                'close_percentile': close_percentile,
                'volume_percentile': volume_percentile
            }
            
        except Exception as e:
            logging.error(f"计算股票 {code} 的百分位数失败: {str(e)}")
            continue
    return percentiles
    
def calculate_strongest_sector(context: ContextInfo) -> List[Tuple[str, float]]:
    """计算今天最强的板块（行业或概念），基于上涨股票的比例。"""
    # 获取 stocks_df
    stocks_df = context.concept.stocks_df
    rising_codes = context.candidates
    
    # 如果候选股票为空，则获取当日涨幅较大的股票作为参考
    if not rising_codes:
        logging.info("候选股票为空，尝试获取当日涨幅较大的股票作为参考")
        try:
            # 获取当前日期
            current_date = context.current_time.split(' ')[0]
            # 获取所有股票的当日涨幅数据
            day_filter = (
                (ds.field('day') == current_date) &
                (ds.field('code').isin(context.stocklist))
            )
            
            day_data = context.daydata.dataset.to_table(
                filter=day_filter,
                columns=['code', 'open', 'close', 'preclose']
            ).to_pandas()
            
            if not day_data.empty:
                # 计算涨幅
                day_data['pct_change'] = (day_data['close'] - day_data['preclose']) / day_data['preclose'] * 100
                # 按涨幅排序，取前100名
                top_stocks = day_data.sort_values('pct_change', ascending=False).head(100)['code'].tolist()
                rising_codes = top_stocks
                logging.info(f"获取到{len(rising_codes)}只当日涨幅较大的股票作为参考")
        except Exception as e:
            logging.error(f"获取当日涨幅较大的股票失败: {e}")
    
    if stocks_df.empty:
        return [("无数据", 0.0)]

    # 1. 计算行业板块强度
    industry_total = stocks_df.groupby('secondary_industry')['code'].count()  # 每个行业总股票数
    rising_industry = stocks_df[stocks_df['code'].isin(rising_codes)].groupby('secondary_industry')['code'].count()  # 每个行业上涨股票数
    industry_strength = (rising_industry / industry_total).fillna(0)  # 计算比例
    
    # 2. 计算概念板块强度
    concept_data = stocks_df[['code', 'concepts']].copy()
    # 将 concepts 列拆分为列表（逗号分隔）
    concept_data['concepts'] = concept_data['concepts'].apply(lambda x: x.split(',') if isinstance(x, str) else [])
    # 获取所有独特概念
    all_concepts = set().union(*concept_data['concepts'])
    concept_strength = {}
    for concept in all_concepts:
        total_stocks = concept_data[concept_data['concepts'].apply(lambda x: concept in x)]['code'].count()
        rising_stocks = concept_data[concept_data['code'].isin(rising_codes) & 
                                    concept_data['concepts'].apply(lambda x: concept in x)]['code'].count()
        if total_stocks > 0:
            concept_strength[concept] = rising_stocks / total_stocks

    # 3. 合并并找出最强板块
    all_strength = {**industry_strength.to_dict(), **concept_strength}
    if all_strength:
        # 按板块强度从高到低排序
        sorted_strength = sorted(all_strength.items(), key=lambda x: x[1], reverse=True)
        # 取前五名
        top_5 = sorted_strength[:5]
        context.top_sectors = top_5  # 修复：使用context.top_sectors而不是Self.top_sectors
        # 返回前五名的结果
        print(f"最强板块: {top_5}")
        return top_5
    else:
        return [("无数据", 0.0)]

def _match_sectors(stock_info: dict, top_sectors: List[str]) -> List[str]:
    """匹配股票所属的顶级板块"""
    matched = []
    for sector in top_sectors:
        if sector == stock_info['secondary_industry']:
            matched.append(f"二级行业: {sector}")
        elif sector == stock_info['sub_industry']:
            matched.append(f"细分行业: {sector}")
        elif sector in stock_info['concepts']:
            matched.append(f"概念: {sector}")
    return matched

def _execute_buy(context: ContextInfo):
    """执行买入操作"""
    candidates = []
    for code in context.candidates:
        try:
            # 检查是否属于最强板块
            stock_info = context.concept.get_stock_info(code)
            if stock_info is None:
                continue
                
            sectors = [stock_info['secondary_industry']] + stock_info['concepts'].split(',') if isinstance(stock_info['concepts'], str) else []
            if not any(s in [x[0] for x in context.top_sectors] for s in sectors):
                continue
                
            # 涨停判断
            today_high = context.mindata.query(
                code=code,
                start=context.current_time,
                end=context.current_time,
                fields=['high']
            )
            
            if today_high.empty:
                continue
                
            today_high_value = today_high.iloc[0]['high']
            
            # 获取前一日收盘价（使用交易日历）
            current_date = context.current_time.split(' ')[0]
            current_idx = context.day.index(current_date) if current_date in context.day else -1
            if current_idx > 0:  # 确保不是第一个交易日
                prev_date = context.day[current_idx - 1]
                prev_close_df = context.daydata.query(
                    code=code,
                    start=prev_date,
                    end=prev_date,
                    fields=['close']
                )
            else:
                # 如果是第一个交易日，则跳过
                continue
            
            if prev_close_df.empty:
                continue
                
            prev_close = prev_close_df.iloc[0]['close']
            
            if today_high_value >= round(prev_close * 1.095, 2):  # 精确涨停判断
                continue
                
            # 记录候选股
            open_price_df = context.mindata.query(
                code=code,
                start=context.current_time,
                end=context.current_time,
                fields=['open']
            )
            
            if open_price_df.empty:
                continue
                
            open_price = open_price_df.iloc[0]['open']
            candidates.append((code, open_price))
            
        except Exception as e:
            logging.error(f"处理候选股 {code} 时出错: {e}")
            continue
            
    # 选择涨幅最大的股票
    if candidates:
        selected = max(candidates, key=lambda x: x[1])
        code, price = selected
        max_shares = int(context.capital // price)
        if max_shares > 0:
            context.MarketPosition[code] = (max_shares, price)
            context.capital -= max_shares * price
            context.trade_log.append({
                'datetime': context.current_time,
                'code': code,
                'action': 'BUY',
                'price': price,
                'shares': max_shares
            })
            logging.info(f"买入 {code}, 价格: {price}, 数量: {max_shares}")

def _manage_positions(context: ContextInfo):
    """持仓管理"""
    to_remove = []
    for code, (shares, cost) in context.MarketPosition.items():
        try:
            # 检查是否有交易记录
            if not context.trade_log:
                continue
                
            # 找到最近的买入记录
            buy_records = [record for record in context.trade_log 
                          if record['code'] == code and record['action'] == 'BUY']
            if not buy_records:
                continue
                
            latest_buy = max(buy_records, key=lambda x: pd.to_datetime(x['datetime']))
            
            # 检查是否到达卖出时间（使用交易日历计算交易日数）
            buy_date = pd.to_datetime(latest_buy['datetime']).strftime('%Y-%m-%d')
            current_date = context.current_time.split(' ')[0]
            
            # 在交易日历中查找买入日期和当前日期的索引
            buy_idx = context.day.index(buy_date) if buy_date in context.day else -1
            current_idx = context.day.index(current_date) if current_date in context.day else -1
            
            # 计算交易日数差
            if buy_idx >= 0 and current_idx >= 0:
                hold_days = current_idx - buy_idx
            else:
                # 如果找不到索引，则使用原来的计算方法作为备选
                hold_days = (pd.to_datetime(context.current_time).date() - 
                             pd.to_datetime(latest_buy['datetime']).date()).days
                
            if hold_days >= 1:  # 持有至少1个交易日后卖出
                # 获取卖出价格（次日开盘价）
                sell_price_df = context.mindata.query(
                    code=code,
                    start=context.current_time,
                    end=context.current_time,
                    fields=['open']
                )
                
                if sell_price_df.empty:
                    continue
                    
                sell_price = sell_price_df.iloc[0]['open']
                
                context.capital += shares * sell_price
                to_remove.append(code)
                context.trade_log.append({
                    'datetime': context.current_time,
                    'code': code,
                    'action': 'SELL',
                    'price': sell_price,
                    'shares': shares
                })
                logging.info(f"卖出 {code}, 价格: {sell_price}, 数量: {shares}")
        except Exception as e:
            logging.error(f"管理持仓 {code} 时出错: {e}")
            continue
            
    # 清理已平仓头寸
    for code in to_remove:
        del context.MarketPosition[code]

def _update_netvalue(context: ContextInfo):
    """更新净值"""
    position_value = 0
    for code, (shares, _) in context.MarketPosition.items():
        try:
            price_df = context.mindata.query(
                code=code,
                start=context.current_time,
                end=context.current_time,
                fields=['close']
            )
            
            if price_df.empty:
                continue
                
            last_price = price_df.iloc[0]['close']
            position_value += shares * last_price
        except Exception as e:
            logging.error(f"更新持仓 {code} 市值时出错: {e}")
            continue
            
    context.Netvalue = context.capital + position_value
    context.net_values.append(context.Netvalue)
    
    # 只在每天收盘时（15:00）打印净值信息
    current_time = pd.to_datetime(context.current_time)
    if current_time.time().strftime('%H:%M') == '15:00':
        logging.info(f"当日收盘净值: {context.Netvalue:.2f}")

def get_sector_stocks(context: ContextInfo, sector: str) -> List[str]:
    """
    获取指定板块的所有股票
    """
    stocks_df = context.concept.stocks_df
    if stocks_df.empty:
        return []
    
    # 尝试从二级行业中查找
    industry_stocks = stocks_df[stocks_df['secondary_industry'] == sector]['code'].tolist()
    if industry_stocks:
        return industry_stocks
    
    # 尝试从细分行业中查找
    sub_industry_stocks = stocks_df[stocks_df['sub_industry'] == sector]['code'].tolist()
    if sub_industry_stocks:
        return sub_industry_stocks
    
    # 尝试从概念中查找
    concept_stocks = []
    for _, row in stocks_df.iterrows():
        if isinstance(row['concepts'], str) and sector in row['concepts'].split(','):
            concept_stocks.append(row['code'])
    
    return concept_stocks

#==================== 回测执行 ====================#

def run_backtest(start_date=None, end_date=None, log_level=logging.INFO, save_result=True):
    """运行回测函数，支持自定义回测区间
    
    参数:
        start_date: 回测开始日期，格式为'YYYY-MM-DD'
        end_date: 回测结束日期，格式为'YYYY-MM-DD'
        log_level: 日志级别，默认为INFO
        save_result: 是否保存回测结果图表，默认为True
        
    返回:
        context: 回测上下文对象，包含所有回测结果和状态
    """
    # 设置日志级别
    logging.getLogger().setLevel(log_level)
    
    # 参数验证
    if start_date and not isinstance(start_date, str):
        raise ValueError("start_date必须是字符串格式，如'2024-10-08'")
    if end_date and not isinstance(end_date, str):
        raise ValueError("end_date必须是字符串格式，如'2024-10-20'")
    
    # 初始化上下文
    try:
        context = ContextInfo(start_date, end_date)
        logging.info(f"成功初始化回测上下文，回测区间: {context.start} 至 {context.end}")
    except Exception as e:
        logging.error(f"初始化回测上下文失败: {e}")
        raise
    
    # 初始化策略
    try:
        init(context)
        logging.info("策略初始化完成")
    except Exception as e:
        logging.error(f"策略初始化失败: {e}")
        raise
    
    # 模拟回测
    print(f"策略回测开始...回测区间: {context.start} 至 {context.end}")
    # 记录净值变化
    context.net_values = [context.Netvalue]
    context.trade_log = []

    # 使用交易日历筛选回测日期
    try:
        start_dt = datetime.strptime(context.start, '%Y-%m-%d').date()
        end_dt = datetime.strptime(context.end, '%Y-%m-%d').date()
                
        filtered_date = []
        for d_str in context.day:
            d_dt = datetime.strptime(d_str, '%Y-%m-%d').date()
            if start_dt <= d_dt <= end_dt:
                filtered_date.append(d_str)
                
        if not filtered_date:
            logging.warning(f"在指定的回测区间内没有找到有效的交易日")
            return context
            
        logging.info(f"回测将执行{len(filtered_date)}个交易日")
    except Exception as e:
        logging.error(f"处理回测日期时出错: {e}")
        raise
    
    # 遍历每个交易日
    try:
        total_days = len(filtered_date)
        for day_idx, day in enumerate(filtered_date):
            # 遍历每个交易时间点
            for min_idx, minute in enumerate(context.min):
                # 设置当前时间
                context.current_time = f"{day} {minute}"
                context.barpos = day_idx * len(context.min) + min_idx
                
                # 执行策略
                handlebar(context)
                
            # 打印进度（每天结束时）
            print(f"回测进度: {day_idx+1}/{total_days} 天完成 ({(day_idx+1)/total_days*100:.1f}%)")
            
            # 每10天保存一次中间结果，防止长时间回测中断导致数据丢失
            if (day_idx + 1) % 10 == 0 and save_result:
                _save_intermediate_results(context, day_idx, total_days)
    except Exception as e:
        logging.error(f"回测执行过程中出错: {e}")
        # 即使出错也尝试保存当前结果
        if save_result:
            _save_intermediate_results(context, day_idx, total_days, is_error=True)
        raise
    
    # 计算回测指标
    try:
        performance_metrics = _calculate_performance_metrics(context)
        
        # 回测结束，输出结果
        print("\n回测完成!")
        print(f"初始资金: {10000000:,.2f}")
        print(f"最终净值: {context.Netvalue:,.2f}")
        print(f"总收益率: {performance_metrics['total_return']:.2f}%")
        print(f"年化收益率: {performance_metrics['annual_return']:.2f}%")
        print(f"最大回撤: {performance_metrics['max_drawdown']:.2f}%")
        print(f"夏普比率: {performance_metrics['sharpe_ratio']:.2f}")
        
        # 将性能指标添加到上下文中
        context.performance_metrics = performance_metrics
    except Exception as e:
        logging.error(f"计算回测指标时出错: {e}")
    
    # 绘制净值曲线
    if save_result:
        try:
            _plot_backtest_results(context)
        except Exception as e:
            logging.error(f"绘制回测结果图表时出错: {e}")
    
    return context

def _save_intermediate_results(context, day_idx, total_days, is_error=False):
    """保存回测中间结果
    
    参数:
        context: 回测上下文
        day_idx: 当前回测天数索引
        total_days: 总回测天数
        is_error: 是否因错误而保存中间结果
    """
    try:
        # 创建保存目录
        import os
        save_dir = "backtest_results"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
        # 生成文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        progress = f"{day_idx+1}of{total_days}"
        status = "error" if is_error else "progress"
        filename = f"{save_dir}/backtest_{status}_{progress}_{timestamp}"
        
        # 保存净值数据
        import pandas as pd
        net_value_df = pd.DataFrame({
            'net_value': context.net_values
        })
        net_value_df.to_csv(f"{filename}_netvalue.csv", index=False)
        
        # 保存交易日志
        if context.trade_log:
            trade_log_df = pd.DataFrame(context.trade_log)
            trade_log_df.to_csv(f"{filename}_tradelog.csv", index=False)
        
        # 绘制并保存净值曲线
        _plot_backtest_results(context, f"{filename}_chart.png")
        
        logging.info(f"已保存中间回测结果到 {filename}")
    except Exception as e:
        logging.error(f"保存中间回测结果失败: {e}")


def _calculate_performance_metrics(context):
    """计算回测性能指标
    
    参数:
        context: 回测上下文
        
    返回:
        dict: 包含各项性能指标的字典
    """
    import numpy as np
    
    # 计算基本指标
    net_values = np.array(context.net_values)
    initial_value = 10000000  # 初始资金
    final_value = context.Netvalue
    total_return = (final_value / initial_value - 1) * 100
    
    # 计算回撤
    max_drawdown = 0
    peak = net_values[0]
    for value in net_values:
        if value > peak:
            peak = value
        drawdown = (peak - value) / peak * 100
        if drawdown > max_drawdown:
            max_drawdown = drawdown
    
    # 计算年化收益率
    try:
        start_date = datetime.strptime(context.start, '%Y-%m-%d')
        end_date = datetime.strptime(context.end, '%Y-%m-%d')
        years = (end_date - start_date).days / 365.0
        annual_return = ((1 + total_return / 100) ** (1 / years) - 1) * 100 if years > 0 else 0
    except:
        annual_return = 0
        logging.warning("计算年化收益率失败，使用默认值0")
    
    # 计算日收益率序列
    daily_returns = []
    for i in range(1, len(net_values)):
        daily_return = (net_values[i] / net_values[i-1]) - 1
        daily_returns.append(daily_return)
    
    # 计算夏普比率 (假设无风险利率为3%)
    risk_free_rate = 0.03
    if len(daily_returns) > 1:
        daily_returns_array = np.array(daily_returns)
        sharpe_ratio = (np.mean(daily_returns_array) * 252 - risk_free_rate) / (np.std(daily_returns_array) * np.sqrt(252)) if np.std(daily_returns_array) > 0 else 0
    else:
        sharpe_ratio = 0
    
    return {
        'total_return': total_return,
        'annual_return': annual_return,
        'max_drawdown': max_drawdown,
        'sharpe_ratio': sharpe_ratio,
        'daily_returns': daily_returns
    }


def _plot_backtest_results(context, save_path='backtest_result.png'):
    """绘制回测结果图表
    
    参数:
        context: 回测上下文
        save_path: 图表保存路径
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    # 创建图表
    plt.figure(figsize=(16, 12))
    
    # 创建子图布局
    gs = plt.GridSpec(3, 1, height_ratios=[3, 1, 1])
    
    # 净值曲线
    ax1 = plt.subplot(gs[0])
    ax1.plot(context.net_values, 'b-', linewidth=2)
    ax1.set_title('策略净值曲线', fontsize=14)
    ax1.set_ylabel('净值', fontsize=12)
    ax1.grid(True)
    
    # 标记最大回撤区间
    if hasattr(context, 'performance_metrics'):
        # 计算最大回撤区间
        net_values = np.array(context.net_values)
        max_drawdown = 0
        peak_idx = 0
        bottom_idx = 0
        temp_peak_idx = 0
        
        for i in range(len(net_values)):
            if net_values[i] > net_values[temp_peak_idx]:
                temp_peak_idx = i
            drawdown = (net_values[temp_peak_idx] - net_values[i]) / net_values[temp_peak_idx]
            if drawdown > max_drawdown:
                max_drawdown = drawdown
                peak_idx = temp_peak_idx
                bottom_idx = i
        
        # 标记最大回撤区间
        if max_drawdown > 0:
            ax1.plot([peak_idx, bottom_idx], [net_values[peak_idx], net_values[bottom_idx]], 'r-', linewidth=2)
            ax1.fill_between(range(peak_idx, bottom_idx+1), 
                            net_values[peak_idx], 
                            net_values[peak_idx:bottom_idx+1], 
                            color='red', alpha=0.3)
    
    # 日收益率
    if hasattr(context, 'performance_metrics') and 'daily_returns' in context.performance_metrics:
        daily_returns = context.performance_metrics['daily_returns']
        ax2 = plt.subplot(gs[1], sharex=ax1)
        ax2.bar(range(len(daily_returns)), daily_returns, color=['g' if r >= 0 else 'r' for r in daily_returns])
        ax2.set_title('日收益率', fontsize=14)
        ax2.set_ylabel('收益率', fontsize=12)
        ax2.grid(True)
    
    # 交易记录
    if context.trade_log:
        ax3 = plt.subplot(gs[2], sharex=ax1)
        buy_indices = []
        sell_indices = []
        
        for log in context.trade_log:
            if 'barpos' in log and 'action' in log:
                if log['action'] == 'buy':
                    buy_indices.append(log['barpos'])
                elif log['action'] == 'sell':
                    sell_indices.append(log['barpos'])
        
        # 绘制买入点和卖出点
        if buy_indices:
            buy_values = [context.net_values[i] if i < len(context.net_values) else context.net_values[-1] for i in buy_indices]
            ax1.scatter(buy_indices, buy_values, color='g', marker='^', s=100, label='买入')
        
        if sell_indices:
            sell_values = [context.net_values[i] if i < len(context.net_values) else context.net_values[-1] for i in sell_indices]
            ax1.scatter(sell_indices, sell_values, color='r', marker='v', s=100, label='卖出')
        
        if buy_indices or sell_indices:
            ax1.legend()
    
    # 添加性能指标文本
    if hasattr(context, 'performance_metrics'):
        metrics = context.performance_metrics
        info_text = f"总收益率: {metrics['total_return']:.2f}%\n"
        info_text += f"年化收益率: {metrics['annual_return']:.2f}%\n"
        info_text += f"最大回撤: {metrics['max_drawdown']:.2f}%\n"
        info_text += f"夏普比率: {metrics['sharpe_ratio']:.2f}"
        
        # 在图表右上角添加文本框
        plt.figtext(0.75, 0.9, info_text, fontsize=12, 
                   bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
    
    # 调整布局并保存
    plt.tight_layout()
    plt.savefig(save_path, dpi=100)
    plt.show()


def integrate_limit_up_analysis(context):
    """集成涨停分析指标到回测系统
    
    参数:
        context: 回测上下文
        
    返回:
        dict: 涨停分析指标
    """
    try:
        from tools.analyze_limit_up_stocks import LimitUpAnalyzer
        
        # 获取当前日期
        current_date = context.current_time.split(' ')[0]
        
        # 初始化涨停分析器
        analyzer = LimitUpAnalyzer()
        
        # 获取板块分析指标
        indicators = analyzer.get_block_analysis_indicators(current_date)
        
        # 记录指标到日志
        logging.info(f"日期: {current_date}, 涨停分析指标: {indicators}")
        
        return indicators
    except Exception as e:
        logging.error(f"集成涨停分析指标失败: {e}")
        return {}


def _adjust_selection_by_limit_up(context):
    """根据涨停分析指标调整选股结果
    
    参数:
        context: 回测上下文，包含limit_up_indicators和candidates属性
    """
    try:
        if not hasattr(context, 'limit_up_indicators') or not context.limit_up_indicators:
            logging.warning("没有可用的涨停分析指标，跳过选股调整")
            return
            
        if not hasattr(context, 'candidates') or not context.candidates:
            logging.warning("没有候选股票，跳过选股调整")
            return
            
        indicators = context.limit_up_indicators
        original_candidates = context.candidates.copy()
        
        # 1. 根据市场情绪调整选股数量
        if 'market_sentiment' in indicators:
            sentiment = indicators['market_sentiment']
            # 市场情绪低迷时减少选股数量，高涨时增加选股数量
            if sentiment < 0.3:
                max_candidates = min(3, len(original_candidates))  # 最多选3只
                logging.info(f"市场情绪低迷({sentiment:.2f})，限制选股数量为{max_candidates}")
            elif sentiment > 0.7:
                max_candidates = min(10, len(original_candidates))  # 最多选10只
                logging.info(f"市场情绪高涨({sentiment:.2f})，扩大选股数量至{max_candidates}")
            else:
                max_candidates = min(5, len(original_candidates))  # 默认最多选5只
                logging.info(f"市场情绪中性({sentiment:.2f})，选股数量为{max_candidates}")
                
            # 如果候选股票数量超过限制，则进行筛选
            if len(original_candidates) > max_candidates:
                context.candidates = original_candidates[:max_candidates]
        
        # 2. 根据涨停质量分数过滤低质量股票
        if 'limit_up_quality_score' in indicators and indicators['limit_up_quality_score'] < 0.4:
            # 涨停质量低时，更加严格地筛选股票
            logging.warning(f"涨停质量较低({indicators['limit_up_quality_score']:.2f})，进行更严格筛选")
            
            # 获取当前日期和前一交易日
            current_date = context.current_time.split(' ')[0]
            date_index = context.day.index(current_date) if current_date in context.day else -1
            
            if date_index > 0:
                prev_date = context.day[date_index - 1]
                
                # 获取候选股票的前一日数据
                try:
                    day_filter = (
                        (ds.field('day') == prev_date) &
                        (ds.field('code').isin(context.candidates))
                    )
                    
                    prev_data = context.daydata.dataset.to_table(
                        filter=day_filter,
                        columns=['code', 'close', 'volume', 'amount']
                    ).to_pandas()
                    
                    if not prev_data.empty:
                        # 计算成交额，按成交额排序（优先选择流动性好的股票）
                        prev_data = prev_data.sort_values('amount', ascending=False)
                        filtered_candidates = prev_data['code'].tolist()
                        
                        # 更新候选列表，保留流动性最好的前70%
                        keep_count = max(1, int(len(filtered_candidates) * 0.7))
                        context.candidates = filtered_candidates[:keep_count]
                        
                        logging.info(f"根据流动性筛选后的候选股票数量: {len(context.candidates)}")
                except Exception as e:
                    logging.error(f"获取前一日数据失败: {e}")
        
        # 3. 根据趋势动量调整选股优先级
        if 'trend_momentum' in indicators:
            momentum = indicators['trend_momentum']
            logging.info(f"当前趋势动量: {momentum:.2f}")
            
            # 如果有强烈的趋势动量，优先选择强势股
            if abs(momentum) > 0.7:
                try:
                    # 获取当前日期
                    current_date = context.current_time.split(' ')[0]
                    
                    # 获取候选股票的当日数据
                    day_filter = (
                        (ds.field('day') == current_date) &
                        (ds.field('code').isin(context.candidates))
                    )
                    
                    day_data = context.daydata.dataset.to_table(
                        filter=day_filter,
                        columns=['code', 'open', 'close', 'preclose']
                    ).to_pandas()
                    
                    if not day_data.empty:
                        # 计算涨幅
                        day_data['pct_change'] = (day_data['close'] - day_data['preclose']) / day_data['preclose'] * 100
                        
                        # 根据趋势动量的方向排序
                        if momentum > 0:  # 上升趋势，优先选择涨幅大的
                            day_data = day_data.sort_values('pct_change', ascending=False)
                        else:  # 下降趋势，优先选择涨幅小的（防御性选股）
                            day_data = day_data.sort_values('pct_change', ascending=True)
                            
                        # 更新候选列表顺序
                        context.candidates = day_data['code'].tolist()
                        logging.info(f"根据趋势动量({momentum:.2f})调整了候选股票优先级")
                except Exception as e:
                    logging.error(f"调整候选股票优先级失败: {e}")
        
        # 记录调整后的候选股票
        logging.info(f"调整后的候选股票数量: {len(context.candidates)}")
        if len(context.candidates) > 0:
            logging.info(f"调整后的前3个候选股票: {context.candidates[:3]}")
            
    except Exception as e:
        logging.error(f"根据涨停分析指标调整选股结果失败: {e}")
        import traceback
        logging.error(traceback.format_exc())
        # 发生错误时保留原始候选列表
        if hasattr(context, 'candidates') and not context.candidates and hasattr(context, '_original_candidates'):
            context.candidates = context._original_candidates


if __name__ == "__main__":
    # 运行回测示例
    context = run_backtest(
        start_date='2023-01-01', 
        end_date='2023-12-31', 
        log_level=logging.INFO,
        save_result=True
    )
    
    # 访问回测结果
    if hasattr(context, 'performance_metrics'):
        print("\n详细性能指标:")
        for key, value in context.performance_metrics.items():
            if key != 'daily_returns':  # 不打印日收益率数组
                print(f"{key}: {value}")
                
    print("\n回测完成，结果已保存到backtest_results目录")
        
    