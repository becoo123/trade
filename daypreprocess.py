# -*- coding: utf-8 -*-
import os
import struct
import math
import time
import datetime
import pandas as pd
from tqdm import tqdm
import concurrent.futures
import threading
import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np
from numba import jit, njit, types
from numba.typed import List

print(os.getcwd())

@njit
def calculate_basic_data(records_data, is_main_board):
    """
    使用JIT编译计算基础数据（不包含未来收益）
    只计算当日的涨跌幅和涨停标志
    """
    enhanced_records = List()
    
    # 根据是否主板确定涨停比例
    limit_ratio = 1.1 if is_main_board else 1.2
    
    for i in range(len(records_data)):
        date_val, open_price, high_price, low_price, close_price, amount_val, volume_val, preclose = records_data[i]
        
        if preclose > 0:
            limit_price = round(preclose * limit_ratio, 2)
            is_limit_up = 1 if abs(close_price - limit_price) < 0.01 else 0
            
            # 当日涨跌幅（相对于昨收）
            high_return = round((high_price - preclose) / preclose * 100, 2)
            low_return = round((low_price - preclose) / preclose * 100, 2)
            close_return = round((close_price - preclose) / preclose * 100, 2)
            open_return = round((open_price - preclose) / preclose * 100, 2)
        else:
            limit_price = 0.0
            is_limit_up = 0
            high_return = low_return = close_return = open_return = 0.0
        
        enhanced_records.append((
            date_val, open_price, high_price, low_price, close_price,
            amount_val, volume_val, limit_price, is_limit_up, preclose,
            high_return, low_return, close_return, open_return
        ))
    
    return enhanced_records

def parse_binary_data(buffer):
    """
    使用原来的struct.unpack方法解析二进制数据
    """
    size = len(buffer)
    row_size = 32
    preclose = 0.0
    records = []
    
    for i in range(0, size, row_size):
        if i + row_size > size:
            break
            
        try:
            row = list(struct.unpack('IIIIIfII', buffer[i:i+row_size]))
            
            date_val = row[0]
            if date_val < 20200101:
                if row[4] > 0:
                    preclose = row[4] / 100.0
                continue
                
            open_price = row[1] / 100.0
            high_price = row[2] / 100.0
            low_price = row[3] / 100.0
            close_price = row[4] / 100.0
            amount_val = row[5]  # amount 是 float 类型
            volume_val = row[6]  # volume 是 int 类型
            
            records.append((
                date_val, open_price, high_price, low_price, close_price,
                amount_val, volume_val, preclose
            ))
            
            preclose = close_price
            
        except (struct.error, ValueError):
            continue
    
    return records

def format_date_batch(date_vals):
    """
    批量格式化日期
    """
    dates = []
    for date_val in date_vals:
        date_str = str(int(date_val))
        if len(date_str) == 8:
            year = int(date_str[:4])
            month = int(date_str[4:6])
            day = int(date_str[6:8])
            dates.append(f"{year:04d}-{month:02d}-{day:02d}")               
    return dates

class LCToParquet:
    def __init__(self, output_path, chunk_size=100000):
        """
        :param output_path: 最终输出的 Parquet 文件路径
        :param chunk_size: 累计达到多少条记录后写入磁盘
        """
        self.output_path = output_path
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        self.chunk_size = chunk_size
        self.chunk_records = []
        self.lock = threading.Lock()
        self.writer = None
        
        # 定义写入 Parquet 的 schema（基础字段）
        self.schema = pa.schema([
            ('code', pa.string()),
            ('day', pa.string()),
            ('open', pa.float64()),
            ('high', pa.float64()),
            ('low', pa.float64()),
            ('close', pa.float64()),
            ('amount', pa.float64()),
            ('volume', pa.int64()),
            ('limit_price', pa.float64()),
            ('flag', pa.int8()),
            ('preclose', pa.float64()),
            ('high_return', pa.float64()),      # 当日最高涨幅
            ('low_return', pa.float64()),       # 当日最低涨幅
            ('close_return', pa.float64()),     # 当日收盘涨幅
            ('open_return', pa.float64())       # 当日开盘涨幅
        ])

    def process_file(self, filepath, filename):
        """
        解析单个 .day 文件
        """
        code = filename[2:8]
        is_main_board = code.startswith(('00', '60'))  # 主板：沪深主板；300xxx创业板/688xxx科创板 用20%
        
        with open(filepath, 'rb') as f:
            buffer = f.read()
        
        # 使用原来的struct.unpack方法解析二进制数据
        raw_records = parse_binary_data(buffer)
        
        if len(raw_records) == 0:
            return
        
        # 转换为numba.typed.List以便JIT函数使用
        numba_records = List()
        for record in raw_records:
            numba_records.append(record)
        
        # 使用JIT编译的函数计算基础数据
        enhanced_records = calculate_basic_data(numba_records, is_main_board)
        
        # 批量格式化日期
        date_vals = [record[0] for record in enhanced_records]
        formatted_dates = format_date_batch(date_vals)
        
        # 构建最终记录
        records = []
        for i, record in enumerate(enhanced_records):
            records.append((
                code, formatted_dates[i], record[1], record[2], record[3],
                record[4], record[5], int(record[6]), record[7], int(record[8]),
                record[9], record[10], record[11], record[12], record[13]
            ))
                
        if records:
            self._append_records(records)

    def _append_records(self, records):
        """
        将解析出的记录追加到 chunk 中,达到阈值则刷新到磁盘
        """
        with self.lock:
            self.chunk_records.extend(records)
            if len(self.chunk_records) >= self.chunk_size:
                self.flush_chunk()

    def flush_chunk(self):
        """
        将当前累计的记录直接写入 Parquet 文件
        """
        if not self.chunk_records:
            return
            
        columns = ['code', 'day', 'open', 'high', 'low', 'close', 'amount', 
                  'volume', 'limit_price', 'flag', 'preclose', 'high_return', 
                  'low_return', 'close_return', 'open_return']
        
        # 转置数据以提高性能
        data_dict = {col: [] for col in columns}
        for record in self.chunk_records:
            for i, col in enumerate(columns):
                data_dict[col].append(record[i])
        
        df = pd.DataFrame(data_dict)
        
        # 转换为PyArrow表
        table = pa.Table.from_pandas(df, schema=self.schema, preserve_index=False)
        
        # 初始化writer
        if self.writer is None:
            self.writer = pq.ParquetWriter(self.output_path, self.schema)
        
        self.writer.write_table(table)
        self.chunk_records.clear()

    def process_dir(self, dir_path, allowed_codes=None, use_multithreading=True, max_workers=None):
        """
        处理指定目录下所有文件
        """
        if not os.path.exists(dir_path):
            print(f"目录不存在:{dir_path}")
            return
            
        files = [f for f in os.listdir(dir_path) if f.endswith('.day')]
        if allowed_codes is not None:
            files = [f for f in files if f[2:8] in allowed_codes]
        
        if not files:
            print(f"目录 {dir_path} 中没有找到符合条件的文件")
            return
        
        if use_multithreading:
            if max_workers is None:
                max_workers = min(8, os.cpu_count() or 1)
                
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = []
                for f in files:
                    future = executor.submit(self.process_file, os.path.join(dir_path, f), f)
                    futures.append(future)
                
                for future in tqdm(concurrent.futures.as_completed(futures),
                                   total=len(futures),
                                   desc=f"Processing {os.path.basename(dir_path)}"):
                    future.result()
        else:
            for f in tqdm(files, desc=f"Processing {os.path.basename(dir_path)}"):
                self.process_file(os.path.join(dir_path, f), f)

    def finalize(self, calculate_future_returns=True, n_days=[1, 3, 5]):
        """
        刷新剩余数据,关闭ParquetWriter,然后读取并计算未来收益
        
        Parameters:
        -----------
        calculate_future_returns : bool
            是否计算未来收益
        n_days : list
            计算未来N日收益的天数列表，如[1, 3, 5]
        """
        with self.lock:
            self.flush_chunk()
            if self.writer is not None:
                self.writer.close()
                self.writer = None
        
        print(f"原始数据已写入: {self.output_path}")
        
        if not calculate_future_returns:
            return
        
        print("开始计算未来收益...")
        
        # 读取刚写入的parquet文件
        df = pd.read_parquet(self.output_path)
        
        # 排序（非常重要！）
        df = df.sort_values(['code', 'day']).reset_index(drop=True)
        
        # 过滤异常数据（价格为0或过大）
        print(f"过滤前记录数: {len(df)}")
        df = df[
            (df['close'] > 0) & 
            (df['open'] > 0) & 
            (df['close'] < 10000) &  # 过滤指数等异常数据
            (df['open'] < 10000)
        ].copy()
        print(f"过滤后记录数: {len(df)}")
        
        # 计算未来收益
        df = self._calculate_future_returns(df, n_days)
        
        # 重新写入文件
        df.to_parquet(self.output_path, index=False, engine='pyarrow')
        print(f"数据已保存为 Parquet 格式: {self.output_path}")
    
    def _calculate_future_returns(self, df, n_days=[1]):
        """
        计算前瞻收益（带下划线前缀标记为未来数据）

        _O2O: 今日开盘买 → 明日开盘卖  (open to open)
        _O2C: 今日开盘买 → 明日收盘卖  (open to close)

        注意: data_processor_v2 已在 _post_process 中计算了这两个字段。
        此函数仅在单独执行后处理时使用，会跳过已存在的列。
        """
        print("计算前瞻收益中...")

        if '_O2O' not in df.columns:
            next_open = df.groupby('code', sort=False)['open'].shift(-1)
            df['_O2O'] = ((next_open - df['open']) / df['open'] * 100).round(2)

        if '_O2C' not in df.columns:
            next_close = df.groupby('code', sort=False)['close'].shift(-1)
            df['_O2C'] = ((next_close - df['open']) / df['open'] * 100).round(2)

        # 过滤除权除息异常值（>31%）
        for col in ['_O2O', '_O2C']:
            df.loc[df[col].abs() > 31, col] = np.nan

        # 删除旧格式的前瞻收益列（如存在）
        old_cols = [c for c in df.columns if any(
            c.startswith(p) for p in ['next_open_return', 'next_close_return',
                                       'next_high_return', 'next_low_return',
                                       'open_to_open_return', 'open_to_close_return',
                                       'open_to_high_return', 'open_to_low_return']
        )]
        if old_cols:
            print(f"  删除旧前瞻收益列: {len(old_cols)}个")
            df.drop(columns=old_cols, inplace=True)

        return df

def create_parquet(allowed_codes, use_multithreading=True, n_days=[1, 3, 5]):
    """
    创建Parquet文件的主函数
    
    Parameters:
    -----------
    allowed_codes : set
        允许的股票代码集合
    use_multithreading : bool
        是否使用多线程
    n_days : list
        计算未来N日收益的天数列表
    """
    output_path = os.path.join("./data", "daydata.parquet")
    processor = LCToParquet(output_path, chunk_size=100000)
    
    dirs = [
        'C://new_tdx//vipdoc//sz//lday',
        'C://new_tdx//vipdoc//sh//lday'
    ]
    
    for d in dirs:
        processor.process_dir(d, allowed_codes=allowed_codes, use_multithreading=use_multithreading)
    
    processor.finalize(calculate_future_returns=True, n_days=n_days)

def main():
    """
    主函数
    """
    # 预编译JIT函数
    print("预编译JIT函数...")
    dummy_list = List()
    dummy_list.append((20200101, 1.0, 1.0, 1.0, 1.0, 100000.0, 1000, 1.0))
    calculate_basic_data(dummy_list, True)
    print("JIT预编译完成")
    
    # 读取股票代码
    stks_df = pd.read_csv('./data/codes.csv', dtype={'code': str}, usecols=['code'])    
    allowed_codes = {code for code in stks_df['code'] 
                    if code.startswith(('0', '3', '6'))}
    
    print(f"共找到 {len(allowed_codes)} 个符合条件的股票代码")
    
    start_time = time.time()
    # 计算1日、3日、5日的未来收益
    create_parquet(allowed_codes, use_multithreading=True, n_days=[1, 3, 5])
    end_time = time.time()
    
    print(f"处理完成,耗时:{end_time - start_time:.2f} 秒")

if __name__ == '__main__':
    main()
    
    # 验证生成的数据
    print("\n" + "="*80)
    print("验证生成的数据")
    print("="*80)
    
    df = pd.read_parquet('./data/daydata.parquet')
    
    print(f"\n总记录数: {len(df):,}")
    print(f"股票数量: {df['code'].nunique():,}")
    print(f"日期范围: {df['day'].min()} 到 {df['day'].max()}")
    print(f"\n字段数量: {len(df.columns)}")
    print(f"字段列表: {df.columns.tolist()}")
    
    # 检查价格范围
    print(f"\n价格统计:")
    print(f"  收盘价范围: {df['close'].min():.2f} 到 {df['close'].max():.2f}")
    print(f"  开盘价范围: {df['open'].min():.2f} 到 {df['open'].max():.2f}")
    print(f"  成交额范围: {df['amount'].min():.0f} 到 {df['amount'].max():.0f}")
    print(f"  成交量范围: {df['volume'].min():,} 到 {df['volume'].max():,}")
    
    # 检查涨停数据
    limit_up_data = df[df['flag'] == 1]
    print(f"\n涨停统计:")
    print(f"  涨停记录数: {len(limit_up_data):,}")
    print(f"  涨停占比: {len(limit_up_data)/len(df)*100:.2f}%")
    
    if len(limit_up_data) > 0:
        print(f"\n涨停板次日收益统计:")
        print(f"  次日开盘平均收益: {limit_up_data['next_open_return_1d'].mean():.2f}%")
        print(f"  次日收盘平均收益: {limit_up_data['next_close_return_1d'].mean():.2f}%")
        print(f"  次日最高平均收益: {limit_up_data['next_high_return_1d'].mean():.2f}%")
        print(f"  次日最低平均收益: {limit_up_data['next_low_return_1d'].mean():.2f}%")
    
    # 验证未来收益计算（选择一只正常股票）
    print("\n" + "="*80)
    print("验证未来收益计算")
    print("="*80)
    
    # 选择平安银行（000001）的数据
    sample_code = '000001'
    sample_df = df[df['code'] == sample_code].tail(10).copy()
    
    if len(sample_df) > 0:
        print(f"\n股票代码: {sample_code} (平安银行)")
        print("\n前10个交易日数据:")
        display_cols = ['day', 'close', 'open', 'next_open_return_1d', 
                       'next_close_return_1d', 'open_to_open_return_1d', 'open_to_close_return_1d']
        print(sample_df[display_cols].to_string(index=False))
        
        # 手动验证第一条记录
        if len(sample_df) >= 2:
            print(f"\n手动验证第一条记录:")
            row1 = sample_df.iloc[0]
            row2 = sample_df.iloc[1]
            
            manual_next_open = (row2['open'] - row1['close']) / row1['close'] * 100
            manual_next_close = (row2['close'] - row1['close']) / row1['close'] * 100
            manual_open_to_open = (row2['open'] - row1['open']) / row1['open'] * 100
            
            print(f"  日期: {row1['day']}")
            print(f"  今日收盘: {row1['close']:.2f}, 次日开盘: {row2['open']:.2f}")
            print(f"  计算的next_open_return_1d: {row1['next_open_return_1d']:.2f}%")
            print(f"  手动计算: {manual_next_open:.2f}%")
            print(f"  差异: {abs(row1['next_open_return_1d'] - manual_next_open):.4f}%")
            
            if abs(row1['next_open_return_1d'] - manual_next_open) < 0.01:
                print("  ✓ 验证通过！")
            else:
                print("  ✗ 验证失败！")
    
    # 全市场收益统计
    print("\n" + "="*80)
    print("全市场未来1日收益统计")
    print("="*80)
    
    # 过滤掉NaN值
    valid_df = df.dropna(subset=['next_open_return_1d', 'next_close_return_1d'])
    
    print(f"\n有效记录数: {len(valid_df):,}")
    print(f"\n基于收盘价买入:")
    print(f"  次日开盘卖出平均收益: {valid_df['next_open_return_1d'].mean():.2f}%")
    print(f"  次日开盘卖出中位数收益: {valid_df['next_open_return_1d'].median():.2f}%")
    print(f"  次日开盘胜率: {(valid_df['next_open_return_1d'] > 0).sum() / len(valid_df) * 100:.2f}%")
    
    print(f"\n  次日收盘卖出平均收益: {valid_df['next_close_return_1d'].mean():.2f}%")
    print(f"  次日收盘卖出中位数收益: {valid_df['next_close_return_1d'].median():.2f}%")
    print(f"  次日收盘胜率: {(valid_df['next_close_return_1d'] > 0).sum() / len(valid_df) * 100:.2f}%")
    
    print(f"\n基于开盘价买入:")
    print(f"  次日开盘卖出平均收益: {valid_df['open_to_open_return_1d'].mean():.2f}%")
    print(f"  次日开盘卖出中位数收益: {valid_df['open_to_open_return_1d'].median():.2f}%")
    print(f"  次日开盘胜率: {(valid_df['open_to_open_return_1d'] > 0).sum() / len(valid_df) * 100:.2f}%")
    
    print(f"\n  次日收盘卖出平均收益: {valid_df['open_to_close_return_1d'].mean():.2f}%")
    print(f"  次日收盘卖出中位数收益: {valid_df['open_to_close_return_1d'].median():.2f}%")
    print(f"  次日收盘胜率: {(valid_df['open_to_close_return_1d'] > 0).sum() / len(valid_df) * 100:.2f}%")
    
    # 收益分布
    print(f"\n收益分布（次日开盘）:")
    print(f"  > 5%: {(valid_df['next_open_return_1d'] > 5).sum() / len(valid_df) * 100:.2f}%")
    print(f"  2% ~ 5%: {((valid_df['next_open_return_1d'] > 2) & (valid_df['next_open_return_1d'] <= 5)).sum() / len(valid_df) * 100:.2f}%")
    print(f"  0% ~ 2%: {((valid_df['next_open_return_1d'] > 0) & (valid_df['next_open_return_1d'] <= 2)).sum() / len(valid_df) * 100:.2f}%")
    print(f"  -2% ~ 0%: {((valid_df['next_open_return_1d'] > -2) & (valid_df['next_open_return_1d'] <= 0)).sum() / len(valid_df) * 100:.2f}%")
    print(f"  < -2%: {(valid_df['next_open_return_1d'] <= -2).sum() / len(valid_df) * 100:.2f}%")
    
    print("\n" + "="*80)
    print("数据生成完成！")
    print("="*80)
