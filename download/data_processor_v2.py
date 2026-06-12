# -*- coding: utf-8 -*-
import os
import struct
import math
import threading
from abc import ABC, abstractmethod
from typing import List, Tuple, Optional, Set
from pathlib import Path
import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm
import concurrent.futures
from numba import njit
from numba.typed import List as NumbaList
#from ..config import Config


# ==================== Numba优化函数 ====================

@njit
def calculate_returns_and_limits(records_data, is_main_board):
    """
    使用JIT编译计算收益率和涨停标志
    返回增强的记录列表
    """
    enhanced_records = NumbaList()
    
    # 根据是否主板确定涨停比例
    limit_ratio = 1.1 if is_main_board else 1.2
    
    for i in range(len(records_data)):
        date_val, open_price, high_price, low_price, close_price, amount_val, volume_val, preclose = records_data[i]
        
        if preclose > 0:
            limit_price = round(preclose * limit_ratio, 2)
            is_limit_up = 1 if abs(close_price - limit_price) < 0.01 else 0
            
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


@njit
def parse_min_binary_batch(h1_array, h2_array):
    """
    使用JIT编译批量验证分钟数据的日期时间有效性
    
    Parameters:
    -----------
    h1_array : 日期编码数组
    h2_array : 时间编码数组
    
    Returns:
    --------
    有效记录的索引列表
    """
    n = len(h1_array)
    valid_indices = NumbaList()
    
    for i in range(n):
        h1 = h1_array[i]
        h2 = h2_array[i]
        
        # 解析日期时间
        year = h1 // 2048 + 2004
        month = (h1 % 2048) // 100
        day = h1 % 2048 % 100
        hour = h2 // 60
        minute = h2 % 60
        
        # 验证有效性
        if (2004 <= year <= 2030 and 
            1 <= month <= 12 and 
            1 <= day <= 31 and
            0 <= hour <= 23 and
            0 <= minute <= 59):
            valid_indices.append(i)
    
    return valid_indices


def parse_binary_data(buffer):
    """
    使用struct.unpack方法解析二进制数据
    返回原始记录列表 (不包含计算字段)
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
            amount_val = row[5]
            volume_val = row[6]
            
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
            if 1900 <= year <= 2100 and 1 <= month <= 12 and 1 <= day <= 31:
                dates.append(f"{year:04d}-{month:02d}-{day:02d}")
            else:
                dates.append('1900-01-01')
        else:
            dates.append('1900-01-01')
    return dates


# ==================== 基类 ====================

class BaseDataProcessor(ABC):
    """数据处理基类"""
    
    def __init__(self, output_path: Path, chunk_size: int):
        self.output_path = output_path
        self.chunk_size = chunk_size
        self.chunk_records = []
        self.lock = threading.Lock()
        self.writer = None
        
    @property
    @abstractmethod
    def schema(self) -> pa.Schema:
        """定义数据schema"""
        pass
    
    @property
    @abstractmethod
    def file_extension(self) -> str:
        """文件扩展名"""
        pass
    
    @property
    @abstractmethod
    def columns(self) -> List[str]:
        """列名"""
        pass
    
    @abstractmethod
    def parse_file(self, filepath: str, code: str) -> List[Tuple]:
        """解析单个文件"""
        pass
    
    def append_records(self, records: List[Tuple]):
        """线程安全地添加记录"""
        with self.lock:
            self.chunk_records.extend(records)
            if len(self.chunk_records) >= self.chunk_size:
                self._flush_chunk()
    
    def _flush_chunk(self):
        """写入数据到parquet"""
        if not self.chunk_records:
            return
        
        df = self._create_dataframe()
        df = self._post_process(df)
        
        table = pa.Table.from_pandas(df, schema=self.schema, preserve_index=False)
        
        if self.writer is None:
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            self.writer = pq.ParquetWriter(self.output_path, self.schema)
        
        self.writer.write_table(table)
        self.chunk_records.clear()
    
    def _create_dataframe(self) -> pd.DataFrame:
        """从记录创建DataFrame - 优化版本使用字典"""
        data_dict = {col: [] for col in self.columns}
        for record in self.chunk_records:
            for i, col in enumerate(self.columns):
                data_dict[col].append(record[i])
        return pd.DataFrame(data_dict)
    
    def _post_process(self, df: pd.DataFrame) -> pd.DataFrame:
        """后处理数据（子类可重写）"""
        return df
    
    def process_directories(self, directories: List[str], 
                          allowed_codes: Optional[Set[str]] = None,
                          use_multithreading: bool = True,
                          max_workers: Optional[int] = None):
        """处理多个目录"""
        for directory in directories:
            self.process_directory(directory, allowed_codes, use_multithreading, max_workers)
    
    def process_directory(self, directory: str, 
                         allowed_codes: Optional[Set[str]] = None,
                         use_multithreading: bool = True,
                         max_workers: Optional[int] = None):
        """处理单个目录"""
        if not os.path.exists(directory):
            print(f"⚠ 目录不存在：{directory}")
            return
            
        files = self._get_files(directory, allowed_codes)
        
        if not files:
            print(f"⚠ 目录 {directory} 中没有找到符合条件的文件")
            return
        
        print(f"找到 {len(files)} 个文件")
        
        if use_multithreading:
            self._process_parallel(directory, files, max_workers)
        else:
            self._process_sequential(directory, files)
    
    def _get_files(self, directory: str, allowed_codes: Optional[Set[str]]) -> List[str]:
        """获取需要处理的文件列表"""
        files = [f for f in os.listdir(directory) if f.endswith(self.file_extension)]
        if allowed_codes:
            files = [f for f in files if f[2:8] in allowed_codes]
        return files
    
    def _process_parallel(self, directory: str, files: List[str], max_workers: Optional[int] = None):
        """并行处理文件"""
        if max_workers is None:
            max_workers = min(8, os.cpu_count() or 1)
            
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(self._process_single_file, directory, f) 
                for f in files
            ]
            
            failed_count = 0
            for future in tqdm(concurrent.futures.as_completed(futures),
                             total=len(futures),
                             desc=f"Processing {os.path.basename(directory)}"):
                try:
                    future.result()
                except Exception as e:
                    failed_count += 1
            
            if failed_count > 0:
                print(f"\n⚠ {failed_count} 个文件处理失败")
    
    def _process_sequential(self, directory: str, files: List[str]):
        """顺序处理文件"""
        for f in tqdm(files, desc=f"Processing {os.path.basename(directory)}"):
            try:
                self._process_single_file(directory, f)
            except Exception as e:
                print(f"Error processing {f}: {e}")
    
    def _process_single_file(self, directory: str, filename: str):
        """处理单个文件"""
        filepath = os.path.join(directory, filename)
        code = filename[2:8]
        records = self.parse_file(filepath, code)
        if records:
            self.append_records(records)
    
    def finalize(self):
        """完成处理"""
        with self.lock:
            self._flush_chunk()
            if self.writer:
                self.writer.close()
                self.writer = None


# ==================== 日线数据处理器 ====================

class DayDataProcessor(BaseDataProcessor):
    """日线数据处理器 - 优化版本"""

    file_extension = '.day'
    columns = ['code', 'day', 'open', 'high', 'low', 'close', 'amount',
               'volume', 'limit_price', 'flag', 'preclose', 'high_return',
               'low_return', 'close_return', 'open_return']

    # 后处理后新增的前瞻收益列
    _forward_columns = ['_O2O', '_O2C']

    @property
    def schema(self) -> pa.Schema:
        return pa.schema([
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
            # 当天收益（基于preclose）
            ('high_return', pa.float64()),
            ('low_return', pa.float64()),
            ('close_return', pa.float64()),
            ('open_return', pa.float64()),
            # 前瞻收益（带下划线前缀标记）
            ('_O2O', pa.float64()),   # 今日开盘买 → 明日开盘卖
            ('_O2C', pa.float64()),   # 今日开盘买 → 明日收盘卖
        ])
    
    def parse_file(self, filepath: str, code: str) -> List[Tuple]:
        """解析日线数据文件"""
        is_main_board = code.startswith(('00', '60'))
        
        try:
            with open(filepath, 'rb') as f:
                buffer = f.read()
        except Exception:
            return []
        
        # 使用优化的二进制解析
        raw_records = parse_binary_data(buffer)
        
        if not raw_records:
            return []
        
        # 转换为Numba类型
        numba_records = NumbaList()
        for record in raw_records:
            numba_records.append(record)
        
        # 使用JIT优化的计算
        enhanced_records = calculate_returns_and_limits(numba_records, is_main_board)
        
        if not enhanced_records:
            return []
        
        # 批量格式化日期
        date_vals = [record[0] for record in enhanced_records]
        formatted_dates = format_date_batch(date_vals)
        
        # 构建最终记录
        records = []
        for i, record in enumerate(enhanced_records):
            if formatted_dates[i] != '1900-01-01':
                records.append((
                    code, formatted_dates[i], record[1], record[2], record[3],
                    record[4], record[5], int(record[6]), record[7], int(record[8]),
                    record[9], record[10], record[11], record[12], record[13]
                ))
        
        return records
    
    def _post_process(self, df: pd.DataFrame) -> pd.DataFrame:
        """后处理: 计算前瞻收益（带下划线前缀标记为未来数据）

        当天收益字段 (high_return, low_return, close_return, open_return)
        保持原样，不做shift，就是当天相对preclose的涨跌幅。

        前瞻收益:
          _O2O = (明日开盘 - 今日开盘) / 今日开盘 * 100  (开盘到开盘)
          _O2C = (明日收盘 - 今日开盘) / 今日开盘 * 100  (开盘到收盘)
        """
        df.sort_values(['code', 'day'], inplace=True)

        next_open = df.groupby('code', sort=False)['open'].shift(-1)
        next_close = df.groupby('code', sort=False)['close'].shift(-1)

        df['_O2O'] = ((next_open - df['open']) / df['open'] * 100).round(2)
        df['_O2C'] = ((next_close - df['open']) / df['open'] * 100).round(2)

        # 过滤除权除息异常值（>31%）
        for col in ['_O2O', '_O2C']:
            df.loc[df[col].abs() > 31, col] = np.nan

        return df


# ==================== 分钟数据处理器 ====================

class MinDataProcessor(BaseDataProcessor):
    """分钟数据处理器 - 完全向量化版本"""
    
    file_extension = '.lc'
    columns = ['code', 'day', 'time', 'open', 'high', 'low', 'close', 'amount', 'volume']
    
    @property
    def schema(self) -> pa.Schema:
        return pa.schema([
            ('code', pa.string()),
            ('day', pa.string()),
            ('time', pa.string()),
            ('open', pa.float32()),
            ('high', pa.float32()),
            ('low', pa.float32()),
            ('close', pa.float32()),
            ('amount', pa.float32()),
            ('volume', pa.int32()),
        ])
    
    def _get_files(self, directory: str, allowed_codes: Optional[Set[str]]) -> List[str]:
        """获取需要处理的文件列表 - 支持多种扩展名"""
        all_files = os.listdir(directory)
        
        # 支持多种分钟数据文件扩展名
        valid_extensions = ('.lc', '.lc1', '.lc5', '.5')
        files = [f for f in all_files if f.lower().endswith(valid_extensions)]
        
        if allowed_codes:
            files = [f for f in files if f[2:8] in allowed_codes]
        
        return files
    
    def parse_file(self, filepath: str, code: str) -> List[Tuple]:
        """解析分钟数据文件 - 完全向量化版本"""
        try:
            file_size = os.path.getsize(filepath)
            if file_size == 0 or file_size % 32 != 0:
                return []
            
            record_count = file_size // 32
            
            # 定义numpy结构化数组的dtype
            dtype = np.dtype([
                ('h1', '<u2'),      # unsigned short, little-endian
                ('h2', '<u2'),      # unsigned short
                ('open', '<f4'),    # float32
                ('high', '<f4'),    # float32
                ('low', '<f4'),     # float32
                ('close', '<f4'),   # float32
                ('amount', '<f4'),  # float32
                ('volume', '<i4'),  # int32
                ('reserve1', '<i4'), # int32 (保留字段)
                ('reserve2', '<i4')  # int32 (保留字段)
            ])
            
            # 使用numpy从文件直接读取为结构化数组
            with open(filepath, 'rb') as f:
                data = np.fromfile(f, dtype=dtype, count=record_count)
            
            if len(data) == 0:
                return []
            
            # 向量化计算日期时间
            h1 = data['h1'].astype(np.int32)
            h2 = data['h2'].astype(np.int32)
            
            years = math.floor(h1 / 2048) + 2004
            months = math.floor(h1 % 2048 / 100)
            days = h1 % 2048 % 100
            hours = math.floor(h2 / 60)
            minutes = h2 % 60
            
            
            # 格式化日期时间（向量化）
            dates = np.array([
                f"{y:04d}-{m:02d}-{d:02d}" 
                for y, m, d in zip(years, months, days)
            ])
            times = np.array([
                f"{h:02d}:{min:02d}" 
                for h, min in zip(hours, minutes)
            ])
            
            # 获取有效的价格数据
            valid_data = data
            
            # 构建记录列表
            records = [
                (
                    code,
                    dates[i],
                    times[i],
                    float(valid_data['open'][i]),
                    float(valid_data['high'][i]),
                    float(valid_data['low'][i]),
                    float(valid_data['close'][i]),
                    float(valid_data['amount'][i]),
                    int(valid_data['volume'][i])
                )
                for i in range(len(valid_data))
            ]
            
            return records
            
        except Exception:
            return []
