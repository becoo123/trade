# -*- coding: utf-8 -*-
import os
from pathlib import Path

class Config:
    # 数据路径配置
    TDX_BASE_PATH = os.getenv('TDX_PATH', 'C://new_tdx//vipdoc')
    DATA_BASE_PATH = Path(__file__).parent / 'data'
    
    # 通达信数据目录
    TDX_DIRS = {
        'day': [
            f'{TDX_BASE_PATH}//sz//lday',
            f'{TDX_BASE_PATH}//sh//lday'
        ],
        'min': [
            f'{TDX_BASE_PATH}//sz//minline',
            f'{TDX_BASE_PATH}//sh//minline'
        ]
    }
    
    # 输出文件路径
    OUTPUT_FILES = {
        'day': DATA_BASE_PATH / 'daydata.parquet',
        'min': DATA_BASE_PATH / 'mindata.parquet',
        'stocks': DATA_BASE_PATH / 'stocks.parquet',
        'codes': DATA_BASE_PATH / 'codes.csv'
    }
    
    # 性能参数
    CHUNK_SIZE = {
        'day': 100000,
        'min': 10000
    }
    MAX_WORKERS = min(8, os.cpu_count())
    
    # 股票代码过滤
    ALLOWED_CODE_PREFIXES = ('0', '3', '6')
    
    # 涨停计算参数
    LIMIT_UP_RATIOS = {
        'main_board': 1.10,  # 主板10%
        'growth_board': 1.20  # 创业板/科创板20%
    }
