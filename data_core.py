# -*- coding: utf-8 -*-
from pathlib import Path
from typing import Optional, List
import pandas as pd
from datetime import datetime
from config import Config
from BK import BlockManager


class StockDataManager:
    """股票数据管理器"""   
    def __init__(self):
        self.config = Config()
        self.block_manager: Optional[BlockManager] = None

    def _ensure_block_manager(self):
        if self.block_manager is None:
            data_dir = self.config.DATA_BASE_PATH / 'blocks'
            self.block_manager = BlockManager(data_dir=str(data_dir))
       
    def _load_parquet(self,
                      data_type: str,
                      start_date: Optional[str] = None,
                      end_date: Optional[str] = None,
                      codes: Optional[List[str]] = None,
                      columns: Optional[List[str]] = None) -> pd.DataFrame:
        file_path = self.config.OUTPUT_FILES.get(data_type)
        if not file_path or not file_path.exists():
            if data_type == 'day':
                raise FileNotFoundError(f"日线数据文件不存在: {file_path}")
            if data_type == 'min':
                raise FileNotFoundError(f"分钟数据文件不存在: {file_path}")
            raise FileNotFoundError(f"{data_type} 数据文件不存在: {file_path}")
        
        filters = []
        if start_date:
            filters.append(('day', '>=', start_date))
        if end_date:
            filters.append(('day', '<=', end_date))
        if codes:
            normalized_codes = [s.split('.')[0] for s in codes]
            normalized_codes = [str(c).zfill(6) for c in normalized_codes]
            filters.append(('code', 'in', normalized_codes))
        
        return pd.read_parquet(
            file_path,
            filters=filters if filters else None,
            columns=columns
        )
    
    def _get_file_info(self, filepath: Path) -> dict:
        """获取文件信息"""
        if filepath.exists():
            stat = filepath.stat()
            return {
                'size_mb': round(stat.st_size / (1024 * 1024), 2),
                'modified': datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M:%S')
            }
        return None
    
    def check_data_exists(self) -> dict:
        """检查数据文件状态"""
        return {
            'day': self._get_file_info(self.config.OUTPUT_FILES['day']),
            'min': self._get_file_info(self.config.OUTPUT_FILES['min'])
        }
    
    def print_data_status(self):
        """打印数据状态"""
        print("\n" + "="*60)
        print("股票数据文件状态")
        print("="*60)
        
        status = self.check_data_exists()
        
        for data_type, info in status.items():
            print(f"\n【{data_type.upper()} 数据】")
            if info:
                print(f"  ✓ 文件存在")
                print(f"  大小: {info['size_mb']} MB")
                print(f"  最后更新: {info['modified']}")
            else:
                print(f"  ✗ 文件不存在")
        
        print("\n" + "="*60 + "\n")
    
    def load_day_data(self, 
                      start_date: Optional[str] = None, 
                      end_date: Optional[str] = None,
                      codes: Optional[List[str]] = None,
                      columns: Optional[List[str]] = None) -> pd.DataFrame:
        """加载日线数据"""
        return self._load_parquet(
            data_type='day',
            start_date=start_date,
            end_date=end_date,
            codes=codes,
            columns=columns
        )
    
    def load_min_data(self, 
                      start_date: Optional[str] = None,
                      end_date: Optional[str] = None,
                      codes: Optional[List[str]] = None,
                      columns: Optional[List[str]] = None) -> pd.DataFrame:
        """加载分钟数据"""
        return self._load_parquet(
            data_type='min',
            start_date=start_date,
            end_date=end_date,
            codes=codes,
            columns=columns
        )
    
    def get_block_types(self) -> List[str]:
        """获取所有板块类型"""
        self._ensure_block_manager()
        return self.block_manager.get_block_types()
    
    def get_block_list(self, block_type: str) -> List[str]:
        """获取指定类型的板块列表"""
        self._ensure_block_manager()
        return self.block_manager.get_block_list(block_type)
    
    def get_stocks_in_block(self, block_name: str, block_type: str) -> List[str]:
        """获取板块中的股票列表"""
        self._ensure_block_manager()
        stocks = self.block_manager.get_stocks_in_block(block_name, block_type)
        return sorted([s.split('.')[0] for s in (stocks or [])])
 
    def get_blocks_by_stock(self, code: str) -> dict:
        """获取股票所属的板块"""
        self._ensure_block_manager()
        return self.block_manager.get_blocks_by_stock(code)
    
    def get_stock_info(self, code: str) -> Optional[pd.Series]:
        """获取股票详细信息"""
        self._ensure_block_manager()
        blocks = self.block_manager.get_blocks_by_stock(code)
        if not blocks:
            return None
        
        info = {
            'code': code,
            'sw1_industry': blocks.get('sw1_industry', []),
            'sw2_industry': blocks.get('sw2_industry', []),
            'csrc1_industry': blocks.get('csrc1_industry', []),
            'csrc2_industry': blocks.get('csrc2_industry', []),
            'concept': blocks.get('concept', []),
            'theme': blocks.get('theme_concept', [])
        }
        
        return pd.Series(info)
    
    def get_stocks_by_industry(self, 
                               primary: Optional[str] = None,
                               secondary: Optional[str] = None) -> List[str]:
        """按行业获取股票列表（内部复用通用板块接口）"""
        self._ensure_block_manager()
        
        if secondary:
            return self.get_stocks_in_block(secondary, 'sw2_industry')
        if primary:
            return self.get_stocks_in_block(primary, 'sw1_industry')
        return []
    
    def get_all_industries(self) -> dict:
        """获取所有行业"""
        self._ensure_block_manager()
        return {
            'primary': self.block_manager.get_block_list('sw1_industry'),
            'secondary': self.block_manager.get_block_list('sw2_industry')
        }
    
    def get_data_info(self, data_type: str = 'day') -> dict:
        """获取数据集详细信息"""
        file_path = self.config.OUTPUT_FILES.get(data_type)
        if not file_path or not file_path.exists():
            raise FileNotFoundError(f"{data_type} 数据文件不存在")
        
        import pyarrow.parquet as pq
        
        parquet_file = pq.ParquetFile(file_path)
        df_sample = pd.read_parquet(file_path, columns=['code', 'day'])
        
        info = {
            'file_size_mb': round(file_path.stat().st_size / (1024 * 1024), 2),
            'total_rows': parquet_file.metadata.num_rows,
            'stock_count': df_sample['code'].nunique(),
            'num_row_groups': parquet_file.num_row_groups
        }
        
        info['date_range'] = (df_sample['day'].min(), df_sample['day'].max())
        
        return info


if __name__ == '__main__':
    manager = StockDataManager()
    
    # 查看数据状态
    manager.print_data_status()
    
    # 加载日线数据
    print("\n加载日线数据:")
    df_day = manager.load_day_data(
        start_date='2024-01-01',
        end_date='2026-12-31',
        codes=['000001','603739']
    )
    print(df_day.head())
    
    print("\n加载日线数据:")
    df_min = manager.load_min_data(
        start_date='2024-01-01',
        end_date='2024-12-31',
        codes=['000001','603739']
    )
    print(df_min.head())
        
    # 获取行业列表
    print("\n行业列表:")
    industries = manager.get_all_industries()
    print(f"一级行业数: {len(industries['primary'])}")
    print(f"二级行业数: {len(industries['secondary'])}")
