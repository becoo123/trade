# -*- coding: utf-8 -*-
"""
板块数据管理模块 - 读取本地数据进行分析 BK.py
"""
from pathlib import Path
import logging
import pandas as pd
from typing import List, Dict, Optional
from collections import Counter
import json

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


class BlockManager:
    """板块数据管理器 - 读取本地数据"""
    BLOCK_TYPES = {
        'sw1_industry': '申万一级行业',
        'sw2_industry': '申万二级行业',
        'sw3_industry': '申万三级行业',
        'csrc1_industry': '证监会一级行业',
        'csrc2_industry': '证监会二级行业',
        'concept': '概念',
        'theme_concept': '主题概念'
    }
    
    def __init__(self, data_dir: str = './data/blocks'):
        """
        初始化管理器
        
        Parameters:
        -----------
        data_dir : str
            数据目录
        """
        self.logger = logging.getLogger("BlockManager")
        self.data_dir = Path(data_dir)
        
        # 检查数据目录
        if not self.data_dir.exists():
            raise FileNotFoundError(
                f"数据目录不存在: {self.data_dir}\n"
                "请先运行 BlockDownloader 下载数据"
            )
        
        self.blocks = {}
        self.excluded_block_keywords = ['加权']
        self.metadata = self._load_metadata()
        self._load_all_blocks()
    
    def _load_metadata(self) -> dict:
        """加载元数据"""
        meta_file = self.data_dir / 'metadata.json'
        if meta_file.exists():
            with open(meta_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}
    
    def _load_all_blocks(self):
        """加载所有板块数据到内存"""
        print("\n" + "="*70)
        print("加载板块数据")
        print("="*70)
        
        if self.metadata:
            print(f"\n数据下载时间: {self.metadata.get('download_time', '未知')}")
        
        for block_type in self.BLOCK_TYPES.keys():
            file_path = self.data_dir / f"{block_type}.parquet"
            
            if file_path.exists():
                df = pd.read_parquet(file_path)
                self.blocks[block_type] = df
                '''   
                print(f"\n{self.BLOCK_TYPES[block_type]}:")
                print(f"  ✓ 文件: {file_path.name}")
                print(f"  板块数: {df['block_name'].nunique()}")
                print(f"  记录数: {len(df)}")
                '''
            else:
                self.logger.warning(f"{self.BLOCK_TYPES[block_type]} 数据不存在")
                self.blocks[block_type] = pd.DataFrame()
        
        print("="*70 + "\n")
    
    # ==================== 统一的查询接口 ====================
    
    def get_block_types(self) -> List[str]:
        """
        获取所有可用的板块类型
        
        Returns:
        --------
        list : 板块类型列表
        """
        return [bt for bt in self.BLOCK_TYPES.keys() if not self.blocks[bt].empty]
    
    def get_block_list(self, block_type: str) -> List[str]:
        """
        获取指定类型的所有板块列表
        
        Parameters:
        -----------
        block_type : str
            板块类型
        
        Returns:
        --------
        list : 板块名称列表（已排序）
        
        Examples:
        ---------
        >>> manager = BlockManager()
        >>> industries = manager.get_block_list('sw1_industry')
        >>> print(industries[:5])
        """
        if block_type not in self.blocks or self.blocks[block_type].empty:
            return []
        
        names = self.blocks[block_type]['block_name'].unique().tolist()
        names = [n for n in names if not self._is_excluded_block(n)]
        return sorted(names)
    
    def get_stocks_in_block(self, 
                           block_name: str, 
                           block_type: str) -> List[str]:
        """
        获取指定板块中的股票列表
        
        Parameters:
        -----------
        block_name : str
            板块名称
        block_type : str
            板块类型
        
        Returns:
        --------
        list : 股票代码列表
        """
        if block_type not in self.blocks or self.blocks[block_type].empty:
            return []
        
        df = self.blocks[block_type]
        stocks = df[df['block_name'] == block_name]['stock_code'].tolist()
        return sorted(stocks)
    
    def get_blocks_by_stock(self, 
                           stock_code: str,
                           block_type: Optional[str] = None) -> Dict[str, List[str]]:
        """
        获取股票所属的板块
        
        Parameters:
        -----------
        stock_code : str
            股票代码
        block_type : str, optional
            板块类型。如果为None，返回所有类型的板块
        
        Returns:
        --------
        dict : {block_type: [block_names]}
        """
        result = {}
        
        # 标准化股票代码
        stock_code = self._normalize_stock_code(stock_code)
        
        # 确定要查询的板块类型
        if block_type:
            block_types = [block_type] if block_type in self.blocks else []
        else:
            block_types = self.blocks.keys()
        
        # 查询所有板块
        for bt in block_types:
            if self.blocks[bt].empty:
                result[bt] = []
                continue
            
            df = self.blocks[bt]
            blocks = df[df['stock_code'] == stock_code]['block_name'].unique().tolist()
            blocks = [b for b in blocks if not self._is_excluded_block(b)]
            result[bt] = sorted(blocks)
        
        return result
    
    def find_common_blocks(self,
                          stock_codes: List[str],
                          block_type: str,
                          top_n: int = 10,
                          method: str = 'weight') -> List[tuple]:
        """
        找出股票列表的共同板块
        
        Parameters:
        -----------
        stock_codes : list
            股票代码列表
        block_type : str
            板块类型
        top_n : int
            返回前N个板块
        method : str
            统计方法：
            - 'count': 按出现次数排序
            - 'weight': 按权重排序（出现次数 / 板块总股票数）
        
        Returns:
        --------
        list : [(板块名, 统计值), ...]
        """
        if block_type not in self.blocks or self.blocks[block_type].empty:
            return []
        
        # 标准化股票代码
        stock_codes = [self._normalize_stock_code(code) for code in stock_codes]
        
        # 统计每个板块的出现次数
        block_counter = Counter()
        for code in stock_codes:
            blocks_dict = self.get_blocks_by_stock(code, block_type)
            if block_type in blocks_dict:
                block_counter.update(blocks_dict[block_type])
        
        if method == 'count':
            return block_counter.most_common(top_n)
        
        elif method == 'weight':
            block_weights = {}
            for block_name, count in block_counter.items():
                stocks = self.get_stocks_in_block(block_name, block_type)
                total = len(stocks)
                weight = count / total if total > 0 else 0
                block_weights[block_name] = weight
            
            sorted_blocks = sorted(
                block_weights.items(),
                key=lambda x: x[1],
                reverse=True
            )
            return sorted_blocks[:top_n]
        
        else:
            raise ValueError(f"不支持的统计方法: {method}")
    
    def get_block_statistics(self, block_type: str) -> pd.DataFrame:
        """
        获取板块统计信息
        
        Parameters:
        -----------
        block_type : str
            板块类型
        
        Returns:
        --------
        pd.DataFrame : 包含 ['block_name', 'stock_count', 'sector_code'] 列
        """
        if block_type not in self.blocks or self.blocks[block_type].empty:
            return pd.DataFrame()
        
        df = self.blocks[block_type]
        df = df[~df['block_name'].apply(self._is_excluded_block)]
        stats = df.groupby(['block_name', 'sector_code']).size().reset_index(name='stock_count')
        stats = stats.sort_values('stock_count', ascending=False).reset_index(drop=True)
        
        return stats
    
    def _is_excluded_block(self, block_name: str) -> bool:
        name = str(block_name)
        for kw in self.excluded_block_keywords:
            if kw and kw in name:
                return True
        return False
    
    def _normalize_stock_code(self, code: str) -> str:
        """
        标准化股票代码格式
        
        Parameters:
        -----------
        code : str
            股票代码（可能是 '000001' 或 '000001.SZ'）
        
        Returns:
        --------
        str : 标准化后的代码（'000001.SZ' 格式）
        """
        code = str(code).zfill(6)
        if '.' in code:
            return code
        
        # 补充交易所后缀
        code = str(code).zfill(6)
        if code.startswith('6'):
            return f"{code}.SH"
        else:
            return f"{code}.SZ"
    
    def print_summary(self):
        """打印数据摘要"""
        print("\n" + "="*70)
        print("板块数据摘要")
        print("="*70)
        
        if self.metadata:
            print(f"\n数据下载时间: {self.metadata.get('download_time', '未知')}")
        
        for block_type in self.BLOCK_TYPES.keys():
            if block_type not in self.blocks or self.blocks[block_type].empty:
                continue
            
            df = self.blocks[block_type]
            print(f"\n{self.BLOCK_TYPES[block_type]}:")
            print(f"  板块数量: {df['block_name'].nunique()}")
            print(f"  总记录数: {len(df)}")
            
            # 股票数分布
            stats = df.groupby('block_name').size()
            print(f"  平均每板块: {stats.mean():.1f} 只股票")
            print(f"  最大板块: {stats.max()} 只")
            print(f"  最小板块: {stats.min()} 只")
        
        print("="*70 + "\n")


if __name__ == "__main__":
    # 测试读取
    manager = BlockManager(data_dir='./data/blocks')
    
    # 打印摘要
    manager.print_summary()
    
    
    # 测试查询
    print("\n" + "="*70)
    print("测试查询功能")
    print("="*70)
    
    # 1. 获取申万一级行业列表
    print("\n1. 申万一级行业:")
    sw1_list = manager.get_block_list('sw1_industry')
    print(f"   共 {len(sw1_list)} 个")
    print(f"   前5个: {sw1_list[:5]}")
    
    # 2. 查询银行行业的股票
    if '银行' in sw1_list:
        print("\n2. 银行行业的股票:")
        banks = manager.get_stocks_in_block('银行', 'sw1_industry')
        print(f"   共 {len(banks)} 只")
        print(f"   前5只: {banks[:5]}")
    
    # 3. 查询股票所属板块
    print("\n3. 查询 000001.SZ 所属板块:")
    blocks = manager.get_blocks_by_stock('000001.SZ')
    for bt, names in blocks.items():
        if names:
            print(f"   {manager.BLOCK_TYPES[bt]}: {names}")
    
    # 4. 找共同概念
    print("\n4. 找共同概念:")
    common = manager.find_common_blocks(
        ['300051', '301171', '300418','300442','300578'],
        block_type='sw3_industry',
        top_n=5,
        method='weight'
    )
    for block, weight in common:
        print(f"   {block}: {weight:.2%}")
    
    # 5. 板块统计
    print("\n5. 申万一级行业统计:")
    stats = manager.get_block_statistics('sw1_industry')
    print(stats.head(20).to_string(index=False))
