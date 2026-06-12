# -*- coding: utf-8 -*-
"""
板块数据下载模块 - 从 xtquant 下载并保存到本地
"""
from pathlib import Path
import logging
import pandas as pd
import time
from datetime import datetime

try:
    from xtquant import xtdata
    from xtquant import xtdatacenter as xtdc
    XTQUANT_AVAILABLE = True
except ImportError:
    XTQUANT_AVAILABLE = False
    logging.warning("xtquant 未安装")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


class BlockDownloader:
    """板块数据下载器"""
    
    # 板块前缀映射（删除了CSRC3）
    PREFIX_MAP = {
        'SW1': 'sw1_industry',      # 申万一级行业
        'SW2': 'sw2_industry',      # 申万二级行业
        'SW3': 'sw3_industry',      # 申万三级行业
        'CSRC1': 'csrc1_industry',  # 证监会一级行业
        'CSRC2': 'csrc2_industry',  # 证监会二级行业
        # 'CSRC3': 'csrc3_industry',  # 删除：证监会三级行业
        'TGN': 'theme_concept',     # 主题概念
        'GN': 'concept'             # 概念
    }
    
    # 板块类型定义（删除了csrc3_industry）
    BLOCK_TYPES = {
        'sw1_industry': '申万一级行业',
        'sw2_industry': '申万二级行业',
        'sw3_industry': '申万三级行业',
        'csrc1_industry': '证监会一级行业',
        'csrc2_industry': '证监会二级行业',
        # 删除：'csrc3_industry': '证监会三级行业',
        'concept': '概念',
        'theme_concept': '主题概念'
    }
    
    def __init__(self, token: str = None, data_dir: str = './data/blocks'):
        """
        初始化下载器
        
        Parameters:
        -----------
        token : str, optional
            xtquant token
        data_dir : str
            数据保存目录
        """
        if not XTQUANT_AVAILABLE:
            raise RuntimeError("xtquant 未安装，请安装: pip install xtquant")
        
        self.logger = logging.getLogger("BlockDownloader")
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # 初始化 xtquant
        if token:
            xtdc.set_token(token)
        
        try:
            xtdc.init()
            self.logger.info("✓ xtquant 初始化成功")
        except Exception as e:
            self.logger.error(f"✗ xtquant 初始化失败: {e}")
            raise
    
    def download_all_blocks(self, force: bool = False):
        """
        下载所有板块数据
        
        Parameters:
        -----------
        force : bool
            是否强制重新下载（即使本地已有数据）
        """
        print("\n" + "="*70)
        print("板块数据下载任务")
        print("="*70)
        
        # 检查是否需要更新
        if not force and self._check_data_exists():
            print("\n本地已有板块数据")
            user_input = input("是否重新下载？(y/n): ").strip().lower()
            if user_input != 'y':
                print("取消下载")
                return
        
        try:
            # 1. 下载板块数据
            print("\n[1/3] 下载最新板块数据...")
            xtdata.download_sector_data()
            time.sleep(2)
            print("✓ 板块数据下载完成")
            
            # 2. 获取所有板块列表
            print("\n[2/3] 获取板块列表...")
            all_sectors = xtdata.get_sector_list()
            print(f"✓ 获取到 {len(all_sectors)} 个板块")
            
            # 3. 下载每个板块的成分股
            print("\n[3/3] 下载板块成分股...")
            self._download_sector_stocks(all_sectors)
            
            # 保存元数据
            self._save_metadata()
            
            print("\n" + "="*70)
            print("✓ 所有板块数据下载完成")
            print("="*70)
            
        except Exception as e:
            self.logger.error(f"下载失败: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
    
    def _download_sector_stocks(self, all_sectors: list):
        """下载所有板块的成分股"""
        # 按类型分类板块
        classified_sectors = self._classify_sectors(all_sectors)
        
        # 统计信息
        total = sum(len(sectors) for sectors in classified_sectors.values())
        processed = 0
        
        # 逐类型下载
        for block_type, sectors in classified_sectors.items():
            if not sectors:
                continue
            
            print(f"\n  下载 {self.BLOCK_TYPES[block_type]} ({len(sectors)} 个)...")
            
            data = []
            for sector_code, block_name in sectors.items():
                try:
                    # 获取成分股
                    stocks = xtdata.get_stock_list_in_sector(sector_code)
                    
                    if stocks:
                        for stock in stocks:
                            data.append({
                                'block_type': block_type,
                                'block_name': block_name,
                                'sector_code': sector_code,
                                'stock_code': stock
                            })
                    
                    processed += 1
                    if processed % 20 == 0:
                        print(f"    进度: {processed}/{total}")
                
                except Exception as e:
                    self.logger.warning(f"获取 {sector_code} 失败: {e}")
            
            # 保存该类型的数据
            if data:
                df = pd.DataFrame(data)
                file_path = self.data_dir / f"{block_type}.parquet"
                df.to_parquet(file_path, index=False)
                print(f"  ✓ 保存到: {file_path}")
                print(f"    板块数: {df['block_name'].nunique()}")
                print(f"    记录数: {len(df)}")
    
    def _classify_sectors(self, all_sectors: list) -> dict:
        """
        按前缀分类板块
        
        Returns:
        --------
        dict : {block_type: {sector_code: block_name}}
        """
        classified = {bt: {} for bt in self.BLOCK_TYPES.keys()}
        
        for sector in all_sectors:
            # 提取前缀
            prefix = None
            for p in self.PREFIX_MAP.keys():
                if sector.startswith(p):
                    prefix = p
                    break
            
            if prefix:
                block_type = self.PREFIX_MAP[prefix]
                # 提取板块名称（去掉前缀）
                block_name = sector[len(prefix):].strip()
                classified[block_type][sector] = block_name
        
        return classified
    
    def _check_data_exists(self) -> bool:
        """检查本地是否已有数据"""
        for block_type in self.BLOCK_TYPES.keys():
            file_path = self.data_dir / f"{block_type}.parquet"
            if file_path.exists():
                return True
        return False
    
    def _save_metadata(self):
        """保存元数据（下载时间等）"""
        metadata = {
            'download_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'block_types': list(self.BLOCK_TYPES.keys())
        }
        
        # 统计每个类型的数据
        for block_type in self.BLOCK_TYPES.keys():
            file_path = self.data_dir / f"{block_type}.parquet"
            if file_path.exists():
                df = pd.read_parquet(file_path)
                metadata[block_type] = {
                    'blocks': int(df['block_name'].nunique()),
                    'records': len(df),
                    'file_size_mb': round(file_path.stat().st_size / 1024 / 1024, 2)
                }
        
        # 保存为JSON
        import json
        meta_file = self.data_dir / 'metadata.json'
        with open(meta_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        print(f"\n元数据已保存: {meta_file}")


if __name__ == "__main__":
    # 使用示例
    TOKEN = "c610ead6e2ae79d26e697a2ccedc995d1e014f74"
    
    downloader = BlockDownloader(token=TOKEN, data_dir='./data/blocks')
    downloader.download_all_blocks(force=False)
