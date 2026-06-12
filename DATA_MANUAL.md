# 股票数据系统说明书

本说明书基于当前项目代码与**实测数据验证**整理，描述：

- 数据来源与目录结构
- 日线 / 分钟数据文件格式（Parquet）
- `StockDataManager` 的主要功能
- 字段对照表（字段名、类型与含义）
- 已知数据问题与注意事项

---

## 1. 数据来源与目录

- 通达信原始数据目录由 `TDX_PATH` 环境变量控制，默认：
  - 日线：`C://new_tdx//vipdoc//sz//lday`，`C://new_tdx//vipdoc//sh//lday`
  - 分钟：`C://new_tdx//vipdoc//sz//minline`，`C://new_tdx//vipdoc//sh//minline`
- 项目数据基准目录：`data`（`config.py:8`）
- 生成的核心文件：
  - 日线数据：`data/daydata.parquet`（由 `daypreprocess.py` 生成）
  - 分钟数据：`data/mindata.parquet`
  - 股票列表：`data/stocks.parquet`
  - 代码表：`data/codes.csv`

---

## 2. 已知数据问题（重要）

> 以下问题已通过实测验证，使用数据前请注意。

### Bug 1：创业板涨停判断错误（影响 flag / limit_price）

**问题**：`daypreprocess.py:147` 中判断主板的逻辑为：

```python
is_main_board = code.startswith(('0', '6', '3'))  # ← 错误！
```

300xxx（创业板）被误判为主板，导致涨停线按 **10%**（而非正确的 **20%**）计算。

**结论**：`flag==1` 对创业板股票意味着「当日涨幅达到 +10%」，而非真实的 +20% 涨停板。

**影响范围**：
- 300xxx 股票的 `flag`、`limit_price` 均受影响
- 使用 `flag==1` 识别"首板"的策略，对创业板等同于"第一根 +10% 大阳线"，并非真正涨停

**修复建议**：重新生成 parquet 时将判断逻辑改为：
```python
is_main_board = code.startswith(('00', '60'))  # 主板：0/6 开头的 6 位码
```

---

## 3. 字段命名约定

### 下划线前缀 = 前瞻数据（未来函数）

以 `_` 开头的字段包含**未来信息**，仅用于回测收益计算，**绝不能用于选股条件**：
- `_O2O` — 今日开盘 → 明日开盘收益
- `_O2C` — 今日开盘 → 明日收盘收益

不带下划线的字段（如 `close_return`）均为**当日已知数据**，可安全用于信号条件。

---

## 4. 日线数据文件（daydata.parquet）

总字段数：**17 列**，日期覆盖 2020-01 至今，全市场股票（约 5000 只，代码以 0/3/6 开头）。

### 4.1 基础字段

| 字段名        | 类型    | 含义                                | 备注                                              |
|--------------|---------|-------------------------------------|---------------------------------------------------|
| `code`       | string  | 股票代码（6 位数字）                 | 不含交易所后缀，例如 `600000`                    |
| `day`        | string  | 交易日期，格式 `YYYY-MM-DD`          |                                                   |
| `open`       | float64 | 日开盘价（元）                       |                                                   |
| `high`       | float64 | 日内最高价（元）                     |                                                   |
| `low`        | float64 | 日内最低价（元）                     |                                                   |
| `close`      | float64 | 收盘价（元）                         |                                                   |
| `amount`     | float64 | 成交额（**元**）                     | `amount/volume ≈ close`                           |
| `volume`     | int64   | 成交量（**股**）                     |                                                   |
| `limit_price`| float64 | 当日涨停价（元）                     | = `preclose × 涨停比例`（创业板误用10%，见 Bug 1）|
| `flag`       | int8    | 涨停标志：`1`=涨停，`0`=未涨停      | 创业板判断有误，见 Bug 1                          |
| `preclose`   | float64 | 前一交易日收盘价（元）               | 首条记录为 0                                      |

### 4.2 当日收益率字段（相对于前一日收盘价，单位：%）

> 均为**当日**数据，无未来函数，可安全用于选股信号条件。

| 字段名         | 公式                                     | 含义                                       |
|---------------|------------------------------------------|--------------------------------------------|
| `close_return` | `(close - preclose) / preclose × 100`   | 当日收盘价涨跌幅（= 通常所说的"涨跌幅"）  |
| `open_return`  | `(open - preclose) / preclose × 100`    | 当日开盘价相对昨收的跳空幅度              |
| `high_return`  | `(high - preclose) / preclose × 100`    | 当日最高价相对昨收的涨幅                  |
| `low_return`   | `(low - preclose) / preclose × 100`     | 当日最低价相对昨收的涨幅                  |

### 4.3 前瞻收益字段（下划线前缀，未来数据！）

> 异常值过滤：`|return| > 31%` 的值已设为 `NaN`（排除除权除息等导致的价格跳变）。

| 字段名  | 公式                                      | 含义                           |
|--------|-------------------------------------------|--------------------------------|
| `_O2O` | `(明日开盘 - 今日开盘) / 今日开盘 × 100`  | 开盘到明日开盘收益（隔日超短用）|
| `_O2C` | `(明日收盘 - 今日开盘) / 今日开盘 × 100`  | 开盘到明日收盘收益              |

**回测收益对齐**：
```
信号日T选股 → T+1开盘买 → T+2开盘卖
= T+1行的 _O2O
= 在T行需要 shift(-1) 才能拿到正确收益
```

---

## 5. 分钟数据文件（mindata.parquet）

### 5.1 字段对照表

| 字段名   | 类型    | 含义                         | 备注               |
|---------|---------|------------------------------|--------------------|
| `code`  | string  | 股票代码（6 位数字）          | 不含后缀           |
| `day`   | string  | 交易日期 `YYYY-MM-DD`        |                    |
| `time`  | string  | 交易时间 `HH:MM`             | 例如 `09:35`       |
| `open`  | float32 | 分钟 K 线开盘价（元）         |                    |
| `high`  | float32 | 分钟内最高价（元）            |                    |
| `low`   | float32 | 分钟内最低价（元）            |                    |
| `close` | float32 | 分钟 K 线收盘价（元）         |                    |
| `amount`| float32 | 分钟内成交额（**元**）        | 与日线 amount 含义一致 |
| `volume`| int32   | 分钟内成交量（**股**）        |                    |

---

## 6. 行业与板块数据

通过 `BlockManager` 管理（`BK.py`），数据目录为 `data/blocks`。

支持的板块类型：

| 板块类型        | 说明             |
|----------------|-----------------|
| `sw1_industry` | 申万一级行业      |
| `sw2_industry` | 申万二级行业      |
| `csrc1_industry`| 证监会一级行业   |
| `csrc2_industry`| 证监会二级行业   |
| `concept`      | 概念板块          |
| `theme_concept`| 主题板块          |

板块数据 DataFrame 结构（`bm.blocks[block_type]`）：

| 字段名       | 含义                              |
|-------------|-----------------------------------|
| `block_type`| 板块类型（如 `sw1_industry`）     |
| `block_name`| 板块名称（如 `基础化工`）          |
| `sector_code`| 行业编码                         |
| `stock_code`| 股票代码（含交易所后缀，如 `000001.SZ`）|

> `stock_code` 包含 `.SZ`/`.SH` 后缀，与 `daydata.parquet` 的 `code` 字段对接时需剥离后缀并补零：
> ```python
> sw1_df['code'] = sw1_df['stock_code'].str.split('.').str[0].str.zfill(6)
> ```

---

## 7. `StockDataManager` 使用说明

核心类定义在 `data_core.py`，封装数据读取与板块查询。

### 7.1 加载日线数据

```python
from data_core import StockDataManager

dm = StockDataManager()

df = dm.load_day_data(
    start_date='2024-01-01',
    end_date='2024-12-31',
    codes=['000001', '600000'],          # 可省略：全市场
    columns=['code', 'day', 'close', 'amount', 'flag']  # 可省略：全列
)
```

### 7.2 板块查询

```python
# 获取板块成份股（返回 6 位不带后缀的代码列表）
stocks = dm.get_stocks_in_block('基础化工', 'sw1_industry')

# 查询股票所属板块
blocks = dm.get_blocks_by_stock('000001')
# {'sw1_industry': ['银行'], 'sw2_industry': ['国有大型银行'], ...}

# 获取所有申万一级行业名称
sw1_list = dm.get_block_list('sw1_industry')
```

### 7.3 使用前瞻收益字段

```python
# 直接用预计算字段，避免自行 shift 出错
df = dm.load_day_data(
    columns=['code', 'day', 'flag', 'close_return', '_O2O', '_O2C']
)

# 涨停后隔日收益
limit_up = df[df['flag'] == 1]
print(limit_up['_O2O'].mean())  # 开盘到明日开盘
```

---

## 8. 回测注意事项

### 8.1 无未来函数清单（可用于选股信号）

以下字段在信号日 T 的收盘后即可获得，可安全用于 T 日信号、T+1 开盘执行：

- `close_return`、`open_return`、`high_return`、`low_return`
- `flag`、`limit_price`、`preclose`、`amount`
- `open`、`close`、`high`、`low`（当日价格）

### 8.2 前瞻字段（仅用于收益计算，不可用于选股）

- `_O2O`、`_O2C` — 包含明日价格信息

### 8.3 自行计算 next_open 的异常值过滤

若在回测中自行计算持仓收益：

```python
df['next_open'] = df.groupby('code')['open'].shift(-1)
df['trade_return'] = df['next_open'] / df['open'] - 1

# 必须过滤价格跳变（除权等），单日持仓收益 ±50% 以上视为异常
df.loc[df['trade_return'].abs() > 0.5, 'trade_return'] = np.nan
```

### 8.4 创业板策略注意事项

如策略依赖 `flag==1`（涨停板）选股：
- **主板**（000xxx / 600xxx）：`flag==1` 正确，表示真实 10% 涨停
- **创业板**（300xxx）：`flag==1` 实为 +10% 大阳线，**不是真实 20% 涨停板**
- **科创板**（688xxx）：未在当前股票代码范围内（`ALLOWED_CODE_PREFIXES = ('0', '3', '6')`）

### 8.5 `amount` 换算

日线 `amount` 为当日总成交额，单位为**元**：
- 成交额（亿元）= `amount / 1e8`
- 换手率需结合流通市值计算（流通市值数据未存入 parquet）
- 用 `amount / rolling_mean_amount` 作为活跃度替代指标
