# alpha_sector_smart.py
import pandas as pd
import numpy as np
from typing import Callable, Dict

# 你原本的 alpha 函式不用改，只在這裡 import 來用
# from alpha import momentum_signal, pe_low_signal, ...

# ------------------------------------------------
# 1) 建三大產業的 mapping（跟你前面說的一樣）
# ------------------------------------------------
def build_sector_map_three_groups(industry_df: pd.DataFrame) -> pd.Series:
    """
    industry_df: 欄位包含 'code', 'sector'
    回傳：index=股票代碼(str), value=三大產業：電子 / 金融 / 傳產
    """
    df = industry_df.copy()
    df["code"] = df["code"].astype(str).str.strip()
    df["sector"] = df["sector"].astype(str).str.strip()

    def map_three(sector_name: str) -> str:
        s = str(sector_name)
        if "電子" in s:
            return "電子"
        if "金融" in s:
            return "金融"
        return "傳產"

    df["sector3"] = df["sector"].apply(map_three)
    sector_map = df.set_index("code")["sector3"]
    return sector_map


# ------------------------------------------------
# 2) 一些小工具
# ------------------------------------------------
def _standardize_returns(returns: pd.DataFrame) -> pd.DataFrame:
    r = returns.copy()
    r.columns = r.columns.astype(str).str.strip()
    r = r.sort_index()
    if not isinstance(r.index, pd.DatetimeIndex):
        raise ValueError("returns.index 必須是 DatetimeIndex（日頻）")
    return r


def _restrict_pool_to_sector(
    mktcap_pool: Dict[pd.Period, set],
    sector_map: pd.Series,
    sector: str,
) -> Dict[pd.Period, set]:
    """
    把原本的 mktcap_pool（Period -> set(all topN)）
    限縮成「某個 sector」的 pool。
    """
    sm = sector_map.copy()
    sm.index = sm.index.astype(str).str.strip()

    pool_s: Dict[pd.Period, set] = {}
    for ym, codes in mktcap_pool.items():
        codes = {str(c).strip() for c in codes}
        sub = [c for c in codes if sm.get(c, None) == sector]
        pool_s[ym] = set(sub)
    return pool_s


# ------------------------------------------------
# 3) 聰明的工廠：自動產生「在產業內跑的 alpha 函式」
# ------------------------------------------------
def sectorize_alpha_fn(
    factor_fn: Callable,
    *,
    sector_map: pd.Series,
    pool_arg_name: str = "mktcap_pool",
) -> Callable:
    """
    回傳一個新的函式 g(...)，功能是：

      - 先按照 sector_map 把股票分產業
      - 對每個產業：
          * 限縮 returns & 其他 DataFrame 到該產業的欄位
          * 限縮 mktcap_pool 到該產業
          * 呼叫原本的 factor_fn(...)
      - 再把各產業的 0/1 矩陣貼回完整的股票集合

    使用方式：
        momentum_sector = sectorize_alpha_fn(momentum_signal, sector_map=sector_map)
        mom_alpha = momentum_sector(returns=returns, mktcap_pool=mktcap_pool, top_frac=0.3)
    """

    # 先標準化 sector_map
    sm = sector_map.copy()
    sm.index = sm.index.astype(str).str.strip()

    def wrapper(*, returns: pd.DataFrame, **kwargs) -> pd.DataFrame:
        # 1) 標準化 returns
        r = _standardize_returns(returns)
        cols = r.columns

        # 只保留在 returns 中出現的股票
        sm_local = sm.reindex(cols)

        # 產出一個總的 alpha 矩陣
        signal_all = pd.DataFrame(0, index=r.index, columns=r.columns, dtype="int8")

        sectors = sorted(sm_local.dropna().unique())
        for sec in sectors:
            cols_sec = cols[sm_local == sec]
            if len(cols_sec) == 0:
                continue

            r_sec = r[cols_sec]

            # 2) 把 kwargs 裡的 mktcap_pool & DataFrame argument 都切成該 sector
            kwargs_sec = {}

            for k, v in kwargs.items():
                if k == pool_arg_name:
                    # 限縮 pool
                    pool_sec = _restrict_pool_to_sector(v, sm_local, sec)
                    kwargs_sec[k] = pool_sec
                elif isinstance(v, pd.DataFrame):
                    df = v.copy()
                    df.columns = df.columns.astype(str).str.strip()
                    kwargs_sec[k] = df[cols_sec]
                else:
                    kwargs_sec[k] = v

            # 3) 呼叫原本的 factor_fn（重點：完全沒改 alpha.py）
            sig_sec = factor_fn(
                returns=r_sec,
                **kwargs_sec,
            )

            # 4) 把結果貼回總矩陣
            signal_all.loc[:, cols_sec] = sig_sec

        signal_all.index.name = returns.index.name
        signal_all.columns.name = returns.columns.name
        return signal_all

    return wrapper
