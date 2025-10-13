import pandas as pd
import numpy as np

def build_sample_pool(mktcap: pd.DataFrame, top_n: int = 200) -> dict:
    pool = {}
    for ym, row in mktcap.iterrows():
        # 當月計算出來的市值 -> 用在下個月
        period = pd.Period(ym, freq="M") + 1
        top_stocks = row.dropna().nlargest(top_n).index
        pool[period] = set(top_stocks)
    return pool
def build_sample_pool_ex_fin(mktcap: pd.DataFrame, fin_df: pd.DataFrame, top_n: int = 200) -> dict[pd.Period, set]:
    """
    以「當月市值」決定「下個月」的 Top-N 宇宙（排除金融股）：
    pool[當月 + 1] = 當月TopN (去掉金融股)。
    """
    # 取金融股代碼 set
    financial_stocks = set(fin_df.iloc[:, 0].astype(str).str.strip())

    mc = mktcap.copy()
    mc.columns = mc.columns.astype(str).str.strip()
    if not isinstance(mc.index, pd.PeriodIndex):
        mc.index = pd.to_datetime(mc.index).to_period("M")

    pool: dict[pd.Period, set] = {}
    for ym, row in mc.iterrows():
        topn = set(row.dropna().nlargest(top_n).index)
        # 去掉金融股
        filtered = topn - financial_stocks
        pool[ym + 1] = filtered
    return pool


def momentum_signal(returns: pd.DataFrame,
                    mktcap_pool: dict,
                    top_frac: float = 0.30,
                    lookback_months: int = 1) -> pd.DataFrame:
    """
    動能訊號（可調回看月數，預設=1 等於原本的「當月MTD」）：
      1) 以當月 m 的 Top200 宇宙做篩選
      2) 在該宇宙內，用過去 lookback_months 個月份（含 m）的日報酬做幾何累積：∏(1+r)-1
      3) 先取全體中的前 top_frac，再從其中保留 > 0
      4) 配置到下一個月 (m+1) 的所有交易日
    回傳：與 returns 同尺寸的 0/1 DataFrame
    """
    r = returns.sort_index()
    signal = pd.DataFrame(0, index=r.index, columns=r.columns, dtype="int8")
    month_key = r.index.to_period("M")

    for m, _ in r.groupby(month_key):
        # 1) 當月宇宙
        universe = list(r.columns.intersection(mktcap_pool.get(m, set())))
        if not universe:
            continue

        # 2) 回看期（含當月）：m - (L-1) ... m
        months = [(m - i) for i in range(lookback_months - 1, -1, -1)]
        win_mask = month_key.isin(months)
        r_win = r.loc[win_mask, universe]

        # 3) 幾何累積報酬（若整段缺值則為 NaN）
        mom = (1.0 + r_win).prod(min_count=1) - 1.0
        mom = mom.dropna()
        if mom.empty:
            continue

        # 4) 先取前 top_frac，再濾 > 0
        k = max(1, int(np.ceil(len(mom) * top_frac)))
        topk = mom.nlargest(k)
        winners = topk[topk > 0].index
        if len(winners) == 0:
            continue

        # 5) 配置到下一個月
        next_mask = (month_key == (m + 1))
        if next_mask.any():
            signal.loc[next_mask, winners] = 1

    signal.index.name = r.index.name
    signal.columns.name = r.columns.name
    return signal


import pandas as pd

def pool_to_alpha(returns: pd.DataFrame, pool: dict) -> pd.DataFrame:
    """
    把 monthly pool (dict: Period -> set of tickers)
    轉換成日頻 alpha 矩陣 (0/1)，大小與 returns 相同。
    
    - returns: DataFrame, index=日 (DatetimeIndex), columns=股票代號
    - pool: dict, key=Period('YYYY-MM','M'), value=set(股票代號)
    """
    r = returns.sort_index()
    signal = pd.DataFrame(0, index=r.index, columns=r.columns, dtype="int8")

    month_key = r.index.to_period("M")

    for m, r_m in r.groupby(month_key):
        if m not in pool:
            continue

        # 取這個月的樣本池
        sample = list(r_m.columns.intersection(pool[m]))

        # 標記到「下一個月」的所有交易日
        next_mask = (month_key == (m + 1))
        if next_mask.any():
            signal.loc[next_mask, sample] = 1

    return signal
