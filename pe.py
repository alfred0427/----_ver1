import pandas as pd
import numpy as np

def build_sample_pool(mktcap: pd.DataFrame, top_n: int = 200) -> dict[pd.Period, set]:
    """
    以「當月市值」決定「下個月」的可投資池（Top-N）。
    mktcap: 月頻，index 可為每月任意日（建議月底），columns=股票代碼
    回傳：{Period('YYYY-MM','M') -> set(TopN tickers)}
    """
    # 1) 統一欄名為字串、去空白
    mktcap = mktcap.copy()
    mktcap.columns = mktcap.columns.astype(str).str.strip()

    # 2) 確保索引是月 PeriodIndex
    if not isinstance(mktcap.index, pd.PeriodIndex):
        mktcap.index = pd.to_datetime(mktcap.index).to_period("M")

    pool: dict[pd.Period, set] = {}
    for ym, row in mktcap.iterrows():
        nxt = ym + 1  # 當月市值 -> 下月可投資池
        top_stocks = row.dropna().nlargest(top_n)
        pool[nxt] = set(top_stocks.index)
    return pool


def pe_low_signal(
    returns: pd.DataFrame,
    pe_ratio: pd.DataFrame,
    mktcap_pool: dict[pd.Period, set],
    bottom_frac: float = 0.30,
    require_positive: bool = True,
) -> pd.DataFrame:
    """
    以「上個月 PE」在 TopN 宇宙中挑選最低本益比的 bottom_frac 標的，整個「本月」持有。
    returns : 日頻，index=交易日(DatetimeIndex)，columns=股票代碼
    pe_ratio: 月頻，index=月(Period/Timestamp皆可)、columns=股票代碼，值=PE
    mktcap_pool : {Period('YYYY-MM','M') -> set(tickers)}，通常來自 build_sample_pool
    回傳：0/1 訊號（int8）
    """
    # ---- 基礎清洗與對齊 ----
    r = returns.sort_index()
    assert isinstance(r.index, pd.DatetimeIndex), "returns.index 必須是 DatetimeIndex（日頻）"
    r_cols = r.columns.astype(str).str.strip()

    pe = pe_ratio.copy()
    pe.columns = pe.columns.astype(str).str.strip()
    if not isinstance(pe.index, pd.PeriodIndex):
        pe.index = pd.to_datetime(pe.index).to_period("M")

    # 把 returns 欄名也標準化成字串
    r = r.copy()
    r.columns = r_cols

    # 建 0/1 訊號容器（省記憶體用 int8）
    signal = pd.DataFrame(0, index=r.index, columns=r.columns, dtype="int8")

    # 以月份分組持有（本月持有 = 上月PE 的結果）
    month_key = r.index.to_period("M")
    unique_months = month_key.unique()

    # ---- 主迴圈（逐月）----
    for m in unique_months:
        prev_m = m - 1  # 依規則，上月為決策月

        # 宇宙：上月的 TopN；與 returns 欄交集以避免 KeyError
        universe = pd.Index(sorted(mktcap_pool.get(prev_m, set()))).astype(str).str.strip()
        universe = r.columns.intersection(universe)
        if universe.empty:
            continue

        # 上月 PE 的橫切面（只取宇宙的欄）
        if prev_m not in pe.index:
            continue
        pe_prev = pd.to_numeric(pe.loc[prev_m, universe], errors="coerce").dropna()

        if require_positive:
            pe_prev = pe_prev[pe_prev > 0]

        if pe_prev.empty:
            continue

        # 取「最低 bottom_frac」的標的
        k = max(1, int(np.ceil(len(pe_prev) * bottom_frac)))
        picks = pe_prev.nsmallest(k).index  # 本月要持有的標的

        # 把這些標的在「本月所有交易日」標 1
        hold_mask = (month_key == m)
        if hold_mask.any():
            signal.loc[hold_mask, picks] = 1

    signal.index.name = r.index.name
    signal.columns.name = r.columns.name
    return signal
