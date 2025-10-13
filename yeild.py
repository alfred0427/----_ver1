import pandas as pd
import numpy as np

# ------------------------------------------------------------
# 產生 Top-N 市值「下月」投資池（和你原本的一樣，但做了型別/索引統一）
# ------------------------------------------------------------
def build_sample_pool(mktcap: pd.DataFrame, top_n: int = 200) -> dict[pd.Period, set]:
    """
    mktcap: 月頻 DataFrame，index 可為任意日期，columns=股票代碼，值=市值
    回傳: {Period('YYYY-MM','M') -> set(TopN tickers)}，代表「下個月」的投資池
    """
    mc = mktcap.copy()
    mc.columns = mc.columns.astype(str).str.strip()
    if not isinstance(mc.index, pd.PeriodIndex):
        mc.index = pd.to_datetime(mc.index).to_period("M")

    pool: dict[pd.Period, set] = {}
    for ym, row in mc.iterrows():
        pool[ym + 1] = set(row.dropna().nlargest(top_n).index)
    return pool


# ------------------------------------------------------------
# 殖利率高因子：上月 DY 在 Top200 宇宙內取「最高的 top_frac」
# 本月整月持有（訊號 0/1）
# ------------------------------------------------------------
def dy_high_signal(
    returns: pd.DataFrame,
    dy_ratio: pd.DataFrame,
    mktcap_pool: dict[pd.Period, set],
    top_frac: float = 0.30,
    require_positive: bool = True,
) -> pd.DataFrame:
    """
    returns : 日頻 DataFrame，index=交易日(DatetimeIndex)，columns=股票代碼
    dy_ratio: 月頻 DataFrame，index=月(Period/Timestamp 皆可)，columns=股票代碼，值=殖利率
              （通常是「該月月底」對應的殖利率）
    mktcap_pool : {Period('YYYY-MM','M') -> set(Top200 tickers)}，來自 build_sample_pool
    top_frac : 取殖利率最高前 x%
    require_positive : 是否只保留 DY > 0（多數情況建議 True）

    回傳：與 returns 同 shape 的 0/1 訊號（int8）
    """
    # 基礎清洗
    r = returns.sort_index().copy()
    assert isinstance(r.index, pd.DatetimeIndex), "returns.index 需為 DatetimeIndex（日頻）"
    r.columns = r.columns.astype(str).str.strip()

    dy = dy_ratio.copy()
    dy.columns = dy.columns.astype(str).str.strip()
    if not isinstance(dy.index, pd.PeriodIndex):
        dy.index = pd.to_datetime(dy.index).to_period("M")

    signal = pd.DataFrame(0, index=r.index, columns=r.columns, dtype="int8")

    # 以月份分組：本月持有 = 由「上月」DY 決定
    month_key = r.index.to_period("M")

    for m in month_key.unique():
        prev_m = m - 1  # 決策月
        # 上月的 Top200 宇宙，和 returns 欄位取交集避免 KeyError
        universe = pd.Index(sorted(mktcap_pool.get(prev_m, set()))).astype(str).str.strip()
        universe = r.columns.intersection(universe)
        if universe.empty or (prev_m not in dy.index):
            continue

        # 取上月 DY 橫切面（只取宇宙），轉數字、剔除 NA
        dy_prev = pd.to_numeric(dy.loc[prev_m, universe], errors="coerce").dropna()
        if require_positive:
            dy_prev = dy_prev[dy_prev > 0]

        if dy_prev.empty:
            continue

        # 取殖利率「最高」的前 top_frac
        k = max(1, int(np.ceil(len(dy_prev) * top_frac)))
        picks = dy_prev.nlargest(k).index  # 注意：和 PE 取最小不同，這裡取最大

        # 本月所有交易日標 1
        mask = (month_key == m)
        if mask.any():
            signal.loc[mask, picks] = 1

    signal.index.name = r.index.name
    signal.columns.name = r.columns.name
    return signal




import pandas as pd
import numpy as np


def yoy_high_signal(
    returns: pd.DataFrame,
    yoy_ratio: pd.DataFrame,
    mktcap_pool: dict[pd.Period, set],
    top_frac: float = 0.30,
    yoy_cap_ratio: float = 200,     # 你的 YoY 是百分比口徑
    yoy_is_percent: bool = True,    # ← 你的數據是百分比（如 248.84）
    require_positive: bool = False, # 依你條件：不強制 >0
) -> pd.DataFrame:
    r = returns.sort_index().copy()
    r.columns = r.columns.astype(str).str.strip()
    assert isinstance(r.index, pd.DatetimeIndex)

    yoy = yoy_ratio.copy()
    yoy.columns = yoy.columns.astype(str).str.strip()
    if not isinstance(yoy.index, pd.PeriodIndex):
        yoy.index = pd.to_datetime(yoy.index).to_period("M")

    signal = pd.DataFrame(0, index=r.index, columns=r.columns, dtype="int8")
    month_key = r.index.to_period("M")

    for m in month_key.unique():
        prev_m = m - 2

        # --- 這一行是關鍵修正：本月 m 的宇宙該用 pool[m] ---
        universe = pd.Index(sorted(mktcap_pool.get(m, set()))).astype(str).str.strip()  # ← 修正
        universe = r.columns.intersection(universe)
        if universe.empty or (prev_m not in yoy.index):
            continue

        yoy_prev = pd.to_numeric(yoy.loc[prev_m, universe], errors="coerce")
        yoy_prev = yoy_prev.replace([np.inf, -np.inf], np.nan).dropna()

        # 百分比→比率（若 yoy_is_percent=True）
        cap = yoy_cap_ratio
        if yoy_is_percent:
            yoy_prev = yoy_prev / 100.0
            cap = cap / 100.0

        if require_positive:
            yoy_prev = yoy_prev[yoy_prev > 0]
        yoy_prev = yoy_prev[yoy_prev <= cap]

        if yoy_prev.empty:
            continue

        k = max(1, int(np.ceil(len(yoy_prev) * top_frac)))
        picks = yoy_prev.nlargest(k).index

        signal.loc[month_key == m, picks] = 1

    signal.index.name = r.index.name
    signal.columns.name = r.columns.name
    return signal
