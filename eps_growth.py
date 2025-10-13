import pandas as pd
import numpy as np

def eps_growth_signal(
    returns: pd.DataFrame,
    eps_est: pd.DataFrame,                 # 預估 EPS（月頻）
    mktcap_pool: dict[pd.Period, set],     # 來自 build_sample_pool（key=Period('YYYY-MM','M')）
    increase_strict: bool = True,          # True: EPS[t] >  EPS[t-1]；False: EPS[t] >= EPS[t-1]
    require_positive: bool = False,        # True: 僅在 EPS[t], EPS[t-1] 皆 > 0 時才納入
) -> pd.DataFrame:
    """
    規則：比較 t 與 t-1 月的預估 EPS，若有成長，則在 t+1 月把該股票納入持有。
    回傳：與 returns 同 shape 的 0/1 訊號（int8）
    """
    # ---- 基礎清洗 ----
    r = returns.sort_index().copy()
    assert isinstance(r.index, pd.DatetimeIndex), "returns.index 必須是 DatetimeIndex（日頻）"
    r.columns = r.columns.astype(str).str.strip()

    eps = eps_est.copy()
    eps.columns = eps.columns.astype(str).str.strip()
    if not isinstance(eps.index, pd.PeriodIndex):
        eps.index = pd.to_datetime(eps.index).to_period("M")

    signal = pd.DataFrame(0, index=r.index, columns=r.columns, dtype="int8")
    month_key = r.index.to_period("M")

    # ---- 主迴圈（逐月持有）----
    for m in month_key.unique():
        # 要決定「本月 m 的持有」，需用 (m-1) 與 (m-2) 的 EPS 來判斷
        t     = m -1  # 當作「觀察月」
        t_1   = m - 2   # 當作「前一月」

        # 宇宙採用 pool[m]（對齊「下月持有 = 由上月市值決定的下月池」的邏輯）
        universe = pd.Index(sorted(mktcap_pool.get(m, set()))).astype(str).str.strip()
        universe = r.columns.intersection(universe)

        if universe.empty or (t not in eps.index) or (t_1 not in eps.index):
            continue

        e_t   = pd.to_numeric(eps.loc[t,   universe], errors="coerce")
        e_t1  = pd.to_numeric(eps.loc[t_1, universe], errors="coerce")

        # 僅保留同時非空的橫切面
        valid = (~e_t.isna()) & (~e_t1.isna())
        if not valid.any():
            continue

        e_t  = e_t[valid]
        e_t1 = e_t1[valid]

        # （可選）要求兩期 EPS 皆為正
        if require_positive:
            pos = (e_t > 0) & (e_t1 > 0)
            if not pos.any():
                continue
            e_t  = e_t[pos]
            e_t1 = e_t1[pos]

        # 成長條件
        if increase_strict:
            picks = (e_t >  e_t1)
        else:
            picks = (e_t >= e_t1)

        picks = e_t.index[picks]
        if len(picks) == 0:
            continue

        # 在「本月 m 的所有交易日」標 1
        hold_mask = (month_key == m)
        signal.loc[hold_mask, picks] = 1

    signal.index.name = r.index.name
    signal.columns.name = r.columns.name
    return signal
