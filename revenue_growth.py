import pandas as pd
import numpy as np

# ---------------------------
# 1) 市值 Top-N（下月）投資池
# ---------------------------
def build_sample_pool(mktcap: pd.DataFrame, top_n: int = 200) -> dict[pd.Period, set]:
    """
    以「當月市值」決定「下個月」的 Top-N 宇宙：
    pool[當月 + 1] = 當月TopN。月度對齊、避免前視。
    """
    mc = mktcap.copy()
    mc.columns = mc.columns.astype(str).str.strip()
    if not isinstance(mc.index, pd.PeriodIndex):
        mc.index = pd.to_datetime(mc.index).to_period("M")

    pool: dict[pd.Period, set] = {}
    for ym, row in mc.iterrows():
        pool[ym + 1] = set(row.dropna().nlargest(top_n).index)
    return pool


# ---------------------------
# 2) 將「公告月份」→「所屬季(Q-DEC)」
# ---------------------------
def align_announce_to_quarter(df: pd.DataFrame) -> pd.DataFrame:
    """
    將公告月份對齊到 Q-DEC（會用該季最後一筆公告作為代表值）
    """
    x = df.copy()
    x.columns = x.columns.astype(str).str.strip()

    if isinstance(x.index, pd.PeriodIndex):
        ts = x.index.to_timestamp()
    else:
        ts = pd.to_datetime(x.index)

    labels = []
    for y, m in zip(ts.year, ts.month):
        if   m in (4, 5):   qy, qn = y,   1
        elif m in (7, 8):   qy, qn = y,   2
        elif m in (10, 11): qy, qn = y,   3
        elif m in (1, 2, 3):qy, qn = y-1, 4
        elif m == 6:        qy, qn = y,   2
        elif m == 9:        qy, qn = y,   3
        elif m == 12:       qy, qn = y,   4
        else:
            labels.append(pd.Period(f"{y}-{m:02d}", "M").asfreq("Q-DEC"))
            continue
        labels.append(pd.Period(f"{qy}Q{qn}", "Q-DEC"))

    qidx = pd.PeriodIndex(labels, freq="Q-DEC")
    return x.groupby(qidx).last()


# ---------------------------
# 3) 連兩季成長判斷
# ---------------------------
def two_consecutive_growth(df_q: pd.DataFrame) -> pd.DataFrame:
    """
    在季別 q 上為 True 的條件：
    df[q] > df[q-1] 且 df[q-1] > df[q-2]
    """
    z = df_q.apply(pd.to_numeric, errors="coerce")
    pos = z.diff().gt(0)
    ok2 = (pos & pos.shift(1)).fillna(False)
    return ok2


# ---------------------------
# 4) 季度 → 實際進場月份（公告截止後 → 下個月初持有）
# ---------------------------
def quarter_entry_month(q: pd.Period) -> pd.Period:
    y = int(q.year)
    if q.quarter == 1:   # Q1 公告 5/15，6 月初開始持有
        return pd.Period(f"{y}-06", "M")
    if q.quarter == 2:   # Q2 公告 8/14，9 月初開始持有
        return pd.Period(f"{y}-09", "M")
    if q.quarter == 3:   # Q3 公告 11/14，12 月初開始持有
        return pd.Period(f"{y}-12", "M")
    return pd.Period(f"{y+1}-04", "M")  # Q4 年報 → 次年 4 月初開始持有


# ---------------------------
# 5) 公告月份 → 該月最後一個交易日
# ---------------------------
def month_last_trading_day(month_period: pd.Period, trading_index: pd.DatetimeIndex) -> pd.Timestamp | None:
    mask = trading_index.to_period("M") == month_period
    if not mask.any():
        return None
    return trading_index[mask][-1]


# ---------------------------
# 6) 主函式：利潤率成長（日頻 0/1 訊號）
# ---------------------------
def margin_growth_signal(
    returns: pd.DataFrame,
    gross: pd.DataFrame,
    operating: pd.DataFrame,
    mktcap_pool: dict[pd.Period, set],
    allow_equal: bool = False,
) -> pd.DataFrame:
    # 1) 對齊 returns
    r = returns.sort_index()
    if not isinstance(r.index, pd.DatetimeIndex):
        raise ValueError("returns.index 必須是 DatetimeIndex（日頻）")
    cols = r.columns.astype(str).str.strip()
    r = r.copy()
    r.columns = cols

    # 2) 季化 + 連兩季成長布林表
    gm_q = align_announce_to_quarter(gross).reindex(columns=cols, copy=False)
    om_q = align_announce_to_quarter(operating).reindex(columns=cols, copy=False)

    if allow_equal:
        gm_ok = (gm_q.diff().ge(0) & gm_q.diff().ge(0).shift(1)).fillna(False)
        om_ok = (om_q.diff().ge(0) & om_q.diff().ge(0).shift(1)).fillna(False)
    else:
        gm_ok = two_consecutive_growth(gm_q)
        om_ok = two_consecutive_growth(om_q)

    # 🚨 修正：避免前視 → shift(1)，進場用的是「上季」的判斷結果
    both_ok = (gm_ok & om_ok).shift(1)

    # 3) 找每一季的「實際進場日」
    decision_tbl = []
    for q in both_ok.index:
        entry_m = quarter_entry_month(q)
        entry_dt = month_last_trading_day(entry_m, r.index)
        if entry_dt is None:
            continue
        decision_tbl.append((q, entry_dt))

    if not decision_tbl:
        return pd.DataFrame(0, index=r.index, columns=cols, dtype="int8")

    # 4) 建立訊號矩陣
    signal = pd.DataFrame(0, index=r.index, columns=cols, dtype="int8")

    for i, (q, start_dt) in enumerate(decision_tbl):
        sel = both_ok.loc[q]
        if sel is None or not sel.any():
            continue
        picks_idx = pd.Index(sel.index[sel.values])

        if i + 1 < len(decision_tbl):
            next_start = decision_tbl[i + 1][1]
            end_pos = r.index.get_indexer_for([next_start])[0] - 1
            if end_pos < 0:
                continue
            end_dt = r.index[end_pos]
        else:
            end_dt = r.index[-1]

        if end_dt < start_dt:
            continue

        date_slice = r.loc[start_dt:end_dt]
        slice_month = date_slice.index.to_period("M")

        for m in slice_month.unique():
            universe = pd.Index(sorted(mktcap_pool.get(m, set()))).astype(str).str.strip()
            uni_cols = signal.columns.intersection(universe)
            final = uni_cols.intersection(picks_idx)
            if final.empty:
                continue
            idx_in_slice = date_slice.index[slice_month == m]
            signal.loc[idx_in_slice, final] = 1

    signal.index.name = r.index.name
    signal.columns.name = r.columns.name
    return signal
