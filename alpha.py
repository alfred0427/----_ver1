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
