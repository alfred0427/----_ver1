# sector_neutral.py
import pandas as pd
import numpy as np


def _standardize_tickers(df: pd.DataFrame) -> pd.DataFrame:
    """欄位轉成字串並去空白，方便跟產業表對齊。"""
    x = df.copy()
    x.columns = x.columns.astype(str).str.strip()
    return x


def build_sector_map(
    sector_df: pd.DataFrame,
    code_col: str = "code",
    sector_col: str = "sector",
) -> pd.Series:
    """
    把「股票代碼–產業」的 DataFrame 轉成 Series：
    index = 股票代碼(str)，values = 產業名稱/代號(str)

    你可以依照自己的欄位名稱改 code_col / sector_col。
    """
    s = (
        sector_df
        .copy()
        .assign(
            **{
                code_col: sector_df[code_col].astype(str).str.strip(),
                sector_col: sector_df[sector_col].astype(str).str.strip(),
            }
        )
        .set_index(code_col)[sector_col]
    )
    return s


def sector_factor_returns(
    alpha: pd.DataFrame,
    returns: pd.DataFrame,
    sector_map: pd.Series,
) -> pd.DataFrame:
    """
    給定：
      - alpha: 0/1 訊號 (日頻，index=日期, columns=股票)
      - returns: 日報酬 (與 alpha 同 shape)
      - sector_map: 股票→產業 (index=股票, value=產業)

    回傳：
      - sector_ret: DataFrame (index=日期, columns=產業)
        每一欄是「該產業內，本因子等權持有的每日報酬」。
    """
    a = _standardize_tickers(alpha)
    r = _standardize_tickers(returns)

    # 只保留交集欄位
    common_cols = a.columns.intersection(r.columns)
    a = a[common_cols]
    r = r[common_cols]

    # 產業 mapping 標準化
    smap = sector_map.copy()
    smap.index = smap.index.astype(str).str.strip()
    smap = smap.astype(str).str.strip()

    # 每檔股票的產業
    col_sector = smap.reindex(a.columns)
    # 去掉沒產業資訊的欄
    valid_cols = col_sector.dropna().index
    a = a[valid_cols]
    r = r[valid_cols]
    col_sector = col_sector.loc[valid_cols]

    sectors = sorted(col_sector.unique())
    sector_ret = pd.DataFrame(index=r.index, columns=sectors, dtype="float64")

    # 對每個產業，算「產業內等權持有」的因子報酬
    for sec in sectors:
        cols_sec = a.columns[col_sector == sec]
        if len(cols_sec) == 0:
            continue

        a_sec = a[cols_sec]
        r_sec = r[cols_sec]

        # 當天在該產業中被選到的股票數
        counts = a_sec.sum(axis=1)
        # 當天該產業選股的加權報酬（其實就是等權：1 * return）
        num = (a_sec * r_sec).sum(axis=1)

        # 沒有持股的天數，結果會是 NaN
        sec_ret = num / counts.replace(0, np.nan)
        sector_ret[sec] = sec_ret

    return sector_ret


def combine_sector_returns(
    sector_ret: pd.DataFrame,
    sector_weight: pd.Series | pd.DataFrame | None = None,
) -> pd.Series:
    """
    把「各產業因子報酬」用「產業權重」加總成總報酬。

    sector_ret:
        index = 日期 (DatetimeIndex)
        columns = 產業
    sector_weight:
        - 若為 None：每日可投資的產業「等權重」；
        - 若為 Series：index=產業, value=固定權重 (總和不用一定=1，會自動 normalize)；
        - 若為 DataFrame：index=Period('YYYY-MM','M') 或 DatetimeIndex (月)，
          columns=產業，值為每月產業權重（會依日期對應到那個月）。

    回傳：
        portfolio_ret: Series (index=日期, values=總報酬)
    """
    sr = sector_ret.copy()

    # 1) 沒給 sector_weight：當天有非 NaN 的產業等權重
    if sector_weight is None:
        w = (~sr.isna()).astype(float)
        w = w.div(w.sum(axis=1), axis=0)  # 每天 row-normalize
        port = (sr * w).sum(axis=1)
        return port

    # 2) sector_weight 是 Series：固定的產業權重
    if isinstance(sector_weight, pd.Series):
        w = sector_weight.copy()
        w.index = w.index.astype(str).str.strip()
        # 對齊欄位
        w = w.reindex(sr.columns).fillna(0.0)
        if w.sum() != 0:
            w = w / w.sum()
        port = (sr * w).sum(axis=1)
        return port

    # 3) sector_weight 是 DataFrame：月度權重
    if isinstance(sector_weight, pd.DataFrame):
        wdf = sector_weight.copy()
        wdf.columns = wdf.columns.astype(str).str.strip()

        # 若 index 不是 PeriodIndex，轉成月 Period
        if isinstance(wdf.index, pd.DatetimeIndex):
            wdf.index = wdf.index.to_period("M")
        elif not isinstance(wdf.index, pd.PeriodIndex):
            wdf.index = pd.to_datetime(wdf.index).to_period("M")

        # 建一個跟 sector_ret 同 index/columns 的權重矩陣
        month_key = sr.index.to_period("M")
        W = pd.DataFrame(index=sr.index, columns=sr.columns, dtype="float64")

        for m in month_key.unique():
            if m not in wdf.index:
                continue
            w_m = wdf.loc[m].reindex(sr.columns).fillna(0.0)
            if w_m.sum() != 0:
                w_m = w_m / w_m.sum()
            mask = (month_key == m)
            W.loc[mask, :] = w_m.values

        port = (sr * W).sum(axis=1)
        return port

    raise TypeError("sector_weight 必須是 None / pd.Series / pd.DataFrame 之一。")


def alpha_sector_neutral_return(
    alpha: pd.DataFrame,
    returns: pd.DataFrame,
    sector_map: pd.Series,
    sector_weight: pd.Series | pd.DataFrame | None = None,
) -> pd.Series:
    """
    一步到位：由 alpha + returns + 產業資訊 → sector-neutral 因子總報酬。

    步驟：
      1) sector_factor_returns：在每個產業內等權選股；
      2) combine_sector_returns：用產業權重加總。

    這個函式可以視為「sector-neutral 版本的 alp_return」。
    """
    sector_ret = sector_factor_returns(alpha, returns, sector_map)
    port_ret = combine_sector_returns(sector_ret, sector_weight)
    return port_ret
