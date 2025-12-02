import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def top_contributors_report_sector_neutral(
    returns: pd.DataFrame,
    tickers: list[str],
    sector_map: pd.Series,                     # index=code(str), value=sector(str)
    start_date: str,
    end_date: str,
    weights: dict[str, float] | None = None,   # 若給了就用你指定的，不再強制 sector-neutral
    top_n: int = 10,
    annualize: int = 252,
    risk_free_annual: float = 0.0,
):
    """
    sector-neutral 版：
      - 若 weights=None，則：
          1) 先把 tickers 分產業
          2) 每個產業給相同總權重（等權產業）
          3) 產業內成分股等權 → 得到 sector-neutral 權重向量 w
      - 其他計算方式與原版 top_contributors_report 相同。
    """
    # 篩選區間 & 欄位
    sub = returns.loc[start_date:end_date, tickers].copy()
    sub = sub.dropna(how="all")
    if sub.empty:
        raise ValueError("選定期間內沒有有效資料。")

    # ---- 建立 sector-neutral 權重 ----
    if weights is None:
        # 對齊產業資訊
        sec = sector_map.copy()
        sec.index = sec.index.astype(str).str.strip()
        sec = sec.reindex(tickers)

        # 只看有產業資訊的
        valid = sec.notna()
        if not valid.any():
            # 如果完全沒有 sector 資訊，就退回全等權
            w = pd.Series(1.0 / len(tickers), index=tickers)
        else:
            sectors = sorted(sec[valid].unique())
            n_sec = len(sectors)
            # 每個有資訊的產業先分配 1/n_sec 的總權重
            # 產業內等權 → w_i = (1 / n_sec) / (#該產業股票數)
            w = pd.Series(0.0, index=tickers, dtype="float64")
            for s in sectors:
                members = sec.index[sec == s]
                if len(members) == 0:
                    continue
                w_s = 1.0 / n_sec / len(members)
                w.loc[members] = w_s

            # 若有無產業資訊的股票，給 0 權重（或你可以依需求再加）
            # 最後再 normalize 一下（避免數值誤差）
            total_w = w.sum()
            if total_w != 0:
                w = w / total_w
            else:
                # 理論上不會，但預防一下
                w = pd.Series(1.0 / len(tickers), index=tickers)
    else:
        # 若你自訂 weights，就尊重你的設定，只做 normalize
        w = pd.Series(weights).reindex(tickers).fillna(0.0)
        total_w = w.sum()
        w = w / total_w if total_w != 0 else pd.Series(1.0/len(tickers), index=tickers)

    # ---- 下面與原本 top_contributors_report 完全相同 ----

    # 各股區間報酬（multiplicative）
    period_ret = (1 + sub).prod() - 1  # 對每檔股票

    # 投組期間報酬 & 各股貢獻
    contribution = w * period_ret
    portfolio_period_return = contribution.sum()

    # 年化指標
    daily_mean = sub.mean()
    daily_std  = sub.std()
    ann_return = (1 + daily_mean)**annualize - 1
    ann_vol    = daily_std * np.sqrt(annualize)
    rf_daily   = (1 + risk_free_annual)**(1/annualize) - 1
    sharpe     = (ann_return - risk_free_annual) / ann_vol

    stats = pd.DataFrame({
        "Annual Return": ann_return,
        "Annual Volatility": ann_vol,
        "Sharpe Ratio": sharpe,
        "Period Return": period_ret,
        "Weight": w,
        "Contribution": contribution
    }).sort_values("Contribution", ascending=False)

    # 取前 N 名做圖
    topN = stats.head(top_n)

    plt.figure(figsize=(10, 5))
    plt.bar(topN.index.astype(str), topN["Contribution"].values)
    plt.title(
        f"Top {top_n} Contributors (Sector-neutral) "
        f"({start_date} ~ {end_date})\n"
        f"Portfolio Period Return = {portfolio_period_return:.2%}"
    )
    plt.xlabel("Ticker")
    plt.ylabel("Contribution to Portfolio Return")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()

    return stats, portfolio_period_return

