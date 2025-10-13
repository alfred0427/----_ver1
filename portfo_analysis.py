import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def top_contributors_report(
    returns: pd.DataFrame,
    tickers: list[str],
    start_date: str,
    end_date: str,
    weights: dict[str, float] | None = None,
    top_n: int = 10,
    annualize: int = 252,
    risk_free_annual: float = 0.0,
):
    """
    依指定區間產生「專業版」績效＋貢獻報告：
    1) 各股：年化報酬、年化波動、Sharpe
    2) 各股：區間報酬(period return) 與 對投組貢獻(contribution)
    3) 取前 top_n 名貢獻最大股票的 bar chart

    returns: index=日期, columns=股票代號, 值=日報酬率
    tickers: 要納入的股票清單（需是 returns 的欄名）
    start_date, end_date: 'YYYY-MM-DD'
    weights: 例如 {"2330":0.2, "2317":0.1, ...}；若 None 則等權
    top_n: 取前幾名貢獻最大
    annualize: 年化天數（台股常用 252）
    risk_free_annual: 年化無風險利率，用於 Sharpe（預設 0）
    """
    # 篩選區間 & 欄位
    sub = returns.loc[start_date:end_date, tickers].copy()
    sub = sub.dropna(how="all")
    if sub.empty:
        raise ValueError("選定期間內沒有有效資料。")

    # 權重：等權或自訂；確保只含現有 tickers 並正規化到 1
    if weights is None:
        w = pd.Series(1.0/len(tickers), index=tickers)
    else:
        w = pd.Series(weights).reindex(tickers).fillna(0.0)
        total_w = w.sum()
        w = w / total_w if total_w != 0 else pd.Series(1.0/len(tickers), index=tickers)

    # 各股區間報酬（multiplicative）
    period_ret = (1 + sub).prod() - 1  # 對每檔股票

    # 投組期間報酬 & 各股對投組的「貢獻」（加總 = 投組期間報酬）
    # 這裡使用簡化的一次性歸因：contribution ≈ w0 * period_ret_i
    contribution = w * period_ret
    portfolio_period_return = contribution.sum()

    # 年化指標（用日資料年化）
    daily_mean = sub.mean()
    daily_std  = sub.std()
    ann_return = (1 + daily_mean)**annualize - 1
    ann_vol    = daily_std * np.sqrt(annualize)
    rf_daily   = (1 + risk_free_annual)**(1/annualize) - 1
    sharpe     = (ann_return - risk_free_annual) / ann_vol

    # 彙整成表格
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
    plt.title(f"Top {top_n} Contributors ({start_date} ~ {end_date})\nPortfolio Period Return = {portfolio_period_return:.2%}")
    plt.xlabel("Ticker")
    plt.ylabel("Contribution to Portfolio Return")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()

    return stats, portfolio_period_return

# # === 範例用法 ===
# # 假設你已經有 returns（日報酬率 DataFrame）以及想看的股票清單：
# tickers = ["2404","2504","2317","2603","6005","5434","6176","2882","2609","2881","2615","2618","2891","2883","2855"]
# stats_df, port_ret = top_contributors_report(
#     returns, tickers, start_date="2024-01-02", end_date="2024-12-31", top_n=10
# )
# print(stats_df.head(15))
