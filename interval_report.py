import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def perf_report(returns_dict: dict, start_date: str, end_date: str, 
                freq: int = 252, rf: float = 0.0):
    """
    產出專業績效報表：
    - 累積報酬曲線圖
    - Sharpe Ratio / Annual Return bar chart
    - 績效總表 (DataFrame)
    """
    rows = []
    cum_dict = {}

    for name, r in returns_dict.items():
        r = r.loc[start_date:end_date].dropna()
        if r.empty:
            rows.append([np.nan, np.nan, np.nan, np.nan])
            cum_dict[name] = pd.Series(dtype=float)
            continue

        # 累積報酬序列
        cum = (1 + r).cumprod() - 1
        cum_dict[name] = cum

        # 指標
        ann_ret = (1 + r).prod() ** (freq / len(r)) - 1
        ann_vol = r.std() * np.sqrt(freq)
        sharpe = (ann_ret - rf) / ann_vol if ann_vol > 0 else np.nan
        total_ret = cum.iloc[-1]
        rows.append([ann_ret, ann_vol, sharpe, total_ret])

    # === 績效表 ===
    perf_table = pd.DataFrame(rows, index=returns_dict.keys(),
                              columns=["Annual Return", "Annual Volatility", "Sharpe Ratio", "Total Return"])

    # === 視覺化 ===
    fig, axes = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [2, 1]})

    # (1) 累積報酬曲線
    ax1 = axes[0]
    for name, cum in cum_dict.items():
        if cum.empty:
            continue
        ax1.plot(cum.index, cum.values, label=name, linewidth=2)
    ax1.set_title(f"Cumulative Returns ({start_date} → {end_date})", fontsize=14, weight="bold")
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Cumulative Return")
    ax1.legend()
    ax1.grid(alpha=0.3)

    # (2) Sharpe ratio / Annual Return 長條圖
    ax2 = axes[1]
    perf_table_sorted = perf_table.sort_values("Sharpe Ratio", ascending=False)
    bars = ax2.bar(perf_table_sorted.index, perf_table_sorted["Sharpe Ratio"], 
                   color="steelblue", alpha=0.8, label="Sharpe Ratio")
    ax2.set_title("Sharpe Ratio Comparison", fontsize=13, weight="bold")
    ax2.set_ylabel("Sharpe Ratio")
    ax2.grid(alpha=0.3, axis="y")
    ax2.set_xticklabels(perf_table_sorted.index, rotation=30, ha="right")

    # 在 bar 上標數值
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2, height, f"{height:.2f}", 
                 ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.show()

    return perf_table




import pandas as pd
import numpy as np

def factor_monthly_heatmap_plotly(returns_dict: dict, end_date: str = None, months: int = 12,
                                  title: str = "Factors Monthly Return Heatmap (Last 12 months)",
                                  save_html: str | None = None):
    """
    使用 Plotly 畫因子月報酬熱力圖（最近 N 個月）。
    - returns_dict: {factor_name: pd.Series(日報酬)}
    - end_date: 結束日期（字串或 Timestamp）。預設用所有 series 的最後一天
    - months: 最近幾個月（預設 12）
    - title: 圖片標題
    - save_html: 若給路徑，會輸出為單一 HTML 檔案

    回傳：用於繪圖的 DataFrame（row=factor, col=月份字串，值=月報酬）
    """
    try:
        import plotly.graph_objects as go
    except ImportError as e:
        raise ImportError("需要 plotly。請先安裝：pip install plotly") from e

    # --- 決定期間 ---
    if end_date is None:
        all_idx = [s.index for s in returns_dict.values() if isinstance(s, pd.Series) and len(s)]
        if not all_idx:
            raise ValueError("returns_dict 內沒有有效的 Series。")
        end_date = max(idx.max() for idx in all_idx)
    end_date = pd.to_datetime(end_date)
    start_date = (end_date - pd.DateOffset(months=months-1)).replace(day=1)

    # --- 計算各因子月報酬（複利） ---
    monthly = {}
    for name, r in returns_dict.items():
        r = pd.Series(r).dropna()
        r = r.loc[start_date:end_date]
        if r.empty:
            monthly[name] = pd.Series(dtype=float)
            continue
        mret = (1 + r).resample("M").prod() - 1
        monthly[name] = mret

    # --- 組成 DataFrame（row=factor, col=month） ---
    df = pd.DataFrame(monthly).T
    df = df.iloc[:, -months:] if df.shape[1] > months else df
    df.columns = [c.strftime("%Y-%m") for c in df.columns]

    # --- z 值與文字 ---
    z = df.values.astype(float) * 100.0
    z = np.where(np.isnan(z), None, z)
    text = df.applymap(lambda x: f"{x*100:.2f}%" if pd.notna(x) else "–").values

    # --- 畫圖 ---
    import plotly.graph_objects as go
    fig = go.Figure(data=go.Heatmap(
        z=z,
        x=df.columns.tolist(),
        y=df.index.tolist(),
        colorscale="RdYlGn",
        zmid=0,
        colorbar=dict(title="Monthly Return (%)"),
        text=text,
        texttemplate="%{text}",
        hovertemplate="<b>%{y}</b><br>%{x}<br>Return: %{text}<extra></extra>",
        xgap=1, ygap=1
    ))

    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor="center"),
        xaxis_title="Month",
        yaxis_title="Factors",
        font=dict(size=12),
        margin=dict(l=80, r=40, t=70, b=50),
    )

    # 🔑 強制每個月都顯示
    fig.update_xaxes(tickmode="array", tickvals=df.columns.tolist())

    if save_html:
        fig.write_html(save_html, include_plotlyjs="cdn")

    fig.show()
    return df




import plotly.express as px

def factor_rank_tile(
    returns_dict: dict,
    end_date: str | None = None,
    periods: int = 12,
    freq: str = "M",
    title: str = "Factor Ranking Tile Chart",
    save_html: str | None = None
):
    """
    依照單期報酬對因子排名，畫出方格圖。
    - returns_dict: {因子名稱: pd.Series(日報酬)}
    - end_date: 結束日期
    - periods: 最近期數 (預設12)
    - freq: "M"=月, "W"=週, "Y"=年
    - title: 圖標題
    - save_html: 可選，若給路徑會輸出 HTML 檔
    """
    # --- 確定結束日期 ---
    if end_date is None:
        all_idx = [s.index for s in returns_dict.values() if isinstance(s, pd.Series) and len(s)]
        if not all_idx:
            raise ValueError("returns_dict 內沒有有效的 Series。")
        end_date = max(idx.max() for idx in all_idx)
    end_date = pd.to_datetime(end_date)

    # --- 起始日期 ---
    if freq == "M":
        start_date = (end_date - pd.DateOffset(months=periods-1)).replace(day=1)
    elif freq == "W":
        start_date = end_date - pd.DateOffset(weeks=periods-1)
    elif freq == "Y":
        start_date = pd.Timestamp(year=end_date.year - (periods-1), month=1, day=1)
    else:
        raise ValueError("freq 只能是 'M', 'W', 'Y'")

    # --- 計算單期報酬 ---
    perf = []
    for name, r in returns_dict.items():
        r = pd.Series(r).dropna()
        r = r.loc[start_date:end_date]
        if r.empty:
            continue
        pret = (1 + r).resample(freq).prod() - 1
        pret = pret.dropna().iloc[-periods:]
        for d, val in pret.items():
            perf.append([d.strftime("%Y-%m") if freq=="M" else str(d.date()), name, val])

    df = pd.DataFrame(perf, columns=["Period", "Factor", "Return"])
    if df.empty:
        raise ValueError("沒有有效資料")

    # --- 每期內排名 ---
    df["Rank"] = df.groupby("Period")["Return"].rank(ascending=False, method="first")

    # --- 整理成排名方格 ---
    df["Label"] = df.apply(lambda x: f"{x['Factor']}<br>{x['Return']*100:.2f}%", axis=1)

    # --- 固定因子顏色 ---
    factors = df["Factor"].unique()
    color_map = {f: px.colors.qualitative.Set2[i % len(px.colors.qualitative.Set2)]
                 for i, f in enumerate(sorted(factors))}

    # --- 畫圖 ---
    fig = px.scatter(
        df,
        x="Period", y="Rank",
        text="Label",
        color="Factor",
        color_discrete_map=color_map,
        title=title,
    )

    # 格子化
    fig.update_traces(
        marker=dict(size=40, symbol="square"),
        textposition="middle center",
        textfont=dict(size=9, color="black")
    )

    fig.update_layout(
        yaxis=dict(autorange="reversed", dtick=1, title="Rank"),
        xaxis=dict(title="Period"),
        legend_title="Factor",
        height=600 + len(factors)*10,
        margin=dict(l=60, r=60, t=80, b=60),
    )

    if save_html:
        fig.write_html(save_html, include_plotlyjs="cdn")

    fig.show()
    return df

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

def factor_rank_tile_full(
    returns_dict: dict,
    end_date: str | None = None,
    periods: int = 12,
    freq: str = "M",   # "M" 月 / "W" 週 / "Y" 年
    title: str = "Factor Ranking — Last 12 Periods",
    save_html: str | None = None,
    text_size: int = 9
):
    """
    依各因子『單期報酬』在每期做排名，畫滿版無縫的方格圖（每格顯示因子名+報酬%）。
    - returns_dict: {因子名稱: pd.Series(日報酬, DatetimeIndex)}
    - end_date: 結束日期（不給就自動抓所有 series 的最後一天）
    - periods: 近 N 期 (預設 12)
    - freq: "M"=月, "W"=週, "Y"=年
    """
        # ==== 自動化標題 ====
    if title is None:
        unit_map = {"M": "Months", "W": "Weeks", "Y": "Years"}
        unit = unit_map.get(freq, "Periods")
        title = f"Factor Ranking — Last {periods} {unit}"

    # ==== 1) 決定時間範圍 ====
    if end_date is None:
        all_idx = [s.index for s in returns_dict.values() if isinstance(s, pd.Series) and len(s)]
        if not all_idx:
            raise ValueError("returns_dict 裡沒有有效的 Series。")
        end_date = max(idx.max() for idx in all_idx)
    end_date = pd.to_datetime(end_date)

    if freq == "M":
        start_date = (end_date - pd.DateOffset(months=periods-1)).replace(day=1)
        period_fmt = lambda p: p.strftime("%Y-%m")
    elif freq == "W":
        start_date = end_date - pd.DateOffset(weeks=periods-1)
        period_fmt = lambda p: p.strftime("%Y-%m-%d")
    elif freq == "Y":
        start_date = pd.Timestamp(year=end_date.year - (periods-1), month=1, day=1)
        period_fmt = lambda p: str(p.year)
    else:
        raise ValueError("freq 只能是 'M', 'W', 或 'Y'")

    # ==== 2) 計算各因子『單期報酬』 ====
    records = []
    for fac, sr in returns_dict.items():
        r = pd.Series(sr).dropna()
        r = r.loc[start_date:end_date]
        if r.empty:
            continue
        ret_p = (1 + r).resample(freq).prod() - 1
        ret_p = ret_p.dropna().iloc[-periods:]
        for d, v in ret_p.items():
            records.append([period_fmt(pd.to_datetime(d)), fac, float(v)])

    df = pd.DataFrame(records, columns=["Period", "Factor", "Return"])
    if df.empty:
        raise ValueError("選定期間內沒有資料。")

    # ==== 3) 每期內依單期報酬排名（高→低） ====
    df["Rank"] = df.groupby("Period")["Return"].rank(ascending=False, method="first").astype(int)
    # 排名軸：1 在上
    ranks = sorted(df["Rank"].unique())
    periods_labels = list(df.sort_values("Period")["Period"].unique())

    # ==== 4) 轉寬表 ====
    pivot_factor = df.pivot(index="Rank", columns="Period", values="Factor").reindex(index=ranks, columns=periods_labels)
    pivot_return = df.pivot(index="Rank", columns="Period", values="Return").reindex(index=ranks, columns=periods_labels)

    # ==== 5) 固定顏色：因子 → 顏色 ====
    factors = sorted(df["Factor"].unique())
    palette = px.colors.qualitative.Set2
    color_map = {f: palette[i % len(palette)] for i, f in enumerate(factors)}
    # 做一個「因子代碼矩陣」與顏色映射（離散色階）
    factor_code = {f: i for i, f in enumerate(factors)}
    Z = pivot_factor.replace(factor_code).to_numpy()

    # 建立離散 colorscale
    # 例如有 K 個因子，z=0..K-1，各段都用固定色
    K = len(factors)
    colorscale = []
    for i, f in enumerate(factors):
        v0 = (i) / max(K-1, 1)
        v1 = (i) / max(K-1, 1)
        colorscale.append([v0, color_map[f]])
        colorscale.append([v1, color_map[f]])

    # ==== 6) 文字（因子名 + 報酬%） ====
    text = np.empty_like(Z, dtype=object)
    for i, rank in enumerate(pivot_factor.index):
        for j, per in enumerate(pivot_factor.columns):
            fac = pivot_factor.iloc[i, j]
            ret = pivot_return.iloc[i, j]
            if pd.isna(fac) or pd.isna(ret):
                text[i, j] = ""
            else:
                text[i, j] = f"{fac}<br>{ret*100:.2f}%"

    # ==== 7) 畫滿版無縫格子 ====
    fig = go.Figure(data=go.Heatmap(
        z=Z,
        x=periods_labels,
        y=pivot_factor.index,                 # Rank 值（1,2,...）
        colorscale=colorscale,
        zmin=0, zmax=max(K-1, 1),
        showscale=False,
        text=text,
        texttemplate="%{text}",
        textfont={"size": text_size, "color": "black"},
        xgap=0, ygap=0                        # 🔑 無縫
    ))

    fig.update_layout(
        title=dict(text=title, x=0.5),
        xaxis=dict(title=None, tickmode="array", tickvals=periods_labels),
        yaxis=dict(title="Rank", autorange="reversed", dtick=1),
        margin=dict(l=40, r=40, t=70, b=40),
        plot_bgcolor="white",
        paper_bgcolor="white",
        height=120 + 30*len(ranks)
    )

    if save_html:
        fig.write_html(save_html, include_plotlyjs="cdn")

    fig.show()
    return df
