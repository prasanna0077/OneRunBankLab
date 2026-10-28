"""Plotly figure builders for OneRun Bank Lab. No Streamlit calls here."""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

import engine as E

NAVY = "#1F3A5F"
INK = "#1B2430"
MUTED = "#64748B"
GRID = "#E5E9F0"
GOOD = "#2E8B57"
BAD = "#C0392B"
WARN = "#E67E22"
BANK_A = "#1F3A5F"
BANK_B = "#E76F51"
COMP = {  # balance-sheet components
    "Loans": "#2A9D8F",
    "Investments & cash": "#8ECDC4",
    "CASA deposits": "#475569",
    "Term deposits": "#94A3B8",
    "Borrowings & other": "#CBD5E1",
    "Equity": "#E9C46A",
}
TYPE_COLOUR = {"PSU": "#1F3A5F", "Private": "#2A9D8F", "SFB": "#E76F51", "My bank": "#111111"}


def inr(x: float, decimals: int = 0) -> str:
    """Indian digit grouping: 1234567 -> 12,34,567."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "–"
    x = float(x)
    neg = round(x, decimals) < 0
    s, _, frac = f"{abs(x):.{decimals}f}".partition(".")
    s_frac = f".{frac}" if decimals else ""
    if len(s) > 3:
        head, tail = s[:-3], s[-3:]
        groups = []
        while len(head) > 2:
            groups.insert(0, head[-2:])
            head = head[:-2]
        if head:
            groups.insert(0, head)
        s = ",".join(groups) + "," + tail
    return ("-" if neg else "") + s + s_frac


def cr(x: float) -> str:
    """Rs crore, switching to lakh crore for very large values."""
    if x is None:
        return "–"
    if abs(x) >= 1e5:
        return f"₹{x / 1e5:,.2f} lakh Cr"
    return f"₹{inr(x)} Cr"


def _layout(fig: go.Figure, height: int = 380, title: str | None = None, **kw) -> go.Figure:
    if title:
        fig.update_layout(title=dict(text=title, x=0, xanchor="left", font=dict(size=15)))
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=64 if title else 24, b=10),
        font=dict(family="Inter, Segoe UI, Helvetica, Arial, sans-serif", size=13, color=INK),
        plot_bgcolor="white",
        paper_bgcolor="white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hoverlabel=dict(font_size=12),
        **kw,
    )
    fig.update_xaxes(gridcolor=GRID, zeroline=False, linecolor=GRID)
    fig.update_yaxes(gridcolor=GRID, zeroline=False, linecolor=GRID)
    return fig


# --------------------------------------------------------------------------- balance sheet
def balance_sheet(r: dict, title: str | None = None) -> go.Figure:
    eq = max(r["eq_end"], 0.0)
    borr = max(r["assets"] - r["deposits"] - eq, 0.0)
    parts = [
        ("Loans", "Assets", r["loans"]),
        ("Investments & cash", "Assets", r["investments"]),
        ("Equity", "Funding", eq),
        ("Borrowings & other", "Funding", borr),
        ("Term deposits", "Funding", r["td"]),
        ("CASA deposits", "Funding", r["casa"]),
    ]
    fig = go.Figure()
    total = r["assets"]
    for name, side, val in parts:
        share = val / total * 100 if total else 0
        fig.add_trace(go.Bar(
            x=[side], y=[val], name=name, marker_color=COMP[name],
            text=[f"{name}<br>{cr(val)} · {share:.0f}%"] if share >= 4 else [""],
            textposition="inside", insidetextanchor="middle",
            textfont=dict(color="white" if name in ("Loans", "CASA deposits") else INK, size=12),
            hovertemplate=f"{name}: %{{y:,.0f}} Cr ({share:.1f}%)<extra></extra>",
        ))
    fig.update_layout(barmode="stack", bargap=0.35, showlegend=False)
    fig.update_yaxes(title_text="₹ crore", tickformat=",")
    return _layout(fig, 460, title)


# --------------------------------------------------------------------------- P&L waterfall
def pnl_waterfall(r: dict, title: str | None = None) -> go.Figure:
    steps = [
        ("Interest income", "relative", r["int_income"]),
        ("Interest expense", "relative", -r["int_expense"]),
        ("Net interest income", "total", r["nii"]),
        ("Fee & other income", "relative", r["fees"]),
        ("Operating cost", "relative", -r["opex"]),
        ("Operating profit", "total", r["ppop"]),
        ("Credit loss", "relative", -r["credit_loss"]),
    ]
    if abs(r["other_loss"]) > 0.5:
        steps.append(("One-off loss", "relative", -r["other_loss"]))
    steps += [("Tax", "relative", -r["tax"]), ("Net profit", "total", r["pat"])]
    fig = go.Figure(go.Waterfall(
        x=[s[0] for s in steps],
        measure=[s[1] for s in steps],
        y=[0 if s[1] == "total" else s[2] for s in steps],
        text=[inr(s[2]) for s in steps],
        textposition="outside",
        connector=dict(line=dict(color=GRID, width=1)),
        increasing=dict(marker=dict(color="#7FB89A")),
        decreasing=dict(marker=dict(color="#E39A91")),
        totals=dict(marker=dict(color=NAVY)),
        hovertemplate="%{x}: %{text} Cr<extra></extra>",
    ))
    fig.update_yaxes(title_text="₹ crore", tickformat=",")
    fig.update_layout(showlegend=False)
    return _layout(fig, 420, title)


# --------------------------------------------------------------------------- stress curves
def crar_curve(curves: list[tuple], x_title: str, current_x: float | None = None,
               title: str | None = None, x_suffix: str = "%") -> go.Figure:
    """curves: list of (name, xs, crar_values, colour)."""
    fig = go.Figure()
    for name, xs, ys, colour in curves:
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="lines", name=name, line=dict(color=colour, width=3),
            hovertemplate=f"{name}<br>%{{x:.2f}}{x_suffix} → CRAR %{{y:.1f}}%<extra></extra>",
        ))
    fig.add_hline(y=E.CRAR_WITH_CCB, line_dash="dash", line_color=WARN,
                  annotation_text="11.5% incl. buffer", annotation_position="top right")
    fig.add_hline(y=E.CRAR_MIN, line_dash="dash", line_color=BAD,
                  annotation_text="9% regulatory minimum", annotation_position="bottom right")
    fig.add_hline(y=0, line_color=MUTED, line_width=1)
    if current_x is not None:
        fig.add_vline(x=current_x, line_dash="dot", line_color=INK,
                      annotation_text="your shock", annotation_position="top left")
    fig.update_xaxes(title_text=x_title, ticksuffix=x_suffix)
    fig.update_yaxes(title_text="CRAR after the shock", ticksuffix="%")
    return _layout(fig, 400, title, showlegend=len(curves) > 1)


def rate_curves(bps, crar, dnii, current_bp, title=None) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=bps, y=crar, name="CRAR (%)", line=dict(color=NAVY, width=3),
                             hovertemplate="%{x:+.0f} bp → CRAR %{y:.1f}%<extra></extra>"))
    fig.add_trace(go.Scatter(x=bps, y=dnii, name="Change in NII (₹ Cr)", yaxis="y2",
                             line=dict(color="#2A9D8F", width=2, dash="dot"),
                             hovertemplate="%{x:+.0f} bp → NII %{y:+,.0f} Cr<extra></extra>"))
    fig.add_hline(y=E.CRAR_WITH_CCB, line_dash="dash", line_color=WARN)
    fig.add_hline(y=E.CRAR_MIN, line_dash="dash", line_color=BAD)
    fig.add_vline(x=current_bp, line_dash="dot", line_color=INK)
    fig.update_layout(yaxis2=dict(overlaying="y", side="right", showgrid=False,
                                  title=dict(text="Change in NII (₹ Cr)"), tickformat=","))
    fig.update_xaxes(title_text="Rate change (basis points)")
    fig.update_yaxes(title_text="CRAR", ticksuffix="%")
    return _layout(fig, 380, title)


def liquidity_bars(usable: float, outflow: float, shortfall: float, title=None) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Bar(x=["Usable liquidity", "Deposit outflow"], y=[usable, outflow],
                         marker_color=["#2A9D8F", BAD if shortfall > 0 else "#94A3B8"],
                         text=[cr(usable), cr(outflow)], textposition="outside",
                         hovertemplate="%{x}: %{y:,.0f} Cr<extra></extra>"))
    fig.update_yaxes(title_text="₹ crore", tickformat=",")
    fig.update_layout(showlegend=False)
    return _layout(fig, 340, title)


# --------------------------------------------------------------------------- what-if
def verdict_heatmap(codes, xs, ys, xlab, ylab, cur_x, cur_y, title=None) -> go.Figure:
    scale = []
    for code in range(6):
        colour = E.VERDICT_COLOUR[code]
        scale += [[code / 6, colour], [(code + 1) / 6, colour]]
    labels = np.vectorize(E.VERDICT_LABEL.get)(codes)
    fig = go.Figure(go.Heatmap(
        z=codes, x=xs, y=ys, zmin=-0.5, zmax=5.5, colorscale=scale, showscale=False,
        customdata=labels, opacity=0.85,
        hovertemplate=f"{xlab}: %{{x:.2f}}<br>{ylab}: %{{y:.2f}}<br><b>%{{customdata}}</b><extra></extra>",
    ))
    fig.add_trace(go.Scatter(x=[cur_x], y=[cur_y], mode="markers+text", text=["Your bank"],
                             textposition="top center", textfont=dict(color="black", size=13),
                             marker=dict(size=16, color="white", line=dict(color="black", width=3)),
                             hoverinfo="skip", showlegend=False))
    for code in range(6):  # legend entries
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", name=E.VERDICT_LABEL[code],
                                 marker=dict(size=12, color=E.VERDICT_COLOUR[code], symbol="square")))
    fig.update_xaxes(title_text=xlab, showgrid=False)
    fig.update_yaxes(title_text=ylab, showgrid=False)
    return _layout(fig, 480, title)


def distance_chart(rows: list[dict], title=None) -> go.Figure:
    """rows: label, cur_pos, fail_pos (0-100 or None), fail_side ('up'/'down'), hover."""
    fig = go.Figure()
    xs, ys = [], []
    rx, ry = [], []
    for r in rows:
        xs += [0, 100, None]
        ys += [r["label"], r["label"], None]
        if r["fail_pos"] is not None:
            a, b = (r["fail_pos"], 100) if r["fail_side"] == "up" else (0, r["fail_pos"])
            rx += [a, b, None]
            ry += [r["label"], r["label"], None]
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line=dict(color="#7FB89A", width=10),
                             name="Survives", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=rx, y=ry, mode="lines", line=dict(color="#E39A91", width=10),
                             name="Fails", hoverinfo="skip"))
    fig.add_trace(go.Scatter(
        x=[r["cur_pos"] for r in rows], y=[r["label"] for r in rows], mode="markers",
        marker=dict(size=13, color="white", line=dict(color="black", width=2.5)),
        name="Now", text=[r["hover"] for r in rows], hovertemplate="%{text}<extra></extra>"))
    fig.update_xaxes(range=[-2, 102], showticklabels=False, title_text="slider range: low → high",
                     showgrid=False)
    fig.update_yaxes(autorange="reversed", showgrid=False)
    return _layout(fig, 70 + 34 * len(rows), title)


def tornado_chart(rows: list[tuple], metric_label: str, title=None) -> go.Figure:
    labels = [r[0] for r in rows]
    fig = go.Figure()
    fig.add_trace(go.Bar(y=labels, x=[r[1] for r in rows], orientation="h", name="Knob down",
                         marker_color="#94A3B8",
                         hovertemplate="%{y} down: %{x:+.2f}<extra></extra>"))
    fig.add_trace(go.Bar(y=labels, x=[r[2] for r in rows], orientation="h", name="Knob up",
                         marker_color=NAVY,
                         hovertemplate="%{y} up: %{x:+.2f}<extra></extra>"))
    fig.update_layout(barmode="overlay")
    fig.update_xaxes(title_text=f"Change in {metric_label} (points)", zeroline=True,
                     zerolinecolor=MUTED)
    return _layout(fig, 90 + 30 * len(rows), title)


def histogram(values, threshold_lines: bool = True, title=None) -> go.Figure:
    fig = go.Figure(go.Histogram(x=values, nbinsx=50, marker_color="#94A3B8",
                                 hovertemplate="CRAR %{x}<br>%{y} years<extra></extra>"))
    if threshold_lines:
        fig.add_vline(x=E.CRAR_WITH_CCB, line_dash="dash", line_color=WARN,
                      annotation_text="11.5%", annotation_position="top")
        fig.add_vline(x=E.CRAR_MIN, line_dash="dash", line_color=BAD,
                      annotation_text="9%", annotation_position="top left")
    fig.update_xaxes(title_text="CRAR at year end", ticksuffix="%")
    fig.update_yaxes(title_text="Simulated years")
    fig.update_layout(showlegend=False, bargap=0.05)
    return _layout(fig, 340, title)


# --------------------------------------------------------------------------- explorer & compare
def peer_scatter(df, xcol, ycol, highlight: str | None = None, extra_point: dict | None = None,
                 title=None) -> go.Figure:
    fig = go.Figure()
    amax = df["Assets Cr"].max()
    for t in ["PSU", "Private", "SFB"]:
        d = df[df["Bank Type"] == t]
        if d.empty:
            continue
        fig.add_trace(go.Scatter(
            x=d[xcol], y=d[ycol], mode="markers", name=t,
            marker=dict(size=10 + 28 * np.sqrt(d["Assets Cr"] / amax), color=TYPE_COLOUR[t],
                        opacity=0.75, line=dict(color="white", width=1)),
            text=d["Bank"],
            hovertemplate=f"<b>%{{text}}</b><br>{xcol}: %{{x}}<br>{ycol}: %{{y}}<extra></extra>",
        ))
    if highlight is not None:
        h = df[df["Bank"] == highlight]
        if not h.empty:
            fig.add_trace(go.Scatter(
                x=h[xcol], y=h[ycol], mode="markers+text", name=highlight, text=[highlight],
                textposition="top center",
                marker=dict(size=22, color="rgba(0,0,0,0)", line=dict(color="black", width=3)),
                hoverinfo="skip"))
    if extra_point is not None:
        fig.add_trace(go.Scatter(
            x=[extra_point["x"]], y=[extra_point["y"]], mode="markers+text", name=extra_point["name"],
            text=[extra_point["name"]], textposition="top center",
            marker=dict(size=18, color="black", symbol="star"),
            hovertemplate=f"<b>{extra_point['name']}</b><br>{xcol}: %{{x:.2f}}<br>{ycol}: %{{y:.2f}}<extra></extra>"))
    fig.update_xaxes(title_text=xcol)
    fig.update_yaxes(title_text=ycol)
    return _layout(fig, 460, title)


SCORE_COLS = ["Funding Score", "Asset Quality Score", "Profitability Score", "Capital Score",
              "Liquidity Score", "Network Score", "Acquisition Score"]


def radar(series: list[tuple], title=None) -> go.Figure:
    """series: list of (name, scores list in SCORE_COLS order, colour)."""
    theta = [c.replace(" Score", "") for c in SCORE_COLS]
    fig = go.Figure()
    for name, scores, colour in series:
        fig.add_trace(go.Scatterpolar(r=list(scores) + [scores[0]], theta=theta + [theta[0]],
                                      fill="toself", name=name, line=dict(color=colour, width=2),
                                      opacity=0.55))
    lo = min(min(s[1]) for s in series)
    fig.update_layout(polar=dict(radialaxis=dict(range=[max(0, lo - 10), 100], gridcolor=GRID),
                                 angularaxis=dict(gridcolor=GRID)))
    return _layout(fig, 400, title)


def branch_mix(rows: list[tuple], title=None) -> go.Figure:
    """rows: (name, rural, semi, urban, metro). Shown as % of branches."""
    cats = [("Rural", "#2A9D8F"), ("Semi-urban", "#8ECDC4"), ("Urban", "#94A3B8"), ("Metro", NAVY)]
    fig = go.Figure()
    for k, (cat, colour) in enumerate(cats):
        vals = []
        for r in rows:
            tot = sum(r[1:5]) or 1
            vals.append(r[1 + k] / tot * 100)
        fig.add_trace(go.Bar(y=[r[0] for r in rows], x=vals, name=cat, orientation="h",
                             marker_color=colour, text=[f"{v:.0f}%" for v in vals],
                             textposition="inside",
                             hovertemplate=f"{cat}: %{{x:.1f}}%<extra></extra>"))
    fig.update_layout(barmode="stack")
    fig.update_xaxes(ticksuffix="%", range=[0, 100])
    return _layout(fig, 110 + 50 * len(rows), title)


def grouped_bars(categories: list[str], series: list[tuple], y_title: str, title=None,
                 suffix: str = "") -> go.Figure:
    """series: list of (name, values, colour)."""
    fig = go.Figure()
    for name, vals, colour in series:
        fig.add_trace(go.Bar(x=categories, y=vals, name=name, marker_color=colour,
                             text=[f"{v:.2f}{suffix}" for v in vals], textposition="outside",
                             hovertemplate=f"{name}<br>%{{x}}: %{{y:.2f}}{suffix}<extra></extra>"))
    fig.update_layout(barmode="group", bargap=0.25)
    fig.update_yaxes(title_text=y_title)
    return _layout(fig, 400, title)


def growth_bars(fys, assets, deposits, advances, title=None) -> go.Figure:
    fig = go.Figure()
    for name, vals, colour in [("Assets", assets, NAVY), ("Deposits", deposits, "#94A3B8"),
                               ("Advances", advances, "#2A9D8F")]:
        fig.add_trace(go.Bar(x=fys, y=[v / 1e5 for v in vals], name=name, marker_color=colour,
                             hovertemplate=f"{name} %{{x}}: ₹%{{y:.2f}} lakh Cr<extra></extra>"))
    fig.update_layout(barmode="group")
    fig.update_yaxes(title_text="₹ lakh crore")
    return _layout(fig, 330, title)
