"""OneRun Bank Lab — how banks make money, take risk, and sometimes break.

Run locally:   streamlit run app.py
"""
from __future__ import annotations

import io
import os
import zlib
from dataclasses import asdict, replace

import numpy as np
import pandas as pd
import streamlit as st

import charts as C
import engine as E

APP_NAME = "OneRun Bank Lab"
DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "OneRun_Bank_Lab_Data.xlsx")
MODE_DATA = "A bank from the dataset"
MODE_OWN = "Create your own bank"
OWN_OPTION = "★ Current knobs"

st.set_page_config(page_title=APP_NAME, page_icon="🏦", layout="wide",
                   initial_sidebar_state="expanded")

st.markdown("""
<style>
.block-container {padding-top: 2.2rem; padding-bottom: 3rem;}
.orb-sub {color:#64748B; font-size:0.95rem; margin-top:-0.6rem;}
.orb-context {font-size:0.95rem; color:#1B2430; margin: 0.2rem 0 0.6rem 0;}
.orb-badge {display:inline-block; padding:0.25rem 0.7rem; border-radius:999px; color:white;
            font-weight:600; font-size:0.95rem;}
.orb-verdict {border-radius:12px; padding:1rem 1.2rem; color:white; margin-bottom:0.6rem;}
.orb-verdict h3 {color:white; margin:0 0 0.2rem 0; font-size:1.5rem;}
.orb-verdict p {margin:0; font-size:1rem; opacity:0.95;}
.orb-card {border:1px solid #E5E9F0; border-radius:10px; padding:0.8rem 1rem; height:100%;}
.orb-card b {color:#1F3A5F;}
.orb-small {color:#64748B; font-size:0.85rem;}
div[data-testid="stMetricValue"] {font-size:1.6rem;}
</style>
""", unsafe_allow_html=True)


# =============================================================================== data
@st.cache_data(show_spinner=False)
def load_data():
    x = pd.ExcelFile(DATA_PATH)
    perf = pd.read_excel(x, "BankPerformance")
    net = pd.read_excel(x, "BranchNetwork")
    ma = pd.read_excel(x, "MAScoring")
    mi = pd.read_excel(x, "ModelInputs")
    dic = pd.read_excel(x, "DataDictionary")
    full = perf.merge(net.drop(columns=["Branches"]), on=["FY", "Bank"], how="left") \
               .merge(ma, on=["FY", "Bank"], how="left")
    return perf, net, ma, mi, dic, full


PERF, NET, MA, MI, DICT, FULL = load_data()
BANKS = sorted(MI["Bank"].unique())
FYS = sorted(MI["FY"].unique())
LATEST_FY = FYS[-1]
BANK_TYPE = dict(zip(MI["Bank"], MI["Bank Type"]))


def bank_inputs(bank: str, fy: str) -> E.Inputs:
    row = MI[(MI["Bank"] == bank) & (MI["FY"] == fy)].iloc[0]
    return E.inputs_from_row(row)


def bank_record(bank: str, fy: str) -> pd.Series:
    return FULL[(FULL["Bank"] == bank) & (FULL["FY"] == fy)].iloc[0]


@st.cache_data(show_spinner=False)
def presets() -> dict:
    out = {"Lab default (mid-size bank)": E.Inputs()}
    latest = MI[MI["FY"] == LATEST_FY]
    names = {"PSU": "Typical PSU bank", "Private": "Typical private bank",
             "SFB": "Typical small finance bank"}
    for t, label in names.items():
        d = latest[latest["Bank Type"] == t]
        if d.empty:
            continue
        vals = {f: float(d[c].median()) for c, f in E.MODEL_COLS.items()}
        out[f"{label} (dataset median)"] = E.Inputs(**vals)
    out["Thin-capital aggressive lender"] = E.Inputs(
        eq_pct=5.5, casa_pct=20, td_rate=7.6, loan_yield=13.5, credit_cost=2.5,
        rw_loans=100, opex_pct=2.6, fee_pct=0.6, t2_pct=0.8)
    return out


PRESETS = presets()


# =============================================================================== state
def knob_key(f: str) -> str:
    return f"k_{f}"


KNOB_FIELDS = ["assets"] + list(E.KNOBS.keys())


def source_inputs() -> E.Inputs:
    ss = st.session_state
    if ss.mode == MODE_DATA:
        return bank_inputs(ss.bank, ss.fy)
    return PRESETS[ss.preset]


def load_current():
    """Copy the selected bank (or preset) into the knobs."""
    ss = st.session_state
    i = source_inputs()
    for f in KNOB_FIELDS:
        v = float(getattr(i, f))
        if f in E.KNOBS:
            _, lo, hi, _, _ = E.KNOBS[f]
            v = min(max(v, lo), hi)
        ss[knob_key(f)] = v
    ss.baseline = i


def current_inputs() -> E.Inputs:
    ss = st.session_state
    return E.Inputs(**{f: float(ss[knob_key(f)]) for f in KNOB_FIELDS})


def is_edited(i: E.Inputs, base: E.Inputs) -> bool:
    return any(abs(getattr(i, f) - getattr(base, f)) > 1e-6 for f in KNOB_FIELDS)


def current_name() -> str:
    ss = st.session_state
    if ss.mode == MODE_DATA:
        return ss.bank
    return ss.own_name.strip() or "My Bank"


ss = st.session_state
if "init" not in ss:
    ss.mode = MODE_DATA
    ss.bank = "HDFC Bank" if "HDFC Bank" in BANKS else BANKS[0]
    ss.fy = LATEST_FY
    ss.preset = list(PRESETS)[0]
    ss.own_name = "My Bank"
    ss.scenarios = {}
    load_current()
    ss.init = True


# =============================================================================== sidebar
with st.sidebar:
    st.markdown(f"## 🏦 {APP_NAME}")
    st.radio("Start from", [MODE_DATA, MODE_OWN], key="mode", on_change=load_current)
    if ss.mode == MODE_DATA:
        c1, c2 = st.columns([3, 2])
        c1.selectbox("Bank", BANKS, key="bank", on_change=load_current)
        c2.selectbox("Year", FYS, key="fy", on_change=load_current)
        st.caption(f"{BANK_TYPE[ss.bank]} bank. Knobs are loaded with its {ss.fy} figures; "
                   "move any knob to run a what-if on this bank.")
    else:
        st.text_input("Name your bank", key="own_name", max_chars=40)
        st.selectbox("Starting template", list(PRESETS), key="preset", on_change=load_current)
        st.caption("Pick a template, then shape the bank with the knobs below.")
    st.button("↺ Reset knobs", on_click=load_current, width="stretch",
              help="Reload the selected bank's figures or the template.")

    st.number_input("Total assets (₹ crore)", min_value=1_000.0, max_value=100_000_000.0,
                    step=10_000.0, format="%.0f", key=knob_key("assets"),
                    help="Balance-sheet size. Every ratio scales with it; ₹1 lakh crore = 1,00,000.")
    st.caption(f"= {C.cr(ss[knob_key('assets')])}")
    for g, flds in E.KNOB_GROUPS.items():
        with st.expander(g, expanded=(g == "Costs & risk")):
            for f in flds:
                label, lo, hi, step, hlp = E.KNOBS[f]
                st.slider(label, min_value=float(lo), max_value=float(hi), step=float(step),
                          format="%.2f", key=knob_key(f), help=hlp)
    st.divider()
    st.caption("Educational simulator. Bank names are real; all figures are illustrative sample "
               "data, partly synthetic. Not investment advice.")

I = current_inputs()
R = E.compute(I)
BASE = ss.baseline
RB = E.compute(BASE)
EDITED = is_edited(I, BASE)
NAME = current_name()
CODE = E.verdict_code(R)


# =============================================================================== helpers
def badge(code: int) -> str:
    return (f"<span class='orb-badge' style='background:{E.VERDICT_COLOUR[code]}'>"
            f"{E.VERDICT_LABEL[code]}</span>")


def verdict_box(r: dict, heading_prefix: str = ""):
    code = E.verdict_code(r)
    st.markdown(
        f"<div class='orb-verdict' style='background:{E.VERDICT_COLOUR[code]}'>"
        f"<h3>{heading_prefix}{E.VERDICT_LABEL[code]}</h3><p>{E.verdict_reason(r)}</p>"
        f"<p style='opacity:0.85;font-size:0.9rem'>{E.VERDICT_MEANING[code]}</p></div>",
        unsafe_allow_html=True)


def pct(x, d=2):
    return "–" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{d}f}%"


def delta(a, b, d=2):
    if not EDITED or a is None or b is None or np.isnan(a) or np.isnan(b):
        return None
    diff = a - b
    return None if abs(diff) < 10 ** (-d) / 2 else f"{diff:+.{d}f} pts"


_CHART_N = [0]


def chart(fig):
    _CHART_N[0] += 1   # stable per-run keys avoid duplicate-element errors
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False},
                    key=f"chart_{_CHART_N[0]}")


def money_table(rows: list[tuple], bold: set[str] = frozenset()) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=["Item", "₹ crore"])
    df["₹ crore"] = [("(" + C.inr(-v) + ")") if v < 0 else C.inr(v) for v in df["₹ crore"]]
    df["Item"] = [f"**{x}**" if x in bold else x for x in df["Item"]]
    return df


def md_table(df: pd.DataFrame):
    """Render a small table as markdown (keeps bold totals and right-aligned numbers)."""
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |",
             "| " + " | ".join([":--"] + ["--:"] * (len(cols) - 1)) + " |"]
    for _, r in df.iterrows():
        cells = [str(r[c]) for c in cols]
        if cells[0].startswith("**"):
            cells = [cells[0]] + [f"**{c}**" for c in cells[1:]]
        lines.append("| " + " | ".join(cells) + " |")
    st.markdown("\n".join(lines))


# =============================================================================== header
st.title(APP_NAME)
st.markdown("<div class='orb-sub'>How banks make money, take risk, and sometimes break · "
            "Developed by Prof. Shalini Velappan</div>", unsafe_allow_html=True)

if ss.mode == MODE_DATA:
    ctx = f"**{NAME}** · {BANK_TYPE[NAME]} · {ss.fy} · " + \
          ("*what-if: knobs edited from the dataset*" if EDITED else "dataset figures")
else:
    ctx = f"**{NAME}** · your own bank · built from *{ss.preset}*" + (" · edited" if EDITED else "")
ctx += f" · Total assets {C.cr(R['assets'])}"
st.markdown(f"<div class='orb-context'>{ctx}</div>", unsafe_allow_html=True)
if R["dep_capped"]:
    st.warning("Deposits plus equity exceed 100% of assets, so deposits were capped. "
               "Lower the deposit or equity knob.")

k = st.columns(6)
k[0].metric("NIM", pct(R["nim"]), delta(R["nim"], RB["nim"]), help="Net interest income / total assets")
k[1].metric("ROA", pct(R["roa"]), delta(R["roa"], RB["roa"]), help="Net profit / total assets")
k[2].metric("ROE", pct(R["roe"], 1), delta(R["roe"], RB["roe"], 1), help="Net profit / year-end equity")
k[3].metric("Cost / Income", pct(R["ci"], 1), delta(R["ci"], RB["ci"], 1), delta_color="inverse",
            help="Operating cost / (net interest income + fee income)")
k[4].metric("CRAR", pct(R["crar"], 1), delta(R["crar"], RB["crar"], 1),
            help="(Equity + Tier 2) / risk-weighted assets. RBI minimum 9%; 11.5% with buffer.")
with k[5]:
    st.markdown("<div style='font-size:0.875rem;color:#31333F;margin-bottom:0.35rem'>Health</div>"
                + badge(CODE), unsafe_allow_html=True)
    if EDITED and E.verdict_code(RB) != CODE:
        st.caption(f"was {E.VERDICT_LABEL[E.verdict_code(RB)]}")

tabs = st.tabs([
    ":material/travel_explore: Bank Explorer",
    ":material/account_balance: Business Model",
    ":material/receipt_long: Financials & Ratios",
    ":material/science: Fragility Lab",
    ":material/tune: What-If Lab",
    ":material/compare_arrows: Compare Banks",
    ":material/lightbulb: Intuition Summary",
    ":material/assignment: Student Assignment",
])

# =============================================================================== 1 explorer
with tabs[0]:
    st.subheader("Bank Explorer")
    st.caption("All 20 banks in the sample dataset. Pick a year, sort the table, and see where your bank sits.")
    c1, c2 = st.columns([1, 3])
    ex_fy = c1.selectbox("Year", FYS, index=FYS.index(ss.fy) if ss.mode == MODE_DATA else len(FYS) - 1,
                         key="ex_fy")
    types = c2.multiselect("Bank type", ["PSU", "Private", "SFB"], default=["PSU", "Private", "SFB"],
                           key="ex_types")
    d = FULL[(FULL["FY"] == ex_fy) & (FULL["Bank Type"].isin(types))].copy()
    d["Assets (₹ lakh Cr)"] = d["Assets Cr"] / 1e5
    show = d[["Bank", "Bank Type", "Assets (₹ lakh Cr)", "CASA %", "NIM %", "ROA %", "ROE %", "GNPA %",
              "Cost-Income %", "CRAR %", "LCR %", "Branches", "Customers Mn", "Rank", "Risk Category"]]
    show = show.rename(columns={"Rank": "Scoring rank", "Risk Category": "Risk"})
    pc = st.column_config
    st.dataframe(
        show.sort_values("Assets (₹ lakh Cr)", ascending=False), hide_index=True, width="stretch",
        column_config={
            "Assets (₹ lakh Cr)": pc.NumberColumn(format="%.2f"),
            "CASA %": pc.NumberColumn("CASA", format="%.0f%%"),
            "NIM %": pc.NumberColumn("NIM", format="%.1f%%"),
            "ROA %": pc.NumberColumn("ROA", format="%.2f%%"),
            "ROE %": pc.NumberColumn("ROE", format="%.0f%%"),
            "GNPA %": pc.NumberColumn("GNPA", format="%.1f%%"),
            "Cost-Income %": pc.NumberColumn("Cost/Income", format="%.0f%%"),
            "CRAR %": pc.ProgressColumn("CRAR", format="%.0f%%", min_value=0, max_value=25),
            "LCR %": pc.NumberColumn("LCR", format="%.0f%%"),
            "Branches": pc.NumberColumn(format="%d"),
            "Customers Mn": pc.NumberColumn("Customers (mn)", format="%.1f"),
        })

    st.markdown("#### Peer map")
    metric_cols = ["NIM %", "ROA %", "ROE %", "GNPA %", "NNPA %", "Credit Cost %", "Cost-Income %",
                   "CRAR %", "CET1 %", "CASA %", "Cost of Deposits %", "Yield on Advances %", "LCR %",
                   "PCR %", "Branches", "Customers Mn"]
    s1, s2 = st.columns(2)
    xcol = s1.selectbox("X axis", metric_cols, index=metric_cols.index("GNPA %"), key="ex_x")
    ycol = s2.selectbox("Y axis", metric_cols, index=metric_cols.index("ROA %"), key="ex_y")
    model_map = {"NIM %": R["nim"], "ROA %": R["roa"], "ROE %": R["roe"], "Cost-Income %": R["ci"],
                 "CRAR %": R["crar"], "CET1 %": R["cet1"], "CASA %": I.casa_pct,
                 "Credit Cost %": I.credit_cost, "Yield on Advances %": I.loan_yield,
                 "Cost of Deposits %": R["cost_of_deposits"]}
    extra = None
    if (ss.mode == MODE_OWN or EDITED) and xcol in model_map and ycol in model_map:
        extra = dict(x=model_map[xcol], y=model_map[ycol], name=f"{NAME} (knobs)")
    chart(C.peer_scatter(d, xcol, ycol, highlight=ss.bank if ss.mode == MODE_DATA else None,
                         extra_point=extra))
    st.caption("Bubble size = total assets. The ring marks the bank chosen in the sidebar; "
               "the star shows your knob settings when both axes are model outputs.")

    st.markdown("#### Bank profile")
    prof_bank = ss.bank if ss.mode == MODE_DATA else st.selectbox("Profile of", BANKS, key="ex_prof")
    rec = bank_record(prof_bank, ex_fy)
    hist = FULL[FULL["Bank"] == prof_bank].sort_values("FY")
    a, b, c = st.columns(3)
    with a:
        st.markdown(f"**{prof_bank}** · {rec['Bank Type']} · {ex_fy}")
        chart(C.growth_bars(list(hist["FY"]), list(hist["Assets Cr"]), list(hist["Deposits Cr"]),
                            list(hist["Advances Cr"]), title="Balance-sheet growth"))
    with b:
        st.markdown(f"**Branch network** · {int(rec['Branches']):,} branches in "
                    f"{int(rec['States Covered'])} states")
        chart(C.branch_mix([(prof_bank, rec["Rural Branches"], rec["Semi Urban Branches"],
                             rec["Urban Branches"], rec["Metro Branches"])]))
        m1, m2 = st.columns(2)
        m1.metric("ATMs", f"{int(rec['ATMs']):,}")
        m2.metric("Employees", f"{int(rec['Employees']):,}")
        m1.metric("Customers", f"{rec['Customers Mn']:.1f} mn")
        m2.metric("Deposits / branch", C.cr(rec["Deposits Cr"] / rec["Branches"]))
    with c:
        st.markdown(f"**Scorecard** · rank {int(rec['Rank'])} of {len(BANKS)} · "
                    f"{rec['Risk Category']} risk")
        chart(C.radar([(prof_bank, [rec[s] for s in C.SCORE_COLS], C.NAVY)]))
    with st.expander("Where the numbers come from"):
        st.markdown("Reported columns come straight from the three sample files. Inputs the simulator "
                    "needs but the files do not contain are **derived** from reported ratios or filled "
                    "with **synthetic** assumptions, listed below. With these, the model reproduces each "
                    "bank's reported NIM, ROA, ROE, CET1 and CRAR exactly.")
        st.dataframe(DICT, hide_index=True, width="stretch")
        st.caption("Note: in the sample files each bank's ratios are the same in FY24, FY25 and FY26; "
                   "only balance-sheet size changes across years.")

# =============================================================================== 2 business model
with tabs[1]:
    st.subheader("How the bank is built")
    left, right = st.columns([3, 2])
    with left:
        chart(C.balance_sheet(R, title=f"{NAME}: what it owns vs how it is funded (year end)"))
    with right:
        st.markdown("**Assets**")
        md_table(money_table([("Loans (advances)", R["loans"]),
                              ("Investments & cash", R["investments"]),
                              ("Total assets", R["assets"])], bold={"Total assets"}))
        st.markdown("**Liabilities & equity**")
        borr_end = R["assets"] - R["deposits"] - R["eq_end"]
        md_table(money_table([("CASA deposits", R["casa"]), ("Term deposits", R["td"]),
                              ("Borrowings & other", borr_end), ("Equity (year end)", R["eq_end"]),
                              ("Total liabilities & equity", R["deposits"] + borr_end + R["eq_end"])],
                             bold={"Total liabilities & equity"}))
    lev = R["leverage"]
    eq_loans = R["eq_end"] / R["loans"] * 100 if R["loans"] else np.nan
    spread = I.loan_yield - R["cost_of_deposits"]
    st.info(
        f"**What this tells you**\n\n"
        f"- {NAME} lends at {I.loan_yield:.2f}% and pays depositors {R['cost_of_deposits']:.2f}% on average: "
        f"a gross spread of **{spread:.2f} points**.\n"
        f"- Every ₹1 of equity supports **₹{lev:.1f} of assets** (leverage {lev:.1f}x).\n"
        f"- Equity is **{eq_loans:.1f}% of the loan book**: losing that share of loans in one year would "
        f"wipe out shareholders.\n"
        f"- {I.casa_pct:.0f}% of deposits are cheap CASA; liquidity above the 18% SLR floor is "
        f"{C.cr(R['usable_liquidity'])}, or {R['usable_liquidity'] / R['deposits'] * 100:.1f}% of deposits."
        if lev == lev else "Equity has been wiped out; leverage is undefined.")

# =============================================================================== 3 financials
with tabs[2]:
    st.subheader("Profit & loss and key ratios")
    left, right = st.columns([3, 2])
    with left:
        chart(C.pnl_waterfall(R, title="From interest income to net profit (₹ crore)"))
    with right:
        rows = [("Interest income", R["int_income"]), ("Interest expense", -R["int_expense"]),
                ("Net interest income", R["nii"]), ("Fee & other income", R["fees"]),
                ("Operating cost", -R["opex"]), ("Operating profit (PPOP)", R["ppop"]),
                ("Credit loss provisions", -R["credit_loss"])]
        if abs(R["other_loss"]) > 0.5:
            rows.append(("One-off loss", -R["other_loss"]))
        rows += [("Profit before tax", R["pbt"]), (f"Tax @ {I.tax_rate:.0f}%", -R["tax"]),
                 ("Net profit", R["pat"])]
        md_table(money_table(rows, bold={"Net interest income", "Operating profit (PPOP)",
                                         "Profit before tax", "Net profit"}))
    st.markdown("#### Why ROE looks high: ROE = ROA × leverage")
    d1, d2, d3 = st.columns(3)
    d1.metric("ROA", pct(R["roa"]))
    d2.metric("× Leverage (assets / equity)", f"{R['leverage']:.1f}x" if R["leverage"] == R["leverage"] else "–")
    d3.metric("= ROE", pct(R["roe"], 1))
    st.markdown("#### Ratios against benchmarks")
    bench = pd.DataFrame([
        ("NIM", pct(R["nim"]), "About 3.0–3.5% for Indian banks", "Net interest income / assets"),
        ("ROA", pct(R["roa"]), "1% or more is healthy", "Net profit / assets"),
        ("ROE", pct(R["roe"], 1), "12–18% typical", "Net profit / year-end equity"),
        ("Cost / Income", pct(R["ci"], 1), "Under 50% is efficient", "Operating cost / (NII + fees)"),
        ("Cost of deposits", pct(R["cost_of_deposits"]), "Lower is better", "Interest on deposits / deposits"),
        ("CRAR", pct(R["crar"], 1), "Min 9%; 11.5% with buffer", "(Equity + Tier 2) / RWA"),
        ("CET1", pct(R["cet1"], 1), "Min 5.5%; 8% with buffer", "Equity / RWA"),
    ], columns=["Ratio", NAME, "Benchmark", "Formula"])
    st.dataframe(bench, hide_index=True, width="stretch")
    if ss.mode == MODE_DATA:
        rec = bank_record(ss.bank, ss.fy)
        st.markdown(f"#### Model vs reported · {ss.bank} {ss.fy}")
        comp = pd.DataFrame([
            ("NIM", rec["NIM %"], RB["nim"], R["nim"]),
            ("ROA", rec["ROA %"], RB["roa"], R["roa"]),
            ("ROE", rec["ROE %"], RB["roe"], R["roe"]),
            ("Cost / Income", rec["Cost-Income %"], RB["ci"], R["ci"]),
            ("CRAR", rec["CRAR %"], RB["crar"], R["crar"]),
            ("CET1", rec["CET1 %"], RB["cet1"], R["cet1"]),
            ("Cost of deposits", rec["Cost of Deposits %"], RB["cost_of_deposits"], R["cost_of_deposits"]),
        ], columns=["Ratio", "Reported (%)", "Model at dataset inputs (%)", "Model with your knobs (%)"])
        st.dataframe(comp.round(2), hide_index=True, width="stretch")
        st.caption("Cost/Income can differ from the reported figure where the sample file's ratios are "
                   "not mutually consistent; the model keeps ROA, ROE and capital exact instead.")

# =============================================================================== 4 fragility
with tabs[3]:
    st.subheader("Fragility Lab: break the bank on purpose")
    exp = st.radio("Experiment", ["Credit shock", "Deposit run", "Rate shock"], horizontal=True,
                   key="fr_exp")
    if exp == "Credit shock":
        st.caption("What if far more loans go bad than planned? The stressed credit cost replaces the "
                   "knob value for one year.")
        shock = st.slider("Stressed credit loss (% of loans)", 0.0, 25.0,
                          float(max(5.0, round(I.credit_cost + 4, 1))), 0.1, key="fr_cc")
        S = E.credit_shock(I, shock)
        a, b = st.columns([2, 3])
        with a:
            verdict_box(S, "After the shock: ")
            md_table(pd.DataFrame([
                ("Credit loss", C.inr(R["credit_loss"]), C.inr(S["credit_loss"])),
                ("Net profit", C.inr(R["pat"]), C.inr(S["pat"])),
                ("Year-end equity", C.inr(R["eq_end"]), C.inr(S["eq_end"])),
                ("CRAR", pct(R["crar"], 1), pct(S["crar"], 1)),
            ], columns=["₹ crore / %", "Before", "After"]))
        with b:
            xs = np.linspace(0, 25, 251)
            chart(C.crar_curve([(NAME, xs, E.sweep(I, "credit_cost", xs)["crar"], C.NAVY)],
                               "Credit loss in the year (% of loans)", shock,
                               title="CRAR falls as credit losses rise"))
        be = E.credit_breakevens(I)
        f = lambda v: "beyond 60%" if v is None else ("already, before any shock" if v == 0 else f"{v:.1f}%")  # noqa: E731
        st.markdown(f"**Break-even points for {NAME}:** profit turns negative at **{f(be['profit_zero'])}** "
                    f"credit loss · the buffer (11.5%) is breached at **{f(be['buffer'])}** · the 9% minimum "
                    f"at **{f(be['minimum'])}** · equity is wiped out at **{f(be['wipeout'])}**.")
    elif exp == "Deposit run":
        st.caption("Depositors pull money out. The bank pays from liquid investments above the 18% SLR "
                   "floor; if that runs out it must sell loans at a discount, and the discount is a loss.")
        c1, c2, c3 = st.columns(3)
        cr_run = c1.slider("CASA withdrawn (%)", 0.0, 100.0, 30.0, 1.0, key="fr_casa")
        td_run = c2.slider("Term deposits withdrawn (%)", 0.0, 100.0, 15.0, 1.0, key="fr_td")
        hc = c3.slider("Fire-sale discount on loans (%)", 0.0, 60.0, 20.0, 1.0, key="fr_hc")
        run = E.deposit_run(I, cr_run, td_run, hc)
        S = run["stressed"]
        a, b = st.columns([2, 3])
        with a:
            if run["runs_out"]:
                st.markdown(f"<div class='orb-verdict' style='background:{E.VERDICT_COLOUR[0]}'>"
                            "<h3>After the run: Failing</h3><p>The bank cannot raise enough cash even by "
                            "selling every loan.</p></div>", unsafe_allow_html=True)
            else:
                verdict_box(S, "After the run: ")
            md_table(pd.DataFrame([
                ("Deposits withdrawn", C.inr(run["outflow"])),
                ("Usable liquidity", C.inr(R["usable_liquidity"])),
                ("Shortfall", C.inr(run["shortfall"])),
                ("Loans sold", C.inr(run["loans_sold"])),
                ("Fire-sale loss", C.inr(run["fire_sale_loss"])),
                ("CRAR after", pct(S["crar"], 1)),
            ], columns=["₹ crore", "Value"]))
        with b:
            chart(C.liquidity_bars(R["usable_liquidity"], run["outflow"], run["shortfall"],
                                   title="Can liquid assets cover the run?"))
            cover = run["liquidity_cover"]
            st.caption(f"Liquidity covers {cover:.0f}% of the outflow." if np.isfinite(cover)
                       else "No outflow.")
        st.markdown("**Lesson:** a bank can be solvent on paper and still fail if it must sell good "
                    "loans cheaply to pay depositors. Liquidity buys time; CASA is cheap but can leave fast.")
    else:
        st.caption("Interest rates move for a year. Floating loans and part of the investment book reprice "
                   "up; term deposits and borrowings reprice partly; CASA barely moves. Bonds held at market "
                   "value lose price when rates rise.")
        bp = st.slider("Rate change (basis points)", -300, 500, 250, 25, key="fr_bp")
        with st.expander("Repricing assumptions"):
            e1, e2, e3 = st.columns(3)
            inv_rp = e1.slider("Investments repricing (%)", 0, 100, 30, key="fr_inv")
            td_rp = e2.slider("Term deposits repricing (%)", 0, 100, 40, key="fr_tdrp")
            br_rp = e3.slider("Borrowings repricing (%)", 0, 100, 50, key="fr_brp")
            e4, e5, e6 = st.columns(3)
            casa_p = e4.slider("CASA pass-through (%)", 0, 100, 10, key="fr_casap")
            afs = e5.slider("Investments marked to market (%)", 0, 100, 25, key="fr_afs")
            dur = e6.slider("Bond duration (years)", 0.5, 10.0, 3.5, 0.5, key="fr_dur")
            st.caption(f"Floating-rate loans come from the sidebar knob: {I.floating_pct:.0f}%.")
        kw = dict(inv_reprice=inv_rp, td_reprice=td_rp, borr_reprice=br_rp, casa_pass=casa_p,
                  afs_share=afs, duration=dur)
        rs = E.rate_shock(I, bp, **kw)
        S = rs["stressed"]
        a, b = st.columns([2, 3])
        with a:
            verdict_box(S, "After the move: ")
            md_table(pd.DataFrame([
                ("Change in net interest income", C.inr(rs["nii_change"])),
                ("Bond mark-to-market loss", C.inr(rs["mtm_loss"])),
                ("Net profit before", C.inr(R["pat"])),
                ("Net profit after", C.inr(S["pat"])),
                ("CRAR after", pct(S["crar"], 1)),
            ], columns=["₹ crore", "Value"]))
        with b:
            bps = np.arange(-300, 525, 25)
            crs, dn = [], []
            for x in bps:
                t = E.rate_shock(I, float(x), **kw)
                crs.append(t["stressed"]["crar"])
                dn.append(t["nii_change"])
            chart(C.rate_curves(bps, crs, dn, bp, title="Rates up: margin may rise, bond values fall"))
        st.markdown("**Lesson:** rising rates help a bank whose loans reprice faster than its deposits, "
                    "but losses on long bonds can outweigh that; this mix, plus a deposit run, brought down "
                    "Silicon Valley Bank in 2023 with little credit loss.")

# =============================================================================== 5 what-if
with tabs[4]:
    st.subheader("What-If Lab: when does the bank thrive, and when does it fail?")
    st.caption("Everything here reacts to the sidebar knobs. Drag any knob and watch the verdict, "
               "the safety margins and the map move.")
    a, b = st.columns([3, 2])
    with a:
        verdict_box(R)
    with b:
        st.markdown("**Health ladder** (first match wins)")
        ladder = "\n".join(
            f"- {badge(c)} {m}" for c, _, _, m in E.VERDICTS)
        st.markdown(ladder, unsafe_allow_html=True)

    st.markdown("#### 1 · Distance to failure, one knob at a time")
    rows = []
    for f, (label, lo, hi, _, _) in E.KNOBS.items():
        cur = getattr(I, f)
        fp = E.distance_to_failure(I, f)
        pos = lambda v: (v - lo) / (hi - lo) * 100  # noqa: E731
        side = None if fp is None else ("up" if fp > cur else "down")
        hover = (f"<b>{label}</b><br>now {cur:.2f}" +
                 ("<br>cannot fail the bank alone in this range" if fp is None
                  else f"<br>fails at {fp:.2f} (room {abs(fp - cur):.2f})"))
        rows.append(dict(label=label, cur_pos=pos(cur), fail_pos=None if fp is None else pos(fp),
                         fail_side=side, hover=hover, fp=fp, cur=cur, f=f))
    killers = [r for r in rows if r["fp"] is not None]
    chart(C.distance_chart(rows, title="Green = the bank survives with every other knob unchanged; "
                                       "red = it fails"))
    if killers:
        txt = "; ".join(f"**{r['label'].split(' (')[0]}** at {r['fp']:.2f} (now {r['cur']:.2f})"
                        for r in sorted(killers, key=lambda r: abs(r['fp'] - r['cur'])))
        st.markdown(f"Knobs that can fail {NAME} on their own: {txt}. "
                    f"The other {len(rows) - len(killers)} cannot, which is the point: well-capitalised "
                    "banks are usually broken by several things going wrong together.")
    else:
        st.markdown(f"No single knob can fail {NAME} within its range. Use the map below to combine two.")

    st.markdown("#### 2 · Two-lever failure map")
    knob_labels = {E.KNOBS[f][0]: f for f in E.KNOBS}
    m1, m2, m3 = st.columns([2, 2, 1])
    xl = m1.selectbox("Horizontal axis", list(knob_labels),
                      index=list(knob_labels).index(E.KNOBS["credit_cost"][0]), key="wi_x")
    yl = m2.selectbox("Vertical axis", list(knob_labels),
                      index=list(knob_labels).index(E.KNOBS["eq_pct"][0]), key="wi_y")
    res = m3.select_slider("Detail", [30, 60, 90], value=60, key="wi_res")
    fx, fy = knob_labels[xl], knob_labels[yl]
    if fx == fy:
        st.info("Pick two different knobs.")
    else:
        xs = np.linspace(E.KNOBS[fx][1], E.KNOBS[fx][2], res)
        ys = np.linspace(E.KNOBS[fy][1], E.KNOBS[fy][2], res)
        codes, _ = E.grid2(I, fx, xs, fy, ys)
        chart(C.verdict_heatmap(codes, xs, ys, xl, yl, getattr(I, fx), getattr(I, fy),
                                title=f"Health of {NAME} across {xl.split(' (')[0].lower()} and "
                                      f"{yl.split(' (')[0].lower()}"))
        share = {E.VERDICT_LABEL[c]: float(np.mean(codes == c) * 100) for c in range(6)}
        st.caption(" · ".join(f"{k}: {v:.0f}% of the map" for k, v in share.items() if v > 0))

    st.markdown("#### 3 · Goal seek")
    g1, g2, g3, g4 = st.columns([2, 2, 1, 1])
    gl = g1.selectbox("Move this knob", list(knob_labels),
                      index=list(knob_labels).index(E.KNOBS["credit_cost"][0]), key="gs_knob")
    gm = g2.selectbox("until this result", list(E.METRICS), format_func=E.METRICS.get, key="gs_metric")
    defaults = {"crar": 9.0, "pat": 0.0, "roe": 15.0, "roa": 1.0, "nim": 3.0, "ci": 50.0, "eq_end": 0.0}
    gt = g3.number_input("reaches", value=float(defaults[gm]), key=f"gs_target_{gm}")
    gf = knob_labels[gl]
    _, glo, ghi, _, _ = E.KNOBS[gf]
    ans = E.solve(I, gf, gm, gt, glo, ghi)
    with g4:
        st.markdown("<div style='height:1.8rem'></div>", unsafe_allow_html=True)
        st.metric("Answer", "–" if ans is None else f"{ans:.2f}")
    now_val = float(np.asarray(R[gm]))
    if ans is None:
        st.markdown(f"Within its range ({glo}–{ghi}), moving **{gl}** cannot bring "
                    f"{E.METRICS[gm]} to {gt:g} (now {now_val:,.2f}).")
    else:
        st.markdown(f"Moving **{gl}** from {getattr(I, gf):.2f} to **{ans:.2f}** takes "
                    f"{E.METRICS[gm]} from {now_val:,.2f} to {gt:g}, everything else unchanged.")

    st.markdown("#### 4 · Save and compare scenarios")
    sc = st.columns(4)
    for slot, col in zip("ABC", sc[:3]):
        if col.button(f"Save as {slot}", key=f"save_{slot}", width="stretch"):
            ss.scenarios[slot] = dict(name=NAME, inputs=I)
    if sc[3].button("Clear", key="clear_sc", width="stretch"):
        ss.scenarios = {}
    if ss.scenarios:
        out = {}
        for slot in sorted(ss.scenarios):
            sinp = ss.scenarios[slot]["inputs"]
            sr = E.compute(sinp)
            col = {"Bank": ss.scenarios[slot]["name"],
                   "Verdict": E.VERDICT_LABEL[E.verdict_code(sr)]}
            for f in KNOB_FIELDS:
                col[E.KNOBS[f][0] if f in E.KNOBS else "Total assets (₹ Cr)"] = getattr(sinp, f)
            col.update({"NIM %": sr["nim"], "ROA %": sr["roa"], "ROE %": sr["roe"],
                        "Cost/Income %": sr["ci"], "CRAR %": sr["crar"], "Net profit (₹ Cr)": sr["pat"]})
            out[f"Scenario {slot}"] = col
        tbl = pd.DataFrame(out)
        if len(out) > 1:   # keep rows that differ plus every result row
            num = tbl.iloc[2:].apply(pd.to_numeric, errors="coerce")
            keep = list(tbl.index[:2]) + [ix for ix in num.index
                                          if num.loc[ix].nunique() > 1 or ix.endswith("%") or "₹" in ix]
            tbl = tbl.loc[list(dict.fromkeys(keep))]
        st.dataframe(tbl.map(lambda v: f"{v:,.2f}" if isinstance(v, float) else v), width="stretch")
    else:
        st.caption("Save the current knobs as A, change something, save as B, and compare side by side.")

    st.markdown("#### 5 · A thousand random years")
    st.caption("Each simulated year draws a random credit loss, interest-rate move and deposit outflow "
               "around today's settings. How often does the bank fail?")
    r1, r2 = st.columns([1, 3])
    sev = r1.select_slider("Volatility", ["Calm", "Normal", "Rough", "Crisis"], value="Normal", key="mc_sev")
    sevv = {"Calm": 0.5, "Normal": 1.0, "Rough": 1.6, "Crisis": 2.4}[sev]
    mc = E.simulate_bad_years(I, n=1000, seed=11, severity=sevv)
    r1.metric("Years failing", f"{mc['fail_share']:.1f}%")
    r1.metric("Years in buffer or failing", f"{mc['buffer_share']:.1f}%")
    r1.metric("Loss-making years", f"{mc['loss_share']:.1f}%")
    with r2:
        crar_vals = np.clip(mc["crar"], -10, 40)
        chart(C.histogram(crar_vals, title="Distribution of year-end CRAR across 1,000 simulated years"))

# =============================================================================== 6 compare
with tabs[5]:
    st.subheader("Compare two banks")
    opts = [OWN_OPTION] + BANKS
    ca, cb = st.columns(2)
    with ca:
        st.markdown("**Bank A**")
        a1, a2 = st.columns([3, 2])
        pick_a = a1.selectbox("Bank A", opts, index=0, key="cmp_a", label_visibility="collapsed",
                             format_func=lambda o: f"{o} ({NAME})" if o == OWN_OPTION else o)
        fy_a = a2.selectbox("Year A", FYS, index=len(FYS) - 1, key="cmp_fya",
                            label_visibility="collapsed", disabled=pick_a == opts[0])
    with cb:
        st.markdown("**Bank B**")
        default_b = "SBI" if NAME != "SBI" else "HDFC Bank"
        b1, b2 = st.columns([3, 2])
        pick_b = b1.selectbox("Bank B", opts, index=opts.index(default_b), key="cmp_b",
                              label_visibility="collapsed",
                              format_func=lambda o: f"{o} ({NAME})" if o == OWN_OPTION else o)
        fy_b = b2.selectbox("Year B", FYS, index=len(FYS) - 1, key="cmp_fyb",
                            label_visibility="collapsed", disabled=pick_b == opts[0])

    def resolve(pick, fy):
        if pick == opts[0]:
            rec = bank_record(ss.bank, ss.fy) if (ss.mode == MODE_DATA and not EDITED) else None
            return NAME + (" (knobs)" if EDITED or ss.mode == MODE_OWN else f" {ss.fy}"), I, rec
        return f"{pick} {fy}", bank_inputs(pick, fy), bank_record(pick, fy)

    na, ia, reca = resolve(pick_a, fy_a)
    nb, ib, recb = resolve(pick_b, fy_b)
    ra, rb = E.compute(ia), E.compute(ib)
    h1, h2 = st.columns(2)
    for col, n_, r_, rec_ in [(h1, na, ra, reca), (h2, nb, rb, recb)]:
        typ = rec_["Bank Type"] if rec_ is not None else "Own design"
        col.markdown(f"### {n_}\n{typ} · {C.cr(r_['assets'])} · {badge(E.verdict_code(r_))}",
                     unsafe_allow_html=True)

    def better(metric, x, y):
        hi_good = {"NIM", "ROA", "ROE", "CRAR", "CET1", "CASA", "PCR", "LCR", "Customers (mn)",
                   "Break-even credit loss to 9% CRAR"}
        lo_good = {"Cost / Income", "Credit cost", "GNPA", "NNPA", "Cost of deposits", "Leverage",
                   "Scoring rank"}
        if x is None or y is None or (isinstance(x, float) and np.isnan(x)) or \
                (isinstance(y, float) and np.isnan(y)) or abs(x - y) < 1e-9:
            return ""
        if metric in hi_good:
            return "A" if x > y else "B"
        if metric in lo_good:
            return "A" if x < y else "B"
        return ""

    bea = E.credit_breakevens(ia)["minimum"]
    beb = E.credit_breakevens(ib)["minimum"]
    lines = [
        ("Total assets (₹ lakh Cr)", ra["assets"] / 1e5, rb["assets"] / 1e5, 2),
        ("CASA", ia.casa_pct, ib.casa_pct, 1),
        ("Cost of deposits", ra["cost_of_deposits"], rb["cost_of_deposits"], 2),
        ("NIM", ra["nim"], rb["nim"], 2),
        ("ROA", ra["roa"], rb["roa"], 2),
        ("ROE", ra["roe"], rb["roe"], 1),
        ("Cost / Income", ra["ci"], rb["ci"], 1),
        ("Credit cost", ia.credit_cost, ib.credit_cost, 2),
        ("CRAR", ra["crar"], rb["crar"], 1),
        ("CET1", ra["cet1"], rb["cet1"], 1),
        ("Leverage", ra["leverage"], rb["leverage"], 1),
        ("Break-even credit loss to 9% CRAR", bea, beb, 1),
    ]
    rep = [("GNPA", "GNPA %", 2), ("NNPA", "NNPA %", 2), ("PCR", "PCR %", 0), ("LCR", "LCR %", 0),
           ("Branches", "Branches", 0), ("Customers (mn)", "Customers Mn", 1), ("Scoring rank", "Rank", 0)]
    for lab, colname, dd in rep:
        lines.append((lab, None if reca is None else float(reca[colname]),
                      None if recb is None else float(recb[colname]), dd))
    tbl = pd.DataFrame([{
        "Measure": lab,
        "A": "–" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:,.{dd}f}",
        "B": "–" if y is None or (isinstance(y, float) and np.isnan(y)) else f"{y:,.{dd}f}",
        "Stronger": better(lab, x, y),
    } for lab, x, y, dd in lines]).rename(columns={"A": na, "B": nb})
    t1, t2 = st.columns([2, 3])
    with t1:
        st.dataframe(tbl, hide_index=True, width="stretch", height=38 * (len(tbl) + 1))
        st.caption("Ratios in %. Branch, customer and scoring rows come from the dataset, so they are blank "
                   "for a bank you designed. 'Stronger' is lower for costs, risk and rank.")
    with t2:
        cats = ["NIM", "ROA", "Cost of deposits", "Credit cost"]
        chart(C.grouped_bars(cats, [
            (na, [ra["nim"], ra["roa"], ra["cost_of_deposits"], ia.credit_cost], C.BANK_A),
            (nb, [rb["nim"], rb["roa"], rb["cost_of_deposits"], ib.credit_cost], C.BANK_B)],
            "%", title="Margins and costs (%)", suffix="%"))
        chart(C.grouped_bars(["ROE", "Cost / Income", "CRAR"], [
            (na, [ra["roe"], ra["ci"], ra["crar"]], C.BANK_A),
            (nb, [rb["roe"], rb["ci"], rb["crar"]], C.BANK_B)],
            "%", title="Returns, efficiency and capital (%)", suffix="%"))

    st.markdown("#### Why their returns differ: every ₹100 of assets")
    def per100(r):
        A = r["assets"]
        return {"Net interest income": r["nii"] / A * 100, "Fee income": r["fees"] / A * 100,
                "Operating cost": -r["opex"] / A * 100, "Credit loss": -(r["credit_loss"] + r["other_loss"]) / A * 100,
                "Tax": -r["tax"] / A * 100}
    pa, pb = per100(ra), per100(rb)
    chart(C.grouped_bars(list(pa), [(na, list(pa.values()), C.BANK_A), (nb, list(pb.values()), C.BANK_B)],
                         "₹ per ₹100 of assets", suffix=""))
    diffs = {k: pb[k] - pa[k] for k in pa}
    top = max(diffs, key=lambda k: abs(diffs[k]))
    roa_gap = rb["roa"] - ra["roa"]
    lev_part = (rb["leverage"] - ra["leverage"]) * ra["roa"] if ra["leverage"] == ra["leverage"] else 0
    leader = nb if roa_gap > 0 else na
    st.markdown(
        f"**Reading it:** {leader} earns the higher ROA by **{abs(roa_gap):.2f} points**. The biggest single "
        f"driver is **{top.lower()}**, worth {abs(diffs[top]):.2f} per ₹100 of assets in "
        f"{'favour of ' + nb if diffs[top] > 0 else 'favour of ' + na}. "
        f"ROE differs by {rb['roe'] - ra['roe']:+.1f} points ({nb} minus {na}); of that, leverage "
        f"(B {rb['leverage']:.1f}x vs A {ra['leverage']:.1f}x) accounts for about {lev_part:+.1f} points.")

    st.markdown("#### Who survives a credit shock longer?")
    xs = np.linspace(0, 25, 251)
    chart(C.crar_curve([(na, xs, E.sweep(ia, "credit_cost", xs)["crar"], C.BANK_A),
                        (nb, xs, E.sweep(ib, "credit_cost", xs)["crar"], C.BANK_B)],
                       "Credit loss in the year (% of loans)",
                       title="CRAR after a one-year credit shock"))
    def _be(v):
        return "is already below 9%" if v == 0 else (
            "stays above 9% even at 60%" if v is None else f"falls below 9% at {v:.1f}% credit loss")
    st.caption(f"{na} {_be(bea)}; {nb} {_be(beb)}.")

    if reca is not None or recb is not None:
        st.markdown("#### Franchise: scorecard and branch network")
        f1, f2 = st.columns(2)
        series, mix = [], []
        for n_, rec_, colr in [(na, reca, C.BANK_A), (nb, recb, C.BANK_B)]:
            if rec_ is not None:
                series.append((n_, [rec_[s] for s in C.SCORE_COLS], colr))
                mix.append((n_, rec_["Rural Branches"], rec_["Semi Urban Branches"],
                            rec_["Urban Branches"], rec_["Metro Branches"]))
        with f1:
            chart(C.radar(series, title="Scorecard (0–100)"))
        with f2:
            chart(C.branch_mix(mix, title="Where the branches are"))

# =============================================================================== 7 intuition
with tabs[6]:
    st.subheader("Big-picture intuition, proved with your bank's numbers")
    all_td = replace(I, casa_pct=0.0)
    cod_all_td = E.compute(all_td)["cost_of_deposits"]
    eq_loans = R["eq_end"] / R["loans"] * 100 if R["loans"] else np.nan
    lev_txt = f"{R['leverage']:.1f}x" if R["leverage"] == R["leverage"] else "–"
    cards = [
        ("A bank is a spread + leverage machine",
         f"NIM {R['nim']:.2f}% × leverage {lev_txt}. Small margins, multiplied by borrowed money."),
        ("CASA is gold",
         f"Deposits cost {R['cost_of_deposits']:.2f}% today; with no CASA they would cost {cod_all_td:.2f}%."),
        ("ROE looks high because equity is small",
         f"ROE {pct(R['roe'], 1)} = ROA {pct(R['roa'])} × {lev_txt}."),
        ("That same leverage makes banks fragile",
         f"Equity is only {eq_loans:.1f}% of loans: a bad enough year erases it."),
        ("Credit risk is the usual killer, but runs and rate shocks are faster",
         "Try the Fragility Lab's deposit run and rate shock: liquidity can fail a solvent bank."),
        ("Regulation exists because banks are structurally fragile",
         f"CRAR {pct(R['crar'], 1)} against the 11.5% requirement: "
         f"{R['crar'] - E.CRAR_WITH_CCB:+.1f} points of headroom."),
    ]
    for row in range(0, 6, 3):
        cols = st.columns(3)
        for col, (t, ev) in zip(cols, cards[row:row + 3]):
            col.markdown(f"<div class='orb-card'><b>{t}</b><br><span class='orb-small'>{ev}</span></div>",
                         unsafe_allow_html=True)
        st.write("")
    st.markdown("#### Which knobs move ROE the most?")
    tm = st.radio("Measure", ["ROE", "CRAR", "ROA"], horizontal=True, key="tor_metric")
    tor = E.tornado(I, tm.lower())
    chart(C.tornado_chart(tor, tm, title=f"Change in {tm} when each knob moves 10% of its range"))
    top = tor[-1]
    st.caption(f"Biggest lever for {tm}: {top[0]} (±{top[3]:.2f} moves {tm} by "
               f"{top[1]:+.2f} / {top[2]:+.2f} points).")

# =============================================================================== 8 assignment
def _secret(name: str):
    try:
        return st.secrets.get(name)
    except Exception:
        return None


COURSE_KEY = _secret("COURSE_KEY") or "onerun-default"


def assignment_for(roll: str) -> dict:
    seed = zlib.crc32((roll.strip().upper() + "|" + COURSE_KEY).encode())
    bank = BANKS[seed % len(BANKS)]
    fy = FYS[(seed // len(BANKS)) % len(FYS)]
    shock = 4.0 + (seed // 997) % 7          # 4% to 10% credit loss
    i = bank_inputs(bank, fy)
    r = E.compute(i)
    be = E.credit_breakevens(i)["minimum"]
    s = E.credit_shock(i, shock)
    answers = {
        "Q1 Net interest income (₹ Cr)": r["nii"],
        "Q2 ROE (%)": r["roe"],
        "Q3 Cost / Income (%)": r["ci"],
        "Q4 Credit loss that takes CRAR to 9% (% of loans)": be,
        f"Q5 CRAR after a {shock:.0f}% credit loss (%)": s["crar"],
    }
    return dict(roll=roll, bank=bank, fy=fy, shock=shock, inputs=i, answers=answers)


def _close(given, correct):
    if given is None or correct is None:
        return False
    tol = max(abs(correct) * 0.02, 1.0 if abs(correct) > 100 else 0.1)
    return abs(given - correct) <= tol


with tabs[7]:
    st.subheader("Student Assignment")
    st.caption("Enter your roll number. You get your own bank, year and stress scenario; the same roll "
               "number always gets the same case.")
    roll = st.text_input("Roll number", key="as_roll", max_chars=30)
    if roll.strip():
        case = assignment_for(roll)
        ci_ = case["inputs"]
        st.markdown(f"**Your case:** {case['bank']} · {case['fy']} · credit shock of {case['shock']:.0f}% "
                    "of loans. Load it in the sidebar (dataset mode) to explore, then answer below.")
        with st.expander("Your bank's inputs"):
            st.dataframe(pd.DataFrame(
                [(E.KNOBS[f][0] if f in E.KNOBS else "Total assets (₹ Cr)", round(getattr(ci_, f), 4))
                 for f in KNOB_FIELDS], columns=["Input", "Value"]), hide_index=True, width="stretch")
        with st.form("as_form"):
            given = {}
            for q in case["answers"]:
                given[q] = st.number_input(q, value=None, format="%.2f", key=f"as_{q[:2]}")
            reflection = st.text_area("Q6 Reflection: which lever would you pull to restore CRAR after "
                                      "the shock, and what would it cost the bank?", key="as_ref")
            submitted = st.form_submit_button("Check my answers")
        if submitted:
            results = []
            for q, correct in case["answers"].items():
                ok = _close(given[q], correct)
                results.append(dict(Question=q, Your_answer=given[q], Correct=round(correct, 2),
                                    Result="✓" if ok else "✗"))
            score = sum(r["Result"] == "✓" for r in results)
            st.metric("Score", f"{score} / {len(results)}")
            st.dataframe(pd.DataFrame(results), hide_index=True, width="stretch")
            buf = io.BytesIO()
            with pd.ExcelWriter(buf, engine="openpyxl") as xw:
                pd.DataFrame([dict(Roll=roll, Bank=case["bank"], FY=case["fy"],
                                   Shock_pct=case["shock"], Score=score,
                                   Submitted=pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"))]) \
                    .to_excel(xw, sheet_name="Summary", index=False)
                pd.DataFrame(results).to_excel(xw, sheet_name="Answers", index=False)
                pd.DataFrame([dict(Reflection=reflection)]).to_excel(xw, sheet_name="Reflection", index=False)
            st.download_button("Download my submission (.xlsx)", buf.getvalue(),
                               file_name=f"OneRun_{roll.strip()}_submission.xlsx",
                               mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    st.divider()
    with st.expander("Instructor"):
        key = _secret("INSTRUCTOR_KEY")
        if not key:
            st.caption("Set INSTRUCTOR_KEY (and optionally COURSE_KEY) in the app's secrets to enable the "
                       "answer key. On Streamlit Cloud: app settings → Secrets.")
        else:
            pw = st.text_input("Instructor key", type="password", key="as_pw")
            if pw == key:
                rolls = st.text_area("Roll numbers, one per line", key="as_rolls")
                if rolls.strip():
                    keyrows = []
                    for rl in [x for x in rolls.splitlines() if x.strip()]:
                        cs = assignment_for(rl)
                        keyrows.append(dict(Roll=rl.strip(), Bank=cs["bank"], FY=cs["fy"],
                                            Shock_pct=cs["shock"],
                                            **{q: round(v, 2) for q, v in cs["answers"].items()}))
                    kdf = pd.DataFrame(keyrows)
                    st.dataframe(kdf, hide_index=True, width="stretch")
                    st.download_button("Download answer key (.csv)", kdf.to_csv(index=False).encode(),
                                       file_name="OneRun_answer_key.csv", mime="text/csv")
            elif pw:
                st.error("Wrong key.")
