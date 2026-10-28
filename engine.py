"""OneRun Bank Lab calculation engine.

Pure functions only (no Streamlit), so the app, the tests and any notebook use the
same numbers. Every function works on scalars and, where noted, on NumPy arrays so
grids of scenarios can be evaluated in one call.

Units: money in Rs crore, rates and ratios in percent unless named *_frac.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields, replace

import numpy as np

# Regulatory reference points (RBI Basel III)
CRAR_MIN = 9.0          # minimum total capital, % of RWA
CRAR_WITH_CCB = 11.5    # minimum + 2.5% capital conservation buffer
SLR = 18.0              # statutory liquidity ratio, % of deposits (encumbered)


@dataclass(frozen=True)
class Inputs:
    assets: float = 100_000.0          # Rs crore
    adv_pct: float = 70.0              # loans, % of assets
    dep_pct: float = 85.0              # deposits, % of assets
    eq_pct: float = 8.0                # opening equity, % of assets
    casa_pct: float = 40.0             # CASA, % of deposits
    casa_rate: float = 2.75
    td_rate: float = 6.75
    borr_rate: float = 7.25
    loan_yield: float = 9.25
    inv_yield: float = 7.0
    fee_pct: float = 0.8               # fee & other income, % of assets
    opex_pct: float = 2.0              # operating cost, % of assets
    credit_cost: float = 1.0           # provisions for credit loss, % of loans
    rw_loans: float = 100.0            # risk weight on loans, %
    t2_pct: float = 1.2                # tier 2 capital, % of assets
    floating_pct: float = 60.0         # loans repricing within a year, %
    tax_rate: float = 25.0
    other_loss: float = 0.0            # one-off loss in Rs crore (stress tests)


INPUT_FIELDS = [f.name for f in fields(Inputs)]

# Slider metadata: label, min, max, step, help. Used by the app and by the
# distance-to-failure / tornado analyses so ranges live in one place.
KNOBS = {
    "adv_pct": ("Loans (% of assets)", 40.0, 85.0, 0.5, "Share of the balance sheet lent out. The rest sits in investments and cash."),
    "dep_pct": ("Deposits (% of assets)", 50.0, 92.0, 0.5, "Share of funding from depositors. The rest is borrowings and equity."),
    "eq_pct": ("Opening equity (% of assets)", 3.0, 20.0, 0.1, "Shareholders' capital at the start of the year. Thin equity = high leverage."),
    "casa_pct": ("CASA (% of deposits)", 10.0, 60.0, 0.5, "Current and savings deposits: cheap, but can leave quickly."),
    "casa_rate": ("CASA rate (%)", 0.0, 5.0, 0.05, "Interest paid on current and savings accounts."),
    "td_rate": ("Term deposit rate (%)", 3.0, 10.0, 0.05, "Interest paid on fixed deposits."),
    "borr_rate": ("Borrowing rate (%)", 4.0, 11.0, 0.05, "Cost of bonds, refinance and interbank borrowing."),
    "loan_yield": ("Loan yield (%)", 6.0, 18.0, 0.05, "Average interest earned on loans."),
    "inv_yield": ("Investment yield (%)", 3.0, 12.0, 0.05, "Average return on government securities and other investments."),
    "fee_pct": ("Fee income (% of assets)", 0.0, 3.0, 0.05, "Fees, commissions, treasury and other income."),
    "opex_pct": ("Operating cost (% of assets)", 0.5, 6.0, 0.05, "Staff, branches, technology and other running costs."),
    "credit_cost": ("Credit cost / NPA provisions (% of loans)", 0.0, 15.0, 0.05, "Loans written off or provided for this year."),
    "rw_loans": ("Loan risk weight (%)", 40.0, 175.0, 1.0, "How risky the regulator treats the loan book. Higher = more capital needed."),
    "t2_pct": ("Tier 2 capital (% of assets)", 0.0, 4.0, 0.05, "Subordinated debt and general provisions counted as capital."),
    "floating_pct": ("Floating-rate loans (%)", 0.0, 100.0, 1.0, "Loans whose rate resets within a year (used in the rate shock)."),
    "tax_rate": ("Tax rate (%)", 0.0, 40.0, 0.5, "Tax on profit before tax (no tax credit on losses)."),
}

KNOB_GROUPS = {
    "Size & mix": ["adv_pct", "dep_pct", "eq_pct", "rw_loans", "t2_pct"],
    "Funding": ["casa_pct", "casa_rate", "td_rate", "borr_rate"],
    "Earnings": ["loan_yield", "inv_yield", "fee_pct", "floating_pct"],
    "Costs & risk": ["opex_pct", "credit_cost", "tax_rate"],
}

# Inputs that make the bank WORSE as they rise (used to orient sensitivity charts)
WORSE_WHEN_UP = {"casa_rate", "td_rate", "borr_rate", "opex_pct", "credit_cost",
                 "rw_loans", "tax_rate", "dep_pct"}


def compute(i: Inputs) -> dict:
    """Balance sheet, P&L and ratios. Works element-wise if any field is an array."""
    A = np.asarray(i.assets, dtype=float)
    loans = A * i.adv_pct / 100
    inv = A - loans
    dep_pct = np.minimum(i.dep_pct, 100 - i.eq_pct)          # funding cannot exceed assets
    deposits = A * dep_pct / 100
    casa = deposits * i.casa_pct / 100
    td = deposits - casa
    eq_open = A * i.eq_pct / 100
    borr = np.maximum(A - deposits - eq_open, 0.0)

    ii_loans = loans * i.loan_yield / 100
    ii_inv = inv * i.inv_yield / 100
    int_income = ii_loans + ii_inv
    ie_casa = casa * i.casa_rate / 100
    ie_td = td * i.td_rate / 100
    ie_borr = borr * i.borr_rate / 100
    int_expense = ie_casa + ie_td + ie_borr
    nii = int_income - int_expense
    fees = A * i.fee_pct / 100
    total_income = nii + fees
    opex = A * i.opex_pct / 100
    ppop = total_income - opex
    credit_loss = loans * i.credit_cost / 100
    pbt = ppop - credit_loss - i.other_loss
    tax = np.maximum(pbt, 0) * i.tax_rate / 100
    pat = pbt - tax

    eq_end = eq_open + pat                      # all profit retained
    t2 = A * i.t2_pct / 100
    capital = eq_end + t2
    rwa = loans * i.rw_loans / 100
    with np.errstate(divide="ignore", invalid="ignore"):
        crar = np.where(rwa > 0, capital / rwa * 100, np.nan)
        cet1 = np.where(rwa > 0, eq_end / rwa * 100, np.nan)
        roe = np.where(eq_end > 0, pat / eq_end * 100, np.nan)
        ci = np.where(total_income > 0, opex / total_income * 100, np.nan)
        leverage = np.where(eq_end > 0, A / eq_end, np.nan)
        cod = np.where(deposits > 0, (ie_casa + ie_td) / deposits * 100, np.nan)
    usable_liquidity = np.maximum(inv - deposits * SLR / 100, 0.0)

    out = dict(
        assets=A, loans=loans, investments=inv, deposits=deposits, casa=casa, td=td,
        borrowings_open=borr, borrowings_end=A - deposits - eq_end, eq_open=eq_open, eq_end=eq_end,
        int_income=int_income, ii_loans=ii_loans, ii_inv=ii_inv,
        int_expense=int_expense, ie_casa=ie_casa, ie_td=ie_td, ie_borr=ie_borr,
        nii=nii, fees=fees, total_income=total_income, opex=opex, ppop=ppop,
        credit_loss=credit_loss, other_loss=np.asarray(i.other_loss, dtype=float) + 0 * A,
        pbt=pbt, tax=tax, pat=pat, t2=t2, capital=capital, rwa=rwa,
        nim=nii / A * 100, roa=pat / A * 100, roe=roe, ci=ci, crar=crar, cet1=cet1,
        leverage=leverage, cost_of_deposits=cod, usable_liquidity=usable_liquidity,
        dep_capped=np.asarray(i.dep_pct > 100 - i.eq_pct),
    )
    return {k: (v.item() if isinstance(v, np.ndarray) and v.ndim == 0 else v) for k, v in out.items()}


# --------------------------------------------------------------------------- verdict
VERDICTS = [  # code, label, colour, one-line meaning
    (0, "Failing", "#C0392B", "Capital is below the 9% regulatory minimum or wiped out."),
    (1, "In the buffer", "#E67E22", "Legal, but inside the 2.5% conservation buffer: dividends restricted."),
    (2, "Loss-making", "#D4A017", "Capital is fine for now, but the bank lost money this year."),
    (3, "Adequate", "#7F8C8D", "Profitable, but return on assets is under 1%."),
    (4, "Healthy", "#2E8B57", "Profitable with ROA of 1% or more and capital above the buffer."),
    (5, "Thriving", "#1B6E4B", "ROE 15%+, ROA 1%+, cost/income under 50%, capital comfortable."),
]
VERDICT_LABEL = {c: l for c, l, _, _ in VERDICTS}
VERDICT_COLOUR = {c: col for c, _, col, _ in VERDICTS}
VERDICT_MEANING = {c: m for c, _, _, m in VERDICTS}


def verdict_code(r: dict):
    """Element-wise verdict code (see VERDICTS)."""
    crar = np.nan_to_num(np.asarray(r["crar"], dtype=float), nan=-1.0)
    eq = np.asarray(r["eq_end"], dtype=float)
    pat = np.asarray(r["pat"], dtype=float)
    roa = np.asarray(r["roa"], dtype=float)
    roe = np.nan_to_num(np.asarray(r["roe"], dtype=float), nan=-1.0)
    ci = np.nan_to_num(np.asarray(r["ci"], dtype=float), nan=999.0)
    code = np.select(
        [(crar < CRAR_MIN) | (eq <= 0),
         crar < CRAR_WITH_CCB,
         pat < 0,
         (roe >= 15) & (roa >= 1) & (ci < 50),
         roa >= 1],
        [0, 1, 2, 5, 4],
        default=3,
    )
    return code.item() if code.ndim == 0 else code


def verdict_reason(r: dict) -> str:
    code = verdict_code(r)
    crar, pat = r["crar"], r["pat"]
    if code == 0:
        if r["eq_end"] <= 0:
            return "Losses have wiped out all shareholders' equity."
        return f"CRAR {crar:.1f}% is below the {CRAR_MIN:.0f}% regulatory minimum."
    if code == 1:
        return f"CRAR {crar:.1f}% is inside the buffer ({CRAR_MIN:.0f}% to {CRAR_WITH_CCB}%)."
    if code == 2:
        return f"The bank lost Rs {abs(pat):,.0f} Cr this year; CRAR is still {crar:.1f}%."
    head = f"CRAR {crar:.1f}% is {crar - CRAR_WITH_CCB:.1f} points above the {CRAR_WITH_CCB}% requirement"
    if code == 5:
        return f"{head}, with ROE {r['roe']:.1f}% and cost/income {r['ci']:.0f}%."
    return f"{head}; ROA is {r['roa']:.2f}%."


# --------------------------------------------------------------------------- grids
def with_values(i: Inputs, **arrays) -> Inputs:
    """Inputs with some fields replaced (scalars or broadcastable arrays)."""
    return replace(i, **arrays)


def _broadcast(r: dict, shape) -> dict:
    """Some knobs (e.g. floating-rate share) do not move every output; make every
    output the full grid shape so callers can index it safely."""
    return {k: np.broadcast_to(np.asarray(v), shape).copy() for k, v in r.items()}


def sweep(i: Inputs, field: str, values) -> dict:
    xs = np.asarray(values, dtype=float)
    return _broadcast(compute(with_values(i, **{field: xs})), xs.shape)


def grid2(i: Inputs, fx: str, xs, fy: str, ys) -> tuple[np.ndarray, dict]:
    X, Y = np.meshgrid(np.asarray(xs, float), np.asarray(ys, float))
    r = _broadcast(compute(with_values(i, **{fx: X, fy: Y})), X.shape)
    return np.asarray(verdict_code(r)), r


# --------------------------------------------------------------------------- solver
METRICS = {
    "crar": "CRAR (%)",
    "pat": "Net profit (Rs Cr)",
    "roe": "ROE (%)",
    "roa": "ROA (%)",
    "nim": "NIM (%)",
    "ci": "Cost/Income (%)",
    "eq_end": "Year-end equity (Rs Cr)",
}


def solve(i: Inputs, field: str, metric: str, target: float, lo: float, hi: float,
          n: int = 2001):
    """Value of `field` in [lo, hi] at which `metric` first crosses `target`,
    searching outward from the current value. Returns None if no crossing."""
    xs = np.linspace(lo, hi, n)
    ys = np.asarray(sweep(i, field, xs)[metric], dtype=float)
    cur = float(getattr(i, field))
    s = np.sign(np.nan_to_num(ys - target, nan=-1.0))
    idx = np.where(s[:-1] * s[1:] <= 0)[0]
    idx = idx[s[idx] != s[idx + 1]] if len(idx) else idx
    if len(idx) == 0:
        return None
    # linear interpolation for each crossing; pick the one closest to current value
    roots = []
    for k in idx:
        y0, y1 = ys[k] - target, ys[k + 1] - target
        t = 0.0 if y1 == y0 else y0 / (y0 - y1)
        roots.append(xs[k] + t * (xs[k + 1] - xs[k]))
    roots = np.array(roots)
    return float(roots[np.argmin(np.abs(roots - cur))])


def credit_breakevens(i: Inputs, hi: float = 60.0) -> dict:
    """Credit cost (% of loans) at which key thresholds are crossed.
    0.0 means the threshold is already breached with zero credit loss; None means
    it is not reached even at `hi`."""
    zero = compute(replace(i, credit_cost=0.0))
    out = {}
    for name, metric, target in [("profit_zero", "pat", 0.0), ("buffer", "crar", CRAR_WITH_CCB),
                                 ("minimum", "crar", CRAR_MIN), ("wipeout", "eq_end", 0.0)]:
        if float(np.nan_to_num(zero[metric], nan=-1e18)) < target:
            out[name] = 0.0
        else:
            out[name] = solve(i, "credit_cost", metric, target, 0.0, hi)
    return out


def distance_to_failure(i: Inputs, field: str, n: int = 801):
    """Closest value of `field` within its slider range where the verdict becomes
    Failing, holding everything else fixed. None if it cannot fail alone."""
    _, lo, hi, _, _ = KNOBS[field]
    lo, hi = min(lo, getattr(i, field)), max(hi, getattr(i, field))
    xs = np.linspace(lo, hi, n)
    codes = np.asarray(verdict_code(sweep(i, field, xs)))
    fail = xs[codes == 0]
    if fail.size == 0:
        return None
    cur = float(getattr(i, field))
    return float(fail[np.argmin(np.abs(fail - cur))])


def tornado(i: Inputs, metric: str = "roe", frac: float = 0.10) -> list[tuple]:
    """Change in `metric` when each knob moves down / up by `frac` of its range."""
    base = float(compute(i)[metric])
    rows = []
    for f, (label, lo, hi, _, _) in KNOBS.items():
        if f in ("tax_rate", "floating_pct", "t2_pct"):
            continue
        step = (hi - lo) * frac
        cur = getattr(i, f)
        down = float(compute(replace(i, **{f: max(lo, cur - step)}))[metric]) - base
        up = float(compute(replace(i, **{f: min(hi, cur + step)}))[metric]) - base
        rows.append((label, down, up, step))
    rows.sort(key=lambda r: max(abs(r[1]), abs(r[2])))
    return rows


# --------------------------------------------------------------------------- stress tests
def credit_shock(i: Inputs, stressed_cost: float) -> dict:
    return compute(replace(i, credit_cost=stressed_cost))


def deposit_run(i: Inputs, casa_run: float, td_run: float, haircut: float = 20.0) -> dict:
    """Depositors withdraw a share of CASA and term deposits. Usable liquidity
    (investments above the SLR floor) pays first; any shortfall forces loan sales
    at a haircut, and the haircut loss hits capital."""
    base = compute(i)
    outflow = base["casa"] * casa_run / 100 + base["td"] * td_run / 100
    shortfall = max(outflow - base["usable_liquidity"], 0.0)
    h = haircut / 100
    loans_sold = shortfall / (1 - h) if h < 1 else float("inf")
    fire_sale_loss = loans_sold * h
    insolvent_liquidity = loans_sold > base["loans"]
    stressed = compute(replace(i, other_loss=i.other_loss + fire_sale_loss))
    return dict(base=base, stressed=stressed, outflow=outflow, shortfall=shortfall,
                loans_sold=loans_sold, fire_sale_loss=fire_sale_loss,
                liquidity_cover=(base["usable_liquidity"] / outflow * 100) if outflow > 0 else np.inf,
                runs_out=insolvent_liquidity)


def rate_shock(i: Inputs, bp: float, inv_reprice: float = 30.0, td_reprice: float = 40.0,
               borr_reprice: float = 50.0, casa_pass: float = 10.0,
               afs_share: float = 25.0, duration: float = 3.5) -> dict:
    """Parallel rate move of `bp` basis points over one year.
    Yields and costs reprice in proportion to the share that resets within a year;
    marked-to-market investments lose value when rates rise (gain when they fall)."""
    d = bp / 100
    shocked = replace(
        i,
        loan_yield=i.loan_yield + d * i.floating_pct / 100,
        inv_yield=i.inv_yield + d * inv_reprice / 100,
        td_rate=i.td_rate + d * td_reprice / 100,
        borr_rate=i.borr_rate + d * borr_reprice / 100,
        casa_rate=np.maximum(i.casa_rate + d * casa_pass / 100, 0.0),
    )
    base = compute(i)
    mtm = base["investments"] * afs_share / 100 * duration * d / 100   # loss if bp > 0
    stressed = compute(replace(shocked, other_loss=i.other_loss + mtm))
    return dict(base=base, stressed=stressed, mtm_loss=mtm,
                nii_change=stressed["nii"] - base["nii"])


def simulate_bad_years(i: Inputs, n: int = 1000, seed: int = 7, severity: float = 1.0,
                       haircut: float = 20.0) -> dict:
    """Monte Carlo of one year with random credit, rate and deposit-run shocks.
    severity scales every shock (1 = normal year volatility, 2 = crisis-like)."""
    rng = np.random.default_rng(seed)
    cc = i.credit_cost * np.exp(rng.normal(0.0, 0.6 * severity, n))      # fat right tail
    bp = rng.normal(0.0, 125.0 * severity, n)
    casa_run = np.clip(np.abs(rng.normal(0.0, 8.0 * severity, n)), 0, 100)
    td_run = np.clip(np.abs(rng.normal(0.0, 4.0 * severity, n)), 0, 100)
    d = bp / 100
    shocked = replace(
        i, credit_cost=cc,
        loan_yield=i.loan_yield + d * i.floating_pct / 100,
        inv_yield=i.inv_yield + d * 0.30,
        td_rate=i.td_rate + d * 0.40,
        borr_rate=i.borr_rate + d * 0.50,
        casa_rate=np.maximum(i.casa_rate + d * 0.10, 0.0),
    )
    base = compute(i)
    mtm = base["investments"] * 0.25 * 3.5 * d / 100
    outflow = base["casa"] * casa_run / 100 + base["td"] * td_run / 100
    shortfall = np.maximum(outflow - base["usable_liquidity"], 0.0)
    h = haircut / 100
    sold = shortfall / (1 - h)
    r = compute(replace(shocked, other_loss=i.other_loss + mtm + sold * h))
    codes = np.asarray(verdict_code(r))
    codes = np.where(sold > base["loans"], 0, codes)
    return dict(codes=codes, crar=np.asarray(r["crar"]), pat=np.asarray(r["pat"]),
                credit_cost=cc, bp=bp, casa_run=casa_run, td_run=td_run,
                fail_share=float(np.mean(codes == 0) * 100),
                buffer_share=float(np.mean(codes <= 1) * 100),
                loss_share=float(np.mean(np.asarray(r["pat"]) < 0) * 100))


# --------------------------------------------------------------------------- data helpers
MODEL_COLS = {  # ModelInputs sheet column -> Inputs field
    "Assets Cr": "assets", "Advances % Assets": "adv_pct", "Deposits % Assets": "dep_pct",
    "Opening Equity % Assets": "eq_pct", "CASA %": "casa_pct", "CASA Rate %": "casa_rate",
    "TD Rate %": "td_rate", "Borrowing Rate %": "borr_rate", "Loan Yield %": "loan_yield",
    "Investment Yield %": "inv_yield", "Fee Income % Assets": "fee_pct",
    "Opex % Assets": "opex_pct", "Credit Cost %": "credit_cost",
    "Loan Risk Weight %": "rw_loans", "Tier 2 % Assets": "t2_pct",
    "Floating Loan Share %": "floating_pct", "Tax Rate %": "tax_rate",
}


def inputs_from_row(row) -> Inputs:
    return Inputs(**{f: float(row[c]) for c, f in MODEL_COLS.items()})


def inputs_to_dict(i: Inputs) -> dict:
    return asdict(i)
