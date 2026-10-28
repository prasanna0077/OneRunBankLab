"""Run with:  python -m pytest -q   (or  python tests/test_engine.py)"""
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import engine as E  # noqa: E402

DATA = os.path.join(ROOT, "data", "OneRun_Bank_Lab_Data.xlsx")


def _rows():
    mi = pd.read_excel(DATA, sheet_name="ModelInputs")
    perf = pd.read_excel(DATA, sheet_name="BankPerformance")
    return mi.merge(perf, on=["FY", "Bank", "Bank Type"], suffixes=("", "_rep"))


def test_model_reproduces_reported_ratios():
    for _, row in _rows().iterrows():
        r = E.compute(E.inputs_from_row(row))
        name = f"{row.Bank} {row.FY}"
        assert abs(r["nim"] - row["NIM %"]) < 1e-6, name
        assert abs(r["roa"] - row["ROA %"]) < 1e-6, name
        assert abs(r["roe"] - row["ROE %"]) < 1e-6, name
        assert abs(r["cet1"] - row["CET1 %"]) < 1e-6, name
        assert abs(r["crar"] - row["CRAR %"]) < 1e-6, name
        assert abs(r["cost_of_deposits"] - row["Cost of Deposits %"]) < 1e-6, name
        if row["Fee Source"] == "Derived":
            assert abs(r["ci"] - row["Cost-Income %"]) < 1e-6, name


def test_balance_sheet_balances():
    r = E.compute(E.Inputs())
    assert abs(r["loans"] + r["investments"] - r["assets"]) < 1e-6
    assert abs(r["deposits"] + r["borrowings_end"] + r["eq_end"] - r["assets"]) < 1e-6


def test_vectorised_matches_scalar():
    i = E.Inputs()
    xs = np.array([0.0, 2.0, 5.0, 9.0])
    vec = E.sweep(i, "credit_cost", xs)
    for k, x in enumerate(xs):
        s = E.compute(E.Inputs(credit_cost=x))
        assert abs(vec["crar"][k] - s["crar"]) < 1e-9
        assert E.verdict_code(s) == E.verdict_code(vec)[k]


def test_breakevens_are_ordered_and_correct():
    i = E.Inputs()
    b = E.credit_breakevens(i)
    assert b["profit_zero"] < b["buffer"] < b["minimum"] < b["wipeout"]
    assert abs(E.credit_shock(i, b["minimum"])["crar"] - E.CRAR_MIN) < 0.01
    assert abs(E.credit_shock(i, b["wipeout"])["eq_end"]) < 5


def test_stresses_hurt():
    i = E.Inputs()
    base = E.compute(i)
    assert E.credit_shock(i, 6)["crar"] < base["crar"]
    run = E.deposit_run(i, 60, 40)
    assert run["shortfall"] > 0 and run["stressed"]["crar"] < base["crar"]
    up = E.rate_shock(i, 300)
    assert up["mtm_loss"] > 0


def test_distance_to_failure_credit_cost():
    i = E.Inputs()
    d = E.distance_to_failure(i, "credit_cost")
    assert d is not None
    assert abs(d - E.credit_breakevens(i)["minimum"]) < 0.05


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
