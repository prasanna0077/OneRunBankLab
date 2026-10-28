# OneRun Bank Lab

How banks make money, take risk, and sometimes break. An educational Streamlit simulator
built on a sample dataset of 20 Indian banks (FY24–FY26). Developed by Prof. Shalini Velappan.

## Run it

```bash
pip install -r requirements.txt
streamlit run app.py
```

Deploy on Streamlit Community Cloud by pushing this folder to GitHub and pointing the app at
`app.py`. Keep the `data/` folder alongside it.

## What's inside

| Tab | What students do |
| --- | --- |
| Bank Explorer | Browse all 20 banks by year, sort the table, plot any two metrics on a peer map, see a bank's growth, branch network and scorecard |
| Business Model | See the balance sheet (what the bank owns vs how it is funded) with totals and live interpretation |
| Financials & Ratios | P&L waterfall, ROE = ROA × leverage, ratios vs benchmarks, model vs reported figures |
| Fragility Lab | Credit shock, deposit run and interest-rate shock, each with a verdict and break-even points |
| What-If Lab | Health verdict, distance to failure for every knob, two-lever failure map, goal seek, saved scenarios, 1,000 random years |
| Compare Banks | Any two banks (or your own design) side by side: ratios, every ₹100 of assets, shock resilience, scorecard, branches |
| Intuition Summary | Six big ideas, each proved with the current bank's numbers, plus which knobs move ROE most |
| Student Assignment | Roll number → a unique bank, year and shock; five graded questions; downloadable submission |

**Sidebar:** start from *a bank from the dataset* (pick bank and year; the knobs load its figures)
or *create your own bank* (pick a template, name it, shape it with the knobs). Moving any knob
turns a real bank into a what-if; *Reset knobs* reloads it.

## Data

`data/OneRun_Bank_Lab_Data.xlsx`

- `BankPerformance`, `BranchNetwork`, `MAScoring`: the three sample files as supplied.
- `ModelInputs`: the simulator's inputs for each bank and year. Values the files do not contain are
  **derived** from reported ratios (green cells) or **synthetic** assumptions (yellow cells).
  With them the model reproduces every bank's reported NIM, ROA, ROE, CET1, CRAR and cost of deposits.
- `DataDictionary`: how each input was built.

Bank names are real; every figure is illustrative sample data. Not investment advice.

## Model in one paragraph

Loans and investments earn their yields; CASA, term deposits and borrowings cost theirs. Net interest
income plus fees, minus operating cost, minus credit-loss provisions and tax, gives net profit, which is
retained in equity. CRAR = (year-end equity + Tier 2) / risk-weighted assets (loans × risk weight;
investments at 0%). Verdicts use RBI thresholds: 9% minimum CRAR, 11.5% with the conservation buffer.
Liquidity for a deposit run is investments above the 18% SLR floor.

## Optional secrets (Student Assignment)

Add in Streamlit Cloud → App settings → Secrets:

```toml
COURSE_KEY = "any-text-unique-to-your-course"   # changes which case each roll number gets
INSTRUCTOR_KEY = "a-password"                   # unlocks the answer-key download
```

## Tests

```bash
python tests/test_engine.py
```

## Files

- `app.py` – the Streamlit app (layout, widgets, text)
- `engine.py` – all calculations, no Streamlit; vectorised for grids and simulations
- `charts.py` – Plotly figure builders
- `tests/test_engine.py` – checks the model against the reported ratios and stress logic
