# Stock Ranker

Phone-first app around the **FinalAPIcode** Colab notebook (Rev A). Type symbols, run, see Score and Rank, open any stock for a deep dive against the SPEC, and simulate other weights live.

The original notebook is untouched. `core.py` copies its model blocks verbatim (marked `# === NOTEBOOK (verbatim) ===`) and the tests prove the port reproduces the notebook's Excel of 13 Nov 2026 exactly (all 27 Scores and Ranks).

## Files

| File | Role |
|---|---|
| `core.py` | Notebook logic: fetch (yfinance), derived metrics, Norm, Pass, weights, Penalty, Score, Rank |
| `streamlit_app.py` | Access code, data fetch, hands results to the UI |
| `frontend/app.html` | The whole UI (source). `frontend/index.html` is generated from it |
| `tools/build.py` | Rebuilds `frontend/index.html`; `--preview FILE` makes an offline preview on the Excel data |
| `tests/` | Parity tests against the notebook's Excel export |

## Deploy on Streamlit Community Cloud

1. Sign in at share.streamlit.io with **Continue with GitHub**.
2. **Create app** → repository `stock-ranker`, branch `main`, main file `streamlit_app.py`, pick a short URL.
3. **Advanced settings → Secrets**: `APP_PASSWORD = "your-code"`.
4. Optional, for the email button to send by itself: also add `SMTP_USER` (your Gmail address), `SMTP_PASSWORD` (a Gmail app password) and `MAIL_TO` (where to send). Without them the button opens your mail app with the ranking filled in.
5. Deploy. On the phone: open the URL, enter the code once, then Add to Home Screen.

## How the model is used

* Score = 100 × Penalty × (Σ wᵢ·Normᵢ + 0.04·BullishScore + 0.02·Norm_PriceToTarget), exactly as in the notebook (PriceToTarget counted twice is kept on purpose).
* The 29 weights must total 0.96 (the notebook's own check). The simulator runs in the browser, so results update while dragging.
* Penalty strength *k* turns a ×m rule into ×(1 − k·(1 − m)). k = 100% with all rules on is the notebook.
* SPEC check uses the notebook `thresholds`. "Strong" = beats a one-sided threshold by 1.5× or more. Dividend yields from Yahoo are in percent and are compared in matching units.

## Develop

```bash
pip install -r requirements.txt pytest
python -m pytest -q
python tools/build.py                      # after editing frontend/app.html
APP_PASSWORD=test STOCK_RANKER_DEMO=1 streamlit run streamlit_app.py   # offline demo data
```
