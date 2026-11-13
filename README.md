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
3. Optional: **Advanced settings → Secrets** `APP_PASSWORD = "your-code"` adds an access code on each new device. Without it the app is open to anyone with the link.
4. Optional, for the email button to send by itself: also add `SMTP_USER` (your Gmail address), `SMTP_PASSWORD` (a Gmail app password) and `MAIL_TO` (where to send). Without them the button opens your mail app with the ranking filled in.
5. Deploy. On the phone: open the URL, enter the code once, then Add to Home Screen.

## How the model is used

* Score = 100 × Penalty × (Σ wᵢ·Normᵢ + 0.04·BullishScore + 0.02·Norm_PriceToTarget), exactly as in the notebook (PriceToTarget counted twice is kept on purpose).
* The 29 weights must total 0.96 (the notebook's own check). The simulator previews in the browser while dragging; ▶ then reruns the Python model on the server with those weights (and penalty settings) and the Ranking shows that run, with each stock's default rank for comparison. The notebook block itself is never edited: the weights are passed in as a new dict.
* Penalty strength *k* turns a ×m rule into ×(1 − k·(1 − m)). k = 100% with all rules on is the notebook.
* SPEC check uses the notebook `thresholds`. "Strong" = beats a one-sided threshold by 1.5× or more. Dividend yields from Yahoo are in percent and are compared in matching units.

## Develop

```bash
pip install -r requirements.txt pytest
python -m pytest -q
python tools/build.py                      # after editing frontend/app.html
STOCK_RANKER_DEMO=1 streamlit run streamlit_app.py   # offline demo data
```
