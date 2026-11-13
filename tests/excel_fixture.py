"""
Rebuild notebook inputs from the Excel the notebook exported on 2026-11-13.

The Excel holds the 29 raw metrics per ticker, their Norm_ values, Penalty,
Score and Rank. It does not hold the underlying yfinance fields (prices,
moving averages, beta ...), so this module reconstructs a set of yfinance-like
`info` dicts that produce exactly the same raw metrics, and momentum inputs
that are consistent with the BullishScore and Penalty the notebook reported.
Those momentum inputs are stand-ins for testing only - never shown as data.
"""
from __future__ import annotations

import itertools
import json
import math
from pathlib import Path

import pandas as pd

import core

FIX = Path(__file__).parent / 'fixtures'
EXCEL = FIX / 'stock_thresholds_and_norms_2026-11-13.xlsx'
NOTEBOOK_OUT = FIX / 'notebook_output_2026-11-13.json'

DIRECT = ['marketCap', 'grossMargins', 'operatingMargins', 'ebitdaMargins', 'profitMargins',
          'returnOnAssets', 'returnOnEquity', 'quickRatio', 'currentRatio', 'trailingPegRatio',
          'revenueGrowth', 'earningsGrowth', 'enterpriseToEbitda', 'operatingCashflow',
          'trailingPE', 'forwardPE', 'debtToEquity', 'fiveYearAvgDividendYield',
          'earningsQuarterlyGrowth', 'dividendYield', 'payoutRatio']

# penalty rules whose inputs are NOT among the exported metrics
HIDDEN_RULES = ['BullishCount_lt_2', 'beta_gt_2', 'BearishFlag']


def load_excel() -> pd.DataFrame:
    """Excel as a frame indexed by ticker, columns renamed to metric keys."""
    df = pd.read_excel(EXCEL).set_index('Ticker')
    cols = list(df.columns)
    rename = {}
    for i, m in enumerate(core.METRICS):
        assert cols[2 * i + 1] == f'Norm_{m}', (cols[2 * i + 1], m)
        rename[cols[2 * i]] = m
    return df.rename(columns=rename)


def notebook_table() -> dict:
    rows = json.loads(NOTEBOOK_OUT.read_text())['rows']
    return {r[0]: {'rank': r[1], 'score': r[2], 'bull': r[3], 'sector': r[4], 'industry': r[5]} for r in rows}


def _nan(x):
    return x is None or (isinstance(x, float) and math.isnan(x))


def observable_penalty(row) -> float:
    """Product of the penalty rules that can be evaluated from exported metrics."""
    frame = pd.DataFrame([row])
    p = 1.0
    for key, _lbl, mult, cond in core.PENALTY_RULES:
        if key in HIDDEN_RULES:
            continue
        if bool(cond(frame).iloc[0]):
            p *= mult
    return p


def solve_momentum(bull: float, residual: float, price_known: bool):
    """Find momentum flags consistent with BullishScore and the hidden penalty part."""
    w = [c[2] for c in core.BULLISH_CONDS]
    for c in itertools.product([False, True], repeat=6):
        for beta_high, bearish in itertools.product([False, True], repeat=2):
            if c[5] and beta_high:          # beta in [1, 1.5] excludes beta > 2
                continue
            if c[1] and bearish:            # 50MA > 200MA excludes the bearish flag
                continue
            if not price_known and (c[0] or c[2] or bearish):
                continue
            score = sum(wi for wi, ci in zip(w, c) if ci)
            res = (0.9 if sum(c) < 2 else 1.0) * (0.9 if beta_high else 1.0) * (0.85 if bearish else 1.0)
            if abs(score - bull) < 1e-9 and abs(res - residual) < 1e-9:
                return c, beta_high, bearish
    raise ValueError(f'no momentum solution for bull={bull}, residual={residual}')


def synthetic_infos() -> tuple[list[str], dict]:
    """(tickers in notebook order, info dicts reproducing the Excel raw metrics)."""
    xl = load_excel()
    table = notebook_table()
    infos = {}
    for sym, r in xl.iterrows():
        info = {k: (None if _nan(r[k]) else float(r[k])) for k in DIRECT}
        ptt = r['PriceToTarget']
        price = None if _nan(ptt) else 100.0 * float(ptt)
        info['targetMeanPrice'] = None if price is None else 100.0
        info['currentPrice'] = price
        x = r['FW_EPSvsEPS']
        info['trailingEps'], info['forwardEps'] = (None, None) if _nan(x) else (1.0, 1.0 + float(x))
        x = r['AnalystsRecommendation']
        info['recommendationMean'] = None if _nan(x) else 1.0 / float(x)
        x = r['NetCash']
        info['totalDebt'], info['totalCash'] = (None, None) if _nan(x) else (1.0, 1.0 + float(x))
        x = r['OPSvsFreeCashFlow']
        ocf = info['operatingCashflow']
        info['freeCashflow'] = None if (_nan(x) or ocf is None) else ocf / float(x)
        x = r['epsFWvsCurrent']
        info['epsCurrentYear'], info['epsForward'] = (None, None) if _nan(x) else (1.0, float(x))

        residual = float(r['Penalty']) / observable_penalty(r)
        c, beta_high, bearish = solve_momentum(table[sym]['bull'], residual, price is not None)
        p = price if price is not None else float('nan')
        a = p * (0.9 if c[0] else 1.1)                       # 50-day average
        b = a * 0.9 if c[1] else (a * 1.5 if bearish else a)  # 200-day average
        info['fiftyDayAverage'] = None if price is None else a
        info['twoHundredDayAverage'] = None if price is None else b
        info['fiftyTwoWeekLow'] = None if price is None else (p / 1.5 if c[2] else p)
        info['volume'], info['averageVolume'] = (2.0, 1.0) if c[3] else (1.0, 2.0)
        info['shortRatio'], info['sharesShort'], info['floatShares'] = (4.0, 2.0, 10.0) if c[4] else (1.0, 1.0, 100.0)
        info['beta'] = 1.2 if c[5] else (2.5 if beta_high else 0.5)
        info['sector'] = table[sym]['sector']
        info['industry'] = table[sym]['industry']
        infos[sym] = info
    return list(xl.index), infos
