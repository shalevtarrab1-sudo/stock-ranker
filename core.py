"""
Stock Ranker - scoring engine.

A faithful port of the Colab notebook FinalAPIcode.ipynb ("Rev A").
The original notebook is NOT modified; this file is new code.

Blocks between  # === NOTEBOOK (verbatim) ===  markers are copied from the
notebook without changes. Everything else only feeds data in (fetching,
type coercion) or reads results out (display metadata, JSON payload).
"""
from __future__ import annotations

import math
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Model definition (copied from the notebook)
# --------------------------------------------------------------------------

# === NOTEBOOK (verbatim) ===
attributes = [
    # Main indicators
    'sector','industry','quoteType','marketCap',
    'operatingMargins','grossMargins','quickRatio','currentRatio',
    'trailingPegRatio','earningsGrowth','revenueGrowth','ebitdaMargins',
    'returnOnAssets','returnOnEquity','profitMargins','fiveYearAvgDividendYield','payoutRatio',
    # Price & performance
    'fiftyDayAverageChangePercent','twoHundredDayAverageChangePercent',
    'totalCash','epsForward','epsCurrentYear','freeCashflow','operatingCashflow',
    'debtToEquity','recommendationMean','SandP52WeekChange','currentPrice',
    'targetHighPrice','targetMeanPrice','52WeekChange','enterpriseToEbitda',
    'trailingEps','forwardEps','overallRisk','compensationRisk','earningsQuarterlyGrowth',
    'enterpriseValue','totalDebt','dividendYield','floatShares','sharesShort',
    'open','fiftyDayAverage','twoHundredDayAverage','beta','shortRatio','fiftyTwoWeekHighChangePercent','fiftyTwoWeekLow','volume','averageVolume','trailingPE','forwardPE'
]

# 4. Define thresholds: (min, max, higher_better)
thresholds = {
    'marketCap':       (5e9,    None, True),   # >=5B
    'grossMargins':    (0.55,   None, True),   # >=55%
    'operatingMargins':(0.25,   None, True),   # >=25%
    'PriceToTarget' : (None, 1.1, False),  # >10% above target = risky
    'ebitdaMargins':   (0.2,   0.38, True),   # 20–38%
    'profitMargins':   (0.2,   None, True),   # >=20%
    'returnOnAssets':  (0.08,   None, True),   # >=8%
    'returnOnEquity':  (0.19,   None, True),   # >=19%
    'quickRatio':      (1.3,    None, True),   # >=1.3
    'currentRatio':    (1.5,    None, True),   # >=1.5
    'trailingPegRatio':(0.8,    1.8,  False),  # 0.8–1.8
    'revenueGrowth':   (0.10,   None, True),   # >=10%
    'earningsGrowth':  (0.3,   None, True),   # >=30%
    'enterpriseToEbitda':  (0,   None, True),   # >=0
    'operatingCashflow':  (2e8,   None, True),   # >=2M
    'trailingPE':      (1,     25,   False),  # 1×–25×
    'forwardPE':       (1,     25,   False),  # 1×–25×
    'debtToEquity':    (None,   100,  False),  # <=100
    'fiveYearAvgDividendYield':  (0.01,   None, True),   # >=1%
    'earningsQuarterlyGrowth':  (0.1,   None, True),   # >=10%
    'dividendYield':   (0.02,   None, True),   # >=2%
    'payoutRatio':     (None,   0.60, False),   # <=60%
       # Computed metrics
   # 'Short/float':           (0.05,  None,  False),  # >=5% short float
    'FW_EPSvsEPS':                          (1.0,    None,  True),   # >=1
    'StockTarget':                         (0.08,    None,  True),   # >0.08
    'AnalystsRecommendation':              (0.4,   None,  True),   # >=0.4
    'NetCash':                             (0.0,    None,  True),   # >=0
    'EquityToDebt':                        (0.3,   None,  True),   # >=0.3
    'OPSvsFreeCashFlow':                   (1.1,   None,  True),   # >=1.1
    'epsFWvsCurrent':                      (0.9,    None,  True),   # >=0.9
    # 'Div_to_NetIncome':                       (0.1, None,   False)   ,

}

weights = {
    'Norm_profitMargins':           0.055,
    'Norm_operatingCashflow':       0.054,
    'Norm_operatingMargins':        0.053,
    'Norm_revenueGrowth':           0.052,
    'Norm_earningsGrowth':          0.052,
    'Norm_epsFWvsCurrent':          0.051,
    'Norm_StockTarget':             0.050,
    'Norm_FW_EPSvsEPS':             0.038,
    'Norm_grossMargins':            0.045,
    'Norm_ebitdaMargins':           0.040,
    'Norm_returnOnAssets':          0.048,
    'Norm_returnOnEquity':          0.037,
    'Norm_debtToEquity':            0.036,
    'Norm_OPSvsFreeCashFlow':       0.034,
    'Norm_trailingPegRatio':        0.031,
    'Norm_marketCap':               0.029,
    'Norm_AnalystsRecommendation':  0.025,
    'Norm_quickRatio':              0.025,
    'Norm_enterpriseToEbitda':      0.025,
    'Norm_fiveYearAvgDividendYield':0.025,
    'Norm_earningsQuarterlyGrowth': 0.025,
    'Norm_dividendYield':           0.025,
    'Norm_EquityToDebt':            0.020,
    'Norm_NetCash':                 0.015,
    'Norm_payoutRatio':             0.015,
    'Norm_forwardPE':               0.015,
    'Norm_currentRatio':            0.010,
    'Norm_PriceToTarget':           0.02,
    'Norm_trailingPE':              0.010,

}
# === END NOTEBOOK ===

DEFAULT_WEIGHTS = dict(weights)
METRICS = list(thresholds)            # the 29 scored metrics, notebook order
BULLISH_WEIGHT = 0.04                 # "df['RawScore'] += df['BullishScore'] * 0.04"
PRICE_TO_TARGET_EXTRA = 0.02          # "df['RawScore'] += df['Norm_PriceToTarget'] * 0.02"

NON_NUMERIC = {'sector', 'industry', 'quoteType'}
DISPLAY_ATTRIBUTES = ['shortName', 'longName', 'currency']   # labels only, never scored

# --------------------------------------------------------------------------
# Display metadata (presentation only - does not affect any number)
# --------------------------------------------------------------------------

CATEGORIES = [                      # grouped by what the notebook weights emphasise
    ('margins',   'Margins'),
    ('forward',   'Forward view'),
    ('growth',    'Growth'),
    ('cash',      'Cash & size'),
    ('balance',   'Balance sheet'),
    ('returns',   'Returns'),
    ('valuation', 'Valuation'),
    ('dividends', 'Dividends'),
]

# key: (label, category, value format, spec text)
# formats: money | pct (fraction -> %) | pctraw (already in %) | x (ratio) | num
METRIC_META = {
    'marketCap':                ('Market cap',                 'cash', 'money',  '≥ $5B'),
    'grossMargins':             ('Gross margin',               'margins',   'pct',    '≥ 55%'),
    'operatingMargins':         ('Operating margin',           'margins',   'pct',    '≥ 25%'),
    'PriceToTarget':            ('Price / analyst target',     'forward',   'x',      '≤ 1.10'),
    'ebitdaMargins':            ('EBITDA margin',              'margins',   'pct',    '20% – 38%'),
    'profitMargins':            ('Profit margin',              'margins',   'pct',    '≥ 20%'),
    'returnOnAssets':           ('Return on assets',           'returns',   'pct',    '≥ 8%'),
    'returnOnEquity':           ('Return on equity',           'returns',   'pct',    '≥ 19%'),
    'quickRatio':               ('Quick ratio',                'balance', 'x',      '≥ 1.3'),
    'currentRatio':             ('Current ratio',              'balance', 'x',      '≥ 1.5'),
    'trailingPegRatio':         ('PEG ratio',                  'valuation',    'x',      '0.8 – 1.8'),
    'revenueGrowth':            ('Revenue growth',             'growth',   'pct',    '≥ 10%'),
    'earningsGrowth':           ('Earnings growth',            'growth',   'pct',    '≥ 30%'),
    'enterpriseToEbitda':       ('EV / EBITDA',                'valuation',    'x',      '≥ 0'),
    'operatingCashflow':        ('Operating cash flow',        'cash', 'money',  '≥ $200M'),
    'trailingPE':               ('P/E (trailing)',             'valuation',    'x',      '1 – 25'),
    'forwardPE':                ('P/E (forward)',              'valuation',    'x',      '1 – 25'),
    'debtToEquity':             ('Debt / equity (%)',          'balance', 'num',    '≤ 100'),
    'fiveYearAvgDividendYield': ('Dividend yield, 5Y avg',     'dividends', 'pctraw', '≥ 1%'),
    'earningsQuarterlyGrowth':  ('Quarterly earnings growth',  'growth',   'pct',    '≥ 10%'),
    'dividendYield':            ('Dividend yield',             'dividends', 'pctraw', '≥ 2%'),
    'payoutRatio':              ('Payout ratio',               'dividends', 'pct',    '≤ 60%'),
    'FW_EPSvsEPS':              ('Forward EPS vs trailing',    'forward',   'pct',    '≥ +100%'),
    'StockTarget':              ('Upside to analyst target',   'forward',   'pct',    '≥ 8%'),
    'AnalystsRecommendation':   ('Analyst rating (1 / mean)',  'forward',   'x',      '≥ 0.40'),
    'NetCash':                  ('Net cash / debt',            'balance', 'x',      '≥ 0'),
    'EquityToDebt':             ('Equity / debt (×10)',        'balance', 'x',      '≥ 0.30'),
    'OPSvsFreeCashFlow':        ('Operating CF / free CF',     'cash', 'x',      '≥ 1.10'),
    'epsFWvsCurrent':           ('EPS next year / this year',  'forward',   'x',      '≥ 0.90'),
}

# yfinance returns these two in percent (0.74 means 0.74%) while the notebook
# thresholds are fractions (0.02 = 2%). The Score never uses the thresholds, so
# this only matters for the SPEC check shown in the app, which compares them in
# consistent units.
PERCENT_UNIT_METRICS = {'dividendYield', 'fiveYearAvgDividendYield'}

# "Exceptional" = beats a one-sided threshold by this factor. Display only.
EXCEPTIONAL_FACTOR = 1.5

# The 16 penalty rules of notebook section 7, in notebook order. Used only to
# show WHICH rules fired; the Penalty itself comes from the verbatim block and
# the two are cross-checked on every run.
PENALTY_RULES = [
    ('marketCap_lt_5e9',         'Market cap below $5B',                             0.8,  lambda d: d['marketCap'] < 5e9),
    ('grossMargins_lt_0',        'Negative gross margin',                            0.6,  lambda d: d['grossMargins'] < 0),
    ('grossMargins_eq_1',        'Gross margin reported as 100% (unreliable data)',  0.8,  lambda d: d['grossMargins'] == 1),
    ('operatingMargins_lt_0',    'Negative operating margin',                        0.7,  lambda d: d['operatingMargins'] < 0),
    ('trailingPE_gt_100',        'Trailing P/E above 100',                           0.85, lambda d: d['trailingPE'] > 100),
    ('profitMargins_lt_0',       'Negative profit margin',                           0.8,  lambda d: d['profitMargins'] < 0),
    ('returnOnAssets_lt_0',      'Negative return on assets',                        0.8,  lambda d: d['returnOnAssets'] < 0),
    ('returnOnEquity_lt_0',      'Negative return on equity',                        0.8,  lambda d: d['returnOnEquity'] < 0),
    ('operatingCashflow_lt_2e8', 'Operating cash flow below $200M',                  0.9,  lambda d: d['operatingCashflow'] < 2e8),
    ('operatingCashflow_lt_0',   'Negative operating cash flow',                     0.8,  lambda d: d['operatingCashflow'] < 0),
    ('FW_EPSvsEPS_lt_0.1',       'Forward EPS growth below 10%',                     0.9,  lambda d: d['FW_EPSvsEPS'] < 0.1),
    ('StockTarget_lt_0.08',      'Upside to analyst target below 8%',                0.6,  lambda d: d['StockTarget'] < 0.08),
    ('revenueGrowth_lt_0',       'Shrinking revenue',                                0.9,  lambda d: d['revenueGrowth'] < 0),
    ('BullishCount_lt_2',        'Fewer than 2 momentum conditions met',             0.9,  lambda d: d['BullishCount'] < 2),
    ('beta_gt_2',                'Beta above 2',                                     0.9,  lambda d: d['beta'] > 2),
    ('BearishFlag',              'Bearish trend: price and 50-day below 200-day average', 0.85, lambda d: d['BearishFlag']),
]

# The six momentum conditions behind BullishScore (label, weight in BullishScore)
BULLISH_CONDS = [
    ('Cond1_Price>50MA',        'Price above 50-day average',                      0.3),
    ('Cond2_50MA>200MA',        '50-day average above 200-day average',            0.3),
    ('Cond3_Price30Above52Low', 'More than 20% above the 52-week low',             0.2),
    ('Cond4_VolumeSpike',       'Volume above average',                            0.05),
    ('Cond5_ShortSqueeze',      'Short-squeeze setup (short ratio > 3, short float > 10%)', 0.05),
    ('Cond6_IdealBeta',         'Beta between 1.0 and 1.5',                        0.1),
]


# --------------------------------------------------------------------------
# Data in
# --------------------------------------------------------------------------

def clean_tickers(raw) -> list[str]:
    """Upper-case, trim and de-duplicate symbols, keeping the order typed."""
    out, seen = [], set()
    for t in raw or []:
        s = str(t).strip().upper()
        if s and s not in seen and len(s) <= 12 and all(ch.isalnum() or ch in '.-^=' for ch in s):
            seen.add(s)
            out.append(s)
    return out


def fetch_info(tickers: list[str], max_workers: int = 5, cache: dict | None = None,
               ttl: float = 1800.0) -> tuple[dict, dict]:
    """Fetch yf.Ticker(sym).info for every symbol (same call as the notebook).

    Runs a few requests in parallel to cut waiting time; the data is identical.
    `cache` (optional) keeps successful responses for `ttl` seconds so that
    re-running a list only fetches new symbols.
    Returns (info_by_ticker, error_by_ticker).
    """
    import yfinance as yf

    now = time.time()
    infos, errors, todo = {}, {}, []
    for sym in tickers:
        hit = cache.get(sym) if cache is not None else None
        if hit and now - hit[0] < ttl:
            infos[sym] = hit[1]
        else:
            todo.append(sym)

    def one(sym):
        last = None
        for attempt in range(2):
            try:
                info = yf.Ticker(sym).info
                if isinstance(info, dict):
                    return sym, info, None
                last = 'empty response'
            except Exception as exc:  # network error, unknown symbol, rate limit
                last = f'{type(exc).__name__}: {exc}'
                time.sleep(0.8 * (attempt + 1))
        return sym, {}, (last or 'no data')[:200]

    if todo:
        with ThreadPoolExecutor(max_workers=max(1, min(max_workers, len(todo)))) as pool:
            for sym, info, err in pool.map(one, todo):
                infos[sym] = info
                if err:
                    errors[sym] = err
                elif cache is not None and info.get('quoteType'):
                    cache[sym] = (now, info)
    return infos, errors


def build_frame(infos: dict, tickers: list[str]) -> pd.DataFrame:
    """Notebook section 3: one row per ticker with the raw yfinance attributes."""
    rows = []
    for sym in tickers:
        info = infos.get(sym) or {}
        row = {'Ticker': sym}
        for attr in attributes:
            row[attr] = info.get(attr, np.nan)
        for attr in DISPLAY_ATTRIBUTES:
            row[attr] = info.get(attr)
        rows.append(row)
    df = pd.DataFrame(rows).set_index('Ticker')
    # Guard: a column that came back entirely empty would be 'object' dtype and
    # crash the comparisons below. Coercing to numbers changes no value.
    for attr in attributes:
        if attr not in NON_NUMERIC:
            df[attr] = pd.to_numeric(df[attr], errors='coerce')
    return df


# --------------------------------------------------------------------------
# The model (notebook sections 3-8)
# --------------------------------------------------------------------------

def score_frame(df: pd.DataFrame, weights: dict | None = None,
                penalty_cfg: dict | None = None) -> pd.DataFrame:
    """Run the notebook logic on a raw-attribute frame. Returns a scored copy.

    weights: Norm_ column -> weight (29 entries, sum 0.96). None = notebook default.
    penalty_cfg: {'k': strength 0-1, 'on': [16 bools]}. None = notebook penalty.
    The notebook block itself always runs unchanged; a custom penalty is
    applied afterwards from the same 16 rules.
    """
    df = df.copy()
    weights = dict(DEFAULT_WEIGHTS if weights is None else weights)

    # === NOTEBOOK (verbatim) ===
    # Compute all the extra columns on df, not df_attributes
    df['Short/float']              = df['sharesShort'] / df['floatShares']
    df['EV_to_EBITDA_over_15']     = df['enterpriseToEbitda'] / 15
    df['FW_EPSvsEPS']              = (df['forwardEps'] - df['trailingEps']) / df['trailingEps']
    df['StockTarget']              = (df['targetMeanPrice'] - df['currentPrice']) / df['currentPrice']
    df['PriceToTarget'] = df['currentPrice'] / df['targetMeanPrice']
    df['Upside20'] = df['StockTarget'] >= 0.20  # 20%+ upside
    df['AnalystsRecommendation']   = 1 / df['recommendationMean']
    df['NetCash']                  = (df['totalCash'] - df['totalDebt']) / df['totalDebt']
    df['EquityToDebt']             = (1 / df['debtToEquity']) * 10
    df['OPSvsFreeCashFlow']        = df['operatingCashflow'] / df['freeCashflow']
    df['epsFWvsCurrent']           = df['epsForward'] / df['epsCurrentYear']
    df['compensationRiskRank']     = (1 / df['compensationRisk']) * 10
    df['NoRisk']                   = df['overallRisk']
    df['Div_to_NetIncome']         = df['payoutRatio'] * 10

    # Composite Bullish Conditions columns
    df['BearishFlag'] = (df['currentPrice'] < df['twoHundredDayAverage']) & (df['fiftyDayAverage'] < df['twoHundredDayAverage'])

    df['Cond1_Price>50MA'] = df['currentPrice'] > df['fiftyDayAverage']
    df['Cond2_50MA>200MA'] = df['fiftyDayAverage'] > df['twoHundredDayAverage']
    df['Cond3_Price30Above52Low'] = (df['currentPrice'] - df['fiftyTwoWeekLow']) / df['fiftyTwoWeekLow'] > 0.2 # i made it 20 % above not 30%
    df['Cond4_VolumeSpike'] = df['volume'] > df['averageVolume']
    df['Cond5_ShortSqueeze'] = (df['shortRatio'] > 3) & (df['sharesShort'] / df['floatShares'] > 0.1)
    df['Cond6_IdealBeta'] = (df['beta'] >= 1) & (df['beta'] <= 1.5)

    # Sum up how many are true per stock
    df['BullishCount'] = df[[f'Cond{i}_' + desc for i, desc in enumerate([
        'Price>50MA','50MA>200MA','Price30Above52Low','VolumeSpike',
        'ShortSqueeze','IdealBeta'
    ], 1)]].sum(axis=1)


    df['BullishScore'] = (
        0.3 * df['Cond1_Price>50MA'].astype(float) +
        0.3 * df['Cond2_50MA>200MA'].astype(float) +
        0.2 * df['Cond3_Price30Above52Low'].astype(float) +
        0.05 * df['Cond4_VolumeSpike'].astype(float) +
        0.05 * df['Cond5_ShortSqueeze'].astype(float) +
        0.1 * df['Cond6_IdealBeta'].astype(float)
    )

    # 5. Normalize metrics 0–1
    norm_cols = []
    for metric, (mn, mx, higher) in thresholds.items():
        col = f"Norm_{metric}"
        #val = df[metric].astype(float)

        vals = df[metric].astype(float)
        # compute dynamic 5th & 95th percentiles

        mn_dyn = vals.quantile(0.01)
        mx_dyn = vals.quantile(0.99)


        if higher:
            if mx_dyn is None:
                  # Only a MIN bound; anything ≥ mn is a perfect score of 1
                #norm = (val / mn).clip(upper=1)
                norm = ((vals - mn_dyn) / (mx_dyn - mn_dyn)).clip(0,1)
            else:
                # norm = ((val - mn) / (mx - mn)).clip(0,1)
                norm = ((vals - mn_dyn) / (mx_dyn - mn_dyn)).clip(0,1)

        else:
            if mn_dyn is None:
                norm = (mx_dyn / vals).clip(upper=1)
            else:
                norm = ((mx_dyn - vals) / (mx_dyn - mn_dyn)).clip(0,1)
        df[col] = norm.fillna(0)
        norm_cols.append(col)

    # 6. Build a pass/fail column for each metric
    for metric, (mn_dyn, mx_dyn, higher) in thresholds.items():
        raw = df[metric].astype(float)
        # Boolean series: True if within your “good” bounds
        if higher:
            if mn_dyn is not None and mx_dyn is not None:
                passed = raw.between(mn_dyn, mx_dyn)
            elif mn_dyn is not None:
                passed = raw >= mn_dyn
            else:
                passed = pd.Series(False, index=raw.index)
        else:
            if mn_dyn is not None and mx_dyn is not None:
                passed = raw.between(mn_dyn, mx_dyn)
            elif mx_dyn is not None:
                passed = raw <= mx_dyn
            else:
                passed = pd.Series(False, index=raw.index)

        df[f"Pass_{metric}"] = passed

    # (Optional) sanity‐check they sum to 1.0
    total_w = sum(weights.values())+ 0.04 # 0.04 for the bullish score i added later on
    if abs(total_w - 1.0) > 1e-6:
        raise ValueError(f"Total weight must be 1.0, got {total_w}")

    # Compute the weighted RawScore
    df['RawScore'] = sum(df[c] * weights[c] for c in norm_cols)
    df['RawScore'] += df['BullishScore'] * 0.04
    df['RawScore'] += df['Norm_PriceToTarget'] * 0.02

    # 7. Penalties for out‐of‐spec criticals
    penalty = pd.Series(1.0, index=df.index)
    penalty[df['marketCap'] < 5e9] *= 0.8
    penalty[df['grossMargins'] < 0] *= 0.6
    penalty[df['grossMargins'] == 1] *= 0.8 # not relaiable data- yfinance place 1 instead NA
    penalty[df['operatingMargins'] < 0] *= 0.7
    penalty[df['trailingPE'] > 100] *= 0.85
    penalty[df['profitMargins'] < 0] *= 0.8
    penalty[df['returnOnAssets'] < 0] *= 0.8
    penalty[df['returnOnEquity'] < 0] *= 0.8
    penalty[df['operatingCashflow'] < 2e8] *= 0.9
    penalty[df['operatingCashflow'] < 0] *= 0.8
    penalty[df['FW_EPSvsEPS'] < 0.1] *= 0.9
    penalty[df['StockTarget'] < 0.08] *= 0.6
    penalty[df['revenueGrowth'] < 0] *= 0.9
    penalty[df['BullishCount'] < 2] *= 0.9
    penalty[df['beta'] > 2] *= 0.9
    penalty[df['BearishFlag']] *= 0.85

    # 8. Final score and rank
    df['Penalty'] = penalty
    df['FinalScore'] = df['RawScore'] * penalty
    df['Score'] = df['FinalScore'] * 100
    df['Rank'] = df['FinalScore'].rank(ascending=False, method='dense').astype(int)
    # === END NOTEBOOK ===

    # Which penalty rules fired (display only), cross-checked against Penalty.
    check = pd.Series(1.0, index=df.index)
    hits = {}
    for key, _label, mult, cond in PENALTY_RULES:
        hit = cond(df).astype(bool)
        hits[f'Hit_{key}'] = hit
        check[hit] *= mult
    if not np.allclose(check.to_numpy(), penalty.to_numpy(), rtol=0, atol=1e-12):
        raise AssertionError('Penalty rule table is out of sync with the notebook penalty block')

    if penalty_cfg is not None and not _is_default_penalty(penalty_cfg):
        custom = custom_penalty(df, penalty_cfg)
        df['Penalty'] = custom
        df['FinalScore'] = df['RawScore'] * custom
        df['Score'] = df['FinalScore'] * 100
        df['Rank'] = df['FinalScore'].rank(ascending=False, method='dense').astype(int)

    return pd.concat([df, pd.DataFrame(hits, index=df.index)], axis=1)


def _is_default_penalty(cfg: dict) -> bool:
    on = cfg.get('on') or [True] * len(PENALTY_RULES)
    return abs(float(cfg.get('k', 1.0)) - 1.0) < 1e-9 and all(on)


def custom_penalty(df: pd.DataFrame, cfg: dict) -> pd.Series:
    """Same 16 rules; a rule that is off is skipped, strength k softens each
    multiplier: x m becomes x (1 - k (1 - m)). k = 1 is the notebook."""
    k = min(max(float(cfg.get('k', 1.0)), 0.0), 1.0)
    on = list(cfg.get('on') or [True] * len(PENALTY_RULES))
    out = pd.Series(1.0, index=df.index)
    for i, (_key, _label, mult, cond) in enumerate(PENALTY_RULES):
        if i < len(on) and not on[i]:
            continue
        out[cond(df).astype(bool)] *= 1 - k * (1 - mult)
    return out


def weights_from_model(model: dict | None) -> dict | None:
    """App model {'w': 29 ints in 0.1% units, total 960} -> notebook weights dict."""
    if not model or not model.get('w'):
        return None
    w = [int(x) for x in model['w']]
    if len(w) != len(METRICS) or any(x < 0 for x in w) or sum(w) != 960:
        raise ValueError('Weights must be 29 non-negative values totalling 96%.')
    return {f'Norm_{m}': x / 1000 for m, x in zip(METRICS, w)}


# --------------------------------------------------------------------------
# Results out
# --------------------------------------------------------------------------

def _num(x):
    """JSON-safe number: NaN / inf / None become None."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _text(x):
    return x if isinstance(x, str) and x.strip() else None


def spec_status(metric: str, value) -> str:
    """Where a raw value sits against its SPEC (thresholds from the notebook).

    'nodata'       value missing
    'fail'         outside the SPEC on the bad side
    'outside'      outside a min-max range on the side the model otherwise rewards
    'pass'         inside the SPEC
    'exceptional'  beats a one-sided SPEC by EXCEPTIONAL_FACTOR or more
    """
    v = _num(value)
    if v is None:
        return 'nodata'
    if metric in PERCENT_UNIT_METRICS:
        v = v / 100.0
    mn, mx, higher = thresholds[metric]
    if mn is not None and mx is not None:            # range, as raw.between(mn, mx)
        if mn <= v <= mx:
            return 'pass'
        good_side = (v > mx) if higher else (0 < v < mn)
        return 'outside' if good_side else 'fail'
    if higher and mn is not None:                    # raw >= mn
        if v < mn:
            return 'fail'
        return 'exceptional' if mn > 0 and v >= mn * EXCEPTIONAL_FACTOR else 'pass'
    if not higher and mx is not None:                # raw <= mx
        if v > mx:
            return 'fail'
        # zero or negative values (e.g. payout ratio 0 = pays no dividend) are
        # in spec but not a strength
        return 'exceptional' if mx > 0 and 0 < v <= mx / EXCEPTIONAL_FACTOR else 'pass'
    return 'fail'                                    # notebook: no bound -> False


def model_meta() -> dict:
    """Static description of the model for the UI (weights, rules, labels)."""
    weight_order = [k[len('Norm_'):] for k in DEFAULT_WEIGHTS]     # notebook dict order
    return {
        'metrics': [{
            'key': m,
            'label': METRIC_META[m][0],
            'cat': METRIC_META[m][1],
            'fmt': METRIC_META[m][2],
            'spec': METRIC_META[m][3],
            'higher': bool(thresholds[m][2]),
            'weight': DEFAULT_WEIGHTS[f'Norm_{m}'],
        } for m in METRICS],
        'categories': [{'key': k, 'label': lbl} for k, lbl in CATEGORIES],
        'rules': [{'key': k, 'label': lbl, 'mult': mult} for k, lbl, mult, _ in PENALTY_RULES],
        'conds': [{'key': k, 'label': lbl, 'weight': w} for k, lbl, w in BULLISH_CONDS],
        'weightOrder': weight_order,
        'bullishWeight': BULLISH_WEIGHT,
        'pttExtra': PRICE_TO_TARGET_EXTRA,
        'pttIndex': METRICS.index('PriceToTarget'),
        'exceptionalFactor': EXCEPTIONAL_FACTOR,
    }


def build_payload(df: pd.DataFrame, errors: dict | None = None,
                  source: str = 'Yahoo Finance via yfinance') -> dict:
    """Turn a scored frame into the JSON the app renders."""
    errors = errors or {}
    stocks = []
    for sym, r in df.iterrows():
        raw = [_num(r[m]) for m in METRICS]
        no_data = sum(v is None for v in raw) >= 20
        stocks.append({
            't': sym,
            'name': _text(r.get('shortName')) or _text(r.get('longName')),
            'sector': _text(r.get('sector')),
            'industry': _text(r.get('industry')),
            'price': _num(r.get('currentPrice')),
            'currency': _text(r.get('currency')),
            'beta': _num(r.get('beta')),
            'raw': raw,
            'norm': [float(r[f'Norm_{m}']) for m in METRICS],
            'status': [spec_status(m, v) for m, v in zip(METRICS, raw)],
            'conds': [bool(r[k]) for k, _, _ in BULLISH_CONDS],
            'bullCount': int(r['BullishCount']),
            'bull': float(r['BullishScore']),
            'bearish': bool(r['BearishFlag']),
            'hits': [bool(r[f'Hit_{k}']) for k, _, _, _ in PENALTY_RULES],
            'residual': 1.0,
            'penalty': float(r['Penalty']),
            'rawScore': float(r['RawScore']),
            'score': float(r['Score']),
            'rank': int(r['Rank']),
            'missing': sum(v is None for v in raw),
            'noData': bool(no_data),
            'error': errors.get(sym),
        })
    return {
        'asOf': datetime.now(timezone.utc).isoformat(timespec='seconds'),
        'source': source,
        'preview': False,
        'stocks': stocks,
    }


def run(tickers, weights: dict | None = None, cache: dict | None = None, fetcher=None,
        model: dict | None = None) -> dict:
    """Fetch, score and package a ranking for the given symbols.

    model: the app's weights/penalty {'w', 'k', 'on'}. When given and not the
    default, the ranking is scored with it in Python, and each stock also
    carries the default-model score/rank for comparison.
    """
    symbols = clean_tickers(tickers)
    if not symbols:
        raise ValueError('Add at least one symbol.')
    if len(symbols) > 60:
        raise ValueError('Up to 60 symbols per run.')
    infos, errors = (fetcher or fetch_info)(symbols, cache=cache)
    if len(errors) == len(symbols):
        first = next(iter(errors.values()))
        raise RuntimeError(f'Yahoo Finance returned no data for any symbol ({first}).')
    frame = build_frame(infos, symbols)
    if model:
        weights = weights_from_model(model) or weights
        pen = {'k': model.get('k', 1.0), 'on': model.get('on')}
    else:
        pen = None
    df = score_frame(frame, weights, pen)
    payload = build_payload(df, errors)
    custom = (weights is not None and weights != DEFAULT_WEIGHTS) or \
        (pen is not None and not _is_default_penalty(pen))
    if custom:
        base = score_frame(frame)
        for st in payload['stocks']:
            r = base.loc[st['t']]
            st['def'] = {'score': float(r['Score']), 'rank': int(r['Rank']),
                         'penalty': float(r['Penalty']), 'rawScore': float(r['RawScore'])}
        payload['model'] = {
            'w': [int(round(weights[f'Norm_{m}'] * 1000)) for m in METRICS]
            if weights is not None else None,
            'k': float(pen['k']) if pen else 1.0,
            'on': list(pen['on']) if pen and pen.get('on') else [True] * len(PENALTY_RULES),
        }
    else:
        payload['model'] = None
    return payload
