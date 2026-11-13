"""The new code must reproduce the Colab notebook's own Excel output."""
import math

import numpy as np
import pytest

import core
from excel_fixture import load_excel, notebook_table, synthetic_infos


@pytest.fixture(scope='module')
def scored():
    tickers, infos = synthetic_infos()
    return core.score_frame(core.build_frame(infos, tickers))


def test_weights_sum_like_the_notebook():
    assert len(core.DEFAULT_WEIGHTS) == 29
    assert abs(sum(core.DEFAULT_WEIGHTS.values()) + core.BULLISH_WEIGHT - 1.0) < 1e-9


def test_raw_metrics_match_excel(scored):
    xl = load_excel()
    for m in core.METRICS:
        a, b = scored[m].astype(float).to_numpy(), xl[m].astype(float).to_numpy()
        assert np.allclose(a, b, rtol=1e-9, atol=1e-12, equal_nan=True), m


def test_norms_match_excel(scored):
    xl = load_excel()
    for m in core.METRICS:
        assert np.allclose(scored[f'Norm_{m}'].to_numpy(), xl[f'Norm_{m}'].to_numpy(), atol=1e-9), m


def test_penalty_score_rank_match_excel(scored):
    xl = load_excel()
    assert np.allclose(scored['Penalty'].to_numpy(), xl['Penalty'].to_numpy(), atol=1e-12)
    assert np.allclose(scored['Score'].to_numpy(), xl['Score'].to_numpy(), atol=1e-8)
    assert list(scored['Rank']) == list(xl['Rank'])


def test_bullish_score_matches_notebook_output(scored):
    table = notebook_table()
    for sym, r in scored.iterrows():
        assert abs(r['BullishScore'] - table[sym]['bull']) < 1e-9, sym
        assert abs(r['Score'] - table[sym]['score']) < 1e-5, sym
        assert int(r['Rank']) == table[sym]['rank'], sym


def test_weights_must_total_one(scored):
    tickers, infos = synthetic_infos()
    bad = dict(core.DEFAULT_WEIGHTS)
    bad['Norm_profitMargins'] += 0.01
    with pytest.raises(ValueError):
        core.score_frame(core.build_frame(infos, tickers), bad)


def test_payload_is_json_safe(scored):
    import json
    payload = core.build_payload(scored)
    text = json.dumps(payload, allow_nan=False)
    assert len(payload['stocks']) == 27 and 'NaN' not in text
    for s in payload['stocks']:
        assert len(s['raw']) == len(s['norm']) == len(s['status']) == 29
        assert len(s['hits']) == 16 and len(s['conds']) == 6


def test_spec_status_rules():
    assert core.spec_status('grossMargins', 0.92) == 'exceptional'   # >= 55% * 1.5
    assert core.spec_status('grossMargins', 0.60) == 'pass'
    assert core.spec_status('grossMargins', 0.40) == 'fail'
    assert core.spec_status('grossMargins', float('nan')) == 'nodata'
    assert core.spec_status('ebitdaMargins', 0.50) == 'outside'      # above 20-38%, higher is better
    assert core.spec_status('ebitdaMargins', 0.10) == 'fail'
    assert core.spec_status('trailingPE', 31.0) == 'fail'
    assert core.spec_status('trailingPE', 0.5) == 'outside'
    assert core.spec_status('debtToEquity', 154.3) == 'fail'
    assert core.spec_status('debtToEquity', 40.0) == 'exceptional'
    assert core.spec_status('payoutRatio', 0.0) == 'pass'            # no dividend is not a strength
    assert core.spec_status('NetCash', 2.6) == 'pass'                # zero threshold: pass / fail only
    assert core.spec_status('dividendYield', 0.74) == 'fail'         # 0.74% vs >= 2%
    assert core.spec_status('dividendYield', 3.4) == 'exceptional'   # 3.4% vs >= 2%


def test_clean_tickers():
    assert core.clean_tickers([' amd', 'AMD', 'brk-b', '', 'bad symbol!', 'ILS=X']) == ['AMD', 'BRK-B', 'ILS=X']


def test_frontend_index_is_built():
    import importlib.util
    from pathlib import Path
    root = Path(__file__).resolve().parent.parent
    spec = importlib.util.spec_from_file_location('build', root / 'tools' / 'build.py')
    build = importlib.util.module_from_spec(spec); spec.loader.exec_module(build)
    assert (root / 'frontend' / 'index.html').read_text(encoding='utf-8') == build.component_html(), \
        'run: python tools/build.py'


def test_preview_payload_matches_excel():
    import importlib.util
    from pathlib import Path
    root = Path(__file__).resolve().parent.parent
    spec = importlib.util.spec_from_file_location('build', root / 'tools' / 'build.py')
    build = importlib.util.module_from_spec(spec); spec.loader.exec_module(build)
    boot = build.preview_payload()          # asserts score and rank per stock
    stocks = boot['payload']['stocks']
    for s in stocks:                        # residual x observable hits == notebook penalty
        p = s['residual']
        for hit, rule in zip(s['hits'], boot['meta']['rules']):
            if hit:
                p *= rule['mult']
        assert abs(p - s['penalty']) < 1e-12, s['t']
