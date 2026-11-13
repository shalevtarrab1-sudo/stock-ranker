"""Server-side runs with the app's weights / penalty settings."""
import pytest

import core
from excel_fixture import synthetic_infos


@pytest.fixture(scope='module')
def fixture_run():
    tickers, infos = synthetic_infos()
    by_upper = {k.upper(): v for k, v in infos.items()}

    def fetch(symbols, cache=None):
        return {t: by_upper[t] for t in symbols}, {}

    return lambda model: core.run(tickers, fetcher=fetch, model=model)


def _default_w():
    return [int(round(core.DEFAULT_WEIGHTS[f'Norm_{m}'] * 1000)) for m in core.METRICS]


def test_default_model_is_notebook(fixture_run):
    base = fixture_run(None)
    same = fixture_run({'w': _default_w(), 'k': 1.0, 'on': [True] * 16})
    assert same['model'] is None
    assert [s['rank'] for s in base['stocks']] == [s['rank'] for s in same['stocks']]


def test_custom_weights_change_ranking(fixture_run):
    w = _default_w()
    i, j = core.METRICS.index('PriceToTarget'), core.METRICS.index('dividendYield')
    move = min(w[i], 50)
    w[i] -= move
    w[j] += move
    p = fixture_run({'w': w, 'k': 0.5, 'on': [True] * 15 + [False]})
    assert p['model']['w'] == w and p['model']['k'] == 0.5
    base = fixture_run(None)
    for s, b in zip(p['stocks'], base['stocks']):
        assert s['def']['rank'] == b['rank']
        assert abs(s['def']['score'] - b['score']) < 1e-9
    assert any(abs(s['score'] - b['score']) > 1e-6 for s, b in zip(p['stocks'], base['stocks']))


def test_bad_weights_rejected(fixture_run):
    with pytest.raises(ValueError):
        fixture_run({'w': [1] * 29})
