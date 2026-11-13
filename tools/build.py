"""
Build the two HTML outputs from frontend/app.html.

    python tools/build.py              -> frontend/index.html (used by the Streamlit app)
    python tools/build.py --preview P  -> P: a self-contained preview page with the
                                          13 Nov 2026 Excel data baked in

Commit frontend/index.html after editing app.html (tests check it is current).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / 'frontend' / 'app.html'
OUT = ROOT / 'frontend' / 'index.html'

HEAD = ('<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n')


def component_html() -> str:
    """Full document for the Streamlit component iframe."""
    src = SRC.read_text(encoding='utf-8')
    head_end = src.index('<div id="app">')
    return (HEAD + src[:head_end] + '</head>\n<body>\n' + src[head_end:] +
            '\n</body>\n</html>\n')


def preview_payload() -> dict:
    """Ranking payload rebuilt from the notebook's Excel export (no network)."""
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / 'tests'))
    import core
    from excel_fixture import HIDDEN_RULES, load_excel, notebook_table, observable_penalty, synthetic_infos

    tickers, infos = synthetic_infos()
    df = core.score_frame(core.build_frame(infos, tickers))
    payload = core.build_payload(df, source='Excel export of 13 Nov 2026')
    payload['asOf'] = '2026-11-13T08:41:00+02:00'
    payload['preview'] = True
    xl = load_excel()
    table = notebook_table()
    hidden = [n for n, (k, *_rest) in enumerate(core.PENALTY_RULES) if k in HIDDEN_RULES]
    for s in payload['stocks']:
        row = xl.loc[s['t']]
        # momentum inputs are not in the Excel: keep the exact combined factor,
        # do not show stand-in conditions, prices or beta
        s['residual'] = float(row['Penalty']) / observable_penalty(row)
        for n in hidden:
            s['hits'][n] = False
        s['conds'] = None
        s['price'] = None
        s['beta'] = None
        s['t'] = s['t'].upper()
        s['sector'] = table[row.name]['sector']
        s['industry'] = table[row.name]['industry']
        assert abs(s['score'] - float(row['Score'])) < 1e-8 and s['rank'] == int(row['Rank'])
    return {'meta': core.model_meta(), 'payload': payload}


def preview_html(boot: dict) -> str:
    src = SRC.read_text(encoding='utf-8')
    blob = json.dumps(boot, separators=(',', ':'), allow_nan=False).replace('</', '<\\/')
    return src.replace('/*__BOOT__*/', 'window.__BOOT__ = ' + blob + ';', 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--preview', type=Path)
    a = ap.parse_args()
    OUT.write_text(component_html(), encoding='utf-8')
    print(f'wrote {OUT.relative_to(ROOT)}')
    if a.preview:
        a.preview.write_text(preview_html(preview_payload()), encoding='utf-8')
        print(f'wrote {a.preview}')


if __name__ == '__main__':
    main()
