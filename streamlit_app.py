"""
Stock Ranker - phone-first front end for the FinalAPIcode Colab notebook.

Streamlit only does three things here: check the access code, fetch data from
Yahoo Finance and run the notebook's scoring (core.py). Everything you see and
the weight simulator run in the browser (frontend/index.html), so dragging a
weight never waits for the server.
"""
from __future__ import annotations

import hashlib
import hmac
import os
import time
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

import core

ROOT = Path(__file__).resolve().parent

st.set_page_config(page_title='Stock Ranker', page_icon='📈', layout='wide',
                   initial_sidebar_state='collapsed')

# Hide Streamlit's own chrome so the app fills the phone screen.
st.markdown("""
<style>
  html, body, .stApp { background: #0B0F16; overflow: hidden !important; }
  @media (prefers-color-scheme: light) { html, body, .stApp { background: #F3F5FA; } }
  header[data-testid="stHeader"], [data-testid="stToolbar"], [data-testid="stDecoration"],
  [data-testid="stStatusWidget"], footer, #MainMenu { display: none !important; }
  .block-container, [data-testid="stMainBlockContainer"] { padding: 0 !important; max-width: 100% !important; }
  [data-testid="stVerticalBlock"] { gap: 0 !important; }
  [data-testid="stAppViewContainer"] > .main, section.main { padding: 0 !important; }
  iframe { border: 0 !important; display: block; }
  [data-stale="true"], .stale-element { opacity: 1 !important; transition: none !important; }
</style>
""", unsafe_allow_html=True)

_ui = components.declare_component('stock_ranker_ui', path=str(ROOT / 'frontend'))


def _password() -> str | None:
    try:
        v = st.secrets.get('APP_PASSWORD')
    except Exception:
        v = None
    v = v or os.environ.get('APP_PASSWORD')
    return str(v) if v else None


def _device_token(pw: str) -> str:
    """What a remembered device keeps instead of the code. Changes if the code changes."""
    return hmac.new(pw.encode(), b'stock-ranker/device/v1', hashlib.sha256).hexdigest()


@st.cache_resource
def _info_cache() -> dict:
    """Yahoo responses shared across sessions for 30 minutes."""
    return {}


def _fetcher():
    if os.environ.get('STOCK_RANKER_DEMO'):          # offline demo for local testing
        import sys
        sys.path.insert(0, str(ROOT / 'tests'))
        from excel_fixture import synthetic_infos
        _, demo = synthetic_infos()
        demo = {k.upper(): v for k, v in demo.items()}

        def fetch(symbols, cache=None):
            time.sleep(1.5)
            infos = {s: dict(demo.get(s, {}), quoteType='EQUITY') if s in demo else {} for s in symbols}
            errors = {s: 'not in demo data' for s in symbols if s not in demo}
            return infos, errors
        return fetch
    return None


ss = st.session_state
ss.setdefault('authed', False)
ss.setdefault('last_id', None)
ss.setdefault('payload', None)
ss.setdefault('payload_id', None)
ss.setdefault('error', None)
ss.setdefault('token', None)
ss.setdefault('fails', 0)

pw = _password()

event = _ui(
    authed=bool(ss.authed),
    needsSetup=pw is None,
    meta=core.model_meta() if ss.authed else None,
    payload=ss.payload if ss.authed else None,
    payloadId=ss.payload_id if ss.authed else None,
    token=ss.token if ss.authed else None,
    ack=ss.last_id,
    error=ss.error,
    key='ui',
    default=None,
)

if isinstance(event, dict) and event.get('id') and event.get('id') != ss.last_id:
    ss.last_id = event['id']
    ss.error = None
    kind = event.get('type')

    if kind == 'auth' and pw:
        ok = False
        if event.get('token'):
            ok = hmac.compare_digest(str(event['token']), _device_token(pw))
        elif event.get('password') is not None:
            ok = hmac.compare_digest(str(event['password']), pw)
        if ok:
            ss.authed, ss.token, ss.fails = True, _device_token(pw), 0
        else:
            ss.fails += 1
            time.sleep(min(5, ss.fails))                # slow down guessing
            ss.error = 'That code is not right.' if event.get('password') is not None else None

    elif kind == 'run' and ss.authed:
        try:
            ss.payload = core.run(event.get('tickers') or [], cache=_info_cache(), fetcher=_fetcher())
            ss.payload_id = event['id']
            bad = [s['t'] for s in ss.payload['stocks'] if s['noData']]
            if bad:
                ss.error = 'No usable data for ' + ', '.join(bad) + '. Check the spelling.'
        except Exception as exc:                        # shown on the Symbols screen
            ss.error = str(exc)[:300]

    st.rerun()
