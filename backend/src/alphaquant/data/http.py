"""HTTP session with retries, backoff and timeouts for data providers."""
from __future__ import annotations

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

DEFAULT_TIMEOUT = 30
USER_AGENT = "AlphaQuant/0.3 (+https://github.com/ShadowBlack33/AlphaQuant)"


def build_session(total_retries: int = 5, backoff: float = 1.0) -> requests.Session:
    retry = Retry(
        total=total_retries,
        backoff_factor=backoff,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET",),
        respect_retry_after_header=True,
        raise_on_status=False,
    )
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT, "Accept": "application/json"})
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    return session
