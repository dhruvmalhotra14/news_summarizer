import trafilatura
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse

# Optional TLS spoofing via curl_cffi
try:
    from curl_cffi import requests as cffi_requests
    HAS_CURL_CFFI = True
except ImportError:
    HAS_CURL_CFFI = False

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}


def clean_text(text: str) -> str:
    if not text:
        return ""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return "\n\n".join(lines).strip()


def fetch_html(url: str) -> str:
    """Fetch raw HTML using curl_cffi (Chrome TLS fingerprint) or requests."""
    try:
        if HAS_CURL_CFFI:
            res = cffi_requests.get(url, headers=HEADERS, impersonate="chrome124", timeout=15)
            if res.status_code == 200:
                return res.text
        else:
            res = requests.get(url, headers=HEADERS, timeout=15)
            if res.status_code == 200:
                return res.text
    except Exception:
        pass
    return ""


def extract_with_trafilatura(html: str, url: str) -> str:
    """Try extracting via Trafilatura using HTML first, then direct URL fetch."""
    if html:
        text = trafilatura.extract(
            html,
            include_comments=False,
            include_tables=False,
            favor_precision=True,
            favor_recall=True,
        )
        if text and len(clean_text(text)) >= 300:
            return clean_text(text)

    # Secondary trafilatura fetch
    downloaded = trafilatura.fetch_url(url)
    if downloaded:
        text = trafilatura.extract(downloaded)
        if text and len(clean_text(text)) >= 300:
            return clean_text(text)
    return ""


def extract_with_jina(url: str) -> str:
    """Fallback using Jina Reader with explicit browser headers."""
    try:
        jina_url = f"https://r.jina.ai/{url}"
        res = requests.get(
            jina_url,
            headers={"User-Agent": "Mozilla/5.0 (compatible; ResearchBot/1.0)"},
            timeout=20,
        )
        if res.status_code == 200 and len(res.text) >= 300:
            return clean_text(res.text)
    except Exception:
        pass
    return ""


def extract_article(url: str) -> str | None:
    url = url.strip()
    if not url or not url.startswith(("http://", "https://")):
        return None

    # Step 1: Fetch HTML
    html = fetch_html(url)

    # Step 2: Try Trafilatura (handles Reuters, IE, and NDTV layouts automatically)
    text = extract_with_trafilatura(html, url)
    if text:
        return text

    # Step 3: Try Jina Reader fallback (bypasses most JS and scraper walls)
    text = extract_with_jina(url)
    if text:
        return text

    return None