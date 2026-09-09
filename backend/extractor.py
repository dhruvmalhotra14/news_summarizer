import json
import re
import requests
import trafilatura
from bs4 import BeautifulSoup
from urllib.parse import urlparse

# TLS fingerprint impersonation to bypass Cloudflare/Akamai
try:
    from curl_cffi import requests as cffi_requests
    HAS_CURL_CFFI = True
except ImportError:
    HAS_CURL_CFFI = False

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/125.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "none",
}

ERROR_PATTERNS = [
    "403 forbidden",
    "access denied",
    "just a moment...",
    "enable javascript",
    "verify you are human",
    "cloudflare",
    "subscribe to continue",
    "please turn javascript on",
]


def clean_text(text: str) -> str:
    if not text:
        return ""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    cleaned = "\n\n".join(lines).strip()

    # Reject if it's merely an anti-bot notice
    sample = cleaned[:600].lower()
    for err in ERROR_PATTERNS:
        if err in sample and len(cleaned) < 800:
            return ""

    return cleaned


def fetch_html(url: str) -> str:
    """Fetch HTML with browser TLS fingerprinting."""
    try:
        if HAS_CURL_CFFI:
            res = cffi_requests.get(
                url,
                headers=HEADERS,
                impersonate="chrome124",
                timeout=15
            )
            if res.status_code == 200:
                return res.text
        else:
            res = requests.get(url, headers=HEADERS, timeout=15)
            if res.status_code == 200:
                return res.text
    except Exception:
        pass
    return ""


def extract_from_json_ld(soup: BeautifulSoup) -> str:
    """
    Indian Express, NDTV, and Reuters put full articleBody
    in the JSON-LD script, which bypasses DOM paywalls and layout scripts.
    """
    try:
        scripts = soup.find_all("script", type="application/ld+json")
        for script in scripts:
            if not script.string:
                continue
            try:
                data = json.loads(script.string)
            except Exception:
                continue

            items = data if isinstance(data, list) else [data]
            for item in items:
                if not isinstance(item, dict):
                    continue

                # Check top-level articleBody
                body = item.get("articleBody")
                if body and len(body.strip()) > 300:
                    return clean_text(body)

                # Check inside @graph (common in WordPress / Indian Express)
                graph = item.get("@graph", [])
                if isinstance(graph, list):
                    for g in graph:
                        if isinstance(g, dict) and g.get("articleBody"):
                            body = g.get("articleBody")
                            if len(body.strip()) > 300:
                                return clean_text(body)
    except Exception:
        pass
    return ""


def extract_with_jina(url: str) -> str:
    """Bypass using Jina's proxy reader."""
    try:
        res = requests.get(
            f"https://r.jina.ai/{url}",
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"},
            timeout=25,
        )
        if res.status_code == 200:
            text = clean_text(res.text)
            if len(text) >= 400:
                return text
    except Exception:
        pass
    return ""


def extract_article(url: str) -> str | None:
    url = url.strip()
    if not url.startswith(("http://", "https://")):
        return None

    # Step 1: Direct Fetch with Browser TLS Fingerprint
    html = fetch_html(url)

    if html:
        soup = BeautifulSoup(html, "html.parser")

        # 1a. Try JSON-LD (most reliable for Indian Express & NDTV)
        json_text = extract_from_json_ld(soup)
        if json_text:
            return json_text

        # 1b. Trafilatura on fetched HTML
        traf_text = trafilatura.extract(
            html,
            include_comments=False,
            include_tables=False,
            favor_precision=True
        )
        if traf_text:
            cleaned = clean_text(traf_text)
            if len(cleaned) >= 400:
                return cleaned

    # Step 2: Trafilatura direct URL download
    downloaded = trafilatura.fetch_url(url)
    if downloaded:
        direct_text = trafilatura.extract(downloaded)
        if direct_text:
            cleaned = clean_text(direct_text)
            if len(cleaned) >= 400:
                return cleaned

    # Step 3: Jina Reader Proxy (Bypasses Cloudflare on Reuters)
    jina_text = extract_with_jina(url)
    if jina_text:
        return jina_text

    return None