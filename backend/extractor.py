import json
import re
import requests
import trafilatura
from bs4 import BeautifulSoup
from urllib.parse import urlparse

# Optional TLS spoofing for Cloudflare-protected sites like Indian Express
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
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "en-IN,en;q=0.9,hi;q=0.8",
    "Referer": "https://www.google.com/",
}

# Strictly reject error pages/bot challenges so they NEVER get passed to Groq
ERROR_KEYWORDS = [
    "unable to retrieve",
    "could not retrieve",
    "403 forbidden",
    "access denied",
    "just a moment...",
    "verify you are human",
    "cloudflare",
    "please enable cookies",
    "turn javascript on",
]


def clean_text(text: str) -> str:
    """Cleans text and strictly blocks scraper error messages."""
    if not text:
        return ""

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    cleaned = "\n\n".join(lines).strip()

    # If the text contains ANY scraper failure keywords, reject it completely
    text_lower = cleaned.lower()
    for err in ERROR_KEYWORDS:
        if err in text_lower and len(cleaned) < 1200:
            return ""

    return cleaned


def fetch_html(url: str) -> str:
    """Fetch raw HTML using curl_cffi (Chrome TLS fingerprint) or requests."""
    try:
        if HAS_CURL_CFFI:
            res = cffi_requests.get(url, headers=HEADERS, impersonate="chrome124", timeout=15)
            if res.status_code == 200 and len(res.text) > 1000:
                return res.text
    except Exception:
        pass

    try:
        res = requests.get(url, headers=HEADERS, timeout=15)
        if res.status_code == 200:
            return res.text
    except Exception:
        pass

    return ""


def extract_json_ld(soup: BeautifulSoup) -> str:
    """Extract articleBody from structured schema (used heavily by Indian Express & NDTV)."""
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

                body = item.get("articleBody")
                if body and len(body.strip()) >= 350:
                    return clean_text(body)

                graph = item.get("@graph", [])
                if isinstance(graph, list):
                    for node in graph:
                        if isinstance(node, dict) and node.get("articleBody"):
                            body = node.get("articleBody")
                            if len(body.strip()) >= 350:
                                return clean_text(body)
    except Exception:
        pass
    return ""


def extract_ndtv(soup: BeautifulSoup) -> str:
    """Dedicated NDTV DOM parser."""
    selectors = [
        "div[class*='sp-cn']",
        "div[class*='story__content']",
        "div[class*='article__content']",
        "div[class*='story']",
        "div[class*='content']",
        "article",
    ]
    for selector in selectors:
        for element in soup.select(selector):
            paragraphs = element.find_all("p")
            text = "\n\n".join(p.get_text(" ", strip=True) for p in paragraphs if len(p.get_text(strip=True)) > 20)
            cleaned = clean_text(text)
            if len(cleaned) >= 350:
                return cleaned
    return ""


def extract_indian_express(soup: BeautifulSoup) -> str:
    """Dedicated Indian Express DOM parser."""
    # Indian Express holds the main story inside pcl-full-content or story-details
    content_div = (
        soup.find("div", id="pcl-full-content")
        or soup.find("div", class_="story-details")
        or soup.find("div", class_="story__content")
        or soup.find("article")
    )
    if content_div:
        # Remove unwanted newsletter/ads widgets inside story
        for tag in content_div(["script", "style", "aside", "nav", "figure"]):
            tag.decompose()
        paragraphs = content_div.find_all("p")
        text = "\n\n".join(p.get_text(" ", strip=True) for p in paragraphs if len(p.get_text(strip=True)) > 20)
        cleaned = clean_text(text)
        if len(cleaned) >= 350:
            return cleaned
    return ""


def extract_article(url: str) -> str | None:
    url = url.strip()
    if not url.startswith(("http://", "https://")):
        return None

    hostname = urlparse(url).netloc.lower()

    # Step 1: Fetch raw page
    html = fetch_html(url)
    if html:
        soup = BeautifulSoup(html, "html.parser")

        # 1a. Try JSON-LD (Cleanest extraction for both NDTV & Indian Express)
        json_text = extract_json_ld(soup)
        if json_text:
            return json_text

        # 1b. Site-specific fallbacks
        if "ndtv.com" in hostname:
            ndtv_text = extract_ndtv(soup)
            if ndtv_text:
                return ndtv_text

        if "indianexpress.com" in hostname:
            ie_text = extract_indian_express(soup)
            if ie_text:
                return ie_text

        # 1c. General Trafilatura extraction
        traf_text = trafilatura.extract(
            html,
            include_comments=False,
            include_tables=False,
            favor_precision=True,
        )
        if traf_text:
            cleaned = clean_text(traf_text)
            if len(cleaned) >= 350:
                return cleaned

    # Step 2: Direct Trafilatura URL Fetcher
    try:
        downloaded = trafilatura.fetch_url(url)
        if downloaded:
            direct_text = trafilatura.extract(downloaded)
            if direct_text:
                cleaned = clean_text(direct_text)
                if len(cleaned) >= 350:
                    return cleaned
    except Exception:
        pass

    # Step 3: Fallback via Jina Reader (with strict rejection of error pages)
    try:
        res = requests.get(
            f"https://r.jina.ai/{url}",
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"},
            timeout=20,
        )
        if res.status_code == 200:
            cleaned = clean_text(res.text)
            if len(cleaned) >= 400:
                return cleaned
    except Exception:
        pass

    return None