import json
import re
from bs4 import BeautifulSoup
import trafilatura
from curl_cffi import requests as curl_requests


def _clean_text(text: str) -> str:
    """Removes excessive newlines and whitespace."""
    if not text:
        return ""
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    return text.strip()


def _extract_json_ld(soup: BeautifulSoup) -> str:
    """Extracts article body directly from JSON-LD schema blocks."""
    for script in soup.find_all("script", type="application/ld+json"):
        if not script.string:
            continue
        try:
            data = json.loads(script.string.strip())
            items = data if isinstance(data, list) else [data]
            for item in items:
                if isinstance(item, dict):
                    if item.get("articleBody"):
                        return item["articleBody"].strip()
                    for graph_item in item.get("@graph", []):
                        if isinstance(graph_item, dict) and graph_item.get("articleBody"):
                            return graph_item["articleBody"].strip()
        except Exception:
            continue
    return ""


def _extract_soup_paragraphs(soup: BeautifulSoup) -> str:
    """Fallback extraction targeting news article body paragraphs."""
    for tag in soup(["script", "style", "nav", "header", "footer", "aside", "form"]):
        tag.decompose()

    container = soup.find(
        "div",
        class_=re.compile(
            r"(story-details|full-details|story_details|ie-content|art-content|wp-block-post-content|article-body|storycontent)",
            re.IGNORECASE,
        ),
    ) or soup.find(["article", "main"])

    target = container if container else soup
    paragraphs = [
        p.get_text(" ", strip=True)
        for p in target.find_all("p")
        if len(p.get_text(strip=True)) > 35
        and not p.get_text(strip=True).startswith(
            ("Also Read", "Click here", "Subscribe", "Explained |", "Follow us on")
        )
    ]

    return "\n\n".join(paragraphs) if len(paragraphs) >= 2 else ""


def _fetch_page(url: str) -> str:
    """
    Tiered fetching strategy:
    1. Search Engine Crawler Identity (bypasses Indian Express / Cloudflare 403)
    2. Desktop Chrome 120 TLS fingerprint
    3. Trafilatura native fetch
    4. Free public CORS proxy fallback
    """
    # Strategy 1: Googlebot Crawler profile (Bypasses Cloudflare on Indian Express & The Hindu)
    try:
        resp = curl_requests.get(
            url,
            headers={
                "User-Agent": "Mozilla/5.0 (compatible; Googlebot/2.1; +http://www.google.com/bot.html)",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.9",
            },
            timeout=12,
            follow_redirects=True,
        )
        if resp.status_code == 200 and len(resp.text) > 1200:
            return resp.text
    except Exception:
        pass

    # Strategy 2: Modern Chrome TLS fingerprint
    try:
        resp = curl_requests.get(
            url,
            impersonate="chrome120",
            headers={"Referer": "https://www.google.com/"},
            timeout=12,
            follow_redirects=True,
        )
        if resp.status_code == 200 and len(resp.text) > 1200:
            return resp.text
    except Exception:
        pass

    # Strategy 3: Trafilatura fetcher
    try:
        downloaded = trafilatura.fetch_url(url)
        if downloaded and len(downloaded) > 1000:
            return downloaded
    except Exception:
        pass

    # Strategy 4: High-speed proxy fallback
    try:
        proxy_url = f"https://api.allorigins.win/raw?url={url}"
        resp = curl_requests.get(proxy_url, timeout=12)
        if resp.status_code == 200 and len(resp.text) > 1200:
            return resp.text
    except Exception:
        pass

    return ""


def extract_article(url: str) -> str:
    """
    Main extraction pipeline:
    Fetches raw HTML -> JSON-LD schema -> Trafilatura -> DOM BeautifulSoup
    """
    clean_url = url.strip()
    html_content = _fetch_page(clean_url)

    if not html_content:
        return ""

    soup = BeautifulSoup(html_content, "html.parser")

    # 1. JSON-LD schema extraction (purest, fastest)
    json_ld_text = _extract_json_ld(soup)
    if json_ld_text and len(json_ld_text) >= 300:
        return _clean_text(json_ld_text)

    # 2. Trafilatura precision extraction
    traf_text = trafilatura.extract(
        html_content,
        include_comments=False,
        include_tables=False,
        no_fallback=False,
    )
    if traf_text and len(traf_text.strip()) >= 300:
        return _clean_text(traf_text)

    # 3. BeautifulSoup DOM paragraphs
    dom_text = _extract_soup_paragraphs(soup)
    if dom_text and len(dom_text.strip()) >= 300:
        return _clean_text(dom_text)

    return ""