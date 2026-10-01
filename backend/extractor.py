import json
import re
import urllib.parse
from bs4 import BeautifulSoup
import trafilatura
from curl_cffi import requests as curl_requests


def _clean_text(text: str) -> str:
    """Cleans up raw parsed strings."""
    if not text:
        return ""
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    return text.strip()


def _extract_from_json_ld(soup: BeautifulSoup) -> str:
    """Attempts to pull articleBody directly from JSON-LD schema blocks."""
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
    """Extracts article text from paragraph elements."""
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
        if len(p.get_text(strip=True)) > 40
        and not p.get_text(strip=True).startswith(
            ("Also Read", "Click here", "Subscribe", "Explained |", "Follow us on")
        )
    ]

    return "\n\n".join(paragraphs) if len(paragraphs) >= 2 else ""


def _parse_html_payload(html: str) -> str:
    """Runs schema, trafilatura, and bs4 against raw html."""
    if not html or len(html) < 800:
        return ""

    soup = BeautifulSoup(html, "html.parser")

    # 1. JSON-LD
    json_ld = _extract_from_json_ld(soup)
    if json_ld and len(json_ld) >= 300:
        return _clean_text(json_ld)

    # 2. Trafilatura
    traf = trafilatura.extract(
        html,
        include_comments=False,
        include_tables=False,
        no_fallback=False,
    )
    if traf and len(traf.strip()) >= 300:
        return _clean_text(traf)

    # 3. DOM fallback
    dom = _extract_soup_paragraphs(soup)
    if dom and len(dom.strip()) >= 300:
        return _clean_text(dom)

    return ""


def extract_article(url: str) -> str:
    """
    Multi-stage resilient pipeline:
    1. Direct TLS request (works on The Hindu, NDTV, BBC, CNN, TOI)
    2. Mobile AMP endpoint
    3. Google Web Cache Mirror (bypasses Cloudflare block on Indian Express)
    4. Jina Reader Engine Proxy
    """
    clean_url = url.strip()

    # Step 1: Direct Fetch
    try:
        resp = curl_requests.get(
            clean_url,
            impersonate="chrome120",
            timeout=10,
            headers={
                "Referer": "https://www.google.com/",
                "Accept-Language": "en-US,en;q=0.9",
            },
        )
        if resp.status_code == 200:
            extracted = _parse_html_payload(resp.text)
            if extracted:
                return extracted
    except Exception:
        pass

    # Step 2: Try Trafilatura fetch
    try:
        html = trafilatura.fetch_url(clean_url)
        if html:
            extracted = _parse_html_payload(html)
            if extracted:
                return extracted
    except Exception:
        pass

    # Step 3: Google WebCache Mirror (bypasses Indian Express Cloudflare wall)
    try:
        cache_url = f"https://webcache.googleusercontent.com/search?q=cache:{clean_url}"
        cache_resp = curl_requests.get(
            cache_url,
            impersonate="chrome120",
            timeout=10,
        )
        if cache_resp.status_code == 200:
            extracted = _parse_html_payload(cache_resp.text)
            if extracted:
                return extracted
    except Exception:
        pass

    # Step 4: Archive.org WayBack fallback
    try:
        archive_api = f"https://archive.org/wayback/available?url={clean_url}"
        api_res = curl_requests.get(archive_api, timeout=6).json()
        snapshot_url = api_res.get("archived_snapshots", {}).get("closest", {}).get("url")
        if snapshot_url:
            arch_resp = curl_requests.get(snapshot_url, impersonate="chrome120", timeout=10)
            if arch_resp.status_code == 200:
                extracted = _parse_html_payload(arch_resp.text)
                if extracted:
                    return extracted
    except Exception:
        pass

    # Step 5: Jina Reader Proxy with stripped query strings
    try:
        clean_target = clean_url.split("?")[0]
        jina_url = f"https://r.jina.ai/{clean_target}"
        jina_resp = curl_requests.get(
            jina_url,
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                "x-return-format": "text",
            },
            timeout=12,
        )
        if jina_resp.status_code == 200:
            content = jina_resp.text.strip()
            if "403 Forbidden" not in content and len(content) >= 300:
                return _clean_text(content)
    except Exception:
        pass

    return ""