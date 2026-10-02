import json
import re
from bs4 import BeautifulSoup
import trafilatura
from curl_cffi import requests as curl_requests


def _clean_text(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    return text.strip()


def _is_error_payload(text: str) -> bool:
    if not text or len(text.strip()) < 250:
        return True
    lower = text.lower()
    error_markers = [
        "403 forbidden",
        "cloudfront",
        "request could not be satisfied",
        "access denied",
        "attention required",
        "robot or human",
        "enable javascript",
        "just a moment...",
    ]
    return any(marker in lower for marker in error_markers)


def _extract_from_json_ld(soup: BeautifulSoup) -> str:
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


def _parse_html_payload(html: str) -> str:
    if not html or _is_error_payload(html):
        return ""

    soup = BeautifulSoup(html, "html.parser")

    # 1. JSON-LD
    json_ld = _extract_from_json_ld(soup)
    if json_ld and not _is_error_payload(json_ld):
        return _clean_text(json_ld)

    # 2. Trafilatura
    traf = trafilatura.extract(html, no_fallback=False)
    if traf and not _is_error_payload(traf):
        return _clean_text(traf)

    # 3. DOM fallback
    dom = _extract_soup_paragraphs(soup)
    if dom and not _is_error_payload(dom):
        return _clean_text(dom)

    return ""


def extract_article(url: str) -> str:
    clean_url = url.strip()

    # Step 1: Direct Fetch via Chrome TLS
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

    # Step 2: Trafilatura standard fetch
    try:
        html = trafilatura.fetch_url(clean_url)
        if html:
            extracted = _parse_html_payload(html)
            if extracted:
                return extracted
    except Exception:
        pass

    # Step 3: Jina Reader Proxy fallback (handles CDN/WAF blocks)
    try:
        clean_target = clean_url.split("?")[0].rstrip("/")
        jina_url = f"https://r.jina.ai/{clean_target}"
        resp = curl_requests.get(
            jina_url,
            timeout=14,
            headers={"Accept": "text/plain"}
        )
        if resp.status_code == 200 and not _is_error_payload(resp.text):
            lines = [l.strip() for l in resp.text.split("\n") if len(l.strip()) > 35]
            filtered = [
                l for l in lines
                if not l.startswith(("Title:", "URL Source:", "Markdown Content:", "http", "Also Read"))
            ]
            if len(filtered) >= 3:
                return _clean_text("\n\n".join(filtered))
    except Exception:
        pass

    return ""