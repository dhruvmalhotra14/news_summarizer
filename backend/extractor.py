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
    """Extracts article body from JSON-LD schema blocks."""
    for script in soup.find_all("script", type="application/ld+json"):
        if not script.string:
            continue
        try:
            data = json.loads(script.string.strip())
            items = data if isinstance(data, list) else [data]
            for item in items:
                if isinstance(item, dict):
                    # Check direct articleBody
                    if item.get("articleBody"):
                        return item["articleBody"].strip()
                    # Check nested @graph
                    for graph_item in item.get("@graph", []):
                        if isinstance(graph_item, dict) and graph_item.get("articleBody"):
                            return graph_item["articleBody"].strip()
        except Exception:
            continue
    return ""


def _extract_soup_paragraphs(soup: BeautifulSoup) -> str:
    """Fallback paragraph extraction targeting main article containers."""
    for tag in soup(["script", "style", "nav", "header", "footer", "aside", "form"]):
        tag.decompose()

    # Priority article containers
    container = soup.find(["article", "main"]) or soup.find(
        "div",
        class_=re.compile(
            r"(story-details|story_details|article-body|article__body|content-body|entry-content|art-content)",
            re.IGNORECASE,
        ),
    )

    target = container if container else soup
    paragraphs = [
        p.get_text(" ", strip=True)
        for p in target.find_all("p")
        if len(p.get_text(strip=True)) > 45
    ]

    return "\n\n".join(paragraphs) if len(paragraphs) >= 2 else ""


def _fetch_html(url: str) -> str:
    """
    Multi-tier fetch engine:
    1. curl_cffi (impersonate chrome120 without conflicting custom headers)
    2. Trafilatura native fetch_url
    """
    # Attempt 1: curl_cffi TLS impersonation
    try:
        resp = curl_requests.get(
            url,
            impersonate="chrome120",
            timeout=12,
            headers={
                "Referer": "https://www.google.com/",
                "Accept-Language": "en-US,en;q=0.9",
            },
        )
        if resp.status_code == 200 and len(resp.text) > 1000:
            return resp.text
    except Exception:
        pass

    # Attempt 2: Trafilatura built-in fetcher
    try:
        downloaded = trafilatura.fetch_url(url)
        if downloaded and len(downloaded) > 1000:
            return downloaded
    except Exception:
        pass

    return ""


def extract_article(url: str) -> str:
    """
    Main extraction pipeline:
    1. Multi-tier HTML fetch
    2. JSON-LD Schema parsing
    3. Trafilatura precision extraction
    4. BeautifulSoup DOM paragraph parsing
    5. Jina Reader fallback (filtered against bot blocks)
    """
    html = _fetch_html(url)

    if html:
        soup = BeautifulSoup(html, "html.parser")

        # 1. Check for JSON-LD schema
        json_ld_text = _extract_json_ld(soup)
        if json_ld_text and len(json_ld_text) >= 300:
            return _clean_text(json_ld_text)

        # 2. Check Trafilatura
        traf_text = trafilatura.extract(
            html,
            include_comments=False,
            include_tables=False,
            no_fallback=False,
        )
        if traf_text and len(traf_text.strip()) >= 300:
            return _clean_text(traf_text)

        # 3. Check BeautifulSoup DOM
        dom_text = _extract_soup_paragraphs(soup)
        if dom_text and len(dom_text.strip()) >= 300:
            return _clean_text(dom_text)

    # 4. Fallback: Jina Reader Proxy
    try:
        jina_url = f"https://r.jina.ai/{url}"
        jina_resp = curl_requests.get(
            jina_url,
            impersonate="chrome120",
            timeout=15,
        )
        if jina_resp.status_code == 200:
            content = jina_resp.text.strip()
            blocked_markers = [
                "403 Forbidden",
                "Access Denied",
                "Robot or human?",
                "Cloudflare",
                "Attention Required!",
            ]
            if not any(marker in content for marker in blocked_markers) and len(content) >= 300:
                return _clean_text(content)
    except Exception:
        pass

    return ""