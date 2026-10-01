import json
import re
from bs4 import BeautifulSoup
import trafilatura
from curl_cffi import requests as curl_requests


def _clean_text(text: str) -> str:
    """Removes excessive newlines, ads, and whitespace."""
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
    """Fallback paragraph extraction targeting main article containers."""
    for tag in soup(["script", "style", "nav", "header", "footer", "aside", "form"]):
        tag.decompose()

    # Match Indian Express and common Indian news layout containers
    container = soup.find(
        "div",
        class_=re.compile(
            r"(story-details|full-details|story_details|ie-content|art-content|wp-block-post-content|article-body)",
            re.IGNORECASE,
        ),
    ) or soup.find(["article", "main"])

    target = container if container else soup
    paragraphs = [
        p.get_text(" ", strip=True)
        for p in target.find_all("p")
        if len(p.get_text(strip=True)) > 40
        and not p.get_text(strip=True).startswith(("Also Read", "Click here", "Subscribe"))
    ]

    return "\n\n".join(paragraphs) if len(paragraphs) >= 2 else ""


def _fetch_html(url: str) -> str:
    """
    Tiered HTML fetching engine:
    1. curl_cffi with Google Referer and realistic Chrome desktop fingerprint
    2. curl_cffi with Googlebot mobile user agent (bypasses Cloudflare on Indian Express)
    3. Trafilatura native fetch_url
    """
    # Attempt 1: Chrome 120 impersonation with search referer
    try:
        resp = curl_requests.get(
            url,
            impersonate="chrome120",
            timeout=12,
            headers={
                "Referer": "https://www.google.com/",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.9",
            },
        )
        if resp.status_code == 200 and len(resp.text) > 1200:
            return resp.text
    except Exception:
        pass

    # Attempt 2: Mobile Chrome profile (often completely bypasses CDN challenges)
    try:
        resp = curl_requests.get(
            url,
            impersonate="chrome120",
            timeout=12,
            headers={
                "User-Agent": "Mozilla/5.0 (Linux; Android 10; K) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Mobile Safari/537.36",
                "Referer": "https://m.facebook.com/",
                "Accept-Language": "en-IN,en;q=0.9",
            },
        )
        if resp.status_code == 200 and len(resp.text) > 1200:
            return resp.text
    except Exception:
        pass

    # Attempt 3: Trafilatura built-in fetcher
    try:
        downloaded = trafilatura.fetch_url(url)
        if downloaded and len(downloaded) > 1000:
            return downloaded
    except Exception:
        pass

    return ""


def extract_article(url: str) -> str:
    """
    Multi-stage extraction pipeline:
    1. Direct HTML fetch + parsing (JSON-LD -> Trafilatura -> DOM)
    2. Fallback via clean Markdown proxy services (txtify.it & Jina)
    """
    html = _fetch_html(url)

    if html:
        soup = BeautifulSoup(html, "html.parser")

        # 1. JSON-LD schema (most accurate and fast)
        json_ld_text = _extract_json_ld(soup)
        if json_ld_text and len(json_ld_text) >= 300:
            return _clean_text(json_ld_text)

        # 2. Trafilatura precision extraction
        traf_text = trafilatura.extract(
            html,
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

    # 4. Proxy Fallback A: Jina Reader Proxy (handles CDN blocks on their cloud IP pool)
    try:
        jina_url = f"https://r.jina.ai/{url}"
        jina_resp = curl_requests.get(
            jina_url,
            impersonate="chrome120",
            timeout=14,
            headers={
                "x-return-format": "text",
                "x-timeout": "10",
            }
        )
        if jina_resp.status_code == 200:
            content = jina_resp.text.strip()
            blocked_markers = ["403 Forbidden", "Access Denied", "Cloudflare", "Robot or human?"]
            if not any(marker in content for marker in blocked_markers) and len(content) >= 300:
                return _clean_text(content)
    except Exception:
        pass

    # 5. Proxy Fallback B: txtify.it reader fallback
    try:
        clean_url_no_proto = url.replace("https://", "").replace("http://", "")
        txtify_url = f"https://txtify.it/{clean_url_no_proto}"
        txt_resp = curl_requests.get(txtify_url, timeout=10)
        if txt_resp.status_code == 200 and len(txt_resp.text.strip()) >= 300:
            return _clean_text(txt_resp.text.strip())
    except Exception:
        pass

    return ""