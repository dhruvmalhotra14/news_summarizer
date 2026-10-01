import json
import re
from bs4 import BeautifulSoup
import trafilatura
from curl_cffi import requests as curl_requests


def _clean_text(text: str) -> str:
    """Removes excessive whitespace and unwanted boilerplate lines."""
    if not text:
        return ""
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    return text.strip()


def _extract_json_ld(soup: BeautifulSoup) -> str:
    """Extracts clean article body directly from schema script tags."""
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
    """Fallback extraction targeting core article text nodes."""
    for tag in soup(["script", "style", "nav", "header", "footer", "aside", "form"]):
        tag.decompose()

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
        and not p.get_text(strip=True).startswith(("Also Read", "Click here", "Subscribe", "Explained |"))
    ]

    return "\n\n".join(paragraphs) if len(paragraphs) >= 2 else ""


def _try_fetch(url: str, headers: dict = None) -> str:
    """Safe helper to fetch text via curl_cffi with Chrome TLS impersonation."""
    try:
        resp = curl_requests.get(
            url,
            impersonate="chrome120",
            timeout=10,
            headers=headers or {"Referer": "https://www.google.com/"},
            follow_redirects=True,
        )
        if resp.status_code == 200 and len(resp.text) > 1000:
            return resp.text
    except Exception:
        pass
    return ""


def extract_article(url: str) -> str:
    """
    High-resilience extraction pipeline:
    1. AMP / Lite direct bypass (essential for Indian Express & Paywalled news)
    2. Direct curl_cffi fetch with browser TLS impersonation
    3. Trafilatura native fetch
    4. Jina Markdown Proxy
    5. r.jina.ai / txtify.it clean text fallbacks
    """
    clean_url = url.strip()
    html_content = ""

    # Strategy 1: If Indian Express, test the lightweight /lite/ or /amp/ endpoint first
    if "indianexpress.com" in clean_url:
        amp_url = clean_url.rstrip("/") + "/lite/"
        html_content = _try_fetch(amp_url)
        if not html_content:
            amp_url_alt = clean_url.rstrip("/") + "/amp/"
            html_content = _try_fetch(amp_url_alt)

    # Strategy 2: Direct fetch with Google Referer & TLS impersonation
    if not html_content:
        html_content = _try_fetch(clean_url)

    # Strategy 3: Trafilatura built-in fetcher
    if not html_content:
        try:
            downloaded = trafilatura.fetch_url(clean_url)
            if downloaded and len(downloaded) > 1000:
                html_content = downloaded
        except Exception:
            pass

    # Parse HTML if any fetch was successful
    if html_content:
        soup = BeautifulSoup(html_content, "html.parser")

        # 1. JSON-LD schema (most accurate)
        json_ld_text = _extract_json_ld(soup)
        if json_ld_text and len(json_ld_text) >= 300:
            return _clean_text(json_ld_text)

        # 2. Trafilatura extraction
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

    # Strategy 4: Proxy Fallback via Jina Reader Engine
    try:
        jina_url = f"https://r.jina.ai/{clean_url}"
        jina_resp = curl_requests.get(
            jina_url,
            impersonate="chrome120",
            timeout=14,
            headers={
                "x-return-format": "text",
                "x-no-cache": "true",
            },
        )
        if jina_resp.status_code == 200:
            content = jina_resp.text.strip()
            blocked_markers = ["403 Forbidden", "Access Denied", "Cloudflare", "Robot or human?"]
            if not any(m.lower() in content.lower() for m in blocked_markers) and len(content) >= 300:
                return _clean_text(content)
    except Exception:
        pass

    # Strategy 5: Proxy Fallback via txtify.it
    try:
        stripped_proto = clean_url.replace("https://", "").replace("http://", "")
        txtify_url = f"https://txtify.it/{stripped_proto}"
        txt_resp = curl_requests.get(txtify_url, timeout=12)
        if txt_resp.status_code == 200 and len(txt_resp.text.strip()) >= 300:
            return _clean_text(txt_resp.text.strip())
    except Exception:
        pass

    return ""