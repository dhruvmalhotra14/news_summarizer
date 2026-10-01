import json
import re
from bs4 import BeautifulSoup
from curl_cffi import requests as curl_requests
import trafilatura

BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Referer": "https://www.google.com/",
    "Sec-Ch-Ua": '"Chromium";v="124", "Google Chrome";v="124", "Not-A.Brand";v="99"',
    "Sec-Ch-Ua-Mobile": "?0",
    "Sec-Ch-Ua-Platform": '"Windows"',
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "cross-site",
    "Sec-Fetch-User": "?1",
    "Upgrade-Insecure-Requests": "1",
}


def _extract_from_json_ld(soup: BeautifulSoup) -> str:
    """Attempts to pull articleBody directly from JSON-LD schema blocks."""
    for script in soup.find_all("script", type="application/ld+json"):
        try:
            data = json.loads(script.string or "")
            items = data if isinstance(data, list) else [data]
            for item in items:
                if isinstance(item, dict):
                    # Check for articleBody in Article / NewsArticle schema
                    if "articleBody" in item and item["articleBody"]:
                        return item["articleBody"].strip()
                    # Check graph array if present
                    if "@graph" in item and isinstance(item["@graph"], list):
                        for sub_item in item["@graph"]:
                            if isinstance(sub_item, dict) and "articleBody" in sub_item:
                                return sub_item["articleBody"].strip()
        except Exception:
            continue
    return ""


def _extract_from_dom(soup: BeautifulSoup) -> str:
    """Fallback manual extraction for Indian Express and similar news structures."""
    # Common container selectors for Indian Express articles
    content_containers = soup.find_all(
        "div",
        class_=re.compile(
            r"(story-details|full-details|story_details|ie-content|art-content|wp-block-post-content)",
            re.IGNORECASE,
        ),
    )

    if content_containers:
        for container in content_containers:
            paragraphs = [
                p.get_text(strip=True)
                for p in container.find_all("p")
                if len(p.get_text(strip=True)) > 40
            ]
            if len(paragraphs) >= 3:
                return "\n\n".join(paragraphs)

    # General fallback: collect all main article paragraphs
    paragraphs = [
        p.get_text(strip=True)
        for p in soup.find_all("p")
        if len(p.get_text(strip=True)) > 50
    ]
    if len(paragraphs) >= 3:
        return "\n\n".join(paragraphs)

    return ""


def extract_article(url: str) -> str:
    """
    Multi-layer article extraction pipeline:
    1. curl_cffi with Chrome 124 TLS impersonation & browser headers
    2. JSON-LD schema extraction (fastest, cleanest article text)
    3. Trafilatura body parsing
    4. BeautifulSoup DOM paragraph parsing
    5. Jina Reader fallback (only if valid article text returned)
    """
    html_content = None

    # Step 1: Fetch with TLS Fingerprint Impersonation
    try:
        response = curl_requests.get(
            url,
            headers=BROWSER_HEADERS,
            impersonate="chrome124",
            timeout=15,
            follow_redirects=True,
        )
        if response.status_code == 200 and response.text:
            html_content = response.text
    except Exception:
        html_content = None

    if html_content:
        soup = BeautifulSoup(html_content, "html.parser")

        # Step 2: Try JSON-LD schema
        json_ld_text = _extract_from_json_ld(soup)
        if json_ld_text and len(json_ld_text) >= 300:
            return json_ld_text

        # Step 3: Try Trafilatura
        trafilatura_text = trafilatura.extract(
            html_content,
            include_comments=False,
            include_tables=False,
            no_fallback=False,
        )
        if trafilatura_text and len(trafilatura_text.strip()) >= 300:
            return trafilatura_text.strip()

        # Step 4: DOM fallback
        dom_text = _extract_from_dom(soup)
        if dom_text and len(dom_text.strip()) >= 300:
            return dom_text.strip()

    # Step 5: Jina Reader Proxy fallback (guarded against error strings)
    try:
        jina_url = f"https://r.jina.ai/{url}"
        jina_response = curl_requests.get(
            jina_url,
            headers={"User-Agent": BROWSER_HEADERS["User-Agent"]},
            impersonate="chrome124",
            timeout=15,
        )
        if jina_response.status_code == 200:
            content = jina_response.text.strip()
            # Verify Jina didn't just return an error page payload
            forbidden_markers = ["403 Forbidden", "Access Denied", "Cloudflare", "Robot or human?"]
            if not any(marker in content for marker in forbidden_markers) and len(content) >= 300:
                return content
    except Exception:
        pass

    return ""