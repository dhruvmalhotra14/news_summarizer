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
    if not text or len(text.strip()) < 200:
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


def _fetch_indian_express_direct(url: str) -> str:
    """
    Extracts Indian Express articles directly via WordPress REST API
    bypassing the Cloudflare HTML challenge completely.
    """
    clean_target = url.split("?")[0].rstrip("/")
    match = re.search(r"-(\d+)$", clean_target)
    
    if match:
        post_id = match.group(1)
        api_url = f"https://indianexpress.com/wp-json/wp/v2/posts/{post_id}"
        
        try:
            # Query the backend API directly with curl_cffi Chrome impersonation
            resp = curl_requests.get(
                api_url,
                impersonate="chrome120",
                timeout=12,
                headers={
                    "Accept": "application/json",
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                }
            )
            if resp.status_code == 200:
                data = resp.json()
                rendered_html = data.get("content", {}).get("rendered", "")
                if rendered_html:
                    soup = BeautifulSoup(rendered_html, "html.parser")
                    # Clean out ad blocks, widgets, and shortcodes
                    for tag in soup(["script", "style", "iframe"]):
                        tag.decompose()
                    paragraphs = [
                        p.get_text(" ", strip=True)
                        for p in soup.find_all("p")
                        if len(p.get_text(strip=True)) > 35
                        and not p.get_text(strip=True).startswith(("Also Read", "Click here", "Subscribe"))
                    ]
                    if paragraphs:
                        return _clean_text("\n\n".join(paragraphs))
        except Exception:
            pass

    return ""


def extract_article(url: str) -> str:
    clean_url = url.strip()

    # Priority route for Indian Express via internal REST API
    if "indianexpress.com" in clean_url:
        ie_text = _fetch_indian_express_direct(clean_url)
        if ie_text and not _is_error_payload(ie_text):
            return ie_text

    # Standard pipeline for all other news domains
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
            soup = BeautifulSoup(resp.text, "html.parser")
            json_ld = _extract_from_json_ld(soup)
            if json_ld and not _is_error_payload(json_ld):
                return _clean_text(json_ld)

            traf = trafilatura.extract(resp.text, no_fallback=False)
            if traf and not _is_error_payload(traf):
                return _clean_text(traf)

            dom = _extract_soup_paragraphs(soup)
            if dom and not _is_error_payload(dom):
                return _clean_text(dom)
    except Exception:
        pass

    # Trafilatura native fallback
    try:
        html = trafilatura.fetch_url(clean_url)
        if html:
            traf = trafilatura.extract(html, no_fallback=False)
            if traf and not _is_error_payload(traf):
                return _clean_text(traf)
    except Exception:
        pass

    return ""