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


# =====================================================
# INDIAN EXPRESS SPECIFIC FETCHERS
# =====================================================

def _fetch_indian_express_direct(url: str) -> str:
    """WordPress REST API se article uthata hai (HTML challenge bypass)."""
    clean_target = url.split("?")[0].rstrip("/")
    match = re.search(r"-(\d+)$", clean_target)
    if not match:
        return ""

    post_id = match.group(1)
    api_url = f"https://indianexpress.com/wp-json/wp/v2/posts/{post_id}"

    try:
        resp = curl_requests.get(
            api_url,
            impersonate="chrome120",
            timeout=12,
            headers={"Accept": "application/json"},
        )
        print("IE API status:", resp.status_code)  # debug: ho jaye to hata dena
        if resp.status_code == 200:
            data = resp.json()
            rendered_html = data.get("content", {}).get("rendered", "")
            if rendered_html:
                soup = BeautifulSoup(rendered_html, "html.parser")
                for tag in soup(["script", "style", "iframe"]):
                    tag.decompose()
                paragraphs = [
                    p.get_text(" ", strip=True)
                    for p in soup.find_all("p")
                    if len(p.get_text(strip=True)) > 35
                    and not p.get_text(strip=True).startswith(
                        ("Also Read", "Click here", "Subscribe")
                    )
                ]
                if paragraphs:
                    return _clean_text("\n\n".join(paragraphs))
    except Exception as e:
        print("IE API error:", e)
    return ""


def _fetch_ie_lite(url: str) -> str:
    """Indian Express ka lite/amp version aksar kam protected hota hai."""
    base = url.split("?")[0].rstrip("/")
    for variant in (base + "/lite/", base + "/amp/"):
        try:
            resp = curl_requests.get(variant, impersonate="chrome120", timeout=12)
            print("IE variant", variant, resp.status_code)  # debug
            if resp.status_code == 200:
                text = trafilatura.extract(resp.text, no_fallback=False)
                if text and not _is_error_payload(text):
                    return _clean_text(text)
        except Exception:
            continue
    return ""


def _fetch_via_jina(url: str) -> str:
    """r.jina.ai free reader proxy, aksar Cloudflare wali sites bhi padh leta hai."""
    try:
        resp = curl_requests.get(
            f"https://r.jina.ai/{url}",
            timeout=25,
            headers={"Accept": "text/plain"},
        )
        print("Jina status:", resp.status_code)  # debug
        if resp.status_code == 200 and resp.text:
            text = resp.text
            if "Markdown Content:" in text:
                text = text.split("Markdown Content:", 1)[1]
            return _clean_text(text)
    except Exception as e:
        print("Jina error:", e)
    return ""


def _extract_from_html(html: str) -> str:
    """Run the full extraction pipeline on raw HTML."""
    soup = BeautifulSoup(html, "html.parser")

    json_ld = _extract_from_json_ld(soup)
    if json_ld and not _is_error_payload(json_ld):
        return _clean_text(json_ld)

    traf = trafilatura.extract(html, no_fallback=False)
    if traf and not _is_error_payload(traf):
        return _clean_text(traf)

    dom = _extract_soup_paragraphs(soup)
    if dom and not _is_error_payload(dom):
        return _clean_text(dom)
    return ""


def _fetch_via_scraper_api(url: str) -> str:
    """
    Paid/free-tier scraping API (ScraperAPI) that uses residential proxies.
    Needs SCRAPER_API_KEY in .streamlit/secrets.toml
    """
    try:
        import streamlit as st
        api_key = st.secrets.get("SCRAPER_API_KEY")
    except Exception:
        api_key = None
    if not api_key:
        return ""

    try:
        resp = curl_requests.get(
            "https://api.scraperapi.com/",
            params={
                "api_key": api_key,
                "url": url,
                "premium": "true",
                "country_code": "in",
            },
            timeout=60,
        )
        print("ScraperAPI status:", resp.status_code)  # debug
        if resp.status_code == 200:
            return _extract_from_html(resp.text)
    except Exception as e:
        print("ScraperAPI error:", e)
    return ""


def _fetch_via_wayback(url: str) -> str:
    """Archived copy from the Wayback Machine (works only if the page was archived)."""
    try:
        resp = curl_requests.get(
            f"https://web.archive.org/web/2/{url}",
            impersonate="chrome120",
            timeout=25,
        )
        print("Wayback status:", resp.status_code)  # debug
        if resp.status_code == 200:
            return _extract_from_html(resp.text)
    except Exception as e:
        print("Wayback error:", e)
    return ""


# =====================================================
# MAIN ENTRY
# =====================================================

def extract_article(url: str) -> str:
    clean_url = url.strip()

    # Indian Express: fallback chain
    if "indianexpress.com" in clean_url:
        for fetcher in (
            _fetch_indian_express_direct,
            _fetch_ie_lite,
            _fetch_via_scraper_api,
            _fetch_via_jina,
            _fetch_via_wayback,
        ):
            ie_text = fetcher(clean_url)
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

    # Last resort: Jina for any site
    jina_text = _fetch_via_jina(clean_url)
    if jina_text and not _is_error_payload(jina_text):
        return jina_text

    return ""