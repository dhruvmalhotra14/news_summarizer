import json
import re
import requests
import trafilatura
from bs4 import BeautifulSoup
from urllib.parse import urlparse

# Standard browser headers
BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/125.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.google.com/",
}

# Googlebot headers: news sites allow Googlebot past all paywalls/403 blocks
GOOGLEBOT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; Googlebot/2.1; +http://www.google.com/bot.html)",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

ERROR_PATTERNS = [
    "403 forbidden",
    "access denied",
    "unable to retrieve",
    "could not retrieve",
    "returns a 403",
    "just a moment...",
    "enable javascript",
    "verify you are human",
    "cloudflare",
    "subscribe to continue",
]


def clean_text(text: str) -> str:
    """Cleans text and ensures it is not an error string."""
    if not text:
        return ""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    cleaned = "\n\n".join(lines).strip()

    # Reject if it matches any bot-block or 403 failure text
    sample = cleaned[:600].lower()
    for err in ERROR_PATTERNS:
        if err in sample:
            return ""

    return cleaned


def extract_from_json_ld(soup: BeautifulSoup) -> str:
    """Extracts articleBody from JSON-LD schema (Used heavily by Indian Express & NDTV)."""
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

                # Check standard articleBody
                body = item.get("articleBody")
                if body and len(body.strip()) > 300:
                    return clean_text(body)

                # Check graph wrapper (common in WordPress / Indian Express)
                graph = item.get("@graph", [])
                if isinstance(graph, list):
                    for node in graph:
                        if isinstance(node, dict) and node.get("articleBody"):
                            body = node.get("articleBody")
                            if len(body.strip()) > 300:
                                return clean_text(body)
    except Exception:
        pass
    return ""


def extract_indian_express(url: str) -> str:
    """Special handler for Indian Express to bypass their 403 anti-bot block."""
    try:
        # Strategy A: Use Googlebot header (Indian Express serves full unblocked HTML to Googlebot)
        res = requests.get(url, headers=GOOGLEBOT_HEADERS, timeout=12)
        if res.status_code == 200:
            soup = BeautifulSoup(res.text, "html.parser")
            
            # Try JSON-LD first
            text = extract_from_json_ld(soup)
            if text:
                return text

            # Try article container
            story_div = soup.find("div", id="pcl-full-content") or soup.find("div", class_="story-details")
            if story_div:
                paragraphs = [p.get_text(strip=True) for p in story_div.find_all("p")]
                cleaned = clean_text("\n\n".join(paragraphs))
                if len(cleaned) >= 300:
                    return cleaned

        # Strategy B: Try Indian Express Lite version
        lite_url = url.replace("indianexpress.com/article/", "indianexpress.com/article/lite/")
        res_lite = requests.get(lite_url, headers=BROWSER_HEADERS, timeout=10)
        if res_lite.status_code == 200:
            soup = BeautifulSoup(res_lite.text, "html.parser")
            text = extract_from_json_ld(soup)
            if text:
                return text
            paragraphs = [p.get_text(strip=True) for p in soup.find_all("p")]
            cleaned = clean_text("\n\n".join(paragraphs))
            if len(cleaned) >= 300:
                return cleaned
    except Exception:
        pass
    return ""


def extract_with_jina(url: str) -> str:
    """Attempts extraction via Jina Reader, rejecting error strings."""
    try:
        jina_url = f"https://r.jina.ai/{url}"
        res = requests.get(
            jina_url,
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"},
            timeout=20,
        )
        if res.status_code == 200:
            cleaned = clean_text(res.text)
            if len(cleaned) >= 300:
                return cleaned
    except Exception:
        pass
    return ""


def extract_with_trafilatura(url: str) -> str:
    """Standard Trafilatura extraction."""
    try:
        downloaded = trafilatura.fetch_url(url)
        if downloaded:
            text = trafilatura.extract(
                downloaded,
                include_comments=False,
                include_tables=False,
                favor_precision=True,
            )
            if text:
                cleaned = clean_text(text)
                if len(cleaned) >= 300:
                    return cleaned
    except Exception:
        pass
    return ""


def extract_article(url: str) -> str | None:
    url = url.strip()
    if not url.startswith(("http://", "https://")):
        return None

    hostname = urlparse(url).netloc.lower()

    # 1. Site-specific priority: Indian Express
    if "indianexpress.com" in hostname:
        text = extract_indian_express(url)
        if text:
            return text

    # 2. Site-specific priority: NDTV & General sites via Trafilatura + JSON-LD
    try:
        res = requests.get(url, headers=BROWSER_HEADERS, timeout=12)
        if res.status_code == 200:
            soup = BeautifulSoup(res.text, "html.parser")
            json_text = extract_from_json_ld(soup)
            if json_text:
                return json_text
            
            traf_text = trafilatura.extract(res.text)
            if traf_text:
                cleaned = clean_text(traf_text)
                if len(cleaned) >= 300:
                    return cleaned
    except Exception:
        pass

    # 3. Direct Trafilatura fetch
    text = extract_with_trafilatura(url)
    if text:
        return text

    # 4. Fallback to Jina Reader (with error checking)
    text = extract_with_jina(url)
    if text:
        return text

    return None