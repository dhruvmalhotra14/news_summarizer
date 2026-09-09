import re
import requests
import trafilatura
from bs4 import BeautifulSoup

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/125.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.google.com/",
}

BLOCKED_PHRASES = [
    "403 forbidden",
    "access denied",
    "just a moment...",
    "enable javascript",
    "robot or human",
    "cloudflare",
    "subscribe to read",
]


def clean_text(text: str) -> str:
    if not text:
        return ""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    cleaned = "\n\n".join(lines).strip()

    lower = cleaned.lower()
    for phrase in BLOCKED_PHRASES:
        if phrase in lower and len(cleaned) < 500:
            return ""
    return cleaned


def extract_with_jina(url: str) -> str:
    """Bypasses Cloudflare/anti-bot protection using Jina proxy."""
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
    """Trafilatura's direct scraper."""
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


def extract_with_requests_soup(url: str) -> str:
    """Fallback manual parser."""
    try:
        res = requests.get(url, headers=HEADERS, timeout=12)
        if res.status_code != 200:
            return ""

        soup = BeautifulSoup(res.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form"]):
            tag.decompose()

        article = soup.find("article")
        if article:
            cleaned = clean_text(article.get_text(separator="\n", strip=True))
            if len(cleaned) >= 300:
                return cleaned

        paragraphs = [p.get_text(strip=True) for p in soup.find_all("p")]
        cleaned = clean_text("\n\n".join(paragraphs))
        if len(cleaned) >= 300:
            return cleaned
    except Exception:
        pass
    return ""


def extract_article(url: str) -> str | None:
    url = url.strip()
    if not url.startswith(("http://", "https://")):
        return None

    # Step 1: Try Jina Reader (handles Indian Express, Reuters, NDTV bot walls)
    text = extract_with_jina(url)
    if text:
        return text

    # Step 2: Try Trafilatura
    text = extract_with_trafilatura(url)
    if text:
        return text

    # Step 3: Try standard BeautifulSoup
    text = extract_with_requests_soup(url)
    if text:
        return text

    return None