import re
import requests
import trafilatura
from bs4 import BeautifulSoup
from urllib.parse import quote

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/125.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
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
    """Cleans up raw text and filters out common anti-bot blocker responses."""
    if not text:
        return ""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    cleaned = "\n\n".join(lines).strip()

    # If the text is merely an error message, drop it
    lower = cleaned.lower()
    for phrase in BLOCKED_PHRASES:
        if phrase in lower and len(cleaned) < 500:
            return ""

    return cleaned


def extract_with_jina(url: str) -> str:
    """
    Bypasses anti-bot walls (Reuters, Indian Express) 
    by routing through Jina's proxy reader.
    """
    try:
        jina_url = f"https://r.jina.ai/{url}"
        res = requests.get(
            jina_url,
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"},
            timeout=25,
        )
        if res.status_code == 200:
            cleaned = clean_text(res.text)
            if len(cleaned) >= 350:
                return cleaned
    except Exception:
        pass
    return ""


def extract_with_trafilatura(url: str) -> str:
    """Standard extraction using Trafilatura."""
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
                if len(cleaned) >= 350:
                    return cleaned
    except Exception:
        pass
    return ""


def extract_with_requests_soup(url: str) -> str:
    """Manual fallback extraction with custom headers."""
    try:
        res = requests.get(url, headers=HEADERS, timeout=15)
        if res.status_code != 200:
            return ""

        soup = BeautifulSoup(res.text, "html.parser")

        # Strip navigation, scripts, ads
        for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form"]):
            tag.decompose()

        article = soup.find("article")
        if article:
            text = article.get_text(separator="\n", strip=True)
            cleaned = clean_text(text)
            if len(cleaned) >= 350:
                return cleaned

        paragraphs = [p.get_text(strip=True) for p in soup.find_all("p")]
        text = "\n\n".join(paragraphs)
        cleaned = clean_text(text)
        if len(cleaned) >= 350:
            return cleaned
    except Exception:
        pass
    return ""


def extract_article(url: str) -> str | None:
    """
    Attempts extraction in order of anti-bot bypass resilience:
    1. Jina Reader (handles Reuters / Indian Express bot challenges)
    2. Trafilatura
    3. BeautifulSoup fallback
    """
    url = url.strip()
    if not url.startswith(("http://", "https://")):
        return None

    # Method 1: Jina Reader (Best for tough anti-bot sites)
    text = extract_with_jina(url)
    if text:
        return text

    # Method 2: Trafilatura
    text = extract_with_trafilatura(url)
    if text:
        return text

    # Method 3: Direct parsing
    text = extract_with_requests_soup(url)
    if text:
        return text

    return None