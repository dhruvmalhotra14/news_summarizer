import requests
import trafilatura

from bs4 import BeautifulSoup
from urllib.parse import urlparse


HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/151.0.0.0 Safari/537.36"
    ),
    "Accept": (
        "text/html,application/xhtml+xml,application/xml;"
        "q=0.9,image/avif,image/webp,*/*;q=0.8"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Connection": "keep-alive",
}


def clean_text(text):
    """Clean extracted article text."""

    if not text:
        return ""

    lines = []

    for line in text.splitlines():

        line = line.strip()

        if line:
            lines.append(line)

    text = "\n\n".join(lines)

    return text.strip()


def extract_with_requests(url):
    """Try direct extraction using requests + Trafilatura."""

    try:

        response = requests.get(
            url,
            headers=HEADERS,
            timeout=15,
            allow_redirects=True
        )

        if response.status_code != 200:
            return ""

        html = response.text


        # -------------------------------------------------
        # Try Trafilatura
        # -------------------------------------------------

        text = trafilatura.extract(
            html,
            include_comments=False,
            include_tables=False,
            include_links=False,
            favor_precision=True,
            favor_recall=True
        )

        if text:

            text = clean_text(text)

            if len(text) >= 500:
                return text


        # -------------------------------------------------
        # BeautifulSoup fallback
        # -------------------------------------------------

        soup = BeautifulSoup(
            html,
            "html.parser"
        )


        # Remove unwanted elements
        for element in soup([
            "script",
            "style",
            "noscript",
            "nav",
            "footer",
            "header",
            "aside",
            "form"
        ]):

            element.decompose()


        # -------------------------------------------------
        # Try article tag
        # -------------------------------------------------

        article = soup.find("article")

        if article:

            text = article.get_text(
                separator="\n",
                strip=True
            )

            text = clean_text(text)

            if len(text) >= 500:
                return text


        # -------------------------------------------------
        # Try paragraphs
        # -------------------------------------------------

        paragraphs = soup.find_all("p")

        text = "\n".join(
            p.get_text(
                " ",
                strip=True
            )
            for p in paragraphs
        )

        text = clean_text(text)

        if len(text) >= 500:
            return text


    except Exception:

        pass


    return ""


def extract_with_jina(url):
    """
    Fallback extractor using Jina Reader.

    Jina Reader converts a publicly accessible URL
    into clean, LLM-friendly text.
    """

    try:

        jina_url = (
            "https://r.jina.ai/" + url
        )

        response = requests.get(
            jina_url,
            headers={
                "User-Agent": "NewsSummarizer/1.0"
            },
            timeout=30
        )

        if response.status_code != 200:
            return ""

        text = response.text

        text = clean_text(text)

        if len(text) >= 500:
            return text


    except Exception:

        pass


    return ""


def extract_article(url):
    """
    Main article extraction function.

    Extraction order:

    1. Direct requests + Trafilatura
    2. BeautifulSoup fallback
    3. Jina Reader fallback
    """

    url = url.strip()

    if not url:
        return None


    # -------------------------------------------------
    # Basic URL validation
    # -------------------------------------------------

    parsed = urlparse(url)

    if parsed.scheme not in (
        "http",
        "https"
    ):

        return None


    # -------------------------------------------------
    # METHOD 1: Direct extraction
    # -------------------------------------------------

    text = extract_with_requests(url)

    if text:
        return text


    # -------------------------------------------------
    # METHOD 2: Jina Reader fallback
    # -------------------------------------------------

    text = extract_with_jina(url)

    if text:
        return text


    return None