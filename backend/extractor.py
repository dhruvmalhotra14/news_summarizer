import requests
import trafilatura
from bs4 import BeautifulSoup
from urllib.parse import urlparse
import json
import re


# =========================================================
# HEADERS
# =========================================================

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
    "Referer": "https://www.google.com/",
    "Connection": "keep-alive",
}


# =========================================================
# CLEAN TEXT
# =========================================================

def clean_text(text):
    """Clean extracted article text."""

    if not text:
        return ""

    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n\n", text)

    lines = []

    for line in text.splitlines():

        line = line.strip()

        if not line:
            continue

        lower = line.lower()

        # Remove common blocking/error text
        if (
            "403 forbidden" in lower
            or "access denied" in lower
            or "request blocked" in lower
            or "checking your browser" in lower
            or "just a moment" in lower
        ):
            continue

        lines.append(line)

    return "\n\n".join(lines).strip()


# =========================================================
# VALIDATE ARTICLE
# =========================================================

def is_valid_article(text):
    """Check whether extracted text looks like article content."""

    if not text:
        return False

    text = clean_text(text)

    if len(text) < 250:
        return False

    lower = text.lower()

    blocked_messages = [
        "403 forbidden",
        "access denied",
        "request blocked",
        "captcha",
        "checking your browser",
        "just a moment",
    ]

    for message in blocked_messages:

        if message in lower and len(text) < 1000:
            return False

    return True


# =========================================================
# GET WEBPAGE
# =========================================================

def get_page(url):
    """Download webpage."""

    try:

        response = requests.get(
            url,
            headers=HEADERS,
            timeout=20,
            allow_redirects=True
        )

        return response

    except Exception:
        return None


# =========================================================
# TRAFILATURA
# =========================================================

def extract_with_trafilatura(html):
    """Extract article using Trafilatura."""

    try:

        text = trafilatura.extract(
            html,
            include_comments=False,
            include_tables=False,
            include_links=False,
            favor_precision=True,
            favor_recall=True
        )

        if is_valid_article(text):
            return clean_text(text)

        # Second attempt
        text = trafilatura.extract(
            html,
            include_comments=False,
            include_tables=False,
            include_links=False,
            favor_recall=True
        )

        if is_valid_article(text):
            return clean_text(text)

    except Exception:
        pass

    return ""


# =========================================================
# BEAUTIFULSOUP
# =========================================================

def extract_with_beautifulsoup(html):
    """Extract article using BeautifulSoup."""

    try:

        soup = BeautifulSoup(html, "html.parser")

        # Remove unnecessary elements
        for element in soup([
            "script",
            "style",
            "noscript",
            "nav",
            "footer",
            "header",
            "aside",
            "form",
            "iframe",
            "svg"
        ]):
            element.decompose()

        # -------------------------------------------------
        # ARTICLE TAG
        # -------------------------------------------------

        article = soup.find("article")

        if article:

            text = article.get_text(
                separator="\n",
                strip=True
            )

            text = clean_text(text)

            if is_valid_article(text):
                return text

        # -------------------------------------------------
        # COMMON ARTICLE CONTAINERS
        # -------------------------------------------------

        selectors = [
            "[class*='article-body']",
            "[class*='article-content']",
            "[class*='story-body']",
            "[class*='story-content']",
            "[class*='content-body']",
            "[class*='post-content']",
            "[class*='articleBody']",
            "[class*='articleBodyContent']",
            "[id*='article-body']",
            "[id*='article-content']",
            "[id*='story-body']",
            "[id*='content-body']",
        ]

        for selector in selectors:

            container = soup.select_one(selector)

            if container:

                text = container.get_text(
                    separator="\n",
                    strip=True
                )

                text = clean_text(text)

                if is_valid_article(text):
                    return text

        # -------------------------------------------------
        # PARAGRAPH FALLBACK
        # -------------------------------------------------

        paragraphs = soup.find_all("p")

        paragraph_text = []

        for paragraph in paragraphs:

            text = paragraph.get_text(
                " ",
                strip=True
            )

            if len(text) >= 40:
                paragraph_text.append(text)

        text = "\n\n".join(paragraph_text)

        text = clean_text(text)

        if is_valid_article(text):
            return text

    except Exception:
        pass

    return ""


# =========================================================
# JSON-LD
# =========================================================

def extract_from_json_ld(html):
    """
    Try extracting articleBody from JSON-LD metadata.
    """

    try:

        soup = BeautifulSoup(html, "html.parser")

        scripts = soup.find_all(
            "script",
            type="application/ld+json"
        )

        for script in scripts:

            try:

                data = json.loads(
                    script.string or script.get_text()
                )

            except Exception:
                continue

            objects = []

            if isinstance(data, dict):

                objects.append(data)

                if "@graph" in data:

                    graph = data["@graph"]

                    if isinstance(graph, list):
                        objects.extend(graph)

            elif isinstance(data, list):

                objects.extend(data)

            for item in objects:

                if not isinstance(item, dict):
                    continue

                article_body = item.get("articleBody")

                if article_body:

                    text = clean_text(article_body)

                    if is_valid_article(text):
                        return text

    except Exception:
        pass

    return ""


# =========================================================
# DIRECT EXTRACTION
# =========================================================

def extract_with_requests(url):
    """
    Try extracting article directly from publisher.
    """

    response = get_page(url)

    if response is None:
        return ""

    html = response.text

    if not html:
        return ""

    # -----------------------------------------------------
    # 1. Trafilatura
    # -----------------------------------------------------

    text = extract_with_trafilatura(html)

    if text:
        return text

    # -----------------------------------------------------
    # 2. JSON-LD
    # -----------------------------------------------------

    text = extract_from_json_ld(html)

    if text:
        return text

    # -----------------------------------------------------
    # 3. BeautifulSoup
    # -----------------------------------------------------

    text = extract_with_beautifulsoup(html)

    if text:
        return text

    return ""


# =========================================================
# JINA READER
# =========================================================

def extract_with_jina(url):
    """
    Use Jina Reader as a fallback for websites
    that block direct automated requests.
    """

    try:

        jina_url = "https://r.jina.ai/" + url

        response = requests.get(
            jina_url,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 "
                    "(Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 "
                    "(KHTML, like Gecko) "
                    "Chrome/151.0.0.0 Safari/537.36"
                ),
                "Accept": "text/plain,text/html,*/*",
            },
            timeout=45,
            allow_redirects=True
        )

        if response.status_code != 200:
            return ""

        text = clean_text(response.text)

        if is_valid_article(text):
            return text

    except Exception:
        pass

    return ""


# =========================================================
# DOMAIN
# =========================================================

def get_domain(url):

    try:

        domain = urlparse(url).netloc.lower()

        if domain.startswith("www."):
            domain = domain[4:]

        return domain

    except Exception:
        return ""


# =========================================================
# MAIN FUNCTION
# =========================================================

def extract_article(url):
    """
    Main article extraction function.

    Returns:
        article text as a string
    """

    url = url.strip()

    if not url:
        return None

    # -----------------------------------------------------
    # URL VALIDATION
    # -----------------------------------------------------

    parsed = urlparse(url)

    if parsed.scheme not in ("http", "https"):
        return None

    domain = get_domain(url)

    # -----------------------------------------------------
    # Publishers that frequently block direct requests
    # -----------------------------------------------------

    blocked_publishers = [
        "indianexpress.com",
        "ndtv.com",
        "hindustantimes.com",
        "timesofindia.indiatimes.com",
        "thehindu.com",
        "economictimes.indiatimes.com",
    ]

    is_blocked_publisher = any(
        publisher in domain
        for publisher in blocked_publishers
    )

    # -----------------------------------------------------
    # 1. JINA FIRST FOR BLOCKED PUBLISHERS
    # -----------------------------------------------------

    if is_blocked_publisher:

        text = extract_with_jina(url)

        if text:
            return text

    # -----------------------------------------------------
    # 2. DIRECT EXTRACTION
    # -----------------------------------------------------

    text = extract_with_requests(url)

    if text:
        return text

    # -----------------------------------------------------
    # 3. JINA FINAL FALLBACK
    # -----------------------------------------------------

    text = extract_with_jina(url)

    if text:
        return text

    # -----------------------------------------------------
    # FAILED
    # -----------------------------------------------------

    return None