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


def extract_json_ld(soup):
    """
    Extract article text from JSON-LD structured data.
    Some news websites expose articleBody here.
    """

    try:
        import json

        scripts = soup.find_all(
            "script",
            type="application/ld+json"
        )

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

                # Direct articleBody
                article_body = item.get("articleBody")

                if article_body:

                    article_body = clean_text(
                        article_body
                    )

                    if len(article_body) >= 500:
                        return article_body

                # Sometimes JSON-LD contains @graph
                graph = item.get("@graph", [])

                if isinstance(graph, list):

                    for graph_item in graph:

                        if not isinstance(
                            graph_item,
                            dict
                        ):
                            continue

                        article_body = graph_item.get(
                            "articleBody"
                        )

                        if article_body:

                            article_body = clean_text(
                                article_body
                            )

                            if len(article_body) >= 500:
                                return article_body

    except Exception:
        pass

    return ""


def extract_site_specific(soup, url):
    """
    Site-specific extraction for websites
    that do not work reliably with generic extraction.
    """

    hostname = urlparse(url).netloc.lower()

    # Remove subdomain
    hostname = hostname.replace("www.", "")

    # =================================================
    # NDTV
    # =================================================

    if "ndtv.com" in hostname:

        selectors = [
            # Common NDTV article containers
            "div[class*='sp-cn']",
            "div[class*='story']",
            "div[class*='article']",
            "div[class*='story__content']",
            "div[class*='article__content']",
            "div[class*='content']",

            # Generic article
            "article"
        ]

        for selector in selectors:

            elements = soup.select(selector)

            for element in elements:

                paragraphs = element.find_all("p")

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

    # =================================================
    # REUTERS
    # =================================================

    if "reuters.com" in hostname:

        selectors = [
            "article",
            "div[data-testid='ArticleBody']",
            "div[class*='article-body']",
            "div[class*='ArticleBody']",
            "div[class*='article-body__content']"
        ]

        for selector in selectors:

            elements = soup.select(selector)

            for element in elements:

                paragraphs = element.find_all("p")

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

    return ""


def extract_with_requests(url):
    """Direct extraction using Requests + multiple methods."""

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

        if not html:
            return ""

        soup = BeautifulSoup(
            html,
            "html.parser"
        )

        # =================================================
        # METHOD 1: JSON-LD
        # =================================================

        text = extract_json_ld(soup)

        if text:
            return text

        # =================================================
        # METHOD 2: Trafilatura
        # =================================================

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

        # =================================================
        # METHOD 3: Site-specific extraction
        # =================================================

        text = extract_site_specific(
            soup,
            url
        )

        if text:
            return text

        # =================================================
        # Remove unwanted elements
        # =================================================

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

        # =================================================
        # METHOD 4: Article tag
        # =================================================

        article = soup.find("article")

        if article:

            text = article.get_text(
                separator="\n",
                strip=True
            )

            text = clean_text(text)

            if len(text) >= 500:
                return text

        # =================================================
        # METHOD 5: Paragraph extraction
        # =================================================

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

    1. JSON-LD
    2. Trafilatura
    3. NDTV / Reuters specific extraction
    4. BeautifulSoup
    5. Jina Reader
    """

    url = url.strip()

    if not url:
        return None

    # =================================================
    # URL VALIDATION
    # =================================================

    parsed = urlparse(url)

    if parsed.scheme not in (
        "http",
        "https"
    ):
        return None

    # =================================================
    # METHOD 1-5: DIRECT EXTRACTION
    # =================================================

    text = extract_with_requests(url)

    if text:
        return text

    # =================================================
    # FINAL FALLBACK: JINA
    # =================================================

    text = extract_with_jina(url)

    if text:
        return text

    return None