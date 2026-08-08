import json
import requests
import trafilatura

from bs4 import BeautifulSoup
from newspaper import Article, Config


# ============================================================
# Browser-like headers
# ============================================================

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
    "Accept-Encoding": "gzip, deflate",
    "Connection": "keep-alive",
}


# ============================================================
# Validate URL
# ============================================================

def is_valid_url(url):
    return (
        url.startswith("http://")
        or url.startswith("https://")
    )


# ============================================================
# Clean extracted text
# ============================================================

def clean_text(text):

    if not text:
        return ""

    lines = []

    for line in text.splitlines():

        line = " ".join(line.split())

        if len(line) > 30:
            lines.append(line)

    return "\n\n".join(lines)


# ============================================================
# METHOD 1
# Trafilatura
# ============================================================

def extract_with_trafilatura(url):

    try:

        downloaded = trafilatura.fetch_url(url)

        if not downloaded:
            return None, None

        # Extract article text
        text = trafilatura.extract(
            downloaded,
            include_comments=False,
            include_tables=False,
            include_links=False,
            favor_precision=True,
        )

        if text:

            text = clean_text(text)

            if len(text) >= 300:

                # Try metadata
                title = "News Article"

                metadata = trafilatura.extract(
                    downloaded,
                    output_format="json",
                    with_metadata=True,
                    include_comments=False,
                    include_tables=False,
                )

                if metadata:

                    try:

                        data = json.loads(metadata)

                        title = (
                            data.get("title")
                            or "News Article"
                        )

                    except Exception:
                        pass

                return title, text

    except Exception:
        pass

    return None, None


# ============================================================
# METHOD 2
# JSON-LD ArticleBody
# ============================================================

def extract_from_json_ld(soup):

    try:

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

            # Sometimes JSON-LD is a list
            if isinstance(data, list):

                objects = data

            else:

                objects = [data]

            for item in objects:

                if not isinstance(item, dict):
                    continue

                # Handle @graph
                if "@graph" in item:

                    graph = item["@graph"]

                    if isinstance(graph, list):
                        objects.extend(graph)

                article_type = item.get("@type", "")

                if isinstance(article_type, list):

                    is_article = any(
                        x in [
                            "NewsArticle",
                            "Article",
                            "ReportageNewsArticle",
                        ]
                        for x in article_type
                    )

                else:

                    is_article = article_type in [
                        "NewsArticle",
                        "Article",
                        "ReportageNewsArticle",
                    ]

                if not is_article:
                    continue

                title = (
                    item.get("headline")
                    or item.get("name")
                    or "News Article"
                )

                article_body = item.get("articleBody")

                if article_body:

                    text = clean_text(
                        article_body
                    )

                    if len(text) >= 300:

                        return title, text

    except Exception:
        pass

    return None, None


# ============================================================
# METHOD 3
# HTML article containers
# ============================================================

def extract_from_html(soup):

    try:

        # Remove unwanted elements
        for element in soup.find_all(
            [
                "script",
                "style",
                "noscript",
                "nav",
                "footer",
                "header",
                "aside",
                "form",
                "iframe",
            ]
        ):
            element.decompose()

        # Common article containers
        selectors = [
            "article",
            "[itemprop='articleBody']",
            ".article-body",
            ".article-content",
            ".articleBody",
            ".story-content",
            ".story-body",
            ".story__content",
            ".content-area",
            ".entry-content",
            ".post-content",
            ".article__content",
            ".story-detail",
        ]

        for selector in selectors:

            container = soup.select_one(selector)

            if not container:
                continue

            paragraphs = container.find_all("p")

            text = "\n".join(
                p.get_text(
                    " ",
                    strip=True
                )
                for p in paragraphs
            )

            text = clean_text(text)

            if len(text) >= 300:

                title = "News Article"

                h1 = soup.find("h1")

                if h1:

                    title = h1.get_text(
                        " ",
                        strip=True
                    )

                return title, text

    except Exception:
        pass

    return None, None


# ============================================================
# METHOD 4
# All paragraph extraction
# ============================================================

def extract_from_paragraphs(soup):

    try:

        paragraphs = soup.find_all("p")

        article_paragraphs = []

        for p in paragraphs:

            text = p.get_text(
                " ",
                strip=True
            )

            # Ignore very short text
            if len(text) >= 40:

                article_paragraphs.append(text)

        text = "\n".join(article_paragraphs)

        text = clean_text(text)

        if len(text) >= 300:

            title = "News Article"

            # Try H1
            h1 = soup.find("h1")

            if h1:

                title = h1.get_text(
                    " ",
                    strip=True
                )

            # Try page title
            elif soup.title:

                title = soup.title.get_text(
                    " ",
                    strip=True
                )

            return title, text

    except Exception:
        pass

    return None, None


# ============================================================
# METHOD 5
# Newspaper3k
# ============================================================

def extract_with_newspaper(url):

    try:

        config = Config()

        config.browser_user_agent = HEADERS[
            "User-Agent"
        ]

        config.request_timeout = 20

        article = Article(
            url,
            config=config
        )

        article.download()

        article.parse()

        text = clean_text(
            article.text
        )

        if len(text) >= 300:

            title = (
                article.title.strip()
                if article.title
                else "News Article"
            )

            return title, text

    except Exception:
        pass

    return None, None


# ============================================================
# MAIN EXTRACTION FUNCTION
# ============================================================

def extract_article(url):

    url = url.strip()

    # --------------------------------------------------------
    # Validate URL
    # --------------------------------------------------------

    if not url:

        raise Exception(
            "Please enter a news article URL."
        )

    if not is_valid_url(url):

        raise Exception(
            "Please enter a valid URL starting with "
            "http:// or https://"
        )


    # ========================================================
    # METHOD 1 - Trafilatura
    # ========================================================

    title, text = extract_with_trafilatura(url)

    if text:

        return title, text


    # ========================================================
    # Download page once for HTML-based methods
    # ========================================================

    soup = None

    try:

        response = requests.get(
            url,
            headers=HEADERS,
            timeout=20,
            allow_redirects=True,
        )

        if response.status_code == 200:

            soup = BeautifulSoup(
                response.text,
                "html.parser"
            )

    except Exception:
        soup = None


    # ========================================================
    # METHOD 2 - JSON-LD
    # ========================================================

    if soup:

        title, text = extract_from_json_ld(
            soup
        )

        if text:

            return title, text


    # ========================================================
    # METHOD 3 - Article HTML containers
    # ========================================================

    if soup:

        title, text = extract_from_html(
            soup
        )

        if text:

            return title, text


    # ========================================================
    # METHOD 4 - Paragraph extraction
    # ========================================================

    if soup:

        title, text = extract_from_paragraphs(
            soup
        )

        if text:

            return title, text


    # ========================================================
    # METHOD 5 - Newspaper3k
    # ========================================================

    title, text = extract_with_newspaper(
        url
    )

    if text:

        return title, text


    # ========================================================
    # Everything failed
    # ========================================================

    raise Exception(
        "Unable to extract the article from this website. "
        "The publisher may be blocking automated access."
    )